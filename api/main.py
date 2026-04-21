"""
api/main.py
-----------
Production-grade FastAPI application for ML model serving.

Key design choices explained inline:

- Lifespan context manager (not deprecated @app.on_event)
  loads artifacts exactly once at startup and is visible in
  the OpenAPI docs.

- A single ModelPredictor instance is shared across all requests
  via module-level instantiation.  FastAPI is async but our sklearn
  predict calls are synchronous — they run in the event loop thread.
  For CPU-heavy models consider wrapping .predict() in
  asyncio.run_in_executor to avoid blocking the loop.

- /health vs /ready serve distinct purposes:
    /health  → liveness  (is the process alive?)
    /ready   → readiness (is the model loaded and accepting traffic?)
  Kubernetes uses both probes separately.

- The request logging middleware records per-request latency so you
  can spot slow predictions without an external APM tool during early
  development.

- MODEL_VERSION is injected via environment variable so the same
  container image can serve different versions without a rebuild.
"""

import logging
import os
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request, status
from fastapi.responses import JSONResponse

from api.schemas import (
    BatchPredictRequest,
    BatchPredictResponse,
    HealthResponse,
    PredictRequest,
    PredictResponse,
    ProbabilityMap,
    ReadinessResponse,
)
from model.predict import ModelPredictor

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

MODEL_VERSION = os.getenv("MODEL_VERSION", "1.0.0")

# Module-level predictor — loaded once, shared across all requests.
predictor = ModelPredictor()
_start_time: float = 0.0


# ── Lifespan ───────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Startup: load model artifacts.
    Shutdown: nothing to release for sklearn, but the hook is here
              for future use (e.g. closing a DB connection pool).
    """
    global _start_time
    _start_time = time.time()
    logger.info("Starting up  |  model version: %s", MODEL_VERSION)
    try:
        predictor.load()
        logger.info("Model loaded successfully.")
    except FileNotFoundError as exc:
        # Surface the error clearly at startup rather than at first request.
        logger.critical("Failed to load model: %s", exc)
        raise
    yield
    logger.info("Shutting down.")


# ── App ────────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="ML Inference API",
    description=(
        "Production-grade REST API for serving a scikit-learn "
        "GradientBoostingClassifier trained on the UCI Breast Cancer dataset.\n\n"
        "Part of the *Production ML Engineering* series at "
        "[EmiTechLogic](https://emitechlogic.com/machine-learning-production-pipeline/)."
    ),
    version=MODEL_VERSION,
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)


# ── Middleware ─────────────────────────────────────────────────────────────────

@app.middleware("http")
async def log_request_timing(request: Request, call_next):
    """Log method, path, status code, and wall-clock latency for every request."""
    t0 = time.perf_counter()
    response = await call_next(request)
    elapsed_ms = (time.perf_counter() - t0) * 1000
    logger.info(
        "%s  %-30s  →  %d  (%.1f ms)",
        request.method,
        request.url.path,
        response.status_code,
        elapsed_ms,
    )
    return response


# ── Exception handlers ─────────────────────────────────────────────────────────

@app.exception_handler(ValueError)
async def value_error_handler(request: Request, exc: ValueError):
    """Surface validation errors from the predictor as 422 responses."""
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={"detail": str(exc)},
    )


# ── Ops endpoints ──────────────────────────────────────────────────────────────

@app.get(
    "/health",
    response_model=HealthResponse,
    tags=["Ops"],
    summary="Liveness probe",
)
def health_check() -> HealthResponse:
    """
    Returns 200 as long as the process is alive.
    Kubernetes uses this to decide whether to restart the container.
    """
    return HealthResponse(
        status="ok",
        model_loaded=predictor.is_loaded,
        model_version=MODEL_VERSION,
        uptime_seconds=round(time.time() - _start_time, 1) if _start_time else None,
    )


@app.get(
    "/ready",
    response_model=ReadinessResponse,
    tags=["Ops"],
    summary="Readiness probe",
    responses={503: {"description": "Model not yet loaded"}},
)
def readiness_check() -> ReadinessResponse:
    """
    Returns 200 only when the model is loaded and ready to serve.
    Kubernetes uses this to decide whether to route traffic to this pod.
    A 503 here means the pod is alive but not yet ready — do not restart it.
    """
    if not predictor.is_loaded:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model artifacts not yet loaded.",
        )
    return ReadinessResponse(status="ready")


# ── Inference endpoints ────────────────────────────────────────────────────────

@app.post(
    "/predict",
    response_model=PredictResponse,
    tags=["Inference"],
    summary="Single-sample prediction",
    status_code=status.HTTP_200_OK,
)
def predict(request: PredictRequest) -> PredictResponse:
    """
    Accepts a 30-element feature vector and returns a class label plus
    per-class probabilities.

    The response includes `model_version` so callers can detect when
    a new model has been promoted without querying the /health endpoint.
    """
    if not predictor.is_loaded:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not ready.",
        )

    result = predictor.predict(request.features)

    return PredictResponse(
        label=result["label"],
        probabilities=ProbabilityMap(**result["probabilities"]),
        model_version=MODEL_VERSION,
        n_features_in=result["n_features_in"],
    )


@app.post(
    "/predict/batch",
    response_model=BatchPredictResponse,
    tags=["Inference"],
    summary="Batch prediction (up to 512 instances)",
    status_code=status.HTTP_200_OK,
)
def predict_batch(request: BatchPredictRequest) -> BatchPredictResponse:
    """
    Accepts up to 512 feature vectors in a single request and returns
    a prediction for each.

    Batching amortizes preprocessing overhead and is substantially
    faster than N sequential /predict calls for offline scoring jobs.
    """
    if not predictor.is_loaded:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not ready.",
        )

    results = predictor.predict_batch(request.instances)

    predictions = [
        PredictResponse(
            label=r["label"],
            probabilities=ProbabilityMap(**r["probabilities"]),
            model_version=MODEL_VERSION,
            n_features_in=len(request.instances[i]),
        )
        for i, r in enumerate(results)
    ]

    return BatchPredictResponse(
        predictions=predictions,
        model_version=MODEL_VERSION,
        n_instances=len(predictions),
    )
