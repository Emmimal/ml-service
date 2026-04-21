"""
api/schemas.py
--------------
Pydantic v2 models for request validation and response serialisation.

Why Pydantic matters here:
- Request payloads are validated before they reach any business logic.
  A malformed payload returns a structured 422 Unprocessable Entity with
  field-level error details — no custom error handling required.
- Response models act as a contract.  If predict.py ever returns an
  unexpected field, the response is still serialised exactly as declared.
- OpenAPI docs are generated automatically from these models, so the
  /docs endpoint always reflects the actual API surface.
"""

from typing import Dict, List, Optional
from pydantic import BaseModel, Field, field_validator, model_validator


# ── Request ────────────────────────────────────────────────────────────────────

class PredictRequest(BaseModel):
    """Single-sample inference request."""

    features: List[float] = Field(
        ...,
        description=(
            "30 numeric features in the order expected by the model. "
            "Values must be finite real numbers (no NaN, no ±inf)."
        ),
        examples=[
            [
                17.99, 10.38, 122.8, 1001.0, 0.1184,
                0.2776, 0.3001, 0.1471, 0.2419, 0.07871,
                1.095, 0.9053, 8.589, 153.4, 0.006399,
                0.04904, 0.05373, 0.01587, 0.03003, 0.006193,
                25.38, 17.33, 184.6, 2019.0, 0.1622,
                0.6656, 0.7119, 0.2654, 0.4601, 0.1189,
            ]
        ],
    )

    @field_validator("features")
    @classmethod
    def validate_features(cls, v: List[float]) -> List[float]:
        if len(v) != 30:
            raise ValueError(
                f"Exactly 30 features required, got {len(v)}."
            )
        import math
        bad = [i for i, x in enumerate(v) if not math.isfinite(x)]
        if bad:
            raise ValueError(
                f"Non-finite values at indices: {bad}. "
                "All features must be finite real numbers."
            )
        return v

    model_config = {"json_schema_extra": {"title": "Single prediction request"}}


class BatchPredictRequest(BaseModel):
    """Multi-sample batch inference request."""

    instances: List[List[float]] = Field(
        ...,
        min_length=1,
        max_length=512,
        description="List of feature vectors.  Max 512 instances per request.",
    )

    @model_validator(mode="after")
    def validate_all_instances(self) -> "BatchPredictRequest":
        import math
        for idx, row in enumerate(self.instances):
            if len(row) != 30:
                raise ValueError(
                    f"Instance {idx}: expected 30 features, got {len(row)}."
                )
            bad = [j for j, x in enumerate(row) if not math.isfinite(x)]
            if bad:
                raise ValueError(
                    f"Instance {idx}: non-finite values at feature indices {bad}."
                )
        return self


# ── Response ───────────────────────────────────────────────────────────────────

class ProbabilityMap(BaseModel):
    class_0: float = Field(..., description="Probability of class 0 (malignant)")
    class_1: float = Field(..., description="Probability of class 1 (benign)")


class PredictResponse(BaseModel):
    label: int = Field(..., description="Predicted class index (0 or 1)")
    probabilities: ProbabilityMap
    model_version: str
    n_features_in: int = Field(..., description="Number of features received")

    model_config = {"protected_namespaces": ()}


class BatchPredictResponse(BaseModel):
    predictions: List[PredictResponse]
    model_version: str
    n_instances: int

    model_config = {"protected_namespaces": ()}


# ── Health / Ops ───────────────────────────────────────────────────────────────

class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_version: str
    uptime_seconds: Optional[float] = None

    model_config = {"protected_namespaces": ()}


class ReadinessResponse(BaseModel):
    status: str
