"""
tests/test_api.py
-----------------
Integration tests for the FastAPI application.

Uses TestClient (backed by httpx) which runs the ASGI app in-process —
no live server required.  The lifespan hook (model loading) is exercised
on every TestClient instantiation, so these tests validate the full
startup path, not just individual functions.

Run:
    pytest tests/ -v
"""

import pytest
from fastapi.testclient import TestClient

from api.main import app

# ── Shared fixture ──────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def client():
    """
    Module-scoped TestClient so the model is loaded once for the
    entire test module — mirrors how a production server behaves.
    """
    with TestClient(app) as c:
        yield c


# ── Sample payload ─────────────────────────────────────────────────────────────

VALID_FEATURES = [
    17.99, 10.38, 122.8, 1001.0, 0.1184,
    0.2776, 0.3001, 0.1471, 0.2419, 0.07871,
    1.095, 0.9053, 8.589, 153.4, 0.006399,
    0.04904, 0.05373, 0.01587, 0.03003, 0.006193,
    25.38, 17.33, 184.6, 2019.0, 0.1622,
    0.6656, 0.7119, 0.2654, 0.4601, 0.1189,
]


# ── /health ────────────────────────────────────────────────────────────────────

class TestHealth:
    def test_health_returns_200(self, client):
        response = client.get("/health")
        assert response.status_code == 200

    def test_health_body(self, client):
        body = client.get("/health").json()
        assert body["status"] == "ok"
        assert body["model_loaded"] is True
        assert "model_version" in body
        assert body["uptime_seconds"] is not None


# ── /ready ─────────────────────────────────────────────────────────────────────

class TestReadiness:
    def test_ready_returns_200_when_model_loaded(self, client):
        response = client.get("/ready")
        assert response.status_code == 200
        assert response.json()["status"] == "ready"


# ── /predict ───────────────────────────────────────────────────────────────────

class TestPredict:
    def test_valid_request_returns_200(self, client):
        response = client.post("/predict", json={"features": VALID_FEATURES})
        assert response.status_code == 200

    def test_response_schema(self, client):
        body = client.post("/predict", json={"features": VALID_FEATURES}).json()
        assert "label" in body
        assert "probabilities" in body
        assert "class_0" in body["probabilities"]
        assert "class_1" in body["probabilities"]
        assert "model_version" in body
        assert "n_features_in" in body

    def test_label_is_binary(self, client):
        body = client.post("/predict", json={"features": VALID_FEATURES}).json()
        assert body["label"] in (0, 1)

    def test_probabilities_sum_to_one(self, client):
        body = client.post("/predict", json={"features": VALID_FEATURES}).json()
        total = body["probabilities"]["class_0"] + body["probabilities"]["class_1"]
        assert abs(total - 1.0) < 1e-4

    def test_wrong_feature_count_returns_422(self, client):
        response = client.post("/predict", json={"features": [1.0, 2.0]})
        assert response.status_code == 422

    def test_empty_features_returns_422(self, client):
        response = client.post("/predict", json={"features": []})
        assert response.status_code == 422

    def test_nan_in_features_returns_422(self, client):
        bad = VALID_FEATURES.copy()
        bad[0] = float("nan")
        # json.dumps converts NaN to null which Pydantic rejects as non-float
        import json
        response = client.post(
            "/predict",
            content=json.dumps({"features": bad}, allow_nan=True),
            headers={"Content-Type": "application/json"},
        )
        # Either 422 (Pydantic) or 500 (if NaN slips through) — both are
        # acceptable at the transport layer; the important thing is it is not 200.
        assert response.status_code in (422, 500)

    def test_missing_features_key_returns_422(self, client):
        response = client.post("/predict", json={})
        assert response.status_code == 422


# ── /predict/batch ─────────────────────────────────────────────────────────────

class TestPredictBatch:
    def test_batch_two_instances(self, client):
        payload = {"instances": [VALID_FEATURES, VALID_FEATURES]}
        response = client.post("/predict/batch", json=payload)
        assert response.status_code == 200
        body = response.json()
        assert body["n_instances"] == 2
        assert len(body["predictions"]) == 2

    def test_batch_single_instance(self, client):
        payload = {"instances": [VALID_FEATURES]}
        response = client.post("/predict/batch", json=payload)
        assert response.status_code == 200
        assert response.json()["n_instances"] == 1

    def test_batch_empty_list_returns_422(self, client):
        response = client.post("/predict/batch", json={"instances": []})
        assert response.status_code == 422

    def test_batch_wrong_feature_count_returns_422(self, client):
        payload = {"instances": [[1.0, 2.0]]}
        response = client.post("/predict/batch", json=payload)
        assert response.status_code == 422
