"""
tests/test_predict.py
---------------------
Unit tests for ModelPredictor, isolated from the FastAPI layer.

These tests verify the predictor's contract independently so failures
are easier to localise than they would be in a full integration test.
"""

import pytest
import numpy as np

from model.predict import ModelPredictor


VALID_FEATURES = [
    17.99, 10.38, 122.8, 1001.0, 0.1184,
    0.2776, 0.3001, 0.1471, 0.2419, 0.07871,
    1.095, 0.9053, 8.589, 153.4, 0.006399,
    0.04904, 0.05373, 0.01587, 0.03003, 0.006193,
    25.38, 17.33, 184.6, 2019.0, 0.1622,
    0.6656, 0.7119, 0.2654, 0.4601, 0.1189,
]


@pytest.fixture(scope="module")
def loaded_predictor():
    """Load once per module — artifact loading is slow relative to prediction."""
    p = ModelPredictor()
    p.load()
    return p


# ── Load behaviour ─────────────────────────────────────────────────────────────

class TestLoad:
    def test_is_loaded_after_load(self, loaded_predictor):
        assert loaded_predictor.is_loaded is True

    def test_unloaded_predictor_raises_on_predict(self):
        p = ModelPredictor()
        with pytest.raises(RuntimeError, match="Call .load()"):
            p.predict(VALID_FEATURES)

    def test_missing_artifact_dir_raises(self, tmp_path):
        from pathlib import Path
        p = ModelPredictor(artifact_dir=tmp_path / "nonexistent")
        with pytest.raises(FileNotFoundError):
            p.load()

    def test_feature_names_populated(self, loaded_predictor):
        assert len(loaded_predictor.feature_names) == 30

    def test_expected_n_features(self, loaded_predictor):
        assert loaded_predictor.expected_n_features == 30


# ── Single prediction ──────────────────────────────────────────────────────────

class TestPredict:
    def test_returns_dict_with_required_keys(self, loaded_predictor):
        result = loaded_predictor.predict(VALID_FEATURES)
        assert set(result.keys()) == {"label", "probabilities", "n_features_in"}

    def test_label_is_binary(self, loaded_predictor):
        result = loaded_predictor.predict(VALID_FEATURES)
        assert result["label"] in (0, 1)

    def test_probabilities_sum_to_one(self, loaded_predictor):
        result = loaded_predictor.predict(VALID_FEATURES)
        total = result["probabilities"]["class_0"] + result["probabilities"]["class_1"]
        assert abs(total - 1.0) < 1e-5

    def test_n_features_in_echoed_correctly(self, loaded_predictor):
        result = loaded_predictor.predict(VALID_FEATURES)
        assert result["n_features_in"] == 30

    def test_wrong_feature_count_raises(self, loaded_predictor):
        with pytest.raises(ValueError, match="Expected 30 features"):
            loaded_predictor.predict([1.0, 2.0, 3.0])

    def test_nan_raises(self, loaded_predictor):
        bad = VALID_FEATURES.copy()
        bad[5] = float("nan")
        with pytest.raises(ValueError, match="NaN or infinite"):
            loaded_predictor.predict(bad)

    def test_inf_raises(self, loaded_predictor):
        bad = VALID_FEATURES.copy()
        bad[0] = float("inf")
        with pytest.raises(ValueError, match="NaN or infinite"):
            loaded_predictor.predict(bad)

    def test_deterministic_output(self, loaded_predictor):
        """Same input must always return the same label and probabilities."""
        r1 = loaded_predictor.predict(VALID_FEATURES)
        r2 = loaded_predictor.predict(VALID_FEATURES)
        assert r1["label"] == r2["label"]
        assert r1["probabilities"] == r2["probabilities"]

    def test_numpy_list_accepted(self, loaded_predictor):
        """Predictor must handle numpy arrays converted to list."""
        features_np = np.array(VALID_FEATURES).tolist()
        result = loaded_predictor.predict(features_np)
        assert result["label"] in (0, 1)


# ── Batch prediction ───────────────────────────────────────────────────────────

class TestPredictBatch:
    def test_batch_returns_correct_length(self, loaded_predictor):
        batch = [VALID_FEATURES] * 5
        results = loaded_predictor.predict_batch(batch)
        assert len(results) == 5

    def test_batch_single_row(self, loaded_predictor):
        results = loaded_predictor.predict_batch([VALID_FEATURES])
        assert len(results) == 1
        assert results[0]["label"] in (0, 1)

    def test_batch_matches_single_predict(self, loaded_predictor):
        """
        Batch output for a single instance must match the output of
        .predict() for the same instance — guarantees consistency
        between the two code paths.
        """
        single = loaded_predictor.predict(VALID_FEATURES)
        batch = loaded_predictor.predict_batch([VALID_FEATURES])
        assert single["label"] == batch[0]["label"]
        assert single["probabilities"] == batch[0]["probabilities"]

    def test_batch_wrong_feature_count_raises(self, loaded_predictor):
        with pytest.raises(ValueError):
            loaded_predictor.predict_batch([[1.0, 2.0]])

    def test_batch_2d_shape_required(self, loaded_predictor):
        with pytest.raises((ValueError, Exception)):
            # Flat list instead of list-of-lists
            loaded_predictor.predict_batch(VALID_FEATURES)  # type: ignore
