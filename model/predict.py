"""
model/predict.py
----------------
Wraps artifact loading and inference in a single, reusable class.

Design decisions worth calling out:

1.  Artifacts are loaded once at class instantiation (or via .load()),
    not on every call to .predict().  Even a lightweight sklearn model
    can add 200-400 ms per request if deserialized on demand under load.

2.  The preprocessor is kept separate from the model.  This mirrors
    train.py and makes training-serving skew immediately visible:
    if the serving transform ever diverges from the training transform
    you will see it in the artifact diff, not buried in a Pipeline object.

3.  Probabilities are always returned alongside the label.  Downstream
    systems often want to apply their own decision thresholds rather than
    the default 0.5 cutoff, and adding this later requires a redeployment.
"""

import json
import logging
from pathlib import Path
from typing import List

import joblib
import numpy as np

logger = logging.getLogger(__name__)

ARTIFACT_DIR = Path(__file__).parent / "artifacts"


class ModelPredictor:
    """
    Load sklearn artifacts from disk and serve predictions.

    Typical lifecycle
    -----------------
    predictor = ModelPredictor()
    predictor.load()                        # once, at startup
    result = predictor.predict(features)    # many times, per request
    """

    def __init__(self, artifact_dir: Path = ARTIFACT_DIR) -> None:
        self._artifact_dir = artifact_dir
        self._preprocessor = None
        self._model = None
        self._metadata: dict = {}

    # ── Startup ────────────────────────────────────────────────────────────────

    def load(self) -> None:
        """
        Deserialize preprocessor, model, and metadata from disk.

        Raises FileNotFoundError if artifacts are missing so the caller
        (e.g. the FastAPI lifespan hook) can fail loudly at startup
        rather than silently at the first prediction request.
        """
        preprocessor_path = self._artifact_dir / "preprocessor.pkl"
        model_path = self._artifact_dir / "model.pkl"
        metadata_path = self._artifact_dir / "metadata.json"

        for path in (preprocessor_path, model_path):
            if not path.exists():
                raise FileNotFoundError(
                    f"Artifact not found: {path}\n"
                    "Run `python -m model.train` to generate artifacts."
                )

        self._preprocessor = joblib.load(preprocessor_path)
        self._model = joblib.load(model_path)

        if metadata_path.exists():
            with open(metadata_path) as f:
                self._metadata = json.load(f)

        logger.info(
            "Loaded %s  |  expects %d features  |  CV ROC-AUC %.4f",
            self._metadata.get("model_class", "model"),
            self._metadata.get("n_features", "?"),
            self._metadata.get("cv_roc_auc_mean", 0.0),
        )

    # ── Properties ─────────────────────────────────────────────────────────────

    @property
    def is_loaded(self) -> bool:
        return self._model is not None and self._preprocessor is not None

    @property
    def expected_n_features(self) -> int:
        return self._metadata.get("n_features", 30)

    @property
    def feature_names(self) -> List[str]:
        return self._metadata.get("feature_names", [])

    # ── Inference ──────────────────────────────────────────────────────────────

    def predict(self, features: List[float]) -> dict:
        """
        Run a single synchronous prediction.

        Parameters
        ----------
        features : list of float
            Raw feature values in the order the model was trained on.

        Returns
        -------
        dict
            label          : predicted class (int)
            probabilities  : {class_0: float, class_1: float}
            n_features_in  : sanity-check field echoed back to caller
        """
        if not self.is_loaded:
            raise RuntimeError(
                "Predictor not ready. Call .load() before .predict()."
            )

        arr = np.array(features, dtype=np.float64).reshape(1, -1)

        if arr.shape[1] != self.expected_n_features:
            raise ValueError(
                f"Expected {self.expected_n_features} features, "
                f"got {arr.shape[1]}."
            )

        if not np.isfinite(arr).all():
            raise ValueError(
                "Feature vector contains NaN or infinite values."
            )

        scaled = self._preprocessor.transform(arr)
        label = int(self._model.predict(scaled)[0])
        proba = self._model.predict_proba(scaled)[0]

        return {
            "label": label,
            "probabilities": {
                "class_0": round(float(proba[0]), 6),
                "class_1": round(float(proba[1]), 6),
            },
            "n_features_in": arr.shape[1],
        }

    def predict_batch(self, batch: List[List[float]]) -> List[dict]:
        """
        Run predictions over a batch of feature vectors.

        Batching amortizes the overhead of the preprocessor transform
        and is significantly faster than calling predict() in a loop
        when serving large offline scoring jobs.
        """
        if not self.is_loaded:
            raise RuntimeError(
                "Predictor not ready. Call .load() before .predict_batch()."
            )

        arr = np.array(batch, dtype=np.float64)

        if arr.ndim != 2 or arr.shape[1] != self.expected_n_features:
            raise ValueError(
                f"Batch must have shape (N, {self.expected_n_features}), "
                f"got {arr.shape}."
            )

        scaled = self._preprocessor.transform(arr)
        labels = self._model.predict(scaled).tolist()
        probas = self._model.predict_proba(scaled).tolist()

        return [
            {
                "label": int(labels[i]),
                "probabilities": {
                    "class_0": round(probas[i][0], 6),
                    "class_1": round(probas[i][1], 6),
                },
            }
            for i in range(len(labels))
        ]
