"""
model/train.py
--------------
Train a GradientBoostingClassifier on the UCI Breast Cancer dataset and
persist the preprocessor and model as separate joblib artifacts.

Keeping them separate matters in production: you can inspect, benchmark,
or swap the preprocessor without touching the model weights, and you get
an exact record of what transformation the serving code must reproduce.

Usage:
    python -m model.train

Output:
    model/artifacts/preprocessor.pkl
    model/artifacts/model.pkl
    model/artifacts/metadata.json
"""

import json
import logging
import time
from pathlib import Path

import joblib
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import (
    classification_report,
    roc_auc_score,
    average_precision_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

ARTIFACT_DIR = Path(__file__).parent / "artifacts"
ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

# ── Hyperparameters ────────────────────────────────────────────────────────────
# These are deliberately conservative for a tutorial baseline.
# In a real system these would come from a config file or experiment tracker.
MODEL_CONFIG = {
    "n_estimators": 200,
    "learning_rate": 0.05,
    "max_depth": 4,
    "subsample": 0.8,
    "random_state": 42,
}
RANDOM_STATE = 42
TEST_SIZE = 0.20


def evaluate(model, X: np.ndarray, y: np.ndarray, split_name: str) -> dict:
    """Return a dict of evaluation metrics for a given split."""
    y_pred = model.predict(X)
    y_proba = model.predict_proba(X)[:, 1]

    report = classification_report(y, y_pred, output_dict=True)
    metrics = {
        "split": split_name,
        "accuracy": round(report["accuracy"], 4),
        "precision_macro": round(report["macro avg"]["precision"], 4),
        "recall_macro": round(report["macro avg"]["recall"], 4),
        "f1_macro": round(report["macro avg"]["f1-score"], 4),
        "roc_auc": round(roc_auc_score(y, y_proba), 4),
        "avg_precision": round(average_precision_score(y, y_proba), 4),
    }
    logger.info("[%s] %s", split_name, metrics)
    return metrics


def train_and_save() -> None:
    logger.info("Loading breast cancer dataset...")
    data = load_breast_cancer()
    X, y = data.data, data.target
    feature_names = list(data.feature_names)

    logger.info("Dataset shape: %s  |  Class balance: %s", X.shape, np.bincount(y))

    # ── Train / test split ─────────────────────────────────────────────────────
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    # ── Preprocessing ──────────────────────────────────────────────────────────
    # StandardScaler is fit on training data only.
    # This is the exact object that must be used at serve time — which is
    # why we persist it separately rather than burying it inside a Pipeline.
    preprocessor = StandardScaler()
    X_train_scaled = preprocessor.fit_transform(X_train)
    X_test_scaled = preprocessor.transform(X_test)

    # ── Cross-validation ───────────────────────────────────────────────────────
    logger.info("Running 5-fold stratified cross-validation...")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    cv_scores = cross_val_score(
        GradientBoostingClassifier(**MODEL_CONFIG),
        X_train_scaled,
        y_train,
        cv=cv,
        scoring="roc_auc",
        n_jobs=-1,
    )
    logger.info(
        "CV ROC-AUC: %.4f ± %.4f", cv_scores.mean(), cv_scores.std()
    )

    # ── Final training ─────────────────────────────────────────────────────────
    logger.info("Training final model on full training split...")
    t0 = time.perf_counter()
    model = GradientBoostingClassifier(**MODEL_CONFIG)
    model.fit(X_train_scaled, y_train)
    train_duration_s = round(time.perf_counter() - t0, 2)
    logger.info("Training completed in %.2fs", train_duration_s)

    train_metrics = evaluate(model, X_train_scaled, y_train, "train")
    test_metrics = evaluate(model, X_test_scaled, y_test, "test")

    print("\n── Held-out Test Set ──────────────────────────────")
    print(classification_report(y_test, model.predict(X_test_scaled),
                                target_names=data.target_names))

    # ── Persist artifacts ──────────────────────────────────────────────────────
    joblib.dump(preprocessor, ARTIFACT_DIR / "preprocessor.pkl")
    joblib.dump(model, ARTIFACT_DIR / "model.pkl")
    logger.info("Artifacts saved to %s", ARTIFACT_DIR)

    # ── Save metadata ──────────────────────────────────────────────────────────
    metadata = {
        "model_class": type(model).__name__,
        "model_config": MODEL_CONFIG,
        "feature_names": feature_names,
        "n_features": len(feature_names),
        "dataset": "sklearn.datasets.load_breast_cancer",
        "train_samples": len(X_train),
        "test_samples": len(X_test),
        "train_duration_s": train_duration_s,
        "cv_roc_auc_mean": round(float(cv_scores.mean()), 4),
        "cv_roc_auc_std": round(float(cv_scores.std()), 4),
        "train_metrics": train_metrics,
        "test_metrics": test_metrics,
    }
    with open(ARTIFACT_DIR / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    logger.info("Metadata written to %s", ARTIFACT_DIR / "metadata.json")
    logger.info("Done.")


if __name__ == "__main__":
    train_and_save()
