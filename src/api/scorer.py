"""
Scoring pipeline.

Loads all model artefacts once at startup and exposes a single
``score_transaction`` function that the API calls per request.
Model loading is expensive so it happens at module import time,
not on every request.
"""

import logging
import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import shap

from src.features.build_features import build_features

logger = logging.getLogger(__name__)

# Model paths are configurable via environment variables so the
# same Docker image can point to different artefact locations
# without rebuilding.
MODELS_DIR = Path(os.getenv("MODELS_DIR", "src/models"))

W_SUPERVISED = 0.7
W_ANOMALY = 0.3

BAND_THRESHOLDS = [
    (0.90, "Critical", "Block / escalate"),
    (0.60, "High",     "Manual review"),
    (0.30, "Medium",   "Monitor"),
    (0.00, "Low",      "Allow"),
]

VALID_TYPES = {"PAYMENT", "TRANSFER", "CASH_OUT", "CASH_IN", "DEBIT"}


def _load_models() -> dict:
    """Load all artefacts from disk and return them in a dictionary.

    Raises
    ------
    FileNotFoundError
        If any required artefact is missing from MODELS_DIR.
    """
    required = [
        "scaler.pkl",
        "feature_columns.pkl",
        "xgb_baseline.pkl",
        "calibrator.pkl",
        "isolation_forest.pkl",
    ]
    for name in required:
        path = MODELS_DIR / name
        if not path.exists():
            raise FileNotFoundError(
                f"Required model artefact not found: {path}. "
                "Run notebooks 02–05 to generate all artefacts."
            )

    scaler = joblib.load(MODELS_DIR / "scaler.pkl")
    feature_columns = joblib.load(MODELS_DIR / "feature_columns.pkl")
    xgb_model = joblib.load(MODELS_DIR / "xgb_baseline.pkl")
    calibrator = None
    iso_model = joblib.load(MODELS_DIR / "isolation_forest.pkl")

    # TreeExplainer is initialised once on the base XGBoost model.
    # Using the base model (not the calibrator wrapper) because
    # TreeExplainer requires direct access to the tree structure.
    explainer = shap.TreeExplainer(xgb_model)

    logger.info("All model artefacts loaded from %s", MODELS_DIR)
    return {
        "scaler": scaler,
        "feature_columns": feature_columns,
        "xgb_model": xgb_model,
        "calibrator": calibrator,
        "iso_model": iso_model,
        "explainer": explainer,
    }


# Load once at module import time.
_models = _load_models()


def _assign_band(score: float) -> tuple[str, str]:
    """Return the risk band and recommended action for a given score."""
    for threshold, band, action in BAND_THRESHOLDS:
        if score >= threshold:
            return band, action
    return "Low", "Allow"


def _normalise_iso_score(raw_score: float, iso_model) -> float:
    """Normalise a single Isolation Forest score to [0, 1].

    The Isolation Forest decision_function returns negative values for
    anomalies and positive for inliers. We flip and scale using the
    model's offset so higher values mean more anomalous, consistent
    with the XGBoost probability direction.
    """
    # offset_ is the average anomaly score over the training set.
    # Subtracting it centres the scores around zero, then we clip
    # and rescale to [0, 1].
    centred = -(raw_score - iso_model.detector_.offset_)
    return float(np.clip((centred + 0.5), 0, 1))


def score_transaction(transaction: dict) -> dict:
    """Run the full scoring pipeline for a single transaction.

    Parameters
    ----------
    transaction : dict
        Raw transaction fields matching the TransactionRequest schema.

    Returns
    -------
    dict
        Scoring result matching the ScoreResponse schema.
    """
    scaler = _models["scaler"]
    feature_columns = _models["feature_columns"]
    xgb_model = _models["xgb_model"]
    iso_model = _models["iso_model"]
    explainer = _models["explainer"]

    # Build a single-row DataFrame so build_features works without
    # modification — it was designed for batch DataFrames.
    raw_df = pd.DataFrame([transaction])

    # One-hot encode all known types so missing dummies are filled
    # with zeros rather than dropped, keeping the column order stable.
    features_df = build_features(raw_df)

    # Align columns to the training set order, filling any missing
    # one-hot columns with 0.
    features_df = features_df.reindex(columns=feature_columns, fill_value=0)

    # Scale using the training-fit RobustScaler.
    scaled = pd.DataFrame(
        scaler.transform(features_df),
        columns=feature_columns,
    )

    # Calibrated XGBoost probability.
    xgb_prob = float(xgb_model.predict_proba(scaled.values)[:, 1][0])

    # Isolation Forest anomaly score, normalised to [0, 1].
    iso_raw = float(iso_model.decision_function(scaled.values)[0])
    iso_norm = _normalise_iso_score(iso_raw, iso_model)

    # Combined risk score.
    risk_score = W_SUPERVISED * xgb_prob + W_ANOMALY * iso_norm
    risk_band, action = _assign_band(risk_score)

    # SHAP top-5 feature contributions.
    shap_values = explainer.shap_values(scaled)
    contributions = sorted(
        [
            {
                "feature": col,
                "value": float(scaled[col].iloc[0]),
                "shap_contribution": float(shap_values[0][i]),
            }
            for i, col in enumerate(feature_columns)
        ],
        key=lambda x: abs(x["shap_contribution"]),
        reverse=True,
    )[:5]

    return {
        "xgb_probability": xgb_prob,
        "anomaly_score": iso_norm,
        "risk_score": risk_score,
        "risk_band": risk_band,
        "action": action,
        "top_features": contributions,
    }
