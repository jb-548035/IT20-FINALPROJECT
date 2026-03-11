"""
model_loader.py
---------------
Responsible for:
  1. Loading the trained ML model, scaler, and encoders from disk
  2. Preprocessing a new customer record to match the training pipeline
  3. Running inference and returning predictions + probabilities

This file is the "bridge" between the raw API input and the trained model.
It mirrors the exact preprocessing steps from data_preprocessing.ipynb
so that predictions are consistent with how the model was trained.
"""

import json
import os
import numpy as np
import pandas as pd
import joblib
from pathlib import Path

# ─── Paths ────────────────────────────────────────────────────────────────────
# Go up two levels from py-app/app/ to reach the project root, then into models/
BASE_DIR   = Path(__file__).resolve().parent.parent.parent
MODELS_DIR = BASE_DIR / "models"


# ─── Singleton loader (load once, reuse across requests) ─────────────────────
_model        = None
_scaler       = None
_le_target    = None
_le_cat       = None
_feature_info = None


def _load_artifacts():
    """
    Load all ML artifacts from disk on first call.
    Uses module-level globals to avoid reloading on every request.
    """
    global _model, _scaler, _le_target, _le_cat, _feature_info

    if _model is not None:
        return  # Already loaded

    model_path = MODELS_DIR / "final_model.pkl"
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model file not found at {model_path}. "
            "Please run the training notebooks first."
        )

    _model     = joblib.load(MODELS_DIR / "final_model.pkl")
    _scaler    = joblib.load(MODELS_DIR / "scaler.pkl")
    _le_target = joblib.load(MODELS_DIR / "label_encoder_target.pkl")
    _le_cat    = joblib.load(MODELS_DIR / "label_encoders_cat.pkl")

    with open(MODELS_DIR / "feature_info.json", "r") as f:
        _feature_info = json.load(f)

    print(f"✅ Model loaded: {type(_model).__name__}")
    print(f"   Features     : {len(_feature_info['all_features'])}")
    print(f"   Classes      : {_le_target.classes_.tolist()}")


def _engineer_features(data: dict) -> dict:
    """
    Recreate the same 5 engineered features from data_preprocessing.ipynb.
    Must match exactly — otherwise predictions will be wrong.
    """
    data["online_to_store_spend_ratio"] = (
        data["avg_online_spend"] /
        (data["avg_store_spend"] + 1)
    )
    data["digital_engagement_score"] = (
        data["daily_internet_hours"] * 0.3 +
        data["social_media_hours"]   * 0.2 +
        data["tech_savvy_score"]     * 0.3 +
        data["monthly_online_orders"] * 0.2
    )
    data["store_preference_indicator"] = (
        data["need_touch_feel_score"] * 0.4 +
        data["monthly_store_visits"]  * 0.4 +
        data["brand_loyalty_score"]   * 0.2
    )
    data["income_spend_efficiency"] = (
        data["monthly_income"] /
        (data["avg_online_spend"] + data["avg_store_spend"] + 1)
    )
    data["price_sensitivity_index"] = (
        data["discount_sensitivity"] * 0.5 +
        data["delivery_fee_sensitivity"] * 0.5
    )
    return data


def _encode_categoricals(data: dict) -> dict:
    """
    Label-encode categorical columns using the saved LabelEncoders.
    If an unseen value is passed, default to the first known class.
    """
    for col in _feature_info["categorical_features"]:
        raw_value = str(data[col])
        le = _le_cat[col]
        if raw_value in le.classes_:
            data[col] = int(le.transform([raw_value])[0])
        else:
            # Graceful fallback for unseen category
            data[col] = 0
    return data


def preprocess_input(raw_input: dict) -> np.ndarray:
    """
    Full preprocessing pipeline:
      raw API input → engineered features → label encoding → scaling → numpy array

    Parameters
    ----------
    raw_input : dict
        Dictionary matching the CustomerFeatures schema (24 raw fields).

    Returns
    -------
    np.ndarray of shape (1, n_features)  — ready for model.predict()
    """
    _load_artifacts()

    data = dict(raw_input)  # Copy to avoid mutating the original

    # Step 1: Feature engineering (adds 5 derived columns)
    data = _engineer_features(data)

    # Step 2: Encode categorical columns (gender, city_tier)
    data = _encode_categoricals(data)

    # Step 3: Select features in the EXACT order the model was trained on
    ordered_features = _feature_info["all_features"]
    feature_values   = [data[f] for f in ordered_features]

    # Step 4: Scale using saved StandardScaler
    df_row   = pd.DataFrame([feature_values], columns=ordered_features)
    scaled   = _scaler.transform(df_row)

    return scaled


def predict(raw_input: dict) -> dict:
    """
    End-to-end prediction for a single customer.

    Parameters
    ----------
    raw_input : dict
        Raw customer data matching CustomerFeatures schema.

    Returns
    -------
    dict with keys: predicted_class, confidence, probabilities, model_used
    """
    _load_artifacts()

    # Preprocess
    X = preprocess_input(raw_input)

    # Predict
    pred_encoded  = _model.predict(X)[0]
    pred_proba    = _model.predict_proba(X)[0]   # Shape: (3,)

    # Decode predicted class
    pred_class    = _le_target.inverse_transform([pred_encoded])[0]
    confidence    = float(np.max(pred_proba))

    # Build probability dict {ClassName: probability}
    class_names   = _le_target.classes_.tolist()
    probabilities = {
        cls: round(float(p), 4)
        for cls, p in zip(class_names, pred_proba)
    }

    return {
        "predicted_class": pred_class,
        "confidence"     : round(confidence, 4),
        "probabilities"  : probabilities,
        "model_used"     : type(_model).__name__
    }


def predict_batch(raw_inputs: list[dict]) -> list[dict]:
    """
    Batch prediction for multiple customers at once.

    Parameters
    ----------
    raw_inputs : list of dicts

    Returns
    -------
    list of prediction dicts
    """
    return [predict(record) for record in raw_inputs]


def get_model_info() -> dict:
    """
    Returns metadata about the loaded model.
    Useful for the /health and /info API endpoints.
    """
    _load_artifacts()
    return {
        "model_type"    : type(_model).__name__,
        "n_features"    : len(_feature_info["all_features"]),
        "target_classes": _le_target.classes_.tolist(),
        "feature_groups": {
            "numerical"   : len(_feature_info["numerical_features"]),
            "categorical" : len(_feature_info["categorical_features"]),
            "engineered"  : len(_feature_info["engineered_features"])
        }
    }