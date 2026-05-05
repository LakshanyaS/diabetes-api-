"""model/predict.py — load model and make predictions with SHAP explanation."""
from __future__ import annotations

import pickle
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import config

try:
    import shap as _shap
    _SHAP = True
except ImportError:
    _SHAP = False


def load_model(model_path: Path | None = None) -> Any:
    if model_path is None:
        model_path = config.MODEL_PATH
    try:
        with open(model_path, "rb") as f:
            return pickle.load(f)
    except FileNotFoundError:
        with open(config.ROOT_MODEL_FALLBACK_PATH, "rb") as f:
            return pickle.load(f)


def predict_risk(inputs: dict[str, float],
                 model: Any | None = None,
                 feature_cols: list | None = None) -> dict[str, Any]:
    """
    Run the champion model pipeline and return prediction + SHAP contributions.

    Returns:
        prob_class_1    : float  — raw probability of Outcome=1
        risk_pct        : float  — prob * 100
        risk_level      : str    — Low / Medium / High
        pred_label      : str    — Diabetes / No Diabetes
        feature_contribs: dict   — {feature: shap_value} per input feature
                                   (empty dict if SHAP unavailable)
    """
    if model is None:
        model = load_model()
    if feature_cols is None:
        feature_cols = config.FEATURE_COLUMNS

    # Keep only features the model was trained on
    input_df = pd.DataFrame([{k: inputs[k] for k in feature_cols if k in inputs}])

    prob_class_1 = float(model.predict_proba(input_df)[0][1])
    risk_pct     = prob_class_1 * 100.0
    pred_label   = "Diabetes" if prob_class_1 >= config.DIABETIC_PROB_THRESHOLD else "No Diabetes"

    if risk_pct < config.RISK_LOW_MAX:
        risk_level = "Low"
    elif risk_pct < config.RISK_MODERATE_MAX:
        risk_level = "Medium"
    else:
        risk_level = "High"

    # SHAP explanation for this single prediction
    feature_contribs: dict[str, float] = {}
    if _SHAP:
        try:
            pre = model.named_steps["pre"]
            clf = model.named_steps["clf"]
            X_transformed = pre.transform(input_df)

            try:
                explainer = _shap.TreeExplainer(clf)
                sv = explainer.shap_values(X_transformed)
                if isinstance(sv, list):
                    sv = sv[1]
            except Exception:
                explainer = _shap.LinearExplainer(clf, X_transformed)
                sv = explainer.shap_values(X_transformed)

            feature_contribs = {
                feat: float(sv[0, i])
                for i, feat in enumerate(feature_cols)
            }
        except Exception:
            pass

    return {
        "prob_class_1":     prob_class_1,
        "risk_pct":         risk_pct,
        "pred_label":       pred_label,
        "risk_level":       risk_level,
        "feature_contribs": feature_contribs,
    }
