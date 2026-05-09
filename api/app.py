"""
api/app.py  —  Optimized Flask API
====================================
- NO shap import on server (saves 300MB RAM)
- Pre-computed shap_values.json is read at startup
- Fast predictions, no slowdowns
- Works perfectly on Render free tier
"""
from __future__ import annotations

import json
import sys
import numpy as np
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from flask import Flask, jsonify, request
from flask_cors import CORS

import config
from model.predict import load_model, predict_risk
from utils.suggestions import get_suggestions
from utils.validation import get_default_ranges, validate_inputs

app  = Flask(__name__)
CORS(app)

# ── Load everything ONCE at startup (fast after this) ──────────────────────
_model        = None
_feature_cols = []
_summary      = {}
_shap_data    = {}   # pre-computed SHAP — loaded from JSON, no shap library needed
_fi_data      = {}   # feature importance
_metrics_data = {}


def _startup():
    """Load model + all JSON artifacts once when server starts."""
    global _model, _feature_cols, _summary, _shap_data, _fi_data, _metrics_data

    print("Loading model...")
    _model = load_model()

    # Training summary
    try:
        with open(config.TRAINING_SUMMARY_PATH) as f:
            _summary = json.load(f)
        _feature_cols = _summary.get("feature_columns", config.FEATURE_COLUMNS)
    except FileNotFoundError:
        _feature_cols = config.FEATURE_COLUMNS

    # Pre-computed SHAP values (generated during training on your laptop)
    try:
        with open(config.ARTIFACTS_DIR / "shap_values.json") as f:
            _shap_data = json.load(f)
        print("SHAP values loaded from file ✅")
    except FileNotFoundError:
        _shap_data = {}
        print("shap_values.json not found — SHAP will show as unavailable")

    # Feature importance
    try:
        with open(config.FEATURE_IMPORTANCE_PATH) as f:
            _fi_data = json.load(f)
    except FileNotFoundError:
        _fi_data = {}

    # Metrics
    try:
        with open(config.METRICS_PATH) as f:
            _metrics_data = json.load(f)
    except FileNotFoundError:
        _metrics_data = {}

    print("Server ready ✅")


# ── GET /health ─────────────────────────────────────────────────────────────
@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status":       "ok",
        "model":        _summary.get("best_model", "unknown"),
        "shap_loaded":  bool(_shap_data),
    })


# ── POST /predict ─────────────────────────────────────────────────────────
@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(force=True, silent=True)
    if not data:
        return jsonify({"success": False, "error": "No JSON body received"}), 400

    ranges = get_default_ranges()
    is_valid, errors = validate_inputs(data, ranges)
    if not is_valid:
        return jsonify({"success": False, "errors": errors}), 422

    try:
        # Fast prediction — no SHAP computation here
        result = predict_risk(data, model=_model, feature_cols=_feature_cols)
    except Exception as exc:
        return jsonify({"success": False, "error": str(exc)}), 500

    # ── SHAP: use pre-computed mean values as approximation ────────────────
    # Instead of computing SHAP per-patient (slow, needs shap library),
    # we use the pre-computed mean |SHAP| values scaled by how far the
    # patient's values deviate from the dataset mean.
    # This gives a good approximation without any heavy computation.
    feature_contribs: dict[str, float] = {}

    if _shap_data and "mean_abs_shap" in _shap_data:
        mean_abs = _shap_data["mean_abs_shap"]      # {feature: mean_shap}
        feat_means = _shap_data.get("feature_means", {})  # dataset means

        prob = result["prob_class_1"]
        direction = 1.0 if prob >= 0.5 else -1.0

        for feat in _feature_cols:
            if feat not in mean_abs:
                continue
            base_shap = mean_abs[feat]

            # Scale by deviation from dataset mean if available
            if feat in feat_means and feat in data:
                patient_val = float(data[feat])
                mean_val    = float(feat_means[feat])
                std_val     = float(_shap_data.get("feature_stds", {}).get(feat, 1.0))
                if std_val > 0:
                    deviation = (patient_val - mean_val) / std_val
                    # Positive deviation → pushes toward higher risk
                    contrib = base_shap * deviation
                else:
                    contrib = base_shap * direction
            else:
                contrib = base_shap * direction

            feature_contribs[feat] = round(float(contrib), 4)

    suggestions = get_suggestions(data, result["risk_level"])

    return jsonify({
        "success":          True,
        "prob_class_1":     result["prob_class_1"],
        "risk_pct":         result["risk_pct"],
        "risk_level":       result["risk_level"],
        "pred_label":       result["pred_label"],
        "feature_contribs": feature_contribs,
        "suggestions":      suggestions,
    })


# ── GET /feature-importance ────────────────────────────────────────────────
@app.route("/feature-importance", methods=["GET"])
def feature_importance():
    if _fi_data:
        return jsonify({"success": True, "feature_importance": _fi_data})
    return jsonify({"success": False,
                    "error": "feature_importance.json not found"}), 404


# ── GET /metrics ───────────────────────────────────────────────────────────
@app.route("/metrics", methods=["GET"])
def metrics():
    if _metrics_data:
        return jsonify({"success": True, "metrics": _metrics_data})
    return jsonify({"success": False,
                    "error": "metrics.json not found"}), 404


# ── GET /shap-summary ──────────────────────────────────────────────────────
@app.route("/shap-summary", methods=["GET"])
def shap_summary():
    if _shap_data and "mean_abs_shap" in _shap_data:
        return jsonify({
            "success":       True,
            "mean_abs_shap": _shap_data["mean_abs_shap"],
            "feature_names": _shap_data.get("feature_names", _feature_cols),
        })
    return jsonify({"success": False,
                    "error": "SHAP data not available"}), 404


if __name__ == "__main__":
    _startup()
    print(f"Running on http://localhost:{config.API_PORT}")
    app.run(host=config.API_HOST, port=config.API_PORT, debug=config.API_DEBUG)