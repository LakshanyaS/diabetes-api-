"""
api/app.py  —  Flask REST API for the Diabetes Prediction mobile app
=====================================================================
Endpoints
---------
GET  /health            → {"status": "ok", "model": "<champion name>"}
POST /predict           → full prediction + SHAP contributions
GET  /feature-importance → permutation importance from training
GET  /metrics           → model comparison metrics

Run:
    cd chronic-disease-main
    python api/app.py

Flutter base URL (local dev):   http://10.0.2.2:5000   (Android emulator)
Flutter base URL (real device): http://<your-PC-IP>:5000
"""
from __future__ import annotations

import json
import sys
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

app   = Flask(__name__)
CORS(app)   # allow Flutter (any origin) to call the API

# Load model once at startup
_model       = None
_feature_cols: list[str] = []
_summary:     dict        = {}


def _get_model():
    global _model, _feature_cols, _summary
    if _model is None:
        _model = load_model()
        # Read feature columns from training summary if available
        try:
            with open(config.TRAINING_SUMMARY_PATH) as f:
                _summary = json.load(f)
            _feature_cols = _summary.get("feature_columns", config.FEATURE_COLUMNS)
        except FileNotFoundError:
            _feature_cols = config.FEATURE_COLUMNS
    return _model


# ── GET /health ─────────────────────────────────────────────────────────────
@app.route("/health", methods=["GET"])
def health():
    _get_model()
    return jsonify({
        "status":  "ok",
        "model":   _summary.get("best_model", "unknown"),
        "smote":   _summary.get("smote_used", False),
        "xgboost": _summary.get("xgboost_used", False),
    })


# ── POST /predict ────────────────────────────────────────────────────────────
@app.route("/predict", methods=["POST"])
def predict():
    """
    Request body (JSON):
    {
        "Pregnancies": 2,
        "Glucose": 148,
        "BloodPressure": 72,
        "SkinThickness": 35,
        "Insulin": 0,
        "BMI": 33.6,
        "DiabetesPedigreeFunction": 0.627,
        "Age": 50
    }

    Response:
    {
        "success": true,
        "prob_class_1": 0.77,
        "risk_pct": 77.0,
        "risk_level": "High",
        "pred_label": "Diabetes",
        "feature_contribs": {"Glucose": 0.42, "BMI": 0.18, ...},
        "suggestions": ["High glucose: ...", ...]
    }
    """
    data = request.get_json(force=True, silent=True)
    if not data:
        return jsonify({"success": False, "error": "No JSON body received"}), 400

    model        = _get_model()
    ranges       = get_default_ranges()
    is_valid, errors = validate_inputs(data, ranges)
    if not is_valid:
        return jsonify({"success": False, "errors": errors}), 422

    try:
        result = predict_risk(data, model=model, feature_cols=_feature_cols)
    except Exception as exc:
        return jsonify({"success": False, "error": str(exc)}), 500

    suggestions = get_suggestions(data, result["risk_level"])

    return jsonify({
        "success":          True,
        "prob_class_1":     result["prob_class_1"],
        "risk_pct":         result["risk_pct"],
        "risk_level":       result["risk_level"],
        "pred_label":       result["pred_label"],
        "feature_contribs": result["feature_contribs"],
        "suggestions":      suggestions,
    })


# ── GET /feature-importance ──────────────────────────────────────────────────
@app.route("/feature-importance", methods=["GET"])
def feature_importance():
    try:
        with open(config.FEATURE_IMPORTANCE_PATH) as f:
            fi = json.load(f)
        return jsonify({"success": True, "feature_importance": fi})
    except FileNotFoundError:
        return jsonify({"success": False,
                        "error": "feature_importance.json not found. Train the model first."}), 404


# ── GET /metrics ─────────────────────────────────────────────────────────────
@app.route("/metrics", methods=["GET"])
def metrics():
    try:
        with open(config.METRICS_PATH) as f:
            m = json.load(f)
        return jsonify({"success": True, "metrics": m})
    except FileNotFoundError:
        return jsonify({"success": False,
                        "error": "metrics.json not found. Train the model first."}), 404


# ── GET /shap-summary ────────────────────────────────────────────────────────
@app.route("/shap-summary", methods=["GET"])
def shap_summary():
    try:
        with open(config.ARTIFACTS_DIR / "shap_values.json") as f:
            sv = json.load(f)
        return jsonify({"success": True,
                        "mean_abs_shap":  sv["mean_abs_shap"],
                        "feature_names":  sv["feature_names"]})
    except FileNotFoundError:
        return jsonify({"success": False,
                        "error": "shap_values.json not found. Train with shap installed."}), 404


if __name__ == "__main__":
    print("Starting Diabetes Prediction API...")
    print(f"  Running on http://localhost:{config.API_PORT}")
    print(f"  Android emulator: http://10.0.2.2:{config.API_PORT}")
    print("  Press Ctrl+C to stop\n")
    _get_model()   # warm up
    app.run(host=config.API_HOST, port=config.API_PORT, debug=config.API_DEBUG)
