from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

import config
from app.pages import (
    render_model_comparison_tab,
    render_predict_tab,
    render_visualizations_tab,
)
from model.predict import load_model
from utils.validation import get_default_ranges


def _load_artifacts() -> tuple[dict[str, Any] | None, dict[str, float] | None]:
    metrics = None
    feature_importance = None

    if config.METRICS_PATH.exists():
        try:
            metrics = json.loads(config.METRICS_PATH.read_text(encoding="utf-8"))
        except Exception:
            metrics = None

    if config.FEATURE_IMPORTANCE_PATH.exists():
        try:
            feature_importance = json.loads(config.FEATURE_IMPORTANCE_PATH.read_text(encoding="utf-8"))
        except Exception:
            feature_importance = None

    return metrics, feature_importance


@st.cache_data(show_spinner=False)
def load_dataframe() -> pd.DataFrame:
    """Load the dataset used for training/visualizations."""
    try:
        return pd.read_csv(config.DATASET_PATH)
    except FileNotFoundError:
        return pd.read_csv(config.ROOT_DATASET_FALLBACK_PATH)


@st.cache_resource(show_spinner=True)
def load_everything():
    df = load_dataframe()
    model = load_model()
    metrics, feature_importance = _load_artifacts()
    return df, model, metrics, feature_importance


def run_app() -> None:
    st.set_page_config(page_title="Diabetes Prediction", page_icon="🩺", layout="wide")

    st.markdown(
        """
        <style>
        .app-title {
            font-family: 'DM Sans', sans-serif;
            font-size: 32px;
            font-weight: 800;
            color: #111827;
        }
        .app-subtitle { color: #6b7280; font-size: 14px; }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("<div class='app-title'>Diabetes Risk Prediction</div>", unsafe_allow_html=True)
    st.markdown("<div class='app-subtitle'>Explainable, rule-assisted decision support (ML-based)</div>", unsafe_allow_html=True)
    st.markdown("---")

    df, model, artifacts_metrics, feature_importance = load_everything()
    ranges = get_default_ranges()

    tab_pred, tab_vis, tab_models = st.tabs(["Predict", "Visualizations", "Model Comparison"])

    with tab_pred:
        render_predict_tab(
            df=df,
            model=model,
            feature_importance=feature_importance,
            artifacts_metrics=artifacts_metrics,
            ranges=ranges,
        )

    with tab_vis:
        render_visualizations_tab(df)

    with tab_models:
        render_model_comparison_tab(artifacts_metrics)

