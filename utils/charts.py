from __future__ import annotations

from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_glucose_vs_outcome(df: pd.DataFrame) -> plt.Figure:
    """Create a glucose vs outcome chart."""
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    outcome_colors = {0: "#6366f1", 1: "#ef4444"}
    for outcome, color in outcome_colors.items():
        subset = df[df["Outcome"] == outcome]
        ax.hist(
            subset["Glucose"],
            bins=25,
            alpha=0.65,
            color=color,
            label="No Diabetes" if outcome == 0 else "Diabetes",
            edgecolor="white",
        )
    ax.set_title("Glucose Distribution by Outcome")
    ax.set_xlabel("Glucose")
    ax.set_ylabel("Count")
    ax.legend()
    fig.tight_layout()
    return fig


def plot_bmi_distribution(df: pd.DataFrame) -> plt.Figure:
    """Create a BMI distribution chart split by outcome."""
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    outcome_colors = {0: "#6366f1", 1: "#ef4444"}
    for outcome, color in outcome_colors.items():
        subset = df[df["Outcome"] == outcome]
        ax.hist(
            subset["BMI"],
            bins=25,
            alpha=0.65,
            color=color,
            label="No Diabetes" if outcome == 0 else "Diabetes",
            edgecolor="white",
        )
    ax.set_title("BMI Distribution by Outcome")
    ax.set_xlabel("BMI")
    ax.set_ylabel("Count")
    ax.legend()
    fig.tight_layout()
    return fig


def plot_feature_importance(importances: dict[str, float], top_k: int = 10) -> plt.Figure:
    """Plot top-K feature importances as a horizontal bar chart."""
    items = sorted(importances.items(), key=lambda kv: kv[1], reverse=True)[:top_k]
    features = [k for k, _ in items]
    values = [v for _, v in items]

    fig, ax = plt.subplots(figsize=(8, 4.5))
    y = np.arange(len(features))
    ax.barh(y, values, color="#22c55e", alpha=0.85)
    ax.set_yticks(y)
    ax.set_yticklabels(features)
    ax.invert_yaxis()
    ax.set_title(f"Top {top_k} Feature Importances (Permutation)")
    ax.set_xlabel("Importance (accuracy drop)")
    fig.tight_layout()
    return fig

