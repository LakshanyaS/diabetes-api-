from __future__ import annotations

from typing import Any

import streamlit as st

import config
from model.predict import predict_risk
from utils.charts import plot_bmi_distribution, plot_feature_importance, plot_glucose_vs_outcome
from utils.pdf_report import build_pdf_report
from utils.suggestions import get_suggestions
from utils.validation import FeatureRange, validate_inputs


def _risk_card_html(risk_level: str, risk_pct: float, pred_label: str) -> str:
    color = {"Low": "#22c55e", "Medium": "#f59e0b", "High": "#ef4444"}.get(risk_level, "#6b7280")
    return f"""
    <div style="background:#ffffff;border-radius:16px;padding:20px;border-top:6px solid {color};
                box-shadow:0 4px 24px rgba(0,0,0,0.08);text-align:center;margin:12px 0;">
      <div style="font-size:38px;font-weight:800;line-height:1;">{risk_level}</div>
      <div style="font-size:22px;font-weight:700;margin-top:6px;">{pred_label}</div>
      <div style="font-size:44px;font-weight:900;margin-top:6px;">{risk_pct:.1f}%</div>
      <div style="color:#6b7280;margin-top:4px;font-weight:600;">Diabetes Risk Score</div>
    </div>
    """


def render_predict_tab(
    df,
    model,
    feature_importance: dict[str, float] | None,
    artifacts_metrics: dict[str, Any] | None,
    ranges: dict[str, FeatureRange],
):
    st.subheader("Diabetes Risk Prediction")

    # Patient input
    patient_name = st.text_input("Patient Name", value="Patient")

    with st.form("prediction_form", clear_on_submit=False):
        col1, col2 = st.columns(2)

        with col1:
            pregnancies = st.selectbox(
                "Pregnancies",
                options=list(range(int(ranges["Pregnancies"].min_value), int(ranges["Pregnancies"].max_value) + 1)),
                index=1,
            )
            glucose = st.slider("Glucose", min_value=int(ranges["Glucose"].min_value), max_value=int(ranges["Glucose"].max_value), value=110)
            skin_thickness = st.slider(
                "Skin Thickness (mm)",
                min_value=int(ranges["SkinThickness"].min_value),
                max_value=int(ranges["SkinThickness"].max_value),
                value=20,
            )
            bmi = st.slider("BMI", min_value=float(ranges["BMI"].min_value), max_value=float(ranges["BMI"].max_value), value=28.0, step=float(ranges["BMI"].step or 0.1))

        with col2:
            blood_pressure = st.number_input(
                "Blood Pressure (mmHg)",
                min_value=int(ranges["BloodPressure"].min_value),
                max_value=int(ranges["BloodPressure"].max_value),
                value=72,
                step=int(ranges["BloodPressure"].step or 1),
            )
            insulin = st.number_input(
                "Insulin (μU/mL)",
                min_value=int(ranges["Insulin"].min_value),
                max_value=int(ranges["Insulin"].max_value),
                value=79,
                step=int(ranges["Insulin"].step or 1),
            )
            age = st.number_input(
                "Age",
                min_value=int(ranges["Age"].min_value),
                max_value=int(ranges["Age"].max_value),
                value=33,
                step=int(ranges["Age"].step or 1),
            )

        submitted = st.form_submit_button("Predict", type="primary")

    if not submitted:
        return

    inputs = {
        "Pregnancies": pregnancies,
        "Glucose": glucose,
        "BloodPressure": blood_pressure,
        "SkinThickness": skin_thickness,
        "Insulin": insulin,
        "BMI": bmi,
        "Age": age,
    }

    is_valid, errors = validate_inputs(inputs, ranges)
    if not is_valid:
        for e in errors:
            st.error(e)
        st.stop()

    result = predict_risk(inputs, model=model)
    risk_pct = float(result["risk_pct"])
    risk_level = str(result["risk_level"])
    pred_label = str(result["pred_label"])

    st.markdown(_risk_card_html(risk_level=risk_level, risk_pct=risk_pct, pred_label=pred_label), unsafe_allow_html=True)

    # Suggestions
    suggestions = get_suggestions(inputs, risk_level)
    st.markdown("### Health Suggestions")
    for s in suggestions:
        st.write(f"• {s}")

    # Feature importance
    if feature_importance:
        st.markdown("### Feature Importance (Permutation)")
        fig = plot_feature_importance(feature_importance, top_k=10)
        st.pyplot(fig, use_container_width=True)
    else:
        st.info("Feature importance not available yet. Train the model to generate it.")

    # Basic charts
    st.markdown("### Dataset Visualizations")
    c1, c2 = st.columns(2)
    with c1:
        fig_glu = plot_glucose_vs_outcome(df)
        st.pyplot(fig_glu, use_container_width=True)
    with c2:
        fig_bmi = plot_bmi_distribution(df)
        st.pyplot(fig_bmi, use_container_width=True)

    # PDF report
    st.markdown("### Download PDF Report")
    pdf_bytes = build_pdf_report(
        patient_name=patient_name,
        inputs=inputs,
        prediction_label=pred_label,
        risk_pct=risk_pct,
        risk_level=risk_level,
        suggestions=suggestions,
    )
    st.download_button(
        label="Download PDF",
        data=pdf_bytes,
        file_name=f"diabetes_report_{patient_name.replace(' ','_')}.pdf",
        mime="application/pdf",
        use_container_width=True,
    )

    st.caption("Disclaimer: This is AI-based decision support and not a medical diagnosis.")


def render_visualizations_tab(df):
    st.subheader("Visualizations")
    st.markdown("Glucose and BMI patterns from the dataset.")

    c1, c2 = st.columns(2)
    with c1:
        fig_glu = plot_glucose_vs_outcome(df)
        st.pyplot(fig_glu, use_container_width=True)
    with c2:
        fig_bmi = plot_bmi_distribution(df)
        st.pyplot(fig_bmi, use_container_width=True)


def render_model_comparison_tab(artifacts_metrics: dict[str, Any] | None):
    st.subheader("Model Comparison")

    if not artifacts_metrics:
        st.info("No training metrics found. Run `python model/train.py --retrain`.")
        return

    rows = []
    for model_name, item in artifacts_metrics.items():
        metrics = item.get("metrics", {})
        rows.append(
            {
                "Model": model_name,
                "Accuracy": metrics.get("accuracy"),
                "Precision": metrics.get("precision"),
                "Recall": metrics.get("recall"),
                "F1": metrics.get("f1"),
            }
        )

    st.dataframe(rows, use_container_width=True)

    st.markdown("### Accuracy Chart")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    names = [r["Model"] for r in rows]
    accs = [r["Accuracy"] for r in rows]
    ax.bar(names, accs, color=["#6366f1", "#22c55e", "#f59e0b"][: len(names)])
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("Accuracy")
    ax.set_title("Model Accuracy Comparison")
    plt.xticks(rotation=15, ha="right")
    fig.tight_layout()
    st.pyplot(fig, use_container_width=True)

