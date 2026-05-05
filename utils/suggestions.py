from __future__ import annotations

from typing import Any


def get_suggestions(inputs: dict[str, float], risk_level: str) -> list[str]:
    """
    Create patient-friendly suggestions based on risk level and key feature triggers.
    This is rule-based decision support (not medical diagnosis).
    """

    suggestions: list[str] = []

    # Feature-based hints
    glucose = float(inputs["Glucose"])
    bmi = float(inputs["BMI"])
    bp = float(inputs["BloodPressure"])
    age = float(inputs["Age"])
    if glucose >= 126:
        suggestions.append("High glucose: reduce sugar/refined carbs and monitor blood sugar regularly.")
    elif glucose >= 100:
        suggestions.append("Borderline glucose: limit sugary drinks and processed foods; consider lifestyle changes now.")
    else:
        suggestions.append("Glucose is in a healthier range: maintain balanced nutrition.")

    if bmi >= 30:
        suggestions.append("BMI suggests obesity: aim for regular physical activity and portion control; consult a dietitian.")
    elif bmi >= 25:
        suggestions.append("BMI is moderately elevated: gradual weight management through diet and exercise can help.")
    else:
        suggestions.append("BMI is relatively healthy: keep current habits and stay active.")

    if bp >= 90:
        suggestions.append("Blood pressure is elevated: reduce salt, manage stress, and follow clinician advice.")
    elif bp >= 80:
        suggestions.append("Blood pressure is slightly elevated: stay hydrated and manage lifestyle factors.")

    if age >= 45 and risk_level in {"Medium", "High"}:
        suggestions.append("Age 45+ with elevated risk: discuss HbA1c screening with your healthcare provider.")

    # Risk-level escalation
    if risk_level == "High":
        suggestions.append("High risk: consult a doctor/endocrinologist soon. Follow a structured diet and exercise plan.")
        suggestions.append("Consider setting up periodic monitoring (glucose/HbA1c) based on clinician guidance.")
    elif risk_level == "Medium":
        suggestions.append("Medium risk: start lifestyle adjustments now (diet, exercise, sleep) and re-check as advised.")
    else:
        suggestions.append("Low risk: maintain healthy habits and do routine check-ups when recommended.")

    # De-duplicate while keeping order
    seen = set()
    deduped: list[str] = []
    for s in suggestions:
        if s not in seen:
            seen.add(s)
            deduped.append(s)
    return deduped

