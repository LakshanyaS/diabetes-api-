from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class FeatureRange:
    min_value: float
    max_value: float
    step: float | None = None

    def contains(self, value: float) -> bool:
        return self.min_value <= value <= self.max_value


def get_default_ranges() -> dict[str, FeatureRange]:
    """
    Validation ranges for dataset inputs.

    These are intentionally broad and aligned with typical Pima Diabetes ranges.
    """

    return {
        "Pregnancies": FeatureRange(0, 20, 1),
        "Glucose": FeatureRange(0, 300, 1),
        "BloodPressure": FeatureRange(0, 150, 1),
        "SkinThickness": FeatureRange(0, 100, 1),
        "Insulin": FeatureRange(0, 900, 1),
        "BMI": FeatureRange(0.0, 70.0, 0.1),
        "Age": FeatureRange(1, 120, 1),
    }


def validate_inputs(inputs: dict[str, Any], ranges: dict[str, FeatureRange]) -> tuple[bool, list[str]]:
    """Validate numeric inputs and return (is_valid, error_messages)."""
    errors: list[str] = []

    for feature, fr in ranges.items():
        if feature not in inputs:
            errors.append(f"Missing input: {feature}")
            continue

        try:
            value = float(inputs[feature])
        except (TypeError, ValueError):
            errors.append(f"Invalid type for {feature}. Expected a number.")
            continue

        if not fr.contains(value):
            errors.append(f"{feature} must be between {fr.min_value} and {fr.max_value}.")

    return (len(errors) == 0), errors

