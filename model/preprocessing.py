from __future__ import annotations
from typing import List
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def make_numeric_preprocess(feature_names: List[str]) -> ColumnTransformer:
    """Impute missing values (median) then scale numeric features."""
    numeric_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler",  StandardScaler()),
    ])
    return ColumnTransformer(
        [("num", numeric_pipeline, feature_names)],
        remainder="drop",
    )
