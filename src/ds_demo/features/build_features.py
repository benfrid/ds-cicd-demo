"""Feature engineering for the Iris classifier."""

from __future__ import annotations

import numpy as np
import pandas as pd


def add_petal_area(df: pd.DataFrame) -> pd.DataFrame:
    """Add a petal_area column (length × width) to the DataFrame."""
    df = df.copy()
    df["petal area (cm^2)"] = df["petal length (cm)"] * df["petal width (cm)"]
    return df


def add_sepal_area(df: pd.DataFrame) -> pd.DataFrame:
    """Add a sepal_area column (length × width) to the DataFrame."""
    df = df.copy()
    df["sepal area (cm^2)"] = df["sepal length (cm)"] * df["sepal width (cm)"]
    return df


def add_petal_sepal_ratio(df: pd.DataFrame) -> pd.DataFrame:
    """Add ratio of petal area to sepal area."""
    df = df.copy()
    sepal_area = df["sepal length (cm)"] * df["sepal width (cm)"]
    petal_area = df["petal length (cm)"] * df["petal width (cm)"]
    df["petal_sepal_area_ratio"] = petal_area / np.where(sepal_area == 0, 1, sepal_area)
    return df


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Apply all feature engineering steps and return enriched DataFrame."""
    df = add_petal_area(df)
    df = add_sepal_area(df)
    df = add_petal_sepal_ratio(df)
    return df


# Feature columns used by the model (original + engineered)
FEATURE_COLUMNS = [
    "sepal length (cm)",
    "sepal width (cm)",
    "petal length (cm)",
    "petal width (cm)",
    "petal area (cm^2)",
    "sepal area (cm^2)",
    "petal_sepal_area_ratio",
]
