"""Load and split the Iris dataset."""

from __future__ import annotations

import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split


def load_iris_dataframe() -> pd.DataFrame:
    """Return the raw Iris dataset as a DataFrame with a 'target' column."""
    iris = load_iris(as_frame=True)
    df = iris.frame  # includes sepal/petal columns + target
    df["species"] = df["target"].map(dict(enumerate(iris.target_names)))
    return df


def split_dataset(
    df: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Split DataFrame into train/test feature and label arrays.

    Returns
    -------
    X_train, X_test, y_train, y_test
    """
    feature_cols = [c for c in df.columns if c not in ("target", "species")]
    X = df[feature_cols]
    y = df["target"]
    return train_test_split(X, y, test_size=test_size, random_state=random_state)


if __name__ == "__main__":
    df = load_iris_dataframe()
    print(df.head())
    print(f"Shape: {df.shape}")
