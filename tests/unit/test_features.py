"""Unit tests for feature engineering."""

import pandas as pd
import pytest

from ds_demo.features.build_features import (
    FEATURE_COLUMNS,
    add_petal_area,
    add_petal_sepal_ratio,
    add_sepal_area,
    build_features,
)


@pytest.fixture
def sample_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "sepal length (cm)": [5.1, 4.9, 6.3],
            "sepal width (cm)": [3.5, 3.0, 3.3],
            "petal length (cm)": [1.4, 1.4, 6.0],
            "petal width (cm)": [0.2, 0.2, 2.5],
        }
    )


def test_add_petal_area(sample_df):
    result = add_petal_area(sample_df)
    assert "petal area (cm^2)" in result.columns
    expected = sample_df["petal length (cm)"] * sample_df["petal width (cm)"]
    pd.testing.assert_series_equal(
        result["petal area (cm^2)"], expected, check_names=False
    )


def test_add_sepal_area(sample_df):
    result = add_sepal_area(sample_df)
    assert "sepal area (cm^2)" in result.columns
    expected = sample_df["sepal length (cm)"] * sample_df["sepal width (cm)"]
    pd.testing.assert_series_equal(
        result["sepal area (cm^2)"], expected, check_names=False
    )


def test_add_petal_sepal_ratio(sample_df):
    result = add_petal_sepal_ratio(sample_df)
    assert "petal_sepal_area_ratio" in result.columns
    # All values should be non-negative
    assert (result["petal_sepal_area_ratio"] >= 0).all()


def test_build_features_returns_all_columns(sample_df):
    result = build_features(sample_df)
    for col in FEATURE_COLUMNS:
        assert col in result.columns, f"Missing column: {col}"


def test_build_features_does_not_mutate_input(sample_df):
    original_cols = list(sample_df.columns)
    build_features(sample_df)
    assert list(sample_df.columns) == original_cols


def test_build_features_row_count_preserved(sample_df):
    result = build_features(sample_df)
    assert len(result) == len(sample_df)
