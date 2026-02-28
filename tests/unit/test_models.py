"""Unit tests for model training and prediction."""

import pathlib

import pytest

from ds_demo.models.predict import IRIS_CLASSES, load_model, predict
from ds_demo.models.train import train


@pytest.fixture(scope="module")
def trained_model_path(tmp_path_factory):
    """Train a model into a temporary directory and return the path."""
    tmp_dir = tmp_path_factory.mktemp("models")
    model_path = tmp_dir / "iris_classifier.joblib"
    train(model_path=model_path)
    return model_path


def test_train_creates_model_file(trained_model_path):
    assert trained_model_path.exists()
    assert trained_model_path.stat().st_size > 0


def test_load_model(trained_model_path):
    model = load_model(trained_model_path)
    assert hasattr(model, "predict")
    assert hasattr(model, "predict_proba")


def test_load_model_missing_file():
    with pytest.raises(FileNotFoundError):
        load_model(pathlib.Path("/nonexistent/path/model.joblib"))


def test_predict_returns_valid_species(trained_model_path):
    model = load_model(trained_model_path)
    result = predict(
        sepal_length=5.1,
        sepal_width=3.5,
        petal_length=1.4,
        petal_width=0.2,
        model=model,
    )
    assert result["species"] in IRIS_CLASSES
    assert result["class_id"] in range(3)


def test_predict_probabilities_sum_to_one(trained_model_path):
    model = load_model(trained_model_path)
    result = predict(5.1, 3.5, 1.4, 0.2, model=model)
    total = sum(result["probabilities"].values())
    assert abs(total - 1.0) < 1e-6


def test_predict_all_classes_present_in_proba(trained_model_path):
    model = load_model(trained_model_path)
    result = predict(5.1, 3.5, 1.4, 0.2, model=model)
    assert set(result["probabilities"].keys()) == set(IRIS_CLASSES)


@pytest.mark.parametrize(
    "sepal_l,sepal_w,petal_l,petal_w,expected_species",
    [
        (5.1, 3.5, 1.4, 0.2, "setosa"),  # classic setosa
        (7.0, 3.2, 4.7, 1.4, "versicolor"),  # classic versicolor
        (6.3, 3.3, 6.0, 2.5, "virginica"),  # classic virginica
    ],
)
def test_predict_known_samples(
    trained_model_path,
    sepal_l,
    sepal_w,
    petal_l,
    petal_w,
    expected_species,
):
    model = load_model(trained_model_path)
    result = predict(sepal_l, sepal_w, petal_l, petal_w, model=model)
    assert result["species"] == expected_species, (
        f"Expected {expected_species}, got {result['species']}"
    )
