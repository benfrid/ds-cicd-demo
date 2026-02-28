"""Load a saved model and run inference."""

from __future__ import annotations

import pathlib

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from ds_demo.features.build_features import FEATURE_COLUMNS, build_features

MODEL_PATH = pathlib.Path("models/iris_classifier.joblib")

# Iris class names in label order
IRIS_CLASSES = ["setosa", "versicolor", "virginica"]


def load_model(model_path: pathlib.Path = MODEL_PATH) -> RandomForestClassifier:
    """Load the serialised classifier from disk."""
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model not found at {model_path}. Run 'make train' first."
        )
    return joblib.load(model_path)


def predict(
    sepal_length: float,
    sepal_width: float,
    petal_length: float,
    petal_width: float,
    model: RandomForestClassifier | None = None,
    model_path: pathlib.Path = MODEL_PATH,
) -> dict[str, object]:
    """Predict the Iris species for a single flower measurement.

    Parameters
    ----------
    sepal_length, sepal_width, petal_length, petal_width : float
        Measurements in centimetres.
    model : optional pre-loaded classifier (avoids repeated disk I/O in the API).
    model_path : path to the saved .joblib file.

    Returns
    -------
    dict with keys: ``species`` (str), ``class_id`` (int), ``probabilities`` (dict).
    """
    if model is None:
        model = load_model(model_path)

    raw = pd.DataFrame(
        {
            "sepal length (cm)": [sepal_length],
            "sepal width (cm)": [sepal_width],
            "petal length (cm)": [petal_length],
            "petal width (cm)": [petal_width],
        }
    )
    enriched = build_features(raw)
    X = enriched[FEATURE_COLUMNS]

    class_id = int(model.predict(X)[0])
    proba = model.predict_proba(X)[0]

    return {
        "species": IRIS_CLASSES[class_id],
        "class_id": class_id,
        "probabilities": {
            species: round(float(p), 4) for species, p in zip(IRIS_CLASSES, proba)
        },
    }
