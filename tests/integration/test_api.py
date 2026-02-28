"""Integration tests for the FastAPI application."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from ds_demo.models.train import MODEL_PATH, train


@pytest.fixture(scope="module", autouse=True)
def ensure_model_trained():
    """Train the model once before running API tests."""
    if not MODEL_PATH.exists():
        MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        train(model_path=MODEL_PATH)


@pytest.fixture(scope="module")
def client():
    """Return a TestClient with the model pre-loaded."""
    # Re-import app so lifespan runs with a fresh model
    from ds_demo.api.app import app

    with TestClient(app) as c:
        yield c


# ---------------------------------------------------------------------------
# Health + info
# ---------------------------------------------------------------------------


def test_health_ok(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_root_returns_model_info(client):
    response = client.get("/")
    assert response.status_code == 200
    body = response.json()
    assert body["name"] == "Iris Classifier"
    assert "version" in body
    assert "classes" in body
    assert len(body["classes"]) == 3


# ---------------------------------------------------------------------------
# Predict endpoint — happy path
# ---------------------------------------------------------------------------

_SETOSA = {
    "sepal_length": 5.1,
    "sepal_width": 3.5,
    "petal_length": 1.4,
    "petal_width": 0.2,
}
_VERSICOLOR = {
    "sepal_length": 7.0,
    "sepal_width": 3.2,
    "petal_length": 4.7,
    "petal_width": 1.4,
}
_VIRGINICA = {
    "sepal_length": 6.3,
    "sepal_width": 3.3,
    "petal_length": 6.0,
    "petal_width": 2.5,
}


@pytest.mark.parametrize(
    "payload,expected_species",
    [
        (_SETOSA, "setosa"),
        (_VERSICOLOR, "versicolor"),
        (_VIRGINICA, "virginica"),
    ],
)
def test_predict_correct_species(client, payload, expected_species):
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    body = response.json()
    assert body["species"] == expected_species
    assert "probabilities" in body
    assert abs(sum(body["probabilities"].values()) - 1.0) < 1e-5


def test_predict_response_schema(client):
    response = client.post("/predict", json=_SETOSA)
    assert response.status_code == 200
    body = response.json()
    assert set(body.keys()) == {"species", "class_id", "probabilities"}
    assert isinstance(body["class_id"], int)
    assert isinstance(body["species"], str)
    assert isinstance(body["probabilities"], dict)


# ---------------------------------------------------------------------------
# Predict endpoint — validation errors
# ---------------------------------------------------------------------------


def test_predict_missing_field(client):
    # Missing petal_width
    payload = {"sepal_length": 5.1, "sepal_width": 3.5, "petal_length": 1.4}
    response = client.post("/predict", json=payload)
    assert response.status_code == 422


def test_predict_negative_value(client):
    payload = {
        "sepal_length": -1.0,
        "sepal_width": 3.5,
        "petal_length": 1.4,
        "petal_width": 0.2,
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 422


def test_predict_non_numeric_value(client):
    payload = {
        "sepal_length": "big",
        "sepal_width": 3.5,
        "petal_length": 1.4,
        "petal_width": 0.2,
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 422
