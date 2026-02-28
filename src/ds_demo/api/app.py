"""FastAPI application for the Iris classifier."""

from __future__ import annotations

import pathlib
from contextlib import asynccontextmanager
from typing import Annotated

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from ds_demo import __version__
from ds_demo.models.predict import IRIS_CLASSES, load_model, predict

MODEL_PATH = pathlib.Path("models/iris_classifier.joblib")

# ---------------------------------------------------------------------------
# Pydantic schemas
# ---------------------------------------------------------------------------


class PredictRequest(BaseModel):
    sepal_length: Annotated[float, Field(gt=0, description="Sepal length in cm")]
    sepal_width: Annotated[float, Field(gt=0, description="Sepal width in cm")]
    petal_length: Annotated[float, Field(gt=0, description="Petal length in cm")]
    petal_width: Annotated[float, Field(gt=0, description="Petal width in cm")]

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "sepal_length": 5.1,
                    "sepal_width": 3.5,
                    "petal_length": 1.4,
                    "petal_width": 0.2,
                }
            ]
        }
    }


class PredictResponse(BaseModel):
    species: str
    class_id: int
    probabilities: dict[str, float]


class HealthResponse(BaseModel):
    status: str


class InfoResponse(BaseModel):
    name: str
    version: str
    model: str
    classes: list[str]


# ---------------------------------------------------------------------------
# Lifespan — load model once at startup
# ---------------------------------------------------------------------------

_model = None  # module-level cache


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _model
    try:
        _model = load_model(MODEL_PATH)
    except FileNotFoundError:
        _model = None  # health check will surface this
    yield
    _model = None


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Iris Classifier API",
    description=(
        "Classifies Iris flowers into *setosa*, *versicolor*, or *virginica* "
        "based on sepal and petal measurements."
    ),
    version=__version__,
    lifespan=lifespan,
)


@app.get("/", response_model=InfoResponse, tags=["info"])
def root() -> InfoResponse:
    """Return model metadata."""
    return InfoResponse(
        name="Iris Classifier",
        version=__version__,
        model="RandomForestClassifier",
        classes=IRIS_CLASSES,
    )


@app.get("/health", response_model=HealthResponse, tags=["info"])
def health() -> HealthResponse:
    """Health check — returns 200 when the model is loaded."""
    if _model is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Run 'make train' and restart the server.",
        )
    return HealthResponse(status="ok")


@app.post("/predict", response_model=PredictResponse, tags=["prediction"])
def predict_species(body: PredictRequest) -> PredictResponse:
    """Predict Iris species from flower measurements."""
    if _model is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Run 'make train' and restart the server.",
        )
    result = predict(
        sepal_length=body.sepal_length,
        sepal_width=body.sepal_width,
        petal_length=body.petal_length,
        petal_width=body.petal_width,
        model=_model,
    )
    return PredictResponse(**result)
