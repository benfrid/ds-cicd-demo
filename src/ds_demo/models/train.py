"""Train the Iris classifier and save it to disk."""

from __future__ import annotations

import pathlib

import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

from ds_demo.data.make_dataset import load_iris_dataframe
from ds_demo.features.build_features import FEATURE_COLUMNS, build_features

MODEL_PATH = pathlib.Path("models/iris_classifier.joblib")


def train(model_path: pathlib.Path = MODEL_PATH) -> None:
    """Train a RandomForestClassifier on the Iris dataset and save it."""
    # 1. Load data
    df = load_iris_dataframe()

    # 2. Feature engineering
    df_features = build_features(df)

    # 3. Train / test split on the enriched features
    from sklearn.model_selection import train_test_split

    X = df_features[FEATURE_COLUMNS]
    y = df_features["target"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # 4. Train model
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)

    # 5. Evaluate
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"Test accuracy: {acc:.4f}")
    print(classification_report(y_test, y_pred))

    # 6. Save model
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(clf, model_path)
    print(f"Model saved to {model_path}")


if __name__ == "__main__":
    train()
