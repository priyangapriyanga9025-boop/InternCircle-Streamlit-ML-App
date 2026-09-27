"""Train and save the Titanic survival prediction model."""

import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier


DATASET_URL = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
FEATURES = ["Pclass", "Sex", "Age", "Fare"]
TARGET = "Survived"
PROJECT_DIR = Path(__file__).resolve().parent
MODEL_DIR = PROJECT_DIR / "models"
MODEL_PATH = MODEL_DIR / "titanic_pipeline.joblib"
METRICS_PATH = MODEL_DIR / "metrics.json"


def load_dataset():
    """Download the Titanic CSV and keep only the columns used by this app."""
    data = pd.read_csv(DATASET_URL)
    return data[FEATURES + [TARGET]]


def build_model_pipeline():
    """Create preprocessing and classifier steps that can be fit without leakage."""
    numeric_features = ["Age", "Fare"]
    preprocessing = ColumnTransformer(
        transformers=[
            ("numeric", SimpleImputer(strategy="median"), numeric_features),
            ("sex", OneHotEncoder(handle_unknown="ignore"), ["Sex"]),
            ("class", "passthrough", ["Pclass"]),
        ]
    )
    return Pipeline(
        steps=[
            ("preprocessing", preprocessing),
            (
                "classifier",
                DecisionTreeClassifier(max_depth=4, random_state=42),
            ),
        ]
    )


def train_and_evaluate(data):
    """Split the data, fit the full pipeline on training rows, and score the test set."""
    features = data[FEATURES]
    target = data[TARGET]
    x_train, x_test, y_train, y_test = train_test_split(
        features,
        target,
        test_size=0.2,
        random_state=42,
        stratify=target,
    )

    model = build_model_pipeline()
    model.fit(x_train, y_train)
    predictions = model.predict(x_test)
    metrics = {
        "accuracy": float(accuracy_score(y_test, predictions)),
        "precision": float(precision_score(y_test, predictions, zero_division=0)),
        "recall": float(recall_score(y_test, predictions, zero_division=0)),
        "f1_score": float(f1_score(y_test, predictions, zero_division=0)),
        "confusion_matrix": confusion_matrix(
            y_test, predictions, labels=[0, 1]
        ).tolist(),
    }
    return model, metrics


def save_model_artifacts(model, metrics):
    """Save the fitted pipeline and its test-set metrics under models/."""
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    with METRICS_PATH.open("w", encoding="utf-8") as metrics_file:
        json.dump(metrics, metrics_file, indent=2)


def main():
    data = load_dataset()
    model, metrics = train_and_evaluate(data)
    save_model_artifacts(model, metrics)

    print("Model trained and saved to:", MODEL_PATH)
    print("Test-set evaluation:")
    for metric_name, value in metrics.items():
        print(f"  {metric_name}: {value}")


if __name__ == "__main__":
    main()