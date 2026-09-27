import pandas as pd
from sklearn.pipeline import Pipeline

from train_model import build_model_pipeline, train_and_evaluate


def sample_data():
    return pd.DataFrame(
        {
            "Pclass": [1, 3, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2],
            "Sex": ["female", "male", "female", "male", "female", "male", "female", "male", "female", "male", "female", "male", "female", "male", "female"],
            "Age": [29, 22, None, 35, 41, 19, 8, 50, 32, None, 25, 45, 17, 38, 28],
            "Fare": [80.0, 7.2, 13.0, 8.1, 70.0, 12.0, 21.0, 55.0, 14.0, 7.8, 60.0, 9.0, 15.0, 90.0, 11.0],
            "Survived": [1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
        }
    )


def test_pipeline_predicts_and_imputes_age_from_fit_data():
    training_rows = sample_data().iloc[:10]
    model = build_model_pipeline()
    model.fit(training_rows[["Pclass", "Sex", "Age", "Fare"]], training_rows["Survived"])

    numeric_imputer = model.named_steps["preprocessing"].named_transformers_["numeric"]
    assert numeric_imputer.statistics_[0] == 30.5
    assert model.predict(pd.DataFrame([{"Pclass": 1, "Sex": "female", "Age": 30, "Fare": 50.0}])).shape == (1,)
    assert isinstance(model, Pipeline)


def test_training_returns_expected_evaluation_metrics():
    model, metrics = train_and_evaluate(sample_data())

    assert set(metrics) == {
        "accuracy",
        "precision",
        "recall",
        "f1_score",
        "confusion_matrix",
    }
    assert len(metrics["confusion_matrix"]) == 2
    assert model.predict(pd.DataFrame([{"Pclass": 3, "Sex": "male", "Age": 20, "Fare": 10.0}])).shape == (1,)