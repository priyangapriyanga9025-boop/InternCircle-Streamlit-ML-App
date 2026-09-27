"""Interactive Streamlit interface for Titanic survival predictions."""

import json

import joblib
import pandas as pd
import streamlit as st

from train_model import (
    METRICS_PATH,
    MODEL_PATH,
    load_dataset,
    save_model_artifacts,
    train_and_evaluate,
)


st.set_page_config(
    page_title="Titanic Survival Predictor",
    page_icon="🚢",
    layout="centered",
)

st.markdown(
    """
    <style>
    .stApp { background-color: #f4f7f5; }
    .block-container { max-width: 900px; padding-top: 2.5rem; }
    h1 { color: #173b39; }
    [data-testid="stMetric"] {
        background: #ffffff;
        border: 1px solid #dce7e2;
        border-radius: 8px;
        padding: 14px 16px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def load_or_train_model():
    """Load saved artifacts or train them once when deploying a fresh checkout."""
    if not MODEL_PATH.exists() or not METRICS_PATH.exists():
        data = load_dataset()
        model, metrics = train_and_evaluate(data)
        save_model_artifacts(model, metrics)
    model = joblib.load(MODEL_PATH)
    with METRICS_PATH.open("r", encoding="utf-8") as metrics_file:
        metrics = json.load(metrics_file)
    return model, metrics


st.title("Titanic Survival Predictor")
st.write(
    "Explore how passenger class, gender, age, and fare relate to a model's "
    "survival prediction."
)

try:
    model, evaluation = load_or_train_model()
except Exception as error:
    st.error(
        "The prediction model could not be loaded or trained. Check your internet "
        "connection and dependencies, then run `py train_model.py` and refresh."
    )
    with st.expander("Technical details"):
        st.code(str(error))
    st.stop()

with st.form("passenger_details"):
    st.subheader("Passenger details")
    left_column, right_column = st.columns(2)
    with left_column:
        passenger_class = st.selectbox("Passenger class", [1, 2, 3], index=2)
        sex = st.selectbox("Gender", ["female", "male"])
    with right_column:
        age = st.slider("Age", min_value=0, max_value=100, value=30)
        fare = st.slider(
            "Fare", min_value=0.0, max_value=512.33, value=32.20, step=0.50
        )
    predict_clicked = st.form_submit_button(
        "Predict survival", type="primary", width="stretch"
    )

if predict_clicked:
    passenger = pd.DataFrame(
        [{"Pclass": passenger_class, "Sex": sex, "Age": age, "Fare": fare}]
    )
    prediction = int(model.predict(passenger)[0])
    probability = model.predict_proba(passenger)[0]
    survived_probability = float(probability[list(model.classes_).index(1)])

    st.divider()
    if prediction == 1:
        st.success("Prediction: Survived")
    else:
        st.error("Prediction: Did Not Survive")
    st.metric("Estimated survival probability", f"{survived_probability:.1%}")
    st.progress(survived_probability)

st.divider()
st.subheader("Model test-set performance")
metric_columns = st.columns(4)
for column, (label, key) in zip(
    metric_columns,
    [("Accuracy", "accuracy"), ("Precision", "precision"), ("Recall", "recall"), ("F1 score", "f1_score")],
):
    column.metric(label, f"{evaluation[key]:.1%}")

with st.expander("Confusion matrix"):
    st.caption("Rows are actual outcomes; columns are model predictions (0, 1).")
    st.dataframe(
        pd.DataFrame(
            evaluation["confusion_matrix"],
            index=["Actual: Did not survive", "Actual: Survived"],
            columns=["Predicted: Did not survive", "Predicted: Survived"],
        ),
        width="stretch",
    )

st.caption(
    "Educational demonstration only. This model's predictions are not historical "
    "certainty or a guarantee about any individual."
)