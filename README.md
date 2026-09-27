# InternCircle Streamlit ML App

An interactive Titanic passenger survival predictor built with Streamlit and scikit-learn. It trains a Decision Tree classifier using passenger class, gender, age, and fare.

## Features

- Age and fare missing values are imputed from training data only.
- Gender is one-hot encoded as part of the fitted scikit-learn pipeline.
- An 80:20 stratified train/test split uses `random_state=42`.
- The Decision Tree uses `max_depth=4` and `random_state=42`.
- The app shows a survival prediction, estimated probability, test metrics, and confusion matrix.
- If model files are absent, the app downloads the dataset, trains the model, and saves the artifacts automatically.

## Requirements

Python 3.11 or newer and an internet connection for the initial dataset download. The app uses the Titanic CSV at [datasciencedojo/datasets](https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv).

## Install and run

From the project folder in PowerShell:

```powershell
py -3 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
python train_model.py
streamlit run app.py
```

Open the local URL printed by Streamlit. On systems where `py -3` is unavailable, use the installed Python 3.11+ command to create the virtual environment.

The training command writes `models/titanic_pipeline.joblib` and `models/metrics.json`. Both are ignored by Git; the app creates them on first launch if needed.

## Test

Run the offline unit tests from the project folder:

```powershell
python -m pytest
```

The tests use a small in-memory sample and do not download the Titanic dataset.

## Push to GitHub

Create a public repository named `InternCircle-Streamlit-ML-App` on GitHub, then run these commands from the project folder (replace the URL with your repository's URL):

```powershell
git init
git add .
git commit -m "Build Titanic survival prediction app"
git branch -M main
git remote add origin https://github.com/YOUR-USERNAME/InternCircle-Streamlit-ML-App.git
git push -u origin main
```

No credentials or secrets belong in this repository. Git may prompt you to authenticate when pushing.

## Deploy with Streamlit Community Cloud

1. Push the project to GitHub using the instructions above.
2. Sign in to [Streamlit Community Cloud](https://share.streamlit.io/) and connect your GitHub account.
3. Create an app from the `InternCircle-Streamlit-ML-App` repository, select the `main` branch, and set the main file path to `app.py`.
4. Deploy. Community Cloud installs `requirements.txt`; on first startup, the app downloads the CSV and trains the model. The resulting artifacts are local to that app instance and may be regenerated after a restart.

## Disclaimer

This is an educational demonstration. Predictions are not historical certainty, and the model is not suitable for consequential decisions.