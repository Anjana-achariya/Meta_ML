import pandas as pd
import os
import json
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

DATA_PATH = "data/meta_training_data.csv"
ARTIFACT_DIR = "artifacts"

def train_meta_model():
    df = pd.read_csv(DATA_PATH)
    feature_cols = [
        "model_encoded",
        "n_rows", "n_features",
        "numeric_ratio", "categorical_ratio",
        "missing_ratio",
        "mean_skewness", "mean_entropy",
        "feature_row_ratio",
        "log_rows", "log_features"
    ]

    X = df[feature_cols]
    y = df["accuracy"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = RandomForestRegressor(
        n_estimators=200,
        random_state=42
    )

    model.fit(X_train, y_train)

    os.makedirs(f"{ARTIFACT_DIR}/meta_models", exist_ok=True)
    os.makedirs(f"{ARTIFACT_DIR}/metadata", exist_ok=True)

    joblib.dump(model, f"{ARTIFACT_DIR}/meta_models/accuracy_model.joblib")

    with open(f"{ARTIFACT_DIR}/metadata/feature_columns.json", "w") as f:
        json.dump(feature_cols, f)

    print("Saved accuracy meta-model and feature schema")

    preds = model.predict(X_test)

    print("MAE:", mean_absolute_error(y_test, preds))
    print("R2 :", r2_score(y_test, preds))

    return model


if __name__ == "__main__":
    train_meta_model()
