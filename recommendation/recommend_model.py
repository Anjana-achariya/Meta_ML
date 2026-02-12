import pandas as pd
import numpy as np
import json
import joblib
from typing import Dict

from meta_features.extract_meta_features import extract_meta_features


# ---------------- PATHS ----------------
ARTIFACT_DIR = "artifacts"

ACC_MODEL_PATH = f"{ARTIFACT_DIR}/meta_models/accuracy_model.joblib"
TIME_MODEL_PATH = f"{ARTIFACT_DIR}/meta_models/training_time_model.joblib"
ENCODER_PATH = f"{ARTIFACT_DIR}/encoders/model_label_encoder.joblib"
FEATURE_SCHEMA_PATH = f"{ARTIFACT_DIR}/metadata/feature_columns.json"


# ---------------- LOAD ARTIFACTS ----------------
acc_model = joblib.load(ACC_MODEL_PATH)
time_model = joblib.load(TIME_MODEL_PATH)
model_encoder = joblib.load(ENCODER_PATH)

with open(FEATURE_SCHEMA_PATH) as f:
    FEATURE_COLS = json.load(f)


# ---------------- UTILS ----------------
def normalize(series: pd.Series) -> pd.Series:
    return (series - series.min()) / (series.max() - series.min() + 1e-9)


def get_model_predictions(meta_features: Dict[str, float]) -> pd.DataFrame:
    rows = []

    for model_name in model_encoder.classes_:
        row = meta_features.copy()
        row["model_encoded"] = model_encoder.transform([model_name])[0]
        rows.append(row)

    X = pd.DataFrame(rows)[FEATURE_COLS]

    acc_preds = acc_model.predict(X)
    time_preds = time_model.predict(X)

    return pd.DataFrame({
        "model": model_encoder.classes_,
        "predicted_accuracy": acc_preds,
        "predicted_training_time": time_preds
    })


def weighted_ranking(
    df: pd.DataFrame,
    accuracy_weight: float = 0.7,
    time_weight: float = 0.3
) -> pd.DataFrame:

    df = df.copy()

    df["norm_accuracy"] = normalize(df["predicted_accuracy"])
    df["norm_time"] = normalize(df["predicted_training_time"])

    df["score"] = (
        accuracy_weight * df["norm_accuracy"]
        - time_weight * df["norm_time"]
    )

    return df.sort_values("score", ascending=False)


def recommend_models(
    dataset_csv: str,
    accuracy_weight: float = 0.7,
    time_weight: float = 0.3
):

    meta_features = extract_meta_features(dataset_csv)

    preds = get_model_predictions(meta_features)

    ranked = weighted_ranking(
        preds,
        accuracy_weight=accuracy_weight,
        time_weight=time_weight
    ).reset_index(drop=True)

    # Normalize score → confidence
    min_score = ranked["score"].min()
    max_score = ranked["score"].max()

    ranked["confidence"] = (
        (ranked["score"] - min_score) /
        (max_score - min_score + 1e-9)
    )

    top_models = ranked.head(3)

    clean_output = []

    for idx, row in top_models.iterrows():

        if row["norm_accuracy"] > 0.7 and row["norm_time"] < 0.5:
            reason = "High accuracy with relatively low training time"
        elif row["norm_accuracy"] > 0.7:
            reason = "High predicted accuracy"
        elif row["norm_time"] < 0.3:
            reason = "Very fast training time"
        else:
            reason = "Balanced accuracy and training efficiency"

        clean_output.append({
            "rank": idx + 1,
            "model": row["model"],
            "predicted_accuracy": round(float(row["predicted_accuracy"]), 4),
            "predicted_training_time": round(float(row["predicted_training_time"]), 4),
            "confidence_score": round(float(row["confidence"]), 2),
            "reason": reason
        })

    return {
        "recommended_models": clean_output
    }


# ---------------- LOCAL TEST ----------------
if __name__ == "__main__":
    print("Running local recommender test...\n")

    result = recommend_models(
        dataset_csv="data/openml_raw/780.csv",
        accuracy_weight=0.8,
        time_weight=0.2
    )

    print(result)
