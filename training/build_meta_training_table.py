import pandas as pd
import os
import json
import joblib
from sklearn.preprocessing import LabelEncoder

META_PATH = "data/meta_features.csv"
PERF_PATH = "data/performance_labels.csv"
OUTPUT_PATH = "data/meta_training_data.csv"
ARTIFACT_DIR = "artifacts"

def build_meta_training_table():
    meta_df = pd.read_csv(META_PATH)
    perf_df = pd.read_csv(PERF_PATH)

    meta_df["dataset"] = meta_df["dataset"].astype(str)
    perf_df["dataset"] = perf_df["dataset"].astype(str)

    merged = perf_df.merge(meta_df, on="dataset", how="inner")
    le = LabelEncoder()
    merged["model_encoded"] = le.fit_transform(merged["model"])

    os.makedirs(f"{ARTIFACT_DIR}/encoders", exist_ok=True)
    joblib.dump(le, f"{ARTIFACT_DIR}/encoders/model_label_encoder.joblib")
    feature_cols = [
        "model_encoded",
        "n_rows", "n_features",
        "numeric_ratio", "categorical_ratio",
        "missing_ratio",
        "mean_skewness", "mean_entropy",
        "feature_row_ratio",
        "log_rows", "log_features"
    ]

    os.makedirs(f"{ARTIFACT_DIR}/metadata", exist_ok=True)
    with open(f"{ARTIFACT_DIR}/metadata/feature_columns.json", "w") as f:
        json.dump(feature_cols, f)

    merged.to_csv(OUTPUT_PATH, index=False)

    print(f"Saved meta-training data to {OUTPUT_PATH}")
    print("Saved model label encoder and feature schema")


if __name__ == "__main__":
    build_meta_training_table()
