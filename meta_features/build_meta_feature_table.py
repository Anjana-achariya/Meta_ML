import os
import pandas as pd
from meta_features.extract_meta_features import extract_meta_features

DATA_DIR = "data/openml_raw"
OUTPUT_PATH = "data/meta_features.csv"

def build_meta_feature_table():
    rows = []

    for file in os.listdir(DATA_DIR):
        if not file.endswith(".csv"):
            continue

        dataset_id = file.replace(".csv", "")
        csv_path = os.path.join(DATA_DIR, file)

        print(f"Extracting meta-features for dataset {dataset_id}")

        try:
            meta = extract_meta_features(csv_path, target_col="target")
            meta["dataset"] = dataset_id
            rows.append(meta)
        except Exception as e:
            print(f"Skipping {dataset_id}: {e}")

    df = pd.DataFrame(rows)
    df.to_csv(OUTPUT_PATH, index=False)
    print(f"Saved meta-features to {OUTPUT_PATH}")

if __name__ == "__main__":
    build_meta_feature_table()
