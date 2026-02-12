import os
import openml
import pandas as pd
from typing import List, Dict

class OpenMLClient:
    """
    Production-safe OpenML dataset fetcher for Meta-ML.
    Ensures:
    - consistent target column
    - dataset size limits
    - caching
    """

    def __init__(
        self,
        cache_dir: str = "data/openml_raw",
        max_rows: int = 200_000,
        max_features: int = 1_000
    ):
        self.cache_dir = cache_dir
        self.max_rows = max_rows
        self.max_features = max_features
        os.makedirs(self.cache_dir, exist_ok=True)

    def list_datasets(
        self,
        limit: int = 100,
        task_type: str = "classification"
    ) -> List[Dict]:

        df = openml.datasets.list_datasets(output_format="dataframe")

        # Basic sanity filters
        df = df[
            (df["NumberOfInstances"] > 50) &
            (df["NumberOfFeatures"] > 1) &
            (df["NumberOfInstances"] <= self.max_rows) &
            (df["NumberOfFeatures"] <= self.max_features)
        ]

        if task_type == "classification":
            df = df[df["NumberOfClasses"] > 1]
        elif task_type == "regression":
            df = df[df["NumberOfClasses"].isna()]

        # Prefer smaller datasets first (faster benchmarking)
        df = df.sort_values("NumberOfInstances", ascending=True)

        return df.head(limit).to_dict(orient="records")

    def fetch_dataset(self, dataset_id: int) -> bool:
        """
        Fetch dataset and attach OpenML default target as `target`.
        Returns True if successful.
        """

        cache_path = os.path.join(self.cache_dir, f"{dataset_id}.csv")

        if os.path.exists(cache_path):
            return True

        try:
            dataset = openml.datasets.get_dataset(dataset_id)

            if not dataset.default_target_attribute:
                print(f"Skipping {dataset_id}: no default target")
                return False

            X, y, _, _ = dataset.get_data(
                dataset_format="dataframe",
                target=dataset.default_target_attribute
            )

            if y is None:
                print(f"Skipping {dataset_id}: target extraction failed")
                return False

            X["target"] = y
            X.to_csv(cache_path, index=False)

            return True

        except Exception as e:
            print(f"Skipping {dataset_id}: {e}")
            return False
