import os
import time
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
from sklearn.metrics import accuracy_score, f1_score

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer

from sklearn.linear_model import LogisticRegression

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    RandomForestClassifier,
    ExtraTreesClassifier,
    GradientBoostingClassifier
)
from sklearn.svm import SVC

DATA_DIR = "data/openml_raw"
OUTPUT_DIR = "data"
os.makedirs(OUTPUT_DIR, exist_ok=True)

MODELS = {
    "logistic_regression": LogisticRegression(max_iter=1000),
   
    "decision_tree": DecisionTreeClassifier(),
    "random_forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "extra_trees": ExtraTreesClassifier(n_estimators=100, random_state=42),
    "gradient_boosting": GradientBoostingClassifier(random_state=42),
    "svm_rbf": SVC(kernel="rbf")
}

def run_models_on_dataset(csv_path: str):
    df = pd.read_csv(csv_path)

    if df.shape[0] < 50:
        return None

    # Detect target column
    target_col = "target" if "target" in df.columns else df.columns[-1]

    X = df.drop(columns=[target_col])
    y = df[target_col]

    # Encode target if categorical
    if y.dtype == "object" or y.dtype.name == "category":
        y = LabelEncoder().fit_transform(y)

    # Identify feature types
    numeric_cols = X.select_dtypes(include=np.number).columns.tolist()
    categorical_cols = X.select_dtypes(exclude=np.number).columns.tolist()
    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_cols),
            ("cat", categorical_transformer, categorical_cols)
        ]
    )
    # Check if stratification is possible
    class_counts = np.bincount(y)

    use_stratify = np.all(class_counts >= 2)

    X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y if use_stratify else None
)



    results = []

    for model_name, model in MODELS.items():
        try:
            pipeline = Pipeline(steps=[
                ("preprocessor", preprocessor),
                ("model", model)
            ])

            start = time.time()
            pipeline.fit(X_train, y_train)
            train_time = time.time() - start

            preds = pipeline.predict(X_test)

            results.append({
                "dataset": os.path.basename(csv_path).replace(".csv", ""),
                "model": model_name,
                "accuracy": accuracy_score(y_test, preds),
                "f1_score": f1_score(y_test, preds, average="weighted"),
                "training_time": train_time
            })

        except Exception as e:
            print(f"Skipping model {model_name} for {csv_path}: {e}")

    return results


def main():
    all_results = []

    for file in os.listdir(DATA_DIR):
        if file.endswith(".csv"):
            path = os.path.join(DATA_DIR, file)
            print(f"Processing {file}")

            res = run_models_on_dataset(path)
            if res:
                all_results.extend(res)

    if not all_results:
        print("No results collected.")
        return

    df = pd.DataFrame(all_results)
    df.to_csv(os.path.join(OUTPUT_DIR, "performance_labels.csv"), index=False)
    print("Saved performance_labels.csv")


if __name__ == "__main__":
    main()
