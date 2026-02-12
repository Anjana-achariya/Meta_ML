import os
import pandas as pd
import numpy as np
from scipy.stats import skew, entropy
from typing import Dict, Optional


def _safe_entropy(series: pd.Series, bins: int = 10) -> float:
    try:
        hist, _ = np.histogram(series.dropna(), bins=bins)
        return entropy(hist + 1)
    except Exception:
        return 0.0


def _get_data_reader(file_path: str, chunksize: int):
    """
    Returns an iterable of DataFrame chunks
    supporting CSV, Excel, and JSON formats.
    """

    file_ext = os.path.splitext(file_path)[1].lower()

    if file_ext == ".csv":
        return pd.read_csv(file_path, chunksize=chunksize)

    elif file_ext in [".xlsx", ".xls"]:
        df = pd.read_excel(file_path)
        return [df]  # simulate chunk iteration

    elif file_ext == ".json":
        try:
            df = pd.read_json(file_path)
        except ValueError:
            # Try JSON Lines format
            df = pd.read_json(file_path, lines=True)
        return [df]

    else:
        raise ValueError(
            "Unsupported file format. Please upload CSV, Excel, or JSON."
        )


def extract_meta_features(
    file_path: str,
    target_col: Optional[str] = None,
    chunksize: int = 50_000,
    max_chunks: Optional[int] = None
) -> Dict[str, float]:

    n_rows = 0
    n_missing = 0
    numeric_cols = None
    categorical_cols = None
    skewness_values = []
    entropy_values = []

    reader = _get_data_reader(file_path, chunksize)

    for idx, chunk in enumerate(reader):

        if max_chunks and idx >= max_chunks:
            break

        if chunk.empty:
            continue

        n_rows += len(chunk)

        if target_col and target_col in chunk.columns:
            chunk = chunk.drop(columns=[target_col])

        if numeric_cols is None:
            numeric_cols = chunk.select_dtypes(include=np.number).columns.tolist()
            categorical_cols = chunk.select_dtypes(exclude=np.number).columns.tolist()

        n_missing += chunk.isna().sum().sum()

        for col in numeric_cols:
            col_data = chunk[col].dropna()
            if len(col_data) > 20:
                skewness_values.append(skew(col_data))
                entropy_values.append(_safe_entropy(col_data))

    n_features = (len(numeric_cols) if numeric_cols else 0) + \
                 (len(categorical_cols) if categorical_cols else 0)

    n_features = max(n_features, 1)
    n_rows = max(n_rows, 1)

    meta_features = {
        "n_rows": float(n_rows),
        "n_features": float(n_features),
        "numeric_ratio": len(numeric_cols) / n_features if numeric_cols else 0.0,
        "categorical_ratio": len(categorical_cols) / n_features if categorical_cols else 0.0,
        "missing_ratio": n_missing / (n_rows * n_features),
        "mean_skewness": float(np.mean(skewness_values)) if skewness_values else 0.0,
        "mean_entropy": float(np.mean(entropy_values)) if entropy_values else 0.0,
        "feature_row_ratio": n_features / n_rows,
        "log_rows": float(np.log1p(n_rows)),
        "log_features": float(np.log1p(n_features)),
    }

    return meta_features
