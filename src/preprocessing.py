"""
preprocessing.py
================
Handles all data cleaning, encoding and scaling steps.

Design Decisions
----------------
WHY StandardScaler over MinMaxScaler?
  KMeans and Isolation Forest rely on Euclidean / distance metrics.
  StandardScaler centres features at 0 with unit variance, so no single
  feature dominates just because it has a large numeric range
  (e.g. TotalVolume in millions vs RefundRate in [0,1]).
  MinMaxScaler is sensitive to outliers which we intentionally injected —
  it would compress most values into a tiny band.

WHY One-Hot Encoding for Category?
  Category is nominal (no ordinal relationship: Travel ≠ 2×Grocery).
  Label encoding would introduce a false ordinal relationship that distance-
  based models would exploit incorrectly.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import joblib
import os

NUMERICAL_FEATURES = [
    "TransactionCount",
    "TotalVolume",
    "RefundRate",
    "ChargebackRate",
]

CATEGORICAL_FEATURES = ["Category"]


# ── Main preprocessing pipeline ──────────────────────────────────────────────

def load_raw(path: str = "data/transactions.csv") -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["Timestamp"])
    print(f"[Preprocessing] Loaded {len(df):,} rows from {path}")
    return df


def handle_missing(df: pd.DataFrame) -> pd.DataFrame:
    """
    Impute or drop missing values.
    Strategy:
      - Numerical: fill with median (robust to outliers).
      - Categorical: fill with mode.
    WHY median for numericals?  Mean is pulled by injected anomalies.
    """
    before = df.isnull().sum().sum()
    for col in NUMERICAL_FEATURES:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].median())
    for col in CATEGORICAL_FEATURES:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].mode()[0])
    after = df.isnull().sum().sum()
    print(f"[Preprocessing] Missing values: {before} → {after}")
    return df


def encode_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    """One-hot encode Category.  drop_first=False keeps all dummies for
    explainability (no reference-category confusion in dashboards)."""
    df = pd.get_dummies(df, columns=["Category"], drop_first=False, dtype=int)
    print(f"[Preprocessing] After OHE, shape: {df.shape}")
    return df


def scale_features(df: pd.DataFrame,
                   scaler_path: str = "models/scaler.pkl",
                   fit: bool = True) -> pd.DataFrame:
    """
    Scale NUMERICAL_FEATURES with StandardScaler.

    Parameters
    ----------
    fit : bool
        True  → fit a new scaler and save it (training time).
        False → load saved scaler and transform only (inference time).
    """
    os.makedirs(os.path.dirname(scaler_path), exist_ok=True)

    # Only scale columns that actually exist (in case dataset changes)
    cols_to_scale = [c for c in NUMERICAL_FEATURES if c in df.columns]

    if fit:
        scaler = StandardScaler()
        df[cols_to_scale] = scaler.fit_transform(df[cols_to_scale])
        joblib.dump(scaler, scaler_path)
        print(f"[Preprocessing] Scaler fitted and saved → {scaler_path}")
    else:
        scaler = joblib.load(scaler_path)
        df[cols_to_scale] = scaler.transform(df[cols_to_scale])
        print(f"[Preprocessing] Scaler loaded from {scaler_path}")

    return df


def preprocess(df: pd.DataFrame,
               scaler_path: str = "models/scaler.pkl",
               fit: bool = True) -> pd.DataFrame:
    """Full preprocessing pipeline: clean → encode → scale."""
    df = handle_missing(df)
    df = encode_categoricals(df)
    df = scale_features(df, scaler_path=scaler_path, fit=fit)
    return df


if __name__ == "__main__":
    raw = load_raw("data/transactions.csv")
    processed = preprocess(raw)
    processed.to_csv("data/processed.csv", index=False)
    print("[Preprocessing] Saved → data/processed.csv")
    print(processed.head())
