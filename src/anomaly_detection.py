"""
anomaly_detection.py
====================
Detects abnormal merchants using Isolation Forest.

WHY Isolation Forest over alternatives?
----------------------------------------
Z-score:
  Assumes Gaussian distribution — invalid for transaction data which is
  right-skewed (long tail of very high-volume merchants).
  Also univariate; can't capture multi-dimensional outliers
  (e.g., low volume + extreme chargeback rate).

DBSCAN:
  Density-based; requires careful tuning of eps and min_samples.
  In high-dimensional spaces (curse of dimensionality) density estimation
  degrades.  Also scales poorly to 100k rows.

Isolation Forest:
  ✓ Works well in high-dimensional space.
  ✓ Makes NO assumption about the underlying distribution.
  ✓ Specifically designed for rare anomalies (contamination parameter).
  ✓ O(n log n) training; efficient on 100k rows.
  ✓ Outputs anomaly score (not just binary flag) → useful for thresholding.

contamination=0.01 → we injected ~1% anomalies, so we tell the model
the expected proportion.  In production this would be estimated from
historical fraud rates.
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
import joblib
import os

ANOMALY_FEATURES = [
    "TransactionCount",
    "TotalVolume",
    "RefundRate",
    "ChargebackRate",
    "AvgTransactionValue",
    "TxnGrowthRate",
]


def train_isolation_forest(df: pd.DataFrame,
                            contamination: float = 0.01,
                            model_path: str = "models/isolation_forest.pkl"
                            ) -> pd.DataFrame:
    """
    Fit Isolation Forest and add anomaly columns to df.

    Returns
    -------
    df with two new columns:
      anomaly_score : float  — lower (more negative) = more anomalous
      is_anomaly    : int    — 1 = anomaly, 0 = normal
    """
    os.makedirs(os.path.dirname(model_path), exist_ok=True)

    # Use only columns that exist in df
    features = [f for f in ANOMALY_FEATURES if f in df.columns]
    X = df[features].fillna(0)

    model = IsolationForest(
        n_estimators=200,       # More trees → more stable scores
        contamination=contamination,
        random_state=42,
        n_jobs=-1,              # Use all CPU cores
    )
    model.fit(X)

    # decision_function: higher = more normal; lower = more anomalous
    df["anomaly_score"] = model.decision_function(X)
    # predict: -1 = anomaly, 1 = normal → map to 1/0
    df["is_anomaly"] = (model.predict(X) == -1).astype(int)

    joblib.dump(model, model_path)
    print(f"[AnomalyDetection] Model saved → {model_path}")
    print(f"[AnomalyDetection] Detected anomalies: {df['is_anomaly'].sum():,} "
          f"({df['is_anomaly'].mean()*100:.2f}%)")

    # If ground truth available, report precision
    if "is_anomaly_injected" in df.columns:
        tp = ((df["is_anomaly"] == 1) & (df["is_anomaly_injected"] == 1)).sum()
        fp = ((df["is_anomaly"] == 1) & (df["is_anomaly_injected"] == 0)).sum()
        fn = ((df["is_anomaly"] == 0) & (df["is_anomaly_injected"] == 1)).sum()
        precision = tp / (tp + fp + 1e-9)
        recall    = tp / (tp + fn + 1e-9)
        print(f"[AnomalyDetection] Precision: {precision:.2f}  Recall: {recall:.2f}")

    return df


def predict_anomaly(df: pd.DataFrame,
                    model_path: str = "models/isolation_forest.pkl") -> pd.DataFrame:
    """Load saved model and score new data."""
    model = joblib.load(model_path)
    features = [f for f in ANOMALY_FEATURES if f in df.columns]
    X = df[features].fillna(0)
    df["anomaly_score"] = model.decision_function(X)
    df["is_anomaly"]    = (model.predict(X) == -1).astype(int)
    return df


if __name__ == "__main__":
    df = pd.read_csv("data/features.csv")
    df = train_isolation_forest(df)
    df.to_csv("data/anomalies.csv", index=False)
    print("[AnomalyDetection] Saved → data/anomalies.csv")
