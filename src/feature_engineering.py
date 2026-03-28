"""
feature_engineering.py
=======================
Derives richer features from raw columns.

Key Techniques
--------------
1. Temporal features from Timestamp.
2. Cyclical encoding of periodic features (month, day-of-week).
   WHY cyclical?  Month 12 (December) is numerically far from Month 1 (January)
   but semantically adjacent.  Feeding raw month numbers breaks that continuity.
   sin/cos projections map the circle correctly:
       month_sin = sin(2π * month / 12)
       month_cos = cos(2π * month / 12)
   This gives the model a smooth, distance-preserving representation.

3. AvgTransactionValue = TotalVolume / TransactionCount
   WHY?  A merchant with 1,000 transactions and ₹1 Cr volume behaves very
   differently from one with 100 transactions and ₹1 Cr volume.

4. TxnGrowthRate: simulated per-merchant transaction trend.
   WHY?  A sudden spike is a stronger risk signal than an absolute high value.
"""

import pandas as pd
import numpy as np
import os

TWO_PI = 2 * np.pi


def add_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    """Extract calendar fields from Timestamp."""
    ts = pd.to_datetime(df["Timestamp"])
    df["Month"]       = ts.dt.month
    df["DayOfWeek"]   = ts.dt.dayofweek          # 0=Monday … 6=Sunday
    df["IsWeekend"]   = (df["DayOfWeek"] >= 5).astype(int)
    df["WeekOfYear"]  = ts.dt.isocalendar().week.astype(int)
    df["Hour"]        = ts.dt.hour
    df["Quarter"]     = ts.dt.quarter
    print("[FeatureEng] Temporal features added.")
    return df


def add_cyclical_encoding(df: pd.DataFrame) -> pd.DataFrame:
    """
    Encode periodic features as (sin, cos) pairs.

    Periods used:
      - Month:      12 months
      - DayOfWeek:  7 days
      - Hour:       24 hours
      - WeekOfYear: 52 weeks
    """
    cycles = {
        "Month":      12,
        "DayOfWeek":  7,
        "Hour":       24,
        "WeekOfYear": 52,
    }
    for col, period in cycles.items():
        if col in df.columns:
            df[f"{col}_sin"] = np.sin(TWO_PI * df[col] / period)
            df[f"{col}_cos"] = np.cos(TWO_PI * df[col] / period)
    print("[FeatureEng] Cyclical encoding done.")
    return df


def add_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute ratio and growth-rate features.

    AvgTransactionValue
    -------------------
    WHY?  Normalises volume by activity; detects micro-transaction fraud
    (many tiny txns inflating count) vs macro-fraud (few huge txns).

    TxnGrowthRate
    -------------
    WHY?  A 10× spike from the merchant's own historical baseline is more
    suspicious than an absolute high count.  We simulate this because the
    synthetic dataset has one row per observation, not a true time-series.
    In production this would be computed from a rolling window.
    """
    # Avoid division by zero
    df["AvgTransactionValue"] = df["TotalVolume"] / df["TransactionCount"].replace(0, 1)

    # Simulate a growth rate: ±10-20% normal; anomalies will have higher values
    # (In production: (current_period_volume - previous_period_volume) / previous)
    np.random.seed(0)
    base_growth = np.random.normal(0.05, 0.12, size=len(df))
    # Bump growth for injected anomalies if the column exists
    if "is_anomaly_injected" in df.columns:
        anomaly_mask = df["is_anomaly_injected"] == 1
        base_growth[anomaly_mask] = np.random.uniform(1.5, 8.0, size=anomaly_mask.sum())
    df["TxnGrowthRate"] = np.round(base_growth, 4)

    print("[FeatureEng] Derived features (AvgTransactionValue, TxnGrowthRate) added.")
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Full feature engineering pipeline."""
    df = add_temporal_features(df)
    df = add_cyclical_encoding(df)
    df = add_derived_features(df)
    return df


if __name__ == "__main__":
    df = pd.read_csv("data/transactions.csv", parse_dates=["Timestamp"])
    df = engineer_features(df)
    os.makedirs("data", exist_ok=True)
    df.to_csv("data/features.csv", index=False)
    print("[FeatureEng] Saved → data/features.csv")
    print(df[["Month","Month_sin","Month_cos","AvgTransactionValue","TxnGrowthRate"]].head())
