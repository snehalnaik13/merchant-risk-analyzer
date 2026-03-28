"""
risk_scoring.py
===============
Two-stage risk scoring:

Stage 1 — Feature Weighting via Random Forest importance
---------------------------------------------------------
WHY Random Forest for weights?
  Random Forest captures non-linear relationships and interactions between
  features (e.g., high ChargebackRate AND high Volume is riskier than either
  alone).  Feature importance from RF gives a principled, data-driven weight
  rather than arbitrary manual weights.

  Correlation-based weights are an alternative but assume linearity and can be
  dominated by highly correlated feature pairs.

Stage 2 — Rule-Based Final Classification
------------------------------------------
WHY rules on top of the model score?
  Regulatory compliance requires explainable, auditable decisions.
  "The model said so" is not acceptable under RBI merchant risk guidelines.
  Rules enforce hard thresholds that can be reviewed by a compliance officer.
  Anomaly flag and NLP signal can override moderate model scores.
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import joblib
import os

# ── Features used for risk score computation ─────────────────────────────────
SCORING_FEATURES = [
    "TransactionCount",
    "TotalVolume",
    "RefundRate",
    "ChargebackRate",
    "nlp_risk_score",
    "AvgTransactionValue",
    "TxnGrowthRate",
    "is_festival_period",
    "Month_sin",
    "Month_cos",
]

# Default weights (used as fallback if RF training not available)
DEFAULT_WEIGHTS = {
    "ChargebackRate":       0.30,
    "RefundRate":           0.20,
    "nlp_risk_score":       0.15,
    "TxnGrowthRate":        0.10,
    "TotalVolume":          0.10,
    "AvgTransactionValue":  0.07,
    "TransactionCount":     0.04,
    "is_festival_period":   0.02,
    "Month_sin":            0.01,
    "Month_cos":            0.01,
}

# ── Risk thresholds ───────────────────────────────────────────────────────────
# These map normalised risk_score [0, 1] to bands.
# Low:    < 0.35
# Medium: 0.35 – 0.65
# High:   > 0.65
LOW_THRESHOLD    = 0.35
MEDIUM_THRESHOLD = 0.65


def train_rf_for_weights(df: pd.DataFrame,
                          target_col: str = "cluster_id",
                          model_path: str = "models/rf_weights.pkl") -> dict:
    """
    Train a Random Forest classifier to get feature importances as weights.

    The cluster_id (from KMeans) serves as a proxy risk target label.
    WHY proxy label?  We have no ground-truth labelled fraud data; cluster_id
    is the best available risk proxy and the RF learns which features most
    strongly separate clusters.
    """
    os.makedirs(os.path.dirname(model_path), exist_ok=True)

    features = [f for f in SCORING_FEATURES if f in df.columns]
    if target_col not in df.columns:
        print("[RiskScoring] cluster_id not found, using default weights.")
        return DEFAULT_WEIGHTS

    X = df[features].fillna(0)
    y = df[target_col].astype(int)

    rf = RandomForestClassifier(
        n_estimators=200,
        max_depth=8,
        random_state=42,
        n_jobs=-1,
        class_weight="balanced",
    )
    rf.fit(X, y)
    joblib.dump(rf, model_path)
    print(f"[RiskScoring] RF model saved → {model_path}")

    # Normalise importances to sum = 1
    raw_importances = rf.feature_importances_
    total = raw_importances.sum()
    weights = {f: round(float(imp / total), 4)
               for f, imp in zip(features, raw_importances)}

    print("[RiskScoring] Feature weights (RF importance):")
    for f, w in sorted(weights.items(), key=lambda x: -x[1]):
        print(f"  {f:<30} {w:.4f}")

    return weights


def compute_risk_score(df: pd.DataFrame,
                       weights: dict = None) -> pd.DataFrame:
    """
    Compute a normalised risk_score ∈ [0, 1] for each row.

    Steps
    -----
    1. Normalise each feature to [0, 1] with min-max scaling
       (done locally here so it's independent of the StandardScaler used for
       clustering — risk score normalisation is for interpretability).
    2. Compute weighted sum.
    3. Clip to [0, 1].
    """
    if weights is None:
        weights = DEFAULT_WEIGHTS

    score = pd.Series(np.zeros(len(df)), index=df.index)

    for feat, w in weights.items():
        if feat not in df.columns:
            continue
        col = df[feat].fillna(0).astype(float)
        col_min, col_max = col.min(), col.max()
        if col_max > col_min:
            norm = (col - col_min) / (col_max - col_min)
        else:
            norm = col * 0   # constant feature → zero contribution
        score += w * norm

    # Anomaly flag boosts score by 0.20 (capped at 1.0)
    if "is_anomaly" in df.columns:
        score += df["is_anomaly"].fillna(0) * 0.20

    df["risk_score"] = score.clip(0, 1).round(4)
    print(f"[RiskScoring] risk_score computed.  "
          f"Mean: {df['risk_score'].mean():.3f}  "
          f"Max: {df['risk_score'].max():.3f}")
    return df


def assign_risk_level(df: pd.DataFrame) -> pd.DataFrame:
    """
    Map continuous risk_score to categorical risk_level using hard thresholds.

    Rule logic
    ----------
    - risk_score > MEDIUM_THRESHOLD  → High Risk
    - risk_score > LOW_THRESHOLD     → Medium Risk
    - otherwise                      → Low Risk

    Override rules (hard rules that take precedence):
    - ChargebackRate > 0.10  → at least Medium Risk
    - ChargebackRate > 0.25  → High Risk
    - is_anomaly == 1        → at least Medium Risk
    - nlp_risk_score > 0.60  → at least Medium Risk
    """
    def _classify(row):
        score = row.get("risk_score", 0)
        cb    = row.get("ChargebackRate", 0)
        anom  = row.get("is_anomaly", 0)
        nlp   = row.get("nlp_risk_score", 0)

        # Model-score baseline
        if score >= MEDIUM_THRESHOLD:
            level = "High Risk"
        elif score >= LOW_THRESHOLD:
            level = "Medium Risk"
        else:
            level = "Low Risk"

        # Hard override rules pushing numeric synchronisation 
        if cb > 0.25:
            level = "High Risk"
            score = max(score, 0.75)
        elif cb > 0.10 and level == "Low Risk":
            level = "Medium Risk"
            score = max(score, 0.45)

        if anom == 1 and level == "Low Risk":
            level = "Medium Risk"
            score = max(score, 0.45)

        if nlp > 0.60 and level == "Low Risk":
            level = "Medium Risk"
            score = max(score, 0.45)

        return pd.Series([level, round(score, 3)])

    df[["risk_level", "risk_score"]] = df.apply(_classify, axis=1)
    print("[RiskScoring] Risk level distribution:")
    print(df["risk_level"].value_counts())
    return df


def full_risk_pipeline(df: pd.DataFrame,
                       train_weights: bool = True) -> pd.DataFrame:
    """End-to-end risk scoring: weights → score → level."""
    if train_weights:
        weights = train_rf_for_weights(df)
    else:
        weights = DEFAULT_WEIGHTS
    df = compute_risk_score(df, weights)
    df = assign_risk_level(df)
    return df


if __name__ == "__main__":
    df = pd.read_csv("data/clustered.csv")
    df = full_risk_pipeline(df, train_weights=True)
    df.to_csv("data/risk_scored.csv", index=False)
    print("[RiskScoring] Saved → data/risk_scored.csv")
    print(df[["MerchantID","risk_score","risk_level"]].head(10))
