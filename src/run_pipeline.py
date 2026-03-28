"""
run_pipeline.py
===============
Orchestrates the complete ML pipeline end-to-end.
Run this once to generate all data and trained models.
"""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

from data_generation   import generate_dataset, save_dataset
from feature_engineering import engineer_features
from nlp_analysis      import add_nlp_scores
from preprocessing     import preprocess, load_raw
from anomaly_detection import train_isolation_forest
from clustering        import elbow_analysis, train_kmeans
from risk_scoring      import full_risk_pipeline

import pandas as pd

def run():
    os.makedirs("data",   exist_ok=True)
    os.makedirs("models", exist_ok=True)

    # ── Step 1: Generate raw dataset ──────────────────────────────────────────
    print("\n" + "="*60)
    print("STEP 1: DATA GENERATION")
    print("="*60)
    df = generate_dataset(100_000)
    save_dataset(df, "data/transactions.csv")

    # ── Step 2: Feature engineering (before scaling) ──────────────────────────
    print("\n" + "="*60)
    print("STEP 2: FEATURE ENGINEERING")
    print("="*60)
    df = engineer_features(df)

    # ── Step 3: NLP analysis ──────────────────────────────────────────────────
    print("\n" + "="*60)
    print("STEP 3: NLP ANALYSIS")
    print("="*60)
    df = add_nlp_scores(df)

    # ── Step 4: Anomaly detection (on unscaled features for interpretability) ─
    print("\n" + "="*60)
    print("STEP 4: ANOMALY DETECTION")
    print("="*60)
    df = train_isolation_forest(df)

    # ── Step 5: Preprocessing / scaling ──────────────────────────────────────
    print("\n" + "="*60)
    print("STEP 5: PREPROCESSING")
    print("="*60)
    # Save unscaled version first (needed for risk scoring and dashboard)
    df.to_csv("data/features_unscaled.csv", index=False)
    df_scaled = preprocess(df.copy(), fit=True)

    # ── Step 6: Clustering (on scaled features) ────────────────────────────────
    print("\n" + "="*60)
    print("STEP 6: CLUSTERING")
    print("="*60)
    elbow_analysis(df_scaled, plot_path="data/elbow_plot.png")
    df_scaled = train_kmeans(df_scaled)

    # Merge cluster labels back to unscaled df
    df["cluster_id"]    = df_scaled["cluster_id"].values
    df["cluster_label"] = df_scaled["cluster_label"].values

    # ── Step 7: Risk scoring ───────────────────────────────────────────────────
    print("\n" + "="*60)
    print("STEP 7: RISK SCORING")
    print("="*60)
    df = full_risk_pipeline(df, train_weights=True)

    # ── Save final dataset ─────────────────────────────────────────────────────
    df.to_csv("data/risk_scored.csv", index=False)
    print("\n" + "="*60)
    print("PIPELINE COMPLETE")
    print("="*60)
    print(f"Final dataset: data/risk_scored.csv  ({len(df):,} rows)")
    print(df[["MerchantID","Category","risk_score","risk_level",
              "is_anomaly","cluster_label"]].head(10))


if __name__ == "__main__":
    run()
