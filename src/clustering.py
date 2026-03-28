"""
clustering.py
=============
Clusters merchants into Low / Medium / High risk groups using KMeans.

WHY KMeans?
-----------
Hierarchical clustering (Ward linkage):
  O(n²) memory and time.  100k rows × ~15 features → intractable.

DBSCAN:
  Produces arbitrary cluster shapes, but struggles when clusters have
  different densities (which they do — low-risk cluster is much denser
  than high-risk).  Also hard to enforce exactly k clusters.

KMeans:
  ✓ O(n·k·i) — scales to 100k rows.
  ✓ Produces compact, spherical clusters matching how risk bands naturally
    separate in refund/chargeback/volume space.
  ✓ Deterministic with fixed random_state.
  ✓ Silhouette score well-understood and interpretable.

WHY k=3?
--------
Business reality: most payment processors use a 3-tier risk classification
(Green / Amber / Red).  The synthetic dataset was engineered with 3 distinct
profiles (see data_generation.py), so the elbow will appear at k=3.
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")   # Non-interactive backend for server environments
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
import joblib
import os

CLUSTER_FEATURES = [
    "TransactionCount",
    "TotalVolume",
    "RefundRate",
    "ChargebackRate",
    "AvgTransactionValue",
    "nlp_risk_score",
]

CLUSTER_LABELS = {0: "Low Risk", 1: "Medium Risk", 2: "High Risk"}


def elbow_analysis(df: pd.DataFrame,
                   max_k: int = 10,
                   plot_path: str = "data/elbow_plot.png") -> dict:
    """
    Run KMeans for k = 1 … max_k, collect inertia and silhouette scores.
    Returns dict of results and saves elbow plot.
    """
    features = [f for f in CLUSTER_FEATURES if f in df.columns]
    X = df[features].fillna(0).values

    inertias    = []
    silhouettes = []
    k_range     = range(2, max_k + 1)

    for k in range(1, max_k + 1):
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(X)
        inertias.append(km.inertia_)
        if k >= 2:
            sil = silhouette_score(X, labels, sample_size=5000, random_state=42)
            silhouettes.append(sil)
        print(f"  k={k}  inertia={km.inertia_:.0f}"
              + (f"  silhouette={silhouettes[-1]:.3f}" if k >= 2 else ""))

    # ── Plot ──────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(range(1, max_k + 1), inertias, "bo-", linewidth=2)
    axes[0].set_xlabel("Number of Clusters (k)")
    axes[0].set_ylabel("Inertia (Within-Cluster Sum of Squares)")
    axes[0].set_title("Elbow Method — Optimal k")
    axes[0].axvline(x=3, color="red", linestyle="--", label="k=3 (selected)")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(range(2, max_k + 1), silhouettes, "gs-", linewidth=2)
    axes[1].set_xlabel("Number of Clusters (k)")
    axes[1].set_ylabel("Silhouette Score")
    axes[1].set_title("Silhouette Scores")
    axes[1].axvline(x=3, color="red", linestyle="--", label="k=3")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    os.makedirs(os.path.dirname(plot_path), exist_ok=True)
    plt.savefig(plot_path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"[Clustering] Elbow plot saved → {plot_path}")

    return {
        "k_range": list(range(1, max_k + 1)),
        "inertias": inertias,
        "silhouettes": silhouettes,
    }


def train_kmeans(df: pd.DataFrame,
                 k: int = 3,
                 model_path: str = "models/kmeans.pkl") -> pd.DataFrame:
    """
    Fit KMeans with k clusters; add cluster_id and cluster_label columns.
    Also saves PCA cluster visualisation.
    """
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    features = [f for f in CLUSTER_FEATURES if f in df.columns]
    X = df[features].fillna(0).values

    km = KMeans(n_clusters=k, random_state=42, n_init=20, max_iter=500)
    labels = km.fit_predict(X)

    # ── Label clusters by mean chargeback rate (ascending → Low to High) ────
    cluster_chargeback_means = {}
    for c in range(k):
        mask = labels == c
        if "ChargebackRate" in df.columns:
            cluster_chargeback_means[c] = df.loc[mask, "ChargebackRate"].mean()
        else:
            cluster_chargeback_means[c] = c

    sorted_clusters = sorted(cluster_chargeback_means, key=cluster_chargeback_means.get)
    remap = {sorted_clusters[i]: i for i in range(k)}
    labels = np.array([remap[l] for l in labels])

    df["cluster_id"]    = labels
    df["cluster_label"] = df["cluster_id"].map({0:"Low Risk",1:"Medium Risk",2:"High Risk"})

    sil = silhouette_score(X, labels, sample_size=5000, random_state=42)
    print(f"[Clustering] k=3  Silhouette Score: {sil:.4f}")
    print(df["cluster_label"].value_counts())

    joblib.dump(km, model_path)
    print(f"[Clustering] KMeans model saved → {model_path}")

    # ── PCA visualisation ─────────────────────────────────────────────────────
    pca = PCA(n_components=2, random_state=42)
    X_2d = pca.fit_transform(X)
    colors = ["#2ecc71", "#f39c12", "#e74c3c"]
    names  = ["Low Risk", "Medium Risk", "High Risk"]

    fig, ax = plt.subplots(figsize=(10, 7))
    for i, (cname, col) in enumerate(zip(names, colors)):
        mask = labels == i
        ax.scatter(X_2d[mask, 0], X_2d[mask, 1], c=col, label=cname,
                   alpha=0.4, s=5, edgecolors="none")
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% var)")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% var)")
    ax.set_title("KMeans Clusters (PCA projection)")
    ax.legend(markerscale=5)
    ax.grid(True, alpha=0.2)
    plt.tight_layout()
    plt.savefig("data/cluster_pca.png", dpi=120, bbox_inches="tight")
    plt.close()
    print("[Clustering] PCA cluster plot saved → data/cluster_pca.png")

    return df


def predict_cluster(df: pd.DataFrame,
                    model_path: str = "models/kmeans.pkl") -> pd.DataFrame:
    """Assign clusters to new data using saved model."""
    km = joblib.load(model_path)
    features = [f for f in CLUSTER_FEATURES if f in df.columns]
    X = df[features].fillna(0).values
    df["cluster_id"] = km.predict(X)
    df["cluster_label"] = df["cluster_id"].map({0:"Low Risk",1:"Medium Risk",2:"High Risk"})
    return df


if __name__ == "__main__":
    df = pd.read_csv("data/anomalies.csv")
    print("[Clustering] Running elbow analysis …")
    elbow_analysis(df)
    print("[Clustering] Fitting KMeans k=3 …")
    df = train_kmeans(df)
    df.to_csv("data/clustered.csv", index=False)
    print("[Clustering] Saved → data/clustered.csv")
