"""
data_generation.py
==================
Generates a synthetic dataset of 100,000 UNIQUE MERCHANTS — one row per merchant.

Each row represents a single merchant's aggregated lifetime profile:
  - TransactionCount  : total transactions processed (lifetime aggregate)
  - TotalVolume       : total ₹ volume processed (lifetime aggregate)
  - RefundRate        : refund rate across all transactions
  - ChargebackRate    : chargeback rate across all transactions
  - Category          : the merchant's business category (one per merchant)
  - Description       : a text description of the merchant's business
  - Timestamp         : merchant onboarding / registration date

WHY one row per merchant?
  Risk analysis is merchant-level, not transaction-level. We assess
  the aggregate behaviour of a merchant across ALL their transactions.
  One row per merchant means each MerchantID is globally unique and
  the dataset represents the full merchant master table.

WHY synthetic data?
  Real merchant data is PCI-DSS sensitive. Synthetic data lets us
  build, test and demo the full ML pipeline without privacy/legal risk.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
import os

# ── Reproducibility ───────────────────────────────────────────────────────────
SEED = 42
np.random.seed(SEED)
random.seed(SEED)

# ── Festival Date Ranges ──────────────────────────────────────────────────────
# WHY date-ranges?  Festivals shift year to year and span multiple days;
# month-level granularity is too coarse.
# The Timestamp (onboarding date) determines whether the merchant joined
# during a high-activity festival season, which influences their initial
# volume profile.
FESTIVAL_PERIODS = [
    # (name, month, start_day, end_day)
    ("Diwali",               10, 20, 28),
    ("Dussehra",             10,  5, 10),
    ("Navratri",             10,  1,  9),
    ("Eid",                   4, 10, 14),
    ("EidAdha",               6, 28, 30),
    ("Christmas",            12, 20, 31),
    ("NewYear",               1,  1,  3),
    ("BigBillionDays",       10,  8, 15),
    ("GreatIndianFestival",  10,  8, 22),
    ("Holi",                  3, 24, 26),
    ("RepublicDay",           1, 24, 27),
    ("IndependenceDay",       8, 13, 16),
]


def is_festival(month: int, day: int) -> bool:
    """Return True if (month, day) falls in any festival window."""
    for _, fm, fstart, fend in FESTIVAL_PERIODS:
        if month == fm and fstart <= day <= fend:
            return True
    return False


def festival_multiplier(month: int, day: int, category: str) -> float:
    """
    Return a volume/activity multiplier based on festival relevance to category.

    WHY category-specific multipliers?
      Grocery spikes on Diwali (gifting, sweets); Travel spikes on holidays;
      Gaming/Crypto spike on long weekends / year-end.
      A uniform multiplier would hide category-level seasonality signals.
    """
    base = 1.0
    for fname, fm, fstart, fend in FESTIVAL_PERIODS:
        if month == fm and fstart <= day <= fend:
            if fname in ("Diwali", "BigBillionDays", "GreatIndianFestival",
                         "Navratri", "Dussehra"):
                bumps = {"Grocery": 3.5, "Travel": 2.0, "Gaming": 2.5,
                         "Crypto": 1.8, "Food": 2.2, "Services": 1.6}
            elif fname in ("Christmas", "NewYear"):
                bumps = {"Travel": 3.5, "Gaming": 3.0, "Crypto": 2.5,
                         "Food": 2.0, "Grocery": 2.0, "Services": 1.5}
            elif fname in ("Eid", "EidAdha"):
                bumps = {"Grocery": 2.8, "Travel": 2.5, "Food": 2.0,
                         "Gaming": 1.5, "Crypto": 1.3, "Services": 1.4}
            elif fname == "Holi":
                bumps = {"Grocery": 2.0, "Food": 2.5, "Gaming": 1.8,
                         "Travel": 1.5, "Crypto": 1.2, "Services": 1.2}
            else:
                bumps = {"Grocery": 1.5, "Travel": 1.5, "Gaming": 1.5,
                         "Crypto": 1.3, "Food": 1.4, "Services": 1.2}
            base = max(base, bumps.get(category, 1.2))
    return base


# ── Category Definitions ──────────────────────────────────────────────────────
CATEGORIES = ["Travel", "Crypto", "Gaming", "Food", "Services", "Grocery"]

CATEGORY_DESCRIPTIONS = {
    "Travel":   [
        "Flight bookings and holiday packages",
        "Hotel reservations and tour packages",
        "International travel ticketing platform",
        "Domestic bus and train booking service",
        "Adventure travel and trekking packages",
        "Corporate travel management platform",
    ],
    "Crypto":   [
        "Cryptocurrency exchange platform",
        "Digital asset trading and investment",
        "Bitcoin and altcoin wallet service",
        "DeFi and crypto staking platform",
        "NFT marketplace and digital collectibles",
        "Crypto portfolio management service",
    ],
    "Gaming":   [
        "Online gaming and in-app purchases",
        "Mobile gaming marketplace",
        "Fantasy sports and e-gaming platform",
        "PC and console game downloads",
        "Esports tournament and betting platform",
        "Game subscription and pass service",
    ],
    "Food":     [
        "Online food delivery aggregator",
        "Restaurant chain payment gateway",
        "Cloud kitchen and meal subscription",
        "Grocery and food delivery service",
        "Healthy meal plan and diet service",
        "Catering and event food management",
    ],
    "Services": [
        "SaaS subscription billing platform",
        "Professional services invoicing",
        "Utility and bill payment gateway",
        "Digital marketing services payment",
        "Freelance marketplace payment processing",
        "Healthcare and telemedicine billing",
    ],
    "Grocery":  [
        "Online grocery and daily essentials",
        "Kirana store digital payment",
        "Supermarket chain payment gateway",
        "Fresh produce home delivery",
        "Organic and natural food store",
        "Wholesale grocery distribution platform",
    ],
}

# ── Risk Cluster Profiles ─────────────────────────────────────────────────────
# WHY 3 clusters?  Industry standard: Green / Amber / Red.
# We engineer clear separation so KMeans elbow shows k=3 and silhouette is high.
#
# Key separation dimensions:
#   Low Risk    -> low chargeback, low refund, moderate volume
#   Medium Risk -> moderate chargeback & refund, higher volume
#   High Risk   -> high chargeback & refund, very high volume
CLUSTER_PROFILES = {
    "low": {
        "txn_count_mean": 400,       "txn_count_std": 80,
        "volume_mean":    500_000,   "volume_std":    80_000,
        "refund_rate_mean":     0.02, "refund_rate_std":     0.005,
        "chargeback_rate_mean": 0.003,"chargeback_rate_std": 0.001,
    },
    "medium": {
        "txn_count_mean": 700,         "txn_count_std": 150,
        "volume_mean":    1_200_000,   "volume_std":    200_000,
        "refund_rate_mean":     0.07,  "refund_rate_std":     0.015,
        "chargeback_rate_mean": 0.012, "chargeback_rate_std": 0.003,
    },
    "high": {
        "txn_count_mean": 1_200,       "txn_count_std": 300,
        "volume_mean":    3_000_000,   "volume_std":    600_000,
        "refund_rate_mean":     0.18,  "refund_rate_std":     0.04,
        "chargeback_rate_mean": 0.045, "chargeback_rate_std": 0.010,
    },
}

# Category -> cluster probability biases (realistic risk distribution)
HIGH_RISK_CATEGORIES   = {"Crypto", "Gaming"}
MEDIUM_RISK_CATEGORIES = {"Travel", "Food"}


def category_cluster_bias(category: str) -> str:
    """
    Assign a risk cluster to a merchant, biased by category.

    WHY bias?  Crypto/Gaming have structurally higher fraud rates (RBI data).
    Grocery/Services are lower risk by nature.
    """
    r = np.random.random()
    if category in HIGH_RISK_CATEGORIES:
        if r < 0.35: return "high"
        elif r < 0.75: return "medium"
        else: return "low"
    elif category in MEDIUM_RISK_CATEGORIES:
        if r < 0.15: return "high"
        elif r < 0.60: return "medium"
        else: return "low"
    else:  # Grocery, Services
        if r < 0.06: return "high"
        elif r < 0.30: return "medium"
        else: return "low"


def generate_dataset(n_merchants: int = 100_000) -> pd.DataFrame:
    """
    Generate one row per unique merchant — 100,000 merchants total.

    Each row is the merchant's aggregated lifetime profile.
    The Timestamp is the merchant's onboarding/registration date, which
    determines whether they onboarded during a festival season.

    Parameters
    ----------
    n_merchants : int
        Number of unique merchants (default 100,000).

    Returns
    -------
    pd.DataFrame with shape (n_merchants, columns).
    Every MerchantID is unique — enforced by assertion.
    """
    print(f"[DataGen] Generating {n_merchants:,} unique merchants (1 row each) ...")

    # ── Unique Merchant IDs ────────────────────────────────────────────────────
    merchant_ids = [f"MID{str(i).zfill(6)}" for i in range(1, n_merchants + 1)]

    # ── Pre-assign stable merchant attributes ─────────────────────────────────
    categories   = np.random.choice(CATEGORIES, size=n_merchants)
    clusters     = np.array([category_cluster_bias(c) for c in categories])
    descriptions = np.array([
        np.random.choice(CATEGORY_DESCRIPTIONS[cat])
        for cat in categories
    ])

    # ── Onboarding timestamps (dynamic to current date) ────────────────────────
    start_date      = datetime(2022, 1, 1)
    end_date        = datetime.today()
    date_range_days = (end_date - start_date).days

    rand_days    = np.random.randint(0, date_range_days, size=n_merchants)
    rand_hours   = np.random.randint(0, 24, size=n_merchants)
    rand_minutes = np.random.randint(0, 60, size=n_merchants)

    timestamps = [
        start_date + timedelta(days=int(d), hours=int(h), minutes=int(m))
        for d, h, m in zip(rand_days, rand_hours, rand_minutes)
    ]

    months = np.array([ts.month for ts in timestamps])
    days   = np.array([ts.day   for ts in timestamps])

    # ── Festival flags & multipliers ──────────────────────────────────────────
    fest_flags = np.array([
        int(is_festival(m, d)) for m, d in zip(months, days)
    ])
    fest_mults = np.array([
        festival_multiplier(m, d, c)
        for m, d, c in zip(months, days, categories)
    ])

    # ── Per-cluster base parameters (vectorised lookup) ───────────────────────
    txn_means  = np.array([CLUSTER_PROFILES[cl]["txn_count_mean"]        for cl in clusters])
    txn_stds   = np.array([CLUSTER_PROFILES[cl]["txn_count_std"]         for cl in clusters])
    vol_means  = np.array([CLUSTER_PROFILES[cl]["volume_mean"]            for cl in clusters])
    vol_stds   = np.array([CLUSTER_PROFILES[cl]["volume_std"]             for cl in clusters])
    rr_means   = np.array([CLUSTER_PROFILES[cl]["refund_rate_mean"]       for cl in clusters])
    rr_stds    = np.array([CLUSTER_PROFILES[cl]["refund_rate_std"]        for cl in clusters])
    cb_means   = np.array([CLUSTER_PROFILES[cl]["chargeback_rate_mean"]   for cl in clusters])
    cb_stds    = np.array([CLUSTER_PROFILES[cl]["chargeback_rate_std"]    for cl in clusters])

    # ── Generate numerical columns ────────────────────────────────────────────
    txn_counts = np.maximum(1, (
        np.random.normal(txn_means * fest_mults, txn_stds)
    ).astype(int))

    volumes = np.maximum(100.0,
        np.random.normal(vol_means * fest_mults, vol_stds)
    ).round(2)

    refund_rates = np.clip(
        np.random.normal(rr_means, rr_stds), 0.0, 1.0
    ).round(4)

    chargeback_rates = np.clip(
        np.random.normal(cb_means, cb_stds), 0.0, 1.0
    ).round(4)

    # ── Assemble DataFrame ────────────────────────────────────────────────────
    df = pd.DataFrame({
        "MerchantID":         merchant_ids,
        "TransactionCount":   txn_counts,
        "TotalVolume":        volumes,
        "RefundRate":         refund_rates,
        "ChargebackRate":     chargeback_rates,
        "Category":           categories,
        "Description":        descriptions,
        "Timestamp":          timestamps,
        "is_festival_period": fest_flags,
        "_cluster_label":     clusters,
    })

    # ── Inject Controlled Anomalies (~1% of merchants) ────────────────────────
    # Bias anomalies strongly towards festival seasons to simulate high risk periods
    n_anomalies = int(n_merchants * 0.01)
    rng = np.random.default_rng(SEED + 1)
    
    # 15x higher probability for a merchant to default as an anomaly during festivals
    weights = np.where(df["is_festival_period"] == 1, 15.0, 1.0)
    probabilities = weights / weights.sum()
    
    anomaly_idx = rng.choice(df.index, size=n_anomalies, replace=False, p=probabilities)

    df.loc[anomaly_idx, "ChargebackRate"] = np.clip(
        rng.uniform(0.15, 0.60, size=n_anomalies), 0, 1
    ).round(4)
    df.loc[anomaly_idx, "TotalVolume"] = (
        df.loc[anomaly_idx, "TotalVolume"] * rng.uniform(5, 20, size=n_anomalies)
    ).round(2)
    df.loc[anomaly_idx, "RefundRate"] = np.clip(
        rng.uniform(0.25, 0.80, size=n_anomalies), 0, 1
    ).round(4)

    df["is_anomaly_injected"] = 0
    df.loc[anomaly_idx, "is_anomaly_injected"] = 1

    # ── Final validation ──────────────────────────────────────────────────────
    assert df["MerchantID"].nunique() == n_merchants, \
        "BUG: Duplicate MerchantIDs detected!"
    assert len(df) == n_merchants, \
        f"BUG: Expected {n_merchants} rows, got {len(df)}"

    print(f"[DataGen] Done. Shape: {df.shape}")
    print(f"[DataGen] Unique merchants: {df['MerchantID'].nunique():,}")
    print(f"[DataGen] Festival onboardings: {df['is_festival_period'].sum():,} "
          f"({df['is_festival_period'].mean()*100:.1f}%)")
    print(f"[DataGen] Injected anomalies: {n_anomalies:,}")
    print(f"[DataGen] Cluster distribution:")
    for label, count in df["_cluster_label"].value_counts().items():
        print(f"          {label:>8}: {count:,}")
    return df


def save_dataset(df: pd.DataFrame, path: str = "data/transactions.csv") -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)
    print(f"[DataGen] Saved -> {path}")


if __name__ == "__main__":
    df = generate_dataset(100_000)
    save_dataset(df, "data/transactions.csv")
    print(df.head())
