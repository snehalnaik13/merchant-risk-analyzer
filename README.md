# 🛡️ AI Merchant Risk Analyzer (India-focused with Seasonality)

A production-quality ML system for classifying merchant risk using synthetic Indian transaction data.

---

## 🏗️ Project Structure

```
ai_merchant_risk_analyzer/
├── data/               ← Generated datasets & plots
├── models/             ← Saved ML models (pkl)
├── notebooks/
│   └── 01_full_pipeline.ipynb
├── src/
│   ├── data_generation.py     ← 100k synthetic rows + Indian seasonality
│   ├── preprocessing.py       ← Cleaning, OHE, StandardScaler
│   ├── feature_engineering.py ← Cyclical encoding, derived features
│   ├── nlp_analysis.py        ← spaCy-style rule-based NLP risk scoring
│   ├── anomaly_detection.py   ← Isolation Forest
│   ├── clustering.py          ← KMeans k=3, elbow, silhouette
│   ├── risk_scoring.py        ← RF feature weights + rule-based classification
│   └── run_pipeline.py        ← End-to-end orchestrator
├── app/
│   └── streamlit_app.py       ← Multi-page dashboard
├── requirements.txt
└── README.md
```

---

## 🚀 Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Run the full ML pipeline (generates data + trains models)
```bash
cd ai_merchant_risk_analyzer
python src/run_pipeline.py
```
This will create:
- `data/transactions.csv` — raw synthetic data
- `data/risk_scored.csv` — final dataset with risk scores
- `data/elbow_plot.png` — KMeans elbow + silhouette chart
- `data/cluster_pca.png` — PCA cluster visualisation
- `models/` — all trained model files

### 3. Launch the Streamlit dashboard
```bash
streamlit run app/streamlit_app.py
```

**Login credentials:**
- Username: `admin` | Password: `risk@123`
- Username: `analyst` | Password: `analyst@456`

---

## 📊 Dashboard Pages

| Page | Description |
|------|-------------|
| 🔍 Merchant Risk Overview | Risk score, gauge, explanation, trend charts |
| ➕ Add Merchant Transaction | Real-time risk analysis for new/existing merchants |
| 📊 Check Trends | Time series, category-wise, festival vs non-festival |
| 📉 Show Plots | Clusters, scatter plots, heatmaps, anomaly distribution |
| 🚨 Top 20 High-Risk | Filterable high-risk merchant table |
| 🗂️ Manage Data | Edit/delete merchant transactions |

---

## 🔬 ML Pipeline

### Data Generation
- 100,000 synthetic rows, ~2,000 merchants
- Indian festival seasonality: Diwali, Eid, Christmas, Big Billion Days, etc.
- Category-specific transaction multipliers during festivals
- 3-cluster structure: Low / Medium / High Risk
- 1% injected anomalies for validation

### Feature Engineering
- Temporal: Month, DayOfWeek, IsWeekend, WeekOfYear
- Cyclical encoding: sin/cos of Month, DayOfWeek (avoids Dec→Jan discontinuity)
- AvgTransactionValue, TxnGrowthRate

### NLP Analysis (rule-based, spaCy-style)
- Tier-1/2/3 risk keyword matching
- Trust keyword discount
- nlp_risk_score ∈ [0, 1]

### Anomaly Detection
- Isolation Forest (contamination=0.01)
- No distribution assumptions; works well in high dimensions

### Clustering
- KMeans k=3 (confirmed by elbow method)
- High silhouette score due to engineered cluster separation

### Risk Scoring
- Feature weights from Random Forest importance
- Weighted combination → risk_score ∈ [0, 1]
- Rule-based overrides for ChargebackRate, anomaly flag, NLP score

---

## 📋 Requirements

See `requirements.txt`. Core dependencies:
- pandas, numpy, scikit-learn
- streamlit, plotly
- matplotlib, seaborn
- joblib, scipy

---

## 🇮🇳 Indian Seasonality Details

| Festival | Month | Days | Categories Boosted |
|----------|-------|------|--------------------|
| Diwali | Oct | 20–28 | Grocery (3.5×), Gaming (2.5×) |
| Big Billion Days | Oct | 8–15 | All categories |
| Christmas / New Year | Dec–Jan | 20–3 | Travel (3.5×), Gaming (3.0×) |
| Eid ul-Fitr | Apr | 10–14 | Grocery (2.8×), Travel (2.5×) |
| Holi | Mar | 24–26 | Food (2.5×), Grocery (2.0×) |

---

*Built with ❤️ for Indian fintech compliance and merchant risk management.*
