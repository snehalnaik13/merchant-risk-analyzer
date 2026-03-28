"""
streamlit_app.py
================
Multi-page Streamlit dashboard for AI Merchant Risk Analyzer.
Pages:
  1. Login
  2. Merchant Risk Overview
  3. Add Merchant Transaction
  4. Check Trends
  5. Show Plots
  6. Top 20 High-Risk Merchants
  7. Manage Data
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os, sys, warnings
from datetime import datetime

warnings.filterwarnings("ignore")

# ── Path setup ─────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(BASE_DIR, "src"))

DATA_PATH = os.path.join(BASE_DIR, "data", "risk_scored.csv")

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AI Merchant Risk Analyzer",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem; font-weight: 700; color: #ffffff;
        text-align: center; padding: 1rem 0;
    }
    .risk-high   { background:#fde8e8; border-left:5px solid #e74c3c;
                   padding:0.8rem; border-radius:6px; margin:4px 0; }
    .risk-medium { background:#fef9e7; border-left:5px solid #f39c12;
                   padding:0.8rem; border-radius:6px; margin:4px 0; }
    .risk-low    { background:#eafaf1; border-left:5px solid #2ecc71;
                   padding:0.8rem; border-radius:6px; margin:4px 0; }
    .metric-card {
        background: #f8f9fa; border-radius:10px; padding:1.2rem;
        text-align:center; border:1px solid #dee2e6;
    }
    .stButton>button { border-radius:8px; font-weight:600; }
</style>
""", unsafe_allow_html=True)

# ── Auth ───────────────────────────────────────────────────────────────────────
CREDENTIALS = {"admin": "risk@123", "analyst": "analyst@456"}

def login_page():
    st.markdown('<div class="main-header">🛡️ AI Merchant Risk Analyzer</div>',
                unsafe_allow_html=True)
    st.markdown("---")
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.subheader("🔐 Secure Login")
        username = st.text_input("Username", placeholder="Enter username")
        password = st.text_input("Password", type="password",
                                 placeholder="Enter password")
        if st.button("Login", use_container_width=True):
            if username in CREDENTIALS and CREDENTIALS[username] == password:
                st.session_state["authenticated"] = True
                st.session_state["username"] = username
                st.rerun()
            else:
                st.error("❌ Invalid credentials. Try admin / risk@123")
        st.caption("Demo credentials — admin / risk@123")

# ── Data loader ────────────────────────────────────────────────────────────────
@st.cache_data(ttl=60)
def load_data(path: str = DATA_PATH) -> pd.DataFrame:
    if not os.path.exists(path):
        st.error(f"Dataset not found at {path}.\n"
                 "Please run: `python src/run_pipeline.py` first.")
        st.stop()
    df = pd.read_csv(path, parse_dates=["Timestamp"])
    return df


def save_data(df: pd.DataFrame):
    df.to_csv(DATA_PATH, index=False)
    st.cache_data.clear()


# ── Colour helpers ─────────────────────────────────────────────────────────────
RISK_COLORS = {"High Risk": "#e74c3c", "Medium Risk": "#f39c12", "Low Risk": "#2ecc71"}

def risk_badge(level: str) -> str:
    c = RISK_COLORS.get(level, "#888")
    return f'<span style="background:{c};color:white;padding:3px 10px;border-radius:12px;font-weight:600">{level}</span>'

def score_gauge(score: float):
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=round(score * 100, 1),
        domain={"x": [0, 1], "y": [0, 1]},
        title={"text": "Risk Score"},
        gauge={
            "axis": {"range": [0, 100]},
            "bar": {"color": "#e74c3c" if score > 0.65 else "#f39c12" if score > 0.35 else "#2ecc71"},
            "steps": [
                {"range": [0, 35],  "color": "#eafaf1"},
                {"range": [35, 65], "color": "#fef9e7"},
                {"range": [65, 100],"color": "#fde8e8"},
            ],
            "threshold": {"line": {"color": "black", "width": 3}, "value": score * 100},
        }
    ))
    fig.update_layout(height=250, margin=dict(t=30, b=0, l=20, r=20))
    return fig


# ═══════════════════════════════════════════════════════════════════════════════
#  PAGE 1 — Merchant Risk Overview
# ═══════════════════════════════════════════════════════════════════════════════
def page_overview(df: pd.DataFrame):
    st.title("🔍 Merchant Risk Overview")
    merchant_ids = df["MerchantID"].unique().tolist()

    col_input, col_main = st.columns([1, 3])
    with col_input:
        mid = st.selectbox("Select MerchantID", sorted(merchant_ids),
                           key="overview_mid")
        st.markdown("---")
        st.caption("Type to search for a merchant ID")

    mdf = df[df["MerchantID"] == mid].copy()
    if mdf.empty:
        st.warning("No data found for this merchant.")
        return

    latest = mdf.sort_values("Timestamp").iloc[-1]
    level  = latest.get("risk_level", "Unknown")
    score  = float(latest.get("risk_score", 0))

    with col_main:
        # ── Top KPIs ──────────────────────────────────────────────────────────
        c1, c2, c3 = st.columns(3)
        c1.metric("Risk Level", level)
        c2.metric("Risk Score", f"{score:.3f}")
        nlp_val = float(latest.get("nlp_risk_score", 0))
        c3.metric("NLP Score", f"{nlp_val:.3f}")
        
        c4, c5, c6 = st.columns(3)
        c4.metric("Category", latest.get("Category","N/A"))
        c5.metric("Refund Rate", f"{latest.get('RefundRate',0)*100:.1f}%")
        c6.metric("Chargeback Rate", f"{latest.get('ChargebackRate',0)*100:.1f}%")

        st.markdown(f"**Risk Level:** {risk_badge(level)}", unsafe_allow_html=True)

        # ── Gauge ─────────────────────────────────────────────────────────────
        col_g, col_exp = st.columns([1, 2])
        with col_g:
            st.plotly_chart(score_gauge(score), use_container_width=True)

        with col_exp:
            st.subheader("📋 Risk Explanation")
            cb   = float(latest.get("ChargebackRate", 0))
            rr   = float(latest.get("RefundRate", 0))
            nlp  = float(latest.get("nlp_risk_score", 0))
            anom = int(latest.get("is_anomaly", 0))
            fest = int(latest.get("is_festival_period", 0))

            reasons = []
            if cb > 0.10:
                reasons.append(f"⚠️ High Chargeback Rate: {cb*100:.1f}%")
            if rr > 0.10:
                reasons.append(f"⚠️ Elevated Refund Rate: {rr*100:.1f}%")
            if nlp > 0.30:
                reasons.append(f"🔤 NLP Risk Signal from Description: {nlp:.2f}")
            if anom:
                reasons.append("🚨 Flagged as Anomaly by Isolation Forest")
            if fest:
                reasons.append("🎉 Transaction during festival period")

            if not reasons:
                reasons.append("✅ No significant risk signals detected")

            for r in reasons:
                st.markdown(f"- {r}")

            st.markdown(f"**Description:** *{latest.get('Description','N/A')}*")




# ═══════════════════════════════════════════════════════════════════════════════
#  PAGE 2 — Add Merchant Transaction
# ═══════════════════════════════════════════════════════════════════════════════
def page_add_transaction(df: pd.DataFrame):
    st.title("➕ Add Merchant")
    st.caption("Each MerchantID is unique (one row per merchant). "
               "Entering a new ID creates a new merchant record. "
               "Entering an existing ID updates that merchant's profile.")

    col_form, col_result = st.columns([1, 2])

    with col_form:
        st.subheader("Transaction Details")
        mid         = st.text_input("MerchantID", placeholder="MID00001")
        txn_count   = st.number_input("Transaction Count", min_value=1, value=100, step=1)
        volume      = st.number_input("Total Volume (₹)", min_value=0.0, value=500000.0, step=1000.0, format="%.2f")
        refund_rate = st.slider("Refund Rate", 0.0, 1.0, 0.02, 0.001, format="%.3f")
        cb_rate     = st.slider("Chargeback Rate", 0.0, 0.5, 0.005, 0.001, format="%.3f")
        category    = st.selectbox("Category",
                                   ["Travel","Crypto","Gaming","Food","Services","Grocery"])
        description = st.text_area("Description",
                                   value="Online payment processing service")
        timestamp   = st.date_input("Transaction Date", value=datetime.today().date())

        submitted = st.button("🚀 Analyze & Add", use_container_width=True)

    if submitted and mid:
        from nlp_analysis import compute_nlp_risk_score
        from risk_scoring import compute_risk_score, assign_risk_level, DEFAULT_WEIGHTS

        ts_full = datetime.combine(timestamp, datetime.min.time())
        month   = ts_full.month

        # Compute derived features
        avg_txn_val = volume / max(txn_count, 1)
        nlp_score   = compute_nlp_risk_score(description)
        month_sin   = np.sin(2 * np.pi * month / 12)
        month_cos   = np.cos(2 * np.pi * month / 12)

        new_row = pd.DataFrame([{
            "MerchantID":         mid,
            "TransactionCount":   txn_count,
            "TotalVolume":        volume,
            "RefundRate":         refund_rate,
            "ChargebackRate":     cb_rate,
            "Category":           category,
            "Description":        description,
            "Timestamp":          ts_full,
            "is_festival_period": 0,
            "nlp_risk_score":     nlp_score,
            "AvgTransactionValue": avg_txn_val,
            "TxnGrowthRate":      0.05,
            "Month":              month,
            "Month_sin":          month_sin,
            "Month_cos":          month_cos,
            "is_anomaly":         0,
            "anomaly_score":      0.0,
            "cluster_id":         0,
            "cluster_label":      "Low Risk",
            "_cluster_label":     "low",
            "is_anomaly_injected": 0,
            "DayOfWeek":          ts_full.weekday(),
            "IsWeekend":          int(ts_full.weekday() >= 5),
            "WeekOfYear":         ts_full.isocalendar()[1],
            "Hour":               0,
            "Quarter":            (month - 1) // 3 + 1,
        }])

        # Upsert: replace existing merchant row, or append if new
        for col in df.columns:
            if col not in new_row.columns:
                new_row[col] = np.nan
        new_row = new_row[[c for c in df.columns if c in new_row.columns]]

        existing_mask = df["MerchantID"] == mid
        if existing_mask.any():
            df_updated = df.copy()
            df_updated = df_updated[~existing_mask]
            df_updated = pd.concat([df_updated, new_row], ignore_index=True)
            msg = f"✅ Merchant {mid} profile updated!"
        else:
            df_updated = pd.concat([df, new_row], ignore_index=True)
            msg = f"✅ New merchant {mid} added to dataset!"

        # Recalculate risk score on the entire dataset to prevent zero-scaled normalisation errors on a single row
        df_updated = compute_risk_score(df_updated, DEFAULT_WEIGHTS)
        df_updated = assign_risk_level(df_updated)

        # Retrieve the updated metrics for the specified merchant
        scored_row = df_updated[df_updated["MerchantID"] == mid].iloc[-1]
        score = float(scored_row["risk_score"])
        level = scored_row["risk_level"]

        with col_result:
            st.subheader("🎯 Risk Analysis Result")
            c1, c2 = st.columns(2)
            c1.plotly_chart(score_gauge(score), use_container_width=True)
            with c2:
                st.markdown(f"**Risk Level:** {risk_badge(level)}",
                            unsafe_allow_html=True)
                st.metric("Risk Score", f"{score:.3f}")
                st.metric("NLP Risk Score", f"{nlp_score:.3f}")
                st.metric("Avg Transaction Value", f"₹{avg_txn_val:,.0f}")

                if cb_rate > 0.10:
                    st.warning(f"⚠️ Chargeback rate {cb_rate*100:.1f}% exceeds threshold!")
                if nlp_score > 0.30:
                    st.warning(f"🔤 NLP detected risk keywords in description")
                if level == "High Risk":
                    st.error("🚨 This merchant is flagged as HIGH RISK")
                elif level == "Medium Risk":
                    st.warning("⚡ This merchant requires MEDIUM RISK monitoring")
                else:
                    st.success("✅ LOW RISK merchant — standard monitoring")

        save_data(df_updated)
        st.success(msg)


# ═══════════════════════════════════════════════════════════════════════════════
#  PAGE 3 — Check Trends
# ═══════════════════════════════════════════════════════════════════════════════
def page_trends(df: pd.DataFrame):
    st.title("📊 Check Trends")

    # ── Time series : Fraud Trend ──────────────────────────────────────────────
    st.subheader("🚨 Monthly Fraud/Anomaly Trend (Seasonality Factor)")
    if "Timestamp" in df.columns and "is_anomaly" in df.columns:
        df["_month"] = pd.to_datetime(df["Timestamp"]).dt.to_period("M").astype(str)
        monthly_fraud = df.groupby("_month")["is_anomaly"].sum().reset_index()
        fig_f = px.line(monthly_fraud, x="_month", y="is_anomaly",
                        title="Number of High-Risk Fraud Transactions by Month (Festival Spikes)",
                        labels={"_month": "Month", "is_anomaly": "Total Fraud Flags"},
                        markers=True)
        fig_f.update_traces(line_color="#e74c3c")
        st.plotly_chart(fig_f, use_container_width=True)

    # ── Time series : Volume Trend ────────────────────────────────────────────
    st.subheader("📈 Monthly Volume Trend (All Merchants)")
    if "Timestamp" in df.columns:
        df["_month"] = pd.to_datetime(df["Timestamp"]).dt.to_period("M").astype(str)
        monthly_vol = df.groupby("_month")["TotalVolume"].sum().reset_index()
        fig = px.line(monthly_vol, x="_month", y="TotalVolume",
                      title="Total Transaction Volume by Month",
                      labels={"_month": "Month", "TotalVolume": "Total Volume (₹)"},
                      markers=True)
        fig.update_traces(line_color="#3498db")
        st.plotly_chart(fig, use_container_width=True)

    # ── Category-wise ─────────────────────────────────────────────────────────
    st.subheader("🏷️ Category-wise Transaction Volume")
    if "Category" in df.columns:
        cat_vol = df.groupby("Category")["TotalVolume"].mean().reset_index().sort_values(
            "TotalVolume", ascending=False)
        fig2 = px.bar(cat_vol, x="Category", y="TotalVolume", color="Category",
                      title="Average Volume by Category",
                      color_discrete_sequence=px.colors.qualitative.Set2)
        st.plotly_chart(fig2, use_container_width=True)

    # ── Festival vs non-festival ───────────────────────────────────────────────
    st.subheader("🎉 Festival vs Non-Festival Period")
    if "is_festival_period" in df.columns:
        df["_period"] = df["is_festival_period"].map(
            {1: "Festival Period", 0: "Non-Festival"})
        fest_comp = df.groupby(["Category", "_period"])["TotalVolume"].mean().reset_index()
        fig3 = px.bar(fest_comp, x="Category", y="TotalVolume", color="_period",
                      barmode="group",
                      color_discrete_map={"Festival Period": "#e74c3c",
                                          "Non-Festival": "#3498db"},
                      title="Average Volume: Festival vs Non-Festival by Category")
        st.plotly_chart(fig3, use_container_width=True)

    # ── Risk level over time ───────────────────────────────────────────────────
    if "risk_level" in df.columns and "Timestamp" in df.columns:
        st.subheader("⚠️ Risk Level Distribution Over Time")
        df["_quarter"] = pd.to_datetime(df["Timestamp"]).dt.to_period("Q").astype(str)
        rl_time = df.groupby(["_quarter", "risk_level"]).size().reset_index(name="Count")
        fig4 = px.bar(rl_time, x="_quarter", y="Count", color="risk_level",
                      barmode="stack",
                      color_discrete_map=RISK_COLORS,
                      title="Risk Level Distribution by Quarter")
        st.plotly_chart(fig4, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════
#  PAGE 4 — Show Plots
# ═══════════════════════════════════════════════════════════════════════════════
def page_plots(df: pd.DataFrame):
    st.title("📉 Show Plots")

    # ── Cluster visualisation ─────────────────────────────────────────────────
    st.subheader("🔵 Cluster Visualisation (PCA 2D)")
    cluster_img = os.path.join(BASE_DIR, "data", "cluster_pca.png")
    elbow_img   = os.path.join(BASE_DIR, "data", "elbow_plot.png")

    from PIL import Image
    c1, c2 = st.columns(2)
    if os.path.exists(cluster_img):
        try:
            img1 = Image.open(cluster_img)
            c1.image(img1, caption="KMeans Clusters (PCA)")
        except Exception as e:
            c1.error("Failed to load clustering plot")
    else:
        c1.info("Run pipeline to generate cluster_pca.png")
        
    if os.path.exists(elbow_img):
        try:
            img2 = Image.open(elbow_img)
            c2.image(img2, caption="Elbow Method + Silhouette")
        except Exception as e:
            c2.error("Failed to load elbow plot")
    else:
        c2.info("Run pipeline to generate elbow_plot.png")

    # ── Risk vs Refund ────────────────────────────────────────────────────────
    st.subheader("📊 Risk Score vs Refund Rate")
    if "risk_score" in df.columns and "RefundRate" in df.columns:
        sample = df.sample(min(3000, len(df)), random_state=42)
        fig = px.scatter(sample, x="RefundRate", y="risk_score",
                         color="risk_level" if "risk_level" in sample.columns else None,
                         color_discrete_map=RISK_COLORS,
                         opacity=0.5, title="Risk Score vs Refund Rate",
                         labels={"RefundRate": "Refund Rate", "risk_score": "Risk Score"})
        st.plotly_chart(fig, use_container_width=True)

    # ── Risk vs Chargeback ────────────────────────────────────────────────────
    st.subheader("📊 Risk Score vs Chargeback Rate")
    if "ChargebackRate" in df.columns:
        fig2 = px.scatter(sample, x="ChargebackRate", y="risk_score",
                          color="risk_level" if "risk_level" in sample.columns else None,
                          color_discrete_map=RISK_COLORS,
                          opacity=0.5, title="Risk Score vs Chargeback Rate",
                          labels={"ChargebackRate": "Chargeback Rate"})
        st.plotly_chart(fig2, use_container_width=True)

    # ── Risk vs Month ─────────────────────────────────────────────────────────
    st.subheader("📅 Risk Score by Month")
    if "Month" in df.columns:
        monthly_risk = df.groupby("Month")["risk_score"].mean().reset_index()
        fig3 = px.line(monthly_risk, x="Month", y="risk_score",
                       title="Average Risk Score by Month",
                       markers=True, line_shape="spline")
        fig3.update_traces(line_color="#e74c3c")
        st.plotly_chart(fig3, use_container_width=True)

    # ── Risk distribution ─────────────────────────────────────────────────────
    st.subheader("🥧 Risk Level Distribution")
    if "risk_level" in df.columns:
        rl_counts = df["risk_level"].value_counts().reset_index()
        rl_counts.columns = ["Risk Level", "Count"]
        fig4 = px.pie(rl_counts, names="Risk Level", values="Count",
                      color="Risk Level", color_discrete_map=RISK_COLORS,
                      title="Merchant Risk Level Distribution")
        st.plotly_chart(fig4, use_container_width=True)

    # ── Category risk heatmap ─────────────────────────────────────────────────
    st.subheader("🗺️ Category × Risk Level Heatmap")
    if "Category" in df.columns and "risk_level" in df.columns:
        heat = df.groupby(["Category", "risk_level"]).size().unstack(fill_value=0)
        fig5 = px.imshow(heat, text_auto=True, aspect="auto",
                         color_continuous_scale="RdYlGn_r",
                         title="Merchant Count: Category × Risk Level")
        st.plotly_chart(fig5, use_container_width=True)

    # ── Anomaly score distribution ────────────────────────────────────────────
    if "anomaly_score" in df.columns:
        st.subheader("🚨 Anomaly Score Distribution")
        fig6 = px.histogram(df, x="anomaly_score", nbins=80,
                            color_discrete_sequence=["#e74c3c"],
                            title="Isolation Forest Anomaly Score Distribution",
                            labels={"anomaly_score": "Anomaly Score (lower=more anomalous)"})
        st.plotly_chart(fig6, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════
#  PAGE 5 — Top 20 High-Risk Merchants
# ═══════════════════════════════════════════════════════════════════════════════
def page_top_risk(df: pd.DataFrame):
    st.title("🚨 Top 20 High-Risk Merchants")

    if "risk_score" not in df.columns:
        st.warning("Risk scores not found. Run the pipeline first.")
        return

    # Filters
    col_f1, col_f2, col_f3 = st.columns(3)
    with col_f1:
        cats = ["All"] + sorted(df["Category"].unique().tolist()) if "Category" in df.columns else ["All"]
        cat_filter = st.selectbox("Filter by Category", cats)
    with col_f2:
        anom_filter = st.checkbox("Anomalies Only", value=False)
    with col_f3:
        festival_filter = st.checkbox("Festival Period Only", value=False)

    filtered = df.copy()
    if cat_filter != "All" and "Category" in filtered.columns:
        filtered = filtered[filtered["Category"] == cat_filter]
    if anom_filter and "is_anomaly" in filtered.columns:
        filtered = filtered[filtered["is_anomaly"] == 1]
    if festival_filter and "is_festival_period" in filtered.columns:
        filtered = filtered[filtered["is_festival_period"] == 1]

    top20 = (filtered.sort_values("risk_score", ascending=False)
             .groupby("MerchantID").first().reset_index()
             .sort_values("risk_score", ascending=False)
             .head(20))

    display_cols = [c for c in ["MerchantID","Category","risk_score","risk_level",
                                 "ChargebackRate","RefundRate","is_anomaly",
                                 "nlp_risk_score","TotalVolume"] if c in top20.columns]

    # Highlight table
    def color_row(row):
        if row.get("risk_level") == "High Risk":
            return ["background-color: #fde8e8; color: #000000;"] * len(row)
        elif row.get("risk_level") == "Medium Risk":
            return ["background-color: #fef9e7; color: #000000;"] * len(row)
        return [""] * len(row)

    styled_df = top20[display_cols].style.apply(color_row, axis=1).format({
        "risk_score": "{:.3f}",
        "nlp_risk_score": "{:.3f}",
        "ChargebackRate": "{:.2%}",
        "RefundRate": "{:.2%}",
        "TotalVolume": "₹{:,.2f}"
    }, na_rep="-")
    
    st.dataframe(
        styled_df,
        use_container_width=True, height=500
    )

    # Bar chart of top 20
    fig = px.bar(top20, x="MerchantID", y="risk_score",
                 color="risk_level" if "risk_level" in top20.columns else None,
                 color_discrete_map=RISK_COLORS,
                 title="Top 20 Merchants by Risk Score")
    fig.update_xaxes(tickangle=45)
    st.plotly_chart(fig, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════
#  PAGE 6 — Manage Data
# ═══════════════════════════════════════════════════════════════════════════════
def page_manage(df: pd.DataFrame):
    st.title("🗂️ Manage Data")

    tab1, tab2 = st.tabs(["📝 Edit Transaction", "🗑️ Delete Merchant"])

    with tab1:
        st.subheader("Edit Merchant Transaction")
        mid = st.selectbox("Select MerchantID to Edit",
                           sorted(df["MerchantID"].unique()), key="edit_mid")
        mdf = df[df["MerchantID"] == mid]
        if not mdf.empty:
            display_mdf = mdf[["MerchantID","Timestamp","Category",
                               "TransactionCount","TotalVolume",
                               "RefundRate","ChargebackRate",
                               "risk_score","risk_level"]].head(20)
            styled_mdf = display_mdf.style.format({
                "RefundRate": "{:.2%}",
                "ChargebackRate": "{:.2%}",
                "TotalVolume": "₹{:,.2f}",
                "risk_score": "{:.3f}",
                "Timestamp": lambda t: t.strftime('%Y-%m-%d %H:%M')
            }, na_rep="-")
            st.dataframe(styled_mdf, use_container_width=True)

            idx_options = mdf.index.tolist()
            if idx_options:
                row_idx = st.selectbox("Select Row Index to Edit", idx_options)
                row = df.loc[row_idx]

                new_refund = st.number_input("New Refund Rate",
                                             min_value=0.0, max_value=1.0,
                                             value=float(row.get("RefundRate", 0)),
                                             step=0.001, format="%.3f")
                new_cb = st.number_input("New Chargeback Rate",
                                         min_value=0.0, max_value=0.5,
                                         value=float(row.get("ChargebackRate", 0)),
                                         step=0.001, format="%.3f")

                if st.button("💾 Save Changes", key="save_edit"):
                    df.at[row_idx, "RefundRate"]     = new_refund
                    df.at[row_idx, "ChargebackRate"] = new_cb
                    save_data(df)
                    st.success("✅ Changes saved!")

    with tab2:
        st.subheader("Delete All Transactions for a Merchant")
        mid_del = st.selectbox("Select MerchantID to Delete",
                               sorted(df["MerchantID"].unique()), key="del_mid")
        count   = (df["MerchantID"] == mid_del).sum()
        st.warning(f"This will delete all {count} rows for merchant {mid_del}")
        if st.button("🗑️ Confirm Delete", key="confirm_del"):
            df_new = df[df["MerchantID"] != mid_del].reset_index(drop=True)
            save_data(df_new)
            st.success(f"✅ Deleted {count} rows for {mid_del}")


# ═══════════════════════════════════════════════════════════════════════════════
#  SIDEBAR NAVIGATION
# ═══════════════════════════════════════════════════════════════════════════════
def sidebar_nav():
    with st.sidebar:
        st.markdown("<h3 style='color: #ffffff; font-weight: 700;'>🛡️ AI Merchant Risk Analyzer</h3>", unsafe_allow_html=True)
        st.markdown(f"👤 Logged in as: **{st.session_state.get('username','?')}**")
        st.markdown("---")

        pages = {
            "🔍 Merchant Risk Overview":       "overview",
            "➕ Add Merchant Transaction":      "add",
            "📊 Check Trends":                 "trends",
            "📉 Show Plots":                   "plots",
            "🚨 Top 20 High-Risk Merchants":   "top_risk",
            "🗂️ Manage Data":                  "manage",
        }
        choice = st.radio("Navigation", list(pages.keys()),
                          label_visibility="collapsed")
        st.markdown("---")
        if st.button("🚪 Logout"):
            st.session_state["authenticated"] = False
            st.rerun()

        st.caption("AI Merchant Risk Analyzer v1.0\nIndia-focused with Seasonality")
        return pages[choice]


# ═══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════════════════
def main():
    if "authenticated" not in st.session_state:
        st.session_state["authenticated"] = False

    if not st.session_state["authenticated"]:
        login_page()
        return

    df = load_data()
    page = sidebar_nav()

    if page == "overview":
        page_overview(df)
    elif page == "add":
        page_add_transaction(df)
    elif page == "trends":
        page_trends(df)
    elif page == "plots":
        page_plots(df)
    elif page == "top_risk":
        page_top_risk(df)
    elif page == "manage":
        page_manage(df)


if __name__ == "__main__":
    main()
