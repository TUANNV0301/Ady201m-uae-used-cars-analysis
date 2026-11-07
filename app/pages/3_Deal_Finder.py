"""
Deal Finder Page: Algorithmic Price Arbitrage and Anomaly Detection
UAE Used-Car Market Analysis (ADY201m - Phase 5)
"""import sys
from pathlib import Path
import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.utils import (
    load_all_listings, 
    load_deal_scores, 
    format_currency, 
    render_sidebar_controls
)
from src.config import DEAL_THRESHOLD_LOG

st.set_page_config(page_title="Deal Finder | UAE Used Cars", page_icon="car", layout="wide")
currency, fx_rate = render_sidebar_controls()

st.title("Deal Finder: Valuation Anomaly Detection")
st.markdown(
    """Identifies market listings priced substantially below fair value (**Undervalued / Bargains**) 
    or above fair value (**Overpriced**), supporting buyer negotiations and dealer inventory optimization.
    """
)

df_listings = load_all_listings()
df_deals = load_deal_scores()

# If precomputed deal scores do not exist, compute benchmark-based deal flags dynamically
if df_deals.empty and not df_listings.empty:
    df_deals = df_listings.copy()
    model_medians = df_deals.groupby(["make_name", "model_name"])["price_aed"].transform("median")
    df_deals["pred_price_aed"] = model_medians
    df_deals["log_residual"] = np.log(df_deals["price_aed"]) - np.log(df_deals["pred_price_aed"])
    df_deals["pct_gap"] = (np.exp(df_deals["log_residual"]) - 1.0) * 100.0
    
    conds = [
        df_deals["log_residual"] > DEAL_THRESHOLD_LOG,
        df_deals["log_residual"] < -DEAL_THRESHOLD_LOG
    ]
    choices = ["Overpriced", "Undervalued"]
    df_deals["deal_flag"] = np.select(conds, choices, default="Fair Price")

if df_deals.empty:
    st.warning("No listings data available.")
    st.stop()

# ----------------------------------------------------
# Filters
# ----------------------------------------------------
with st.sidebar:
    st.markdown("### Deal Filters")
    flag_filter = st.multiselect(
        "Deal Status",
        options=["Undervalued", "Fair Price", "Overpriced"],
        default=["Undervalued", "Overpriced"]
    )
    makes_filter = st.multiselect(
        "Vehicle Make",
        options=sorted(df_deals["make_name"].unique().tolist()),
        default=[]
    )

filtered = df_deals[df_deals["deal_flag"].isin(flag_filter)]
if makes_filter:
    filtered = filtered[filtered["make_name"].isin(makes_filter)]

# Metric cards
c1, c2, c3 = st.columns(3)
with c1:
    undervalued_count = (df_deals["deal_flag"] == "Undervalued").sum()
    st.metric("Total Undervalued Deals", f"{undervalued_count:,}", help="Priced >22% below market benchmark")
with c2:
    fair_count = (df_deals["deal_flag"] == "Fair Price").sum()
    st.metric("Fairly Priced Listings", f"{fair_count:,}")
with c3:
    overpriced_count = (df_deals["deal_flag"] == "Overpriced").sum()
    st.metric("Total Overpriced Listings", f"{overpriced_count:,}", help="Priced >28% above market benchmark")

st.markdown("---")

# Visual: Asking vs. Fair Price
st.markdown("#### Asking Price vs. Fair Benchmark")
color_map = {
    "Undervalued": "#2ca02c",
    "Fair Price": "#1f77b4",
    "Overpriced": "#d62728"
}
fig_deals = px.scatter(
    filtered.head(1500), # Cap points for responsive rendering
    x="pred_price_aed",
    y="price_aed",
    color="deal_flag",
    color_discrete_map=color_map,
    hover_data=["make_name", "model_name", "model_year", "pct_gap"],
    log_x=True,
    log_y=True,
    labels={"pred_price_aed": f"Benchmark Price ({currency})", "price_aed": f"Asking Price ({currency})", "deal_flag": "Deal Flag"}
)
fig_deals.update_layout(margin=dict(l=0, r=0, t=20, b=0))
st.plotly_chart(fig_deals, use_container_width=True)

# Deal Table
st.markdown("#### Deal Table (Showing Top Opportunities)")
table_cols = ["car_id", "make_name", "model_name", "model_year", "price_aed", "pred_price_aed", "pct_gap", "deal_flag", "emirate_name"]
show_table = filtered[table_cols].sort_values("pct_gap")

st.dataframe(show_table.head(100), use_container_width=True)
