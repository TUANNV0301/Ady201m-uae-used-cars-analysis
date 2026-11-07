"""
Streamlit Home Page: Executive Market Overview
UAE Used-Car Market Analysis (ADY201m - Phase 5)
"""import sys
from pathlib import Path
import streamlit as st
import pandas as pd
import plotly.express as px

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.utils import load_all_listings, format_currency, render_sidebar_controls

st.set_page_config(
    page_title="UAE Used Cars | Market Intelligence",
    page_icon="car",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Render common sidebar controls
currency, fx_rate = render_sidebar_controls()

# Page Header
st.title("UAE Used-Car Market Intelligence & Valuation")
st.markdown(
    """**ADY201m Applied Data Analysis Final Project (FPT University)**  
    An end-to-end analytical decision system combining a normalized **SQLite relational database**, 
    exploratory statistical analysis, **machine-learning price valuation**, and an interactive dashboard.
    """
)
st.markdown("---")

# Load Dataset
df = load_all_listings()

if df.empty:
    st.warning("Database is empty or not initialized. Run `python src/build_db.py` to populate data.")
    st.stop()

# ----------------------------------------------------
# 1. Executive Market KPIs (Aligned with SQL Query Q1)
# ----------------------------------------------------
st.subheader("Market Summary Indicators")

col1, col2, col3, col4, col5 = st.columns(5)

total_listings = len(df)
median_price = df["price_aed"].median()
avg_price = df["price_aed"].mean()
dubai_share = (df["emirate_name"].str.lower() == "dubai").mean() * 100
top_make = df["make_name"].value_counts().index[0].title()

with col1:
    st.metric("Total Listings", f"{total_listings:,}")
with col2:
    st.metric("Median Asking Price", format_currency(median_price, currency, fx_rate))
with col3:
    st.metric("Average Asking Price", format_currency(avg_price, currency, fx_rate))
with col4:
    st.metric("Dubai Supply Share", f"{dubai_share:.1f}%")
with col5:
    st.metric("Top Brand Volume", top_make)

st.markdown("---")

# ----------------------------------------------------
# 2. Market Overview Visuals
# ----------------------------------------------------
st.subheader("High-Level Market Structure")

chart_col1, chart_col2 = st.columns(2)

with chart_col1:
    st.markdown("#### Top 10 Makes by Listing Volume")
    make_counts = df["make_name"].value_counts().head(10).reset_index()
    make_counts.columns = ["Make", "Count"]
    fig_makes = px.bar(
        make_counts,
        x="Count",
        y="Make",
        orientation="h",
        color="Count",
        color_continuous_scale="Blues",
        labels={"Count": "Number of Listings", "Make": "Make"}
    )
    fig_makes.update_layout(yaxis={"categoryorder": "total ascending"}, margin=dict(l=0, r=0, t=20, b=0))
    st.plotly_chart(fig_makes, use_container_width=True)

with chart_col2:
    st.markdown("#### Asking Price Distribution (Log-Scale)")
    price_series = df["price_aed"] * (fx_rate if currency == "VND" else 1.0)
    fig_hist = px.histogram(
        df,
        x="price_aed",
        nbins=50,
        log_x=True,
        labels={"price_aed": f"Price ({currency})"},
        color_discrete_sequence=["#1f77b4"]
    )
    fig_hist.update_layout(margin=dict(l=0, r=0, t=20, b=0), showlegend=False)
    st.plotly_chart(fig_hist, use_container_width=True)

st.markdown("---")

# ----------------------------------------------------
# 3. Project Objectives & Roadmap
# ----------------------------------------------------
st.subheader("Project Navigation Guide")

n1, n2, n3, n4 = st.columns(4)

with n1:
    st.info("""**1. Market Explorer**  
    Multi-dimensional filterable analytics covering body types, fuel types, regional distribution, and model year trends.
    """)

with n2:
    st.info("""**2. Price Predictor**  
    Interactive valuation engine estimating fair market price and uncertainty bands for any specific car configuration.
    """)

with n3:
    st.info("""**3. Deal Finder**  
    Algorithmic arbitrage tool detecting undervalued bargains and overpriced listings across all 10,000 listings.
    """)

with n4:
    st.info("""**4. SQL Console**  
    Interactive query runner with presets for all 12 core business queries and custom query execution.
    """)
