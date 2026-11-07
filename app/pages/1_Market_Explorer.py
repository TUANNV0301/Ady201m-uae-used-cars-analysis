"""
Market Explorer Page: Interactive Filtering and Visual Analytics
UAE Used-Car Market Analysis (ADY201m - Phase 5)
"""import sys
from pathlib import Path
import streamlit as st
import pandas as pd
import plotly.express as px

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.utils import load_all_listings, format_currency, render_sidebar_controls

st.set_page_config(page_title="Market Explorer | UAE Used Cars", page_icon="car", layout="wide")
currency, fx_rate = render_sidebar_controls()

st.title("Market Explorer")
st.markdown("Filter and inspect the 10,000 listings across makes, body types, locations, and pricing segments.")

df = load_all_listings()
if df.empty:
    st.warning("Database records not available.")
    st.stop()

# ----------------------------------------------------
# Filters
# ----------------------------------------------------
with st.sidebar:
    st.markdown("### Granular Filters")
    all_makes = ["All Makes"] + sorted(df["make_name"].unique().tolist())
    selected_make = st.selectbox("Select Make", all_makes)
    
    all_body_types = ["All Body Types"] + sorted(df["body_type_name"].unique().tolist())
    selected_body = st.selectbox("Select Body Style", all_body_types)
    
    all_emirates = ["All Locations"] + sorted(df["emirate_name"].unique().tolist())
    selected_emirate = st.selectbox("Select Location", all_emirates)
    
    min_yr = int(df["model_year"].min())
    max_yr = int(df["model_year"].max())
    year_range = st.slider("Model Year Range", min_yr, max_yr, (min_yr, max_yr))
    
    min_p = int(df["price_aed"].min())
    max_p = int(df["price_aed"].max())
    price_range = st.slider("Price Range (AED)", min_p, max_p, (min_p, max_p))

# Filter Application
filtered_df = df.copy()
if selected_make != "All Makes":
    filtered_df = filtered_df[filtered_df["make_name"] == selected_make]
if selected_body != "All Body Types":
    filtered_df = filtered_df[filtered_df["body_type_name"] == selected_body]
if selected_emirate != "All Locations":
    filtered_df = filtered_df[filtered_df["emirate_name"] == selected_emirate]

filtered_df = filtered_df[
    (filtered_df["model_year"] >= year_range[0]) &
    (filtered_df["model_year"] <= year_range[1]) &
    (filtered_df["price_aed"] >= price_range[0]) &
    (filtered_df["price_aed"] <= price_range[1])
]

# Summary metrics
m1, m2, m3, m4 = st.columns(4)
with m1:
    st.metric("Filtered Listings", f"{len(filtered_df):,}")
with m2:
    st.metric("Median Asking Price", format_currency(filtered_df["price_aed"].median() if not filtered_df.empty else 0, currency, fx_rate))
with m3:
    st.metric("Average Mileage", f"{filtered_df['mileage_km'].mean():,.0f} km" if not filtered_df.empty else "0 km")
with m4:
    st.metric("Top Model", filtered_df["model_name"].value_counts().index[0].title() if not filtered_df.empty else "N/A")

st.markdown("---")

if filtered_df.empty:
    st.info("No listings found matching the selected filters. Please expand your criteria.")
    st.stop()

# Charts Grid
c1, c2 = st.columns(2)

with c1:
    st.markdown("#### Mileage vs. Price Relationship")
    fig_scatter = px.scatter(
        filtered_df,
        x="mileage_km",
        y="price_aed",
        color="body_type_name",
        hover_data=["make_name", "model_name", "model_year"],
        log_y=True,
        labels={"mileage_km": "Odometer (km)", "price_aed": f"Price ({currency})", "body_type_name": "Body Style"}
    )
    fig_scatter.update_layout(margin=dict(l=0, r=0, t=20, b=0))
    st.plotly_chart(fig_scatter, use_container_width=True)

with c2:
    st.markdown("#### Price Trend by Model Year")
    yearly = filtered_df.groupby("model_year")["price_aed"].median().reset_index()
    fig_year = px.line(
        yearly,
        x="model_year",
        y="price_aed",
        markers=True,
        labels={"model_year": "Model Year", "price_aed": f"Median Price ({currency})"}
    )
    fig_year.update_layout(margin=dict(l=0, r=0, t=20, b=0))
    st.plotly_chart(fig_year, use_container_width=True)

# Data Table with CSV download
st.markdown("---")
st.markdown("#### Filtered Listings Dataset")
display_cols = ["car_id", "make_name", "model_name", "model_year", "price_aed", "mileage_km", "body_type_name", "emirate_name", "condition_name"]
st.dataframe(filtered_df[display_cols].head(100), use_container_width=True)

csv_data = filtered_df.to_csv(index=False).encode("utf-8")
st.download_button(
    label="Download Filtered Listings (CSV)",
    data=csv_data,
    file_name="uae_filtered_listings.csv",
    mime="text/csv"
)
