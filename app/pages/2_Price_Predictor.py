"""
Price Predictor Page: Fair Market Valuation Engine
UAE Used-Car Market Analysis (ADY201m - Phase 5)
"""import sys
from pathlib import Path
import streamlit as st
import numpy as np
import pandas as pd

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.utils import (
    load_all_listings, 
    load_trained_model, 
    format_currency, 
    render_sidebar_controls
)
from src.config import REFERENCE_YEAR, TARGET_FEATURES

st.set_page_config(page_title="Price Predictor | UAE Used Cars", page_icon="car", layout="wide")
currency, fx_rate = render_sidebar_controls()

st.title("Used Car Price Valuation Predictor")
st.markdown("Estimate an evidence-based fair market asking price for an individual used car listing in the UAE.")

df = load_all_listings()
model, metadata = load_trained_model()

if df.empty:
    st.warning("Database records not available.")
    st.stop()

# Form Inputs Layout
st.markdown("### Vehicle Specifications")

c1, c2, c3 = st.columns(3)

with c1:
    makes = sorted(df["make_name"].unique().tolist())
    selected_make = st.selectbox("Vehicle Make", makes, index=makes.index("toyota") if "toyota" in makes else 0)
    
    # Filter models by selected make
    available_models = sorted(df[df["make_name"] == selected_make]["model_name"].unique().tolist())
    selected_model = st.selectbox("Vehicle Model", available_models)
    
    model_year = st.slider("Model Year", 2005, 2024, 2018)
    mileage_km = st.number_input("Odometer Mileage (km)", min_value=1000, max_value=400000, value=75000, step=5000)

with c2:
    body_types = sorted(df["body_type_name"].unique().tolist())
    selected_body = st.selectbox("Body Style", body_types, index=body_types.index("SUV") if "SUV" in body_types else 0)
    
    cylinders = st.selectbox("Engine Cylinders", [3, 4, 6, 8, 10, 12], index=2)
    transmission = st.selectbox("Transmission", ["Automatic Transmission", "Manual"])
    fuel_type = st.selectbox("Fuel Type", ["Gasoline", "Diesel", "Hybrid", "Electric"])

with c3:
    locations = sorted(df["emirate_name"].unique().tolist())
    selected_location = st.selectbox("Location / Emirate", locations, index=locations.index("Dubai") if "Dubai" in locations else 0)
    
    conditions = sorted(df["condition_name"].unique().tolist())
    selected_condition = st.selectbox("Stated Vehicle Condition", conditions, index=conditions.index("No damage") if "No damage" in conditions else 0)

st.markdown("#### Installed Optional Equipment")
feat_cols = st.columns(3)
selected_features = []
for i, feat in enumerate(TARGET_FEATURES):
    with feat_cols[i % 3]:
        if st.checkbox(feat, value=(i in [0, 3, 4])): # Sunroof, Bluetooth, Rear camera default
            selected_features.append(feat)

st.markdown("---")

# Prediction Execution
if st.button("Calculate Estimated Fair Price", type="primary", use_container_width=True):
    car_age = max(1, REFERENCE_YEAR - model_year)
    mileage_per_year = mileage_km / car_age
    log_mileage = np.log(max(1, mileage_km))
    n_features = len(selected_features)
    
    # Model baseline comparison from data
    model_subset = df[(df["make_name"] == selected_make) & (df["model_name"] == selected_model)]
    if not model_subset.empty:
        benchmark_median = model_subset["price_aed"].median()
        benchmark_count = len(model_subset)
    else:
        make_subset = df[df["make_name"] == selected_make]
        benchmark_median = make_subset["price_aed"].median() if not make_subset.empty else df["price_aed"].median()
        benchmark_count = len(make_subset)

    # If scikit-learn trained model is present
    if model is not None and metadata is not None:
        input_data = pd.DataFrame([{
            "make": selected_make,
            "model": selected_model,
            "model_year": model_year,
            "car_age": car_age,
            "mileage_km": mileage_km,
            "log_mileage": log_mileage,
            "mileage_per_year": mileage_per_year,
            "cylinders": cylinders,
            "body_type": selected_body,
            "transmission": transmission,
            "fuel_type": fuel_type,
            "location": selected_location,
            "condition": selected_condition,
            "n_features": n_features,
            "has_sunroof": int("Sunroof" in selected_features),
            "has_leather_seats": int("Leather seats" in selected_features),
            "has_navigation": int("Navigation" in selected_features),
            "has_bluetooth": int("Bluetooth" in selected_features),
            "has_rear_camera": int("Rear camera" in selected_features),
            "has_adaptive_cruise": int("Adaptive cruise control" in selected_features),
            "is_accident_history": int(selected_condition == "Accident history"),
            "region_group": selected_location if selected_location in ["Dubai", "Abu Dhabi", "Sharjah"] else "Other Northern Emirates"}])
        
        try:
            pred_log = model.predict(input_data)[0]
            smearing = metadata.get("smearing_factor", 1.0)
            pred_aed = np.exp(pred_log) * smearing
            res_std = metadata.get("test_rmse_log", 0.35)
            lower_aed = np.exp(pred_log - 1.28 * res_std) * smearing
            upper_aed = np.exp(pred_log + 1.28 * res_std) * smearing
        except Exception:
            pred_aed = benchmark_median
            lower_aed = benchmark_median * 0.8
            upper_aed = benchmark_median * 1.2
    else:
        # Statistical baseline fallback when models/price_model.joblib is not yet generated
        pred_aed = benchmark_median
        lower_aed = benchmark_median * 0.78
        upper_aed = benchmark_median * 1.25

    # Display Valuation Results
    st.success("### Valuation Output")
    r1, r2, r3 = st.columns(3)
    
    with r1:
        st.metric(
            "Estimated Fair Market Price", 
            format_currency(pred_aed, currency, fx_rate)
        )
    with r2:
        st.metric(
            "80% Confidence Valuation Band", 
            f"{format_currency(lower_aed, currency, fx_rate)} - {format_currency(upper_aed, currency, fx_rate)}")
    with r3:
        st.metric(
            f"Historical Median ({selected_make.title()} {selected_model.title()})", 
            format_currency(benchmark_median, currency, fx_rate),
            help=f"Computed from {benchmark_count} actual listings in the database.")

    if model is None:
        st.info("Note: Running with statistical historical benchmark. Run `notebooks/04_modeling.ipynb` to fit and save the full machine learning model.")
