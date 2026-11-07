"""
Feature Engineering and Data Preprocessing Pipeline
UAE Used-Car Market Analysis (ADY201m)
"""

import re
from typing import Tuple, List, Optional
import numpy as np
import pandas as pd

from src.config import (
    REFERENCE_YEAR,
    AED_TO_VND,
    OUTLIER_CUTOFF_KM_PER_YEAR,
    TARGET_FEATURES,
    FEATURE_FLAG_COLS,
    REGION_MAPPING
)

# Regex to parse description: matches "... with <features>. Condition: <condition>."
DESC_PATTERN = re.compile(r"with\s+(.*?)\.\s*Condition:\s*(.*?)\.?$", re.IGNORECASE)


def parse_description(description: str) -> Tuple[List[str], str]:
    """
    Parses a listing's Description text into a list of features and a condition label.
    Example: '2016 toyota camry with Rear camera, Leather seats. Condition: Engine repaired.'
    Returns: (['Rear camera', 'Leather seats'], 'Engine repaired')
    """
    if not isinstance(description, str) or not description.strip():
        return [], "Unknown"
        
    match = DESC_PATTERN.search(description.strip())
    if match:
        raw_features, raw_condition = match.groups()
        features = [f.strip() for f in raw_features.split(",") if f.strip()]
        condition = raw_condition.strip()
        return features, condition
    return [], "Unknown"


def clean_raw_dataframe(raw_df: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans raw CSV data:
    - Standardizes column names
    - Trims whitespace from strings (including location)
    - Converts unknown / missing cylinders to NaN
    - Parses features and condition from Description
    """
    df = raw_df.copy()
    
    # Standardize column naming
    col_map = {
        "Make": "make",
        "Model": "model",
        "Year": "model_year",
        "Price": "price_aed",
        "Mileage": "mileage_km",
        "Body Type": "body_type",
        "Cylinders": "cylinders",
        "Transmission": "transmission",
        "Fuel Type": "fuel_type",
        "Color": "color",
        "Location": "location",
        "Description": "description"
    }
    df = df.rename(columns=col_map)
    
    # Trim text fields
    for col in ["make", "model", "body_type", "transmission", "fuel_type", "color", "location"]:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()
            if col in ["make", "model"]:
                df[col] = df[col].str.lower()
                
    # Clean Cylinders: blank and "Unknown" to NaN, then numeric
    df["cylinders_raw"] = df["cylinders"]
    df["cylinders"] = pd.to_numeric(
        df["cylinders"].astype(str).str.strip().replace({"Unknown": np.nan, "nan": np.nan, "": np.nan}),
        errors="coerce"
    )
    
    # Parse Description
    parsed = df["description"].apply(parse_description)
    df["features_list"] = parsed.apply(lambda x: x[0])
    df["condition"] = parsed.apply(lambda x: x[1])
    
    return df


def impute_cylinders(df: pd.DataFrame) -> pd.DataFrame:
    """
    Imputes missing cylinders:
    1. Mode within (make, model)
    2. Fallback: Mode within body_type
    3. Fallback: Global mode (6)
    Records flag cylinders_imputed.
    """
    df = df.copy()
    df["cylinders_imputed"] = df["cylinders"].isna().astype(int)
    
    # 1. Mode by (make, model)
    mm_mode = df.groupby(["make", "model"])["cylinders"].transform(
        lambda s: s.mode().iloc[0] if not s.mode().empty else np.nan
    )
    df["cylinders"] = df["cylinders"].fillna(mm_mode)
    
    # 2. Mode by body_type
    bt_mode = df.groupby("body_type")["cylinders"].transform(
        lambda s: s.mode().iloc[0] if not s.mode().empty else np.nan
    )
    df["cylinders"] = df["cylinders"].fillna(bt_mode)
    
    # 3. Global mode fallback
    global_mode = df["cylinders"].mode().iloc[0] if not df["cylinders"].mode().empty else 6.0
    df["cylinders"] = df["cylinders"].fillna(global_mode).astype(int)
    
    return df


def engineer_features(df: pd.DataFrame, reference_year: int = REFERENCE_YEAR) -> pd.DataFrame:
    """
    Computes all planned domain and statistical features:
    - car_age, mileage_per_year, log_mileage, log_price
    - n_features, has_<feature> binary flags
    - is_accident_history
    - region_group
    - implausible_usage flag
    - price_vnd (display only)
    """
    df = df.copy()
    
    # Ensure cylinders are imputed if missing
    if df["cylinders"].isna().any():
        df = impute_cylinders(df)
    elif "cylinders_imputed" not in df.columns:
        df["cylinders_imputed"] = 0

    # Age and Usage Features
    df["car_age"] = (reference_year - df["model_year"]).clip(lower=1)
    df["mileage_per_year"] = (df["mileage_km"] / df["car_age"]).round(1)
    df["log_mileage"] = np.log(df["mileage_km"].clip(lower=1))
    
    if "price_aed" in df.columns:
        df["log_price"] = np.log(df["price_aed"].clip(lower=1))
        df["price_vnd"] = (df["price_aed"] * AED_TO_VND).round(0)
    
    # Optional Equipment Flags
    if "features_list" not in df.columns:
        parsed = df["description"].apply(parse_description)
        df["features_list"] = parsed.apply(lambda x: x[0])
        df["condition"] = parsed.apply(lambda x: x[1])

    df["n_features"] = df["features_list"].apply(len)
    
    for feat_name, col_name in zip(TARGET_FEATURES, FEATURE_FLAG_COLS):
        df[col_name] = df["features_list"].apply(
            lambda flist: int(any(feat_name.lower() in f.lower() for f in flist))
        )
        
    # Condition Indicator
    df["is_accident_history"] = (df["condition"] == "Accident history").astype(int)
    
    # Regional Grouping
    df["region_group"] = df["location"].map(REGION_MAPPING).fillna("Other Northern Emirates")
    
    # Anomaly Flags
    df["implausible_usage"] = (df["mileage_per_year"] > OUTLIER_CUTOFF_KM_PER_YEAR).astype(int)
    
    return df


def compute_make_tiers(
    train_df: pd.DataFrame, 
    apply_df: Optional[pd.DataFrame] = None
) -> Tuple[pd.DataFrame, dict]:
    """
    Computes four price tiers (Budget, Mid, Premium, Ultra-Luxury) based on
    make median prices fitted strictly on train_df to prevent data leakage.
    Returns: (dataframe with make_tier, tier_mapping_dict)
    """
    target_df = apply_df if apply_df is not None else train_df
    target_df = target_df.copy()
    
    make_medians = train_df.groupby("make")["price_aed"].median()
    q25, q50, q75 = make_medians.quantile([0.25, 0.50, 0.75])
    
    def get_tier(make_name: str) -> str:
        med = make_medians.get(make_name, q50)
        if med < q25:
            return "Budget"
        elif med < q50:
            return "Mid"
        elif med < q75:
            return "Premium"
        else:
            return "Ultra-Luxury"
            
    tier_map = {m: get_tier(m) for m in make_medians.index}
    target_df["make_tier"] = target_df["make"].map(tier_map).fillna("Mid")
    return target_df, tier_map
