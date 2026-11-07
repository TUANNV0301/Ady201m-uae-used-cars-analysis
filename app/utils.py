"""
Streamlit Application Utilities
UAE Used-Car Market Analysis (ADY201m - Phase 5)
"""

import sys
import json
import sqlite3
from pathlib import Path
from typing import Tuple, Dict, Any, Optional
import pandas as pd
import streamlit as st
import joblib

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import (
    DB_PATH,
    AED_TO_VND,
    FX_DATE,
    MODEL_PIPELINE_PATH,
    MODEL_METADATA_PATH,
    DEAL_SCORES_PATH,
    to_rel_path
)


@st.cache_data(show_spinner="Loading market dataset from SQLite...")
def load_all_listings() -> pd.DataFrame:
    """Reads all rows from the database view v_listings."""
    if not DB_PATH.exists():
        st.error(f"Database not found at {to_rel_path(DB_PATH)}. Run 'python src/build_db.py' first.")
        return pd.DataFrame()
        
    uri = f"file:{DB_PATH.resolve().as_posix()}?mode=ro"
    conn = sqlite3.connect(uri, uri=True)
    try:
        df = pd.read_sql_query("SELECT * FROM v_listings", conn)
        return df
    finally:
        conn.close()


@st.cache_resource(show_spinner="Loading trained valuation model...")
def load_trained_model():
    """Loads the serialized scikit-learn model and associated metadata."""
    if not MODEL_PIPELINE_PATH.exists() or not MODEL_METADATA_PATH.exists():
        return None, None
        
    model = joblib.load(MODEL_PIPELINE_PATH)
    with open(MODEL_METADATA_PATH, "r", encoding="utf-8") as f:
        metadata = json.load(f)
    return model, metadata


@st.cache_data(show_spinner="Loading deal valuation scores...")
def load_deal_scores() -> pd.DataFrame:
    """Loads precomputed out-of-fold deal scores and valuation flags."""
    if not DEAL_SCORES_PATH.exists():
        return pd.DataFrame()
    return pd.read_csv(DEAL_SCORES_PATH)


def format_currency(
    val: float, 
    currency: str = "AED", 
    rate: float = AED_TO_VND
) -> str:
    """Formats numeric values into AED or VND formatted strings."""
    if pd.isna(val):
        return "N/A"
    if currency == "VND":
        vnd_val = val * rate
        if vnd_val >= 1e9:
            return f"{vnd_val / 1e9:,.2f} B VND"
        return f"{vnd_val / 1e6:,.1f} M VND"
    else:
        return f"{val:,.0f} AED"


def render_sidebar_controls():
    """Renders the common sidebar controls for currency selection and filtering."""
    st.sidebar.title("Configuration & Filters")
    
    # 1. Currency Selector
    currency = st.sidebar.radio(
        "Display Currency",
        options=["AED", "VND"],
        index=0 if st.session_state.get("currency", "AED") == "AED" else 1,
        key="currency_radio"
    )
    st.session_state["currency"] = currency
    
    # FX Rate display/editor
    if currency == "VND":
        custom_rate = st.sidebar.number_input(
            f"AED to VND Rate (As of {FX_DATE})",
            min_value=1000.0,
            max_value=20000.0,
            value=float(st.session_state.get("fx_rate", AED_TO_VND)),
            step=50.0,
            key="fx_rate_input"
        )
        st.session_state["fx_rate"] = custom_rate
        st.sidebar.caption("Note: VND figures are conversion benchmarks, not Vietnam domestic retail prices.")
    else:
        st.session_state["fx_rate"] = AED_TO_VND

    st.sidebar.markdown("---")
    return st.session_state["currency"], st.session_state["fx_rate"]
