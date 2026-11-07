"""
SQL Console Page: Interactive Relational Query Runner
UAE Used-Car Market Analysis (ADY201m - Phase 5)
"""import sys
import time
import sqlite3
import re
from pathlib import Path
import streamlit as st
import pandas as pd

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.utils import render_sidebar_controls
from src.config import DB_PATH, SQL_DIR, to_rel_path

st.set_page_config(page_title="SQL Console | UAE Used Cars", page_icon="car", layout="wide")
currency, fx_rate = render_sidebar_controls()

st.title("SQL Database Console")
st.markdown("Execute pre-configured analytical queries (Q1–Q12) or write custom read-only `SELECT` queries against the SQLite database.")

if not DB_PATH.exists():
    st.error(f"Database file not found at {to_rel_path(DB_PATH)}.")
    st.stop()

# ----------------------------------------------------
# 1. Preset Queries Catalog
# ----------------------------------------------------
PRESET_QUERIES = {
    "Q1: Market Overview KPIs (AED & VND)": """SELECT 
    COUNT(*) AS total_listings,
    ROUND(AVG(c.price_aed), 2) AS avg_price_aed,
    MIN(c.price_aed) AS min_price_aed,
    MAX(c.price_aed) AS max_price_aed,
    ROUND(AVG(c.price_aed * fx.aed_to_vnd) / 1e6, 2) AS avg_price_million_vnd,
    ROUND(MIN(c.price_aed * fx.aed_to_vnd) / 1e6, 2) AS min_price_million_vnd,
    ROUND(MAX(c.price_aed * fx.aed_to_vnd) / 1e6, 2) AS max_price_million_vnd
FROM cars c
CROSS JOIN (SELECT aed_to_vnd FROM fx_rates ORDER BY rate_date DESC LIMIT 1) fx;""",

    "Q2: High-Volume Makes Average Pricing (>= 30 listings)": """SELECT 
    mk.make_name,
    COUNT(c.car_id) AS listing_count,
    ROUND(AVG(c.price_aed), 0) AS avg_price_aed,
    ROUND(AVG(c.mileage_km), 0) AS avg_mileage_km
FROM cars c
JOIN models m ON c.model_id = m.model_id
JOIN makes mk ON m.make_id = mk.make_id
GROUP BY mk.make_name
HAVING COUNT(c.car_id) >= 30
ORDER BY avg_price_aed DESC;""",

    "Q3: Geographic Market Share by Location": """SELECT 
    e.emirate_name,
    COUNT(c.car_id) AS listing_count,
    ROUND(100.0 * COUNT(c.car_id) / SUM(COUNT(c.car_id)) OVER (), 2) AS share_pct,
    ROUND(AVG(c.price_aed), 0) AS avg_price_aed
FROM cars c
JOIN emirates e ON c.emirate_id = e.emirate_id
GROUP BY e.emirate_name
ORDER BY listing_count DESC;""",

    "Q4: Year-Over-Year Price Dynamics (LAG Window Function)": """WITH yearly_stats AS (
    SELECT 
        model_year,
        COUNT(*) AS n_listings,
        AVG(price_aed) AS avg_price
    FROM cars
    GROUP BY model_year
)
SELECT 
    model_year,
    n_listings,
    ROUND(avg_price, 0) AS avg_price_aed,
    ROUND(100.0 * (avg_price - LAG(avg_price) OVER (ORDER BY model_year)) 
          / LAG(avg_price) OVER (ORDER BY model_year), 2) AS yoy_change_pct
FROM yearly_stats
ORDER BY model_year;""",

    "Q5: Top 3 Most Expensive Models per Make": """WITH model_stats AS (
    SELECT 
        mk.make_name,
        m.model_name,
        COUNT(c.car_id) AS listings,
        AVG(c.price_aed) AS avg_price
    FROM cars c
    JOIN models m ON c.model_id = m.model_id
    JOIN makes mk ON m.make_id = mk.make_id
    GROUP BY mk.make_name, m.model_name
    HAVING COUNT(c.car_id) >= 5
),
ranked_models AS (
    SELECT 
        make_name,
        model_name,
        listings,
        ROUND(avg_price, 0) AS avg_price_aed,
        ROW_NUMBER() OVER (PARTITION BY make_name ORDER BY avg_price DESC) AS price_rank
    FROM model_stats
)
SELECT make_name, model_name, listings, avg_price_aed, price_rank
FROM ranked_models
WHERE price_rank <= 3
ORDER BY make_name, price_rank
LIMIT 30;""",

    "Q6: Supply Concentration & Cumulative Share (Pareto)": """WITH make_counts AS (
    SELECT 
        mk.make_name,
        COUNT(c.car_id) AS n_listings
    FROM cars c
    JOIN models m ON c.model_id = m.model_id
    JOIN makes mk ON m.make_id = mk.make_id
    GROUP BY mk.make_name
)
SELECT 
    make_name,
    n_listings,
    ROUND(100.0 * n_listings / SUM(n_listings) OVER (), 2) AS make_share_pct,
    ROUND(100.0 * SUM(n_listings) OVER (ORDER BY n_listings DESC) / SUM(n_listings) OVER (), 2) AS cumulative_share_pct
FROM make_counts
ORDER BY n_listings DESC
LIMIT 15;""",

    "Q7: Overpriced Inventory Identification (> 2x Model Avg)": """WITH model_benchmarks AS (
    SELECT 
        model_id,
        AVG(price_aed) AS model_avg_price
    FROM cars
    GROUP BY model_id
    HAVING COUNT(*) >= 5
)
SELECT 
    mk.make_name,
    COUNT(c.car_id) AS overpriced_count,
    ROUND(AVG(c.price_aed), 0) AS avg_overpriced_aed
FROM cars c
JOIN model_benchmarks mb ON c.model_id = mb.model_id
JOIN models m ON c.model_id = m.model_id
JOIN makes mk ON m.make_id = mk.make_id
WHERE c.price_aed > 2.0 * mb.model_avg_price
GROUP BY mk.make_name
ORDER BY overpriced_count DESC
LIMIT 10;""",

    "Q8: Optional Feature Price Premium": """WITH feature_with AS (
    SELECT 
        f.feature_id,
        f.feature_name,
        COUNT(cf.car_id) AS cars_with_feature,
        AVG(c.price_aed) AS avg_price_with
    FROM features f
    JOIN car_features cf ON f.feature_id = cf.feature_id
    JOIN cars c ON cf.car_id = c.car_id
    GROUP BY f.feature_id, f.feature_name
),
totals AS (
    SELECT COUNT(*) AS total_cars, SUM(price_aed) AS total_price FROM cars
)
SELECT 
    fw.feature_name,
    fw.cars_with_feature,
    ROUND(fw.avg_price_with, 0) AS avg_price_with_feature,
    ROUND((t.total_price - (fw.avg_price_with * fw.cars_with_feature)) / (t.total_cars - fw.cars_with_feature), 0) AS avg_price_without_feature,
    ROUND(fw.avg_price_with - ((t.total_price - (fw.avg_price_with * fw.cars_with_feature)) / (t.total_cars - fw.cars_with_feature)), 0) AS absolute_premium_aed
FROM feature_with fw
CROSS JOIN totals t
ORDER BY absolute_premium_aed DESC;""",

    "Q9: Condition Effect Relative to Model Median": """WITH model_benchmarks AS (
    SELECT 
        model_id,
        AVG(price_aed) AS model_avg_price
    FROM cars
    GROUP BY model_id
    HAVING COUNT(*) >= 5
)
SELECT 
    cd.condition_name,
    COUNT(c.car_id) AS listing_count,
    ROUND(AVG(c.price_aed), 0) AS raw_avg_price_aed,
    ROUND(AVG(1.0 * c.price_aed / mb.model_avg_price), 3) AS avg_relative_price_ratio
FROM cars c
JOIN conditions cd ON c.condition_id = cd.condition_id
JOIN model_benchmarks mb ON c.model_id = mb.model_id
GROUP BY cd.condition_name
ORDER BY avg_relative_price_ratio DESC;""",

    "Q10: Mileage Bands across Price Quartiles (NTILE(4))": """WITH ranked_cars AS (
    SELECT 
        car_id,
        price_aed,
        mileage_km,
        NTILE(4) OVER (ORDER BY price_aed) AS price_quartile,
        CASE 
            WHEN mileage_km < 50000 THEN '1. Under 50k km'WHEN mileage_km < 100000 THEN '2. 50k - 100k km'WHEN mileage_km < 150000 THEN '3. 100k - 150k km'ELSE '4. 150k+ km'END AS mileage_band
    FROM cars
)
SELECT 
    mileage_band,
    SUM(CASE WHEN price_quartile = 1 THEN 1 ELSE 0 END) AS Q1_low_price,
    SUM(CASE WHEN price_quartile = 2 THEN 1 ELSE 0 END) AS Q2_mid_low_price,
    SUM(CASE WHEN price_quartile = 3 THEN 1 ELSE 0 END) AS Q3_mid_high_price,
    SUM(CASE WHEN price_quartile = 4 THEN 1 ELSE 0 END) AS Q4_high_price,
    COUNT(*) AS total_listings
FROM ranked_cars
GROUP BY mileage_band
ORDER BY mileage_band;""",

    "Q11: 3-Year Moving Average by Model Year": """WITH yearly_prices AS (
    SELECT 
        model_year,
        COUNT(*) AS listings,
        AVG(price_aed) AS yearly_avg_price
    FROM cars
    GROUP BY model_year
)
SELECT 
    model_year,
    listings,
    ROUND(yearly_avg_price, 0) AS yearly_avg_aed,
    ROUND(AVG(yearly_avg_price) OVER (
        ORDER BY model_year 
        ROWS BETWEEN 2 PRECEDING AND CURRENT ROW
    ), 0) AS three_year_moving_avg_aed
FROM yearly_prices
ORDER BY model_year;""",

    "Q12: Median Price by Body Type (Window Function)": """WITH ordered_prices AS (
    SELECT 
        b.body_type_name,
        c.price_aed,
        ROW_NUMBER() OVER (PARTITION BY b.body_type_name ORDER BY c.price_aed) AS row_num,
        COUNT(*) OVER (PARTITION BY b.body_type_name) AS total_rows
    FROM cars c
    JOIN body_types b ON c.body_type_id = b.body_type_id
)
SELECT 
    body_type_name,
    total_rows AS listing_count,
    ROUND(AVG(price_aed), 0) AS median_price_aed
FROM ordered_prices
WHERE row_num IN ((total_rows + 1) / 2, (total_rows + 2) / 2)
GROUP BY body_type_name, total_rows
ORDER BY median_price_aed DESC;"""
}

# Dropdown selection
selected_preset = st.selectbox("Select Preset Query (Q1 - Q12)", options=list(PRESET_QUERIES.keys()))
default_sql = PRESET_QUERIES[selected_preset]

user_query = st.text_area("SQL Statement (Read-only SELECT supported)", value=default_sql, height=220)

col_run, col_clear = st.columns([1, 6])
with col_run:
    execute_clicked = st.button("Run Query", type="primary")

if execute_clicked:
    cleaned_query = user_query.strip()
    
    # Security / Read-only enforcement
    forbidden_tokens = ["DROP", "DELETE", "INSERT", "UPDATE", "ALTER", "TRUNCATE", "REPLACE"]
    has_forbidden = any(re.search(rf"\b{token}\b", cleaned_query, re.IGNORECASE) for token in forbidden_tokens)
    
    if not cleaned_query.upper().startswith("SELECT") and not cleaned_query.upper().startswith("WITH"):
        st.error("Execution Rejected: Only 'SELECT' or 'WITH' queries are permitted.")
    elif has_forbidden:
        st.error("Execution Rejected: Modification statements (DROP, DELETE, INSERT, UPDATE, etc.) are strictly disallowed.")
    else:
        uri = f"file:{DB_PATH.resolve().as_posix()}?mode=ro"conn = sqlite3.connect(uri, uri=True)
        try:
            start_time = time.time()
            result_df = pd.read_sql_query(cleaned_query, conn)
            elapsed = time.time() - start_time
            
            st.success(f"Query returned **{len(result_df):,} rows** in **{elapsed*1000:.1f} ms**.")
            st.dataframe(result_df.head(500), use_container_width=True)
            
            if len(result_df) > 500:
                st.caption("Display capped at first 500 rows.")
        except Exception as e:
            st.error(f"SQL Execution Error: {e}")
        finally:
            conn.close()
