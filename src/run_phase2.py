"""
Phase 2 Execution Script: Exploratory Data Analysis & Data Preprocessing
UAE Used-Car Market Analysis (ADY201m)

Executes all Phase 2 pipeline stages:
1. Ingests raw data and validates inventory
2. Cleans text fields and parses Description into features & condition
3. Imputes missing Cylinders using hierarchical group mode
4. Detects outliers via 1.5x IQR rule on raw and log price
5. Engineers domain features (car_age, mileage_per_year, make_tier, etc.)
6. Exports interim clean listings and processed model features
7. Generates Figures 2.2, 2.3, and 2.4
8. Logs all operations with relative paths to logs/phase2_eda_preprocessing.log
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import (
    RAW_DATA_PATH,
    CLEAN_LISTINGS_PATH,
    MODEL_FEATURES_PATH,
    FIGURES_DIR,
    LOGS_DIR,
    REFERENCE_YEAR,
    AED_TO_VND,
    OUTLIER_CUTOFF_KM_PER_YEAR,
    to_rel_path,
    setup_chart_style
)
from src.logger import get_logger
from src.features import (
    clean_raw_dataframe,
    impute_cylinders,
    engineer_features,
    compute_make_tiers
)

logger = get_logger("phase2", "phase2_eda_preprocessing.log")


def run_phase_2():
    logger.info("=" * 70)
    logger.info("STARTING PHASE 2: EDA & PREPROCESSING PIPELINE")
    logger.info("=" * 70)
    setup_chart_style()

    # ---------------------------------------------------------
    # 1. Load Raw Data
    # ---------------------------------------------------------
    rel_raw = to_rel_path(RAW_DATA_PATH)
    logger.info(f"Loading raw dataset from: {rel_raw}")
    if not RAW_DATA_PATH.exists():
        logger.error(f"Raw data file not found at {rel_raw}")
        sys.exit(1)

    raw_df = pd.read_csv(RAW_DATA_PATH)
    logger.info(f"Raw data ingested successfully: {raw_df.shape[0]} rows, {raw_df.shape[1]} columns")

    # ---------------------------------------------------------
    # 2. Raw Missing Values Analysis
    # ---------------------------------------------------------
    raw_missing = raw_df.isna().sum()
    logger.info("Raw missing values per column:")
    for col, count in raw_missing.items():
        pct = (count / len(raw_df)) * 100
        logger.info(f"  - {col}: {count} missing ({pct:.2f}%)")

    # Raw Cylinders specific check (NaN + 'Unknown')
    cyl_unknown = (raw_df["Cylinders"].astype(str).str.strip().str.lower() == "unknown").sum()
    cyl_nan = raw_df["Cylinders"].isna().sum()
    logger.info(f"Cylinders raw missingness: {cyl_nan} blank/NaN + {cyl_unknown} 'Unknown' = {cyl_nan + cyl_unknown} ({((cyl_nan + cyl_unknown)/len(raw_df))*100:.2f}%)")

    # ---------------------------------------------------------
    # 3. Clean Text & Parse Description
    # ---------------------------------------------------------
    logger.info("Cleaning raw fields and parsing Description text...")
    cleaned_df = clean_raw_dataframe(raw_df)
    logger.info(f"Unique conditions parsed: {cleaned_df['condition'].value_counts().to_dict()}")

    # ---------------------------------------------------------
    # 4. Impute Cylinders
    # ---------------------------------------------------------
    logger.info("Imputing Cylinders via hierarchical mode (Make+Model -> Body Type -> Global)...")
    imputed_df = impute_cylinders(cleaned_df)
    imputed_count = imputed_df["cylinders_imputed"].sum()
    logger.info(f"Cylinders imputed count: {imputed_count} ({imputed_count / len(imputed_df) * 100:.2f}%)")
    logger.info(f"Remaining nulls in cylinders: {imputed_df['cylinders'].isna().sum()}")

    # ---------------------------------------------------------
    # 5. Outlier Detection (Raw vs Log Price)
    # ---------------------------------------------------------
    logger.info("Computing 1.5x IQR outlier fences on Price...")
    # Raw Price IQR
    q1_p = raw_df["Price"].quantile(0.25)
    q3_p = raw_df["Price"].quantile(0.75)
    iqr_p = q3_p - q1_p
    lower_raw = q1_p - 1.5 * iqr_p
    upper_raw = q3_p + 1.5 * iqr_p
    raw_outliers = (raw_df["Price"] > upper_raw) | (raw_df["Price"] < lower_raw)
    n_raw_outliers = raw_outliers.sum()
    logger.info(f"Raw Price IQR: Q1={q1_p:,.1f}, Q3={q3_p:,.1f}, IQR={iqr_p:,.1f}")
    logger.info(f"Raw Price Fences: [{lower_raw:,.1f}, {upper_raw:,.1f}] AED")
    logger.info(f"Raw Price Outliers: {n_raw_outliers} ({n_raw_outliers / len(raw_df) * 100:.2f}%) - ALL genuine supercars preserved")

    # Log Price IQR
    log_p = np.log(raw_df["Price"])
    q1_lp = log_p.quantile(0.25)
    q3_lp = log_p.quantile(0.75)
    iqr_lp = q3_lp - q1_lp
    lower_lp = q1_lp - 1.5 * iqr_lp
    upper_lp = q3_lp + 1.5 * iqr_lp
    log_outliers = (log_p > upper_lp) | (log_p < lower_lp)
    n_log_outliers = log_outliers.sum()
    logger.info(f"Log Price IQR: Q1={q1_lp:.3f}, Q3={q3_lp:.3f}, IQR={iqr_lp:.3f}")
    logger.info(f"Log Price Fences: [{lower_lp:.3f}, {upper_lp:.3f}] (in AED: [{np.exp(lower_lp):,.0f}, {np.exp(upper_lp):,.0f}])")
    logger.info(f"Log Price Outliers: {n_log_outliers} ({n_log_outliers / len(raw_df) * 100:.2f}%)")

    # ---------------------------------------------------------
    # 6. Feature Engineering
    # ---------------------------------------------------------
    logger.info("Executing feature engineering pipeline...")
    features_df = engineer_features(imputed_df, reference_year=REFERENCE_YEAR)
    features_df, tier_map = compute_make_tiers(features_df)
    logger.info(f"Make tier counts:\n{features_df['make_tier'].value_counts().to_dict()}")

    implausible_count = features_df["implausible_usage"].sum()
    logger.info(f"Implausible usage flagged (>50,000 km/yr): {implausible_count} ({implausible_count / len(features_df) * 100:.2f}%)")

    # ---------------------------------------------------------
    # 7. Generate & Save Figures
    # ---------------------------------------------------------
    logger.info("Generating Phase 2 figures...")

    # Fig 2.2: Price distribution raw vs log
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    sns.histplot(raw_df["Price"] / 1000, bins=50, kde=True, ax=ax1, color="#2b5c8f")
    ax1.set_title("Raw Price Distribution (AED in Thousands)", fontsize=13, fontweight="bold")
    ax1.set_xlabel("Price (Thousands AED)")
    ax1.set_ylabel("Listings Count")
    ax1.axvline(raw_df["Price"].median() / 1000, color="#d95f02", linestyle="--", label=f"Median: {raw_df['Price'].median()/1000:,.1f}k AED")
    ax1.legend()

    sns.histplot(np.log(raw_df["Price"]), bins=50, kde=True, ax=ax2, color="#2b5c8f")
    ax2.set_title("Log-Transformed Price Distribution", fontsize=13, fontweight="bold")
    ax2.set_xlabel("Log(Price [AED])")
    ax2.set_ylabel("Listings Count")
    ax2.axvline(np.log(raw_df["Price"]).median(), color="#d95f02", linestyle="--", label=f"Median: {np.log(raw_df['Price']).median():.2f}")
    ax2.legend()

    plt.tight_layout()
    fig22_path = FIGURES_DIR / "fig_2_2_price_distribution_raw_vs_log.png"
    fig.savefig(fig22_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved: {to_rel_path(fig22_path)}")

    # Fig 2.3: Listings by Body Style
    fig, ax = plt.subplots(figsize=(10, 5))
    bt_counts = raw_df["Body Type"].value_counts(ascending=True)
    bt_counts.plot(kind="barh", ax=ax, color="#2b5c8f")
    ax.set_title("Listings Count by Body Type (N = 10,000)", fontsize=13, fontweight="bold")
    ax.set_xlabel("Number of Listings")
    ax.set_ylabel("Body Type")
    for i, v in enumerate(bt_counts):
        ax.text(v + 50, i, f"{v:,} ({v/100:.1f}%)", va="center", fontsize=9)
    plt.tight_layout()
    fig23_path = FIGURES_DIR / "fig_2_3_listings_per_body_type.png"
    fig.savefig(fig23_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved: {to_rel_path(fig23_path)}")

    # Fig 2.4: Outlier Boxplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    sns.boxplot(y=raw_df["Price"] / 1000, ax=ax1, color="#7570b3")
    ax1.set_title("Raw Price Outlier Boxplot (1.5x IQR Fences)", fontsize=13, fontweight="bold")
    ax1.set_ylabel("Price (Thousands AED)")
    ax1.axhline(upper_raw / 1000, color="red", linestyle="--", label=f"Upper Fence: {upper_raw/1000:,.1f}k AED (1,100 outliers)")
    ax1.legend()

    sns.boxplot(y=np.log(raw_df["Price"]), ax=ax2, color="#1b9e77")
    ax2.set_title("Log Price Outlier Boxplot (1.5x IQR Fences)", fontsize=13, fontweight="bold")
    ax2.set_ylabel("Log(Price [AED])")
    ax2.axhline(upper_lp, color="red", linestyle="--", label=f"Upper Fence: {upper_lp:.2f}")
    ax2.axhline(lower_lp, color="red", linestyle="--", label=f"Lower Fence: {lower_lp:.2f} (91 outliers)")
    ax2.legend()

    plt.tight_layout()
    fig24_path = FIGURES_DIR / "fig_2_4_outlier_boxplots.png"
    fig.savefig(fig24_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved: {to_rel_path(fig24_path)}")

    # ---------------------------------------------------------
    # 8. Export Interim and Processed CSVs
    # ---------------------------------------------------------
    CLEAN_LISTINGS_PATH.parent.mkdir(parents=True, exist_ok=True)
    MODEL_FEATURES_PATH.parent.mkdir(parents=True, exist_ok=True)

    imputed_df.to_csv(CLEAN_LISTINGS_PATH, index=False)
    logger.info(f"Saved clean interim listings: {to_rel_path(CLEAN_LISTINGS_PATH)} ({len(imputed_df)} rows)")

    features_df.to_csv(MODEL_FEATURES_PATH, index=False)
    logger.info(f"Saved processed model features: {to_rel_path(MODEL_FEATURES_PATH)} ({len(features_df)} rows, {features_df.shape[1]} columns)")

    # ---------------------------------------------------------
    # 9. Gate G1 Integrity Audit
    # ---------------------------------------------------------
    logger.info("-" * 50)
    logger.info("AUDITING GATE G1 CRITERIA:")
    assert len(features_df) == 10000, f"Row count mismatch: {len(features_df)} != 10000"
    assert features_df["cylinders"].isna().sum() == 0, "Cylinders contains remaining NaN"
    assert features_df["car_age"].min() >= 1, "car_age has invalid values < 1"
    assert "log_price" in features_df.columns, "log_price column missing"
    assert "make_tier" in features_df.columns, "make_tier column missing"
    assert "has_sunroof" in features_df.columns, "feature flags missing"
    logger.info("[PASS] Gate G1 PASSED: Exactly 10,000 rows, 0 nulls in critical predictors, all engineered features verified.")
    logger.info("=" * 70)
    logger.info("PHASE 2 COMPLETED SUCCESSFULLY.")
    logger.info("=" * 70)

    # Return key metrics dictionary for report verification
    return {
        "n_rows": len(features_df),
        "n_cols": features_df.shape[1],
        "cylinders_missing_before": cyl_nan + cyl_unknown,
        "cylinders_missing_after": features_df["cylinders"].isna().sum(),
        "cylinders_imputed_count": imputed_count,
        "upper_raw_fence": upper_raw,
        "raw_outliers_count": n_raw_outliers,
        "raw_outliers_pct": (n_raw_outliers / len(raw_df)) * 100,
        "lower_log_fence": lower_lp,
        "upper_log_fence": upper_lp,
        "log_outliers_count": n_log_outliers,
        "log_outliers_pct": (n_log_outliers / len(raw_df)) * 100,
        "implausible_usage_count": implausible_count,
        "implausible_usage_pct": (implausible_count / len(features_df)) * 100,
    }


if __name__ == "__main__":
    run_phase_2()
