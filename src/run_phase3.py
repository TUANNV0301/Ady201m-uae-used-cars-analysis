"""
Phase 3: Python Analysis & Visualization Catalog
UAE Used-Car Market Analysis (ADY201m)

Generates 12 report figures, analysis tables (crosstabs, correlations,
within-model regression, feature premiums, condition Kruskal-Wallis test)
and exports all outputs to figures/ and logs to logs/phase3_analysis.log.
"""

import sys
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from scipy import stats
import plotly.express as px
import plotly.io as pio

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import (
    MODEL_FEATURES_PATH, FIGURES_DIR, AED_TO_VND,
    REFERENCE_YEAR, to_rel_path, setup_chart_style
)
from src.logger import get_logger

logger = get_logger("phase3", "phase3_analysis.log")
setup_chart_style()

TIER_COLORS = {
    "Budget": "#2ca02c",
    "Mid": "#1f77b4",
    "Premium": "#ff7f0e",
    "Ultra-Luxury": "#d62728",
}
PALETTE = "colorblind"


def save_fig(fig, fname):
    path = FIGURES_DIR / fname
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved: {to_rel_path(path)}")
    return path


def load_data():
    df = pd.read_csv(MODEL_FEATURES_PATH)
    logger.info(f"Loaded model features: {df.shape[0]} rows, {df.shape[1]} cols")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# FIG 3.1 – Listings per Make (Top 15 horizontal bar)
# ─────────────────────────────────────────────────────────────────────────────
def fig_3_1(df):
    top15 = df["make"].value_counts().head(15).sort_values()
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.barh(top15.index, top15.values, color="#1f77b4")
    ax.set_title("Figure 3.1. Listing Volume by Vehicle Make (Top 15)", fontweight="bold", fontsize=12)
    ax.set_xlabel("Number of Listings")
    ax.set_ylabel("Vehicle Make")
    for bar in bars:
        ax.text(bar.get_width() + 10, bar.get_y() + bar.get_height() / 2,
                f"{bar.get_width():,.0f}", va="center", fontsize=8)
    plt.tight_layout()
    return save_fig(fig, "fig_3_1_listings_per_make.png")


# ─────────────────────────────────────────────────────────────────────────────
# FIG 3.2 – Median Price by Make (Top 15 by volume)
# ─────────────────────────────────────────────────────────────────────────────
def fig_3_2(df):
    top15_makes = df["make"].value_counts().head(15).index
    med_price = df[df["make"].isin(top15_makes)].groupby("make")["price_aed"].median().sort_values(ascending=False)
    tier_map = df.drop_duplicates("make").set_index("make")["make_tier"]
    colors = [TIER_COLORS.get(tier_map.get(m, "Mid"), "#1f77b4") for m in med_price.index]

    fig, ax = plt.subplots(figsize=(12, 5))
    bars = ax.bar(range(len(med_price)), med_price.values / 1000, color=colors)
    ax.set_xticks(range(len(med_price)))
    ax.set_xticklabels([m.title() for m in med_price.index], rotation=35, ha="right", fontsize=9)
    ax.set_title("Figure 3.2. Median Asking Price by Make – Top 15 Volume Brands", fontweight="bold", fontsize=12)
    ax.set_ylabel("Median Price (Thousands AED)")
    ax2 = ax.twinx()
    ax2.set_ylim(ax.get_ylim()[0] * AED_TO_VND / 1e6, ax.get_ylim()[1] * AED_TO_VND / 1e6)
    ax2.set_ylabel("Median Price (Million VND)")
    # Legend for tiers
    from matplotlib.patches import Patch
    handles = [Patch(color=c, label=t) for t, c in TIER_COLORS.items()]
    ax.legend(handles=handles, title="Make Tier", loc="upper right", fontsize=8)
    plt.tight_layout()
    return save_fig(fig, "fig_3_2_median_price_per_make.png")


# ─────────────────────────────────────────────────────────────────────────────
# FIG 3.3 – Log Price by Body Type (Boxplot)
# ─────────────────────────────────────────────────────────────────────────────
def fig_3_3(df):
    order = df.groupby("body_type")["log_price"].median().sort_values(ascending=False).index
    fig, ax = plt.subplots(figsize=(12, 5))
    sns.boxplot(data=df, x="body_type", y="log_price", order=order,
                palette="Blues_r", ax=ax)
    ax.set_title("Figure 3.3. Log Price Distribution by Body Type", fontweight="bold", fontsize=12)
    ax.set_xlabel("Body Type")
    ax.set_ylabel("ln(Price [AED])")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=35, ha="right", fontsize=8)
    plt.tight_layout()
    return save_fig(fig, "fig_3_3_log_price_by_body_type.png")


# ─────────────────────────────────────────────────────────────────────────────
# FIG 3.4 – Median Price & Listing Count by Model Year (Dual-axis line)
# ─────────────────────────────────────────────────────────────────────────────
def fig_3_4(df):
    yearly = df.groupby("model_year").agg(
        median_price=("price_aed", "median"),
        count=("price_aed", "count")
    ).reset_index()
    fig, ax1 = plt.subplots(figsize=(12, 5))
    color1, color2 = "#1f77b4", "#d62728"
    ax1.plot(yearly["model_year"], yearly["median_price"] / 1000,
             color=color1, linewidth=2, marker="o", markersize=4, label="Median Price (AED)")
    ax1.set_xlabel("Model Year")
    ax1.set_ylabel("Median Price (Thousands AED)", color=color1)
    ax1.tick_params(axis="y", labelcolor=color1)
    ax2 = ax1.twinx()
    ax2.bar(yearly["model_year"], yearly["count"], alpha=0.3, color=color2, label="Listings")
    ax2.set_ylabel("Number of Listings", color=color2)
    ax2.tick_params(axis="y", labelcolor=color2)
    ax1.set_title("Figure 3.4. Median Price and Listing Count by Model Year (2005–2024)",
                  fontweight="bold", fontsize=12)
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left", fontsize=9)
    plt.tight_layout()
    return save_fig(fig, "fig_3_4_price_and_count_by_year.png")


# ─────────────────────────────────────────────────────────────────────────────
# FIG 3.5 – Mileage vs Price (Scatter, log y-axis, colored by make tier)
# ─────────────────────────────────────────────────────────────────────────────
def fig_3_5(df):
    sample = df.sample(n=min(3000, len(df)), random_state=42)
    fig, ax = plt.subplots(figsize=(10, 6))
    for tier, grp in sample.groupby("make_tier"):
        ax.scatter(grp["mileage_km"] / 1000, grp["price_aed"] / 1000,
                   alpha=0.35, s=15, label=tier,
                   color=TIER_COLORS.get(tier, "#999999"))
    ax.set_yscale("log")
    ax.set_title("Figure 3.5. Mileage vs. Asking Price by Make Tier (n=3,000 sample)",
                 fontweight="bold", fontsize=12)
    ax.set_xlabel("Odometer (Thousands km)")
    ax.set_ylabel("Price (Thousands AED) – Log Scale")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:,.0f}k"))
    ax.legend(title="Make Tier", fontsize=9)
    plt.tight_layout()
    return save_fig(fig, "fig_3_5_mileage_vs_price_by_tier.png")


# ─────────────────────────────────────────────────────────────────────────────
# FIG 3.6 – Spearman Correlation Matrix (Heatmap)
# ─────────────────────────────────────────────────────────────────────────────
def fig_3_6(df):
    cols = ["log_price", "car_age", "log_mileage", "cylinders", "n_features",
            "is_accident_history", "implausible_usage"]
    labels = ["ln(Price)", "Car Age", "ln(Mileage)", "Cylinders", "N Features",
              "Accident Flag", "Implausible Use"]
    corr = df[cols].rename(columns=dict(zip(cols, labels))).corr(method="spearman")
    fig, ax = plt.subplots(figsize=(9, 7))
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="RdBu_r",
                center=0, vmin=-1, vmax=1, linewidths=0.5, ax=ax,
                annot_kws={"size": 9})
    ax.set_title("Figure 3.6. Spearman Correlation Matrix of Key Numerical Predictors",
                 fontweight="bold", fontsize=12)
    plt.tight_layout()
    return save_fig(fig, "fig_3_6_spearman_correlation_matrix.png")


# ─────────────────────────────────────────────────────────────────────────────
# FIG 3.7 – Listings & Median Price by Region Group (Grouped bar)
# ─────────────────────────────────────────────────────────────────────────────
def fig_3_7(df):
    rg = df.groupby("region_group").agg(
        count=("price_aed", "count"),
        median_price=("price_aed", "median")
    ).reset_index().sort_values("count", ascending=False)
    x = np.arange(len(rg))
    w = 0.35
    fig, ax1 = plt.subplots(figsize=(9, 5))
    ax1.bar(x - w/2, rg["count"], w, label="Listings", color="#1f77b4")
    ax1.set_ylabel("Number of Listings", color="#1f77b4")
    ax1.tick_params(axis="y", labelcolor="#1f77b4")
    ax2 = ax1.twinx()
    ax2.bar(x + w/2, rg["median_price"] / 1000, w, label="Median Price (k AED)", color="#ff7f0e")
    ax2.set_ylabel("Median Price (Thousands AED)", color="#ff7f0e")
    ax2.tick_params(axis="y", labelcolor="#ff7f0e")
    ax1.set_xticks(x)
    ax1.set_xticklabels(rg["region_group"], fontsize=9)
    ax1.set_title("Figure 3.7. Listings and Median Price by UAE Region Group",
                  fontweight="bold", fontsize=12)
    lines1, lbl1 = ax1.get_legend_handles_labels()
    lines2, lbl2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, lbl1 + lbl2, loc="upper right", fontsize=9)
    plt.tight_layout()
    return save_fig(fig, "fig_3_7_listings_price_by_region.png")


# ─────────────────────────────────────────────────────────────────────────────
# FIG 3.8 – Price Relative to Model Median by Condition (Boxplot)
# ─────────────────────────────────────────────────────────────────────────────
def fig_3_8(df):
    model_med = df.groupby(["make", "model"])["price_aed"].transform("median")
    df2 = df.copy()
    df2["price_ratio"] = df2["price_aed"] / model_med.replace(0, np.nan)
    df2 = df2[df2["price_ratio"].between(0.1, 5)]
    order = df2.groupby("condition")["price_ratio"].median().sort_values(ascending=False).index
    fig, ax = plt.subplots(figsize=(11, 5))
    sns.boxplot(data=df2, x="condition", y="price_ratio", order=order,
                palette="Set2", ax=ax, showfliers=False)
    ax.axhline(1.0, color="red", linestyle="--", linewidth=1, label="Model Median = 1.0")
    ax.set_title("Figure 3.8. Price Ratio vs. Model Median by Vehicle Condition",
                 fontweight="bold", fontsize=12)
    ax.set_xlabel("Vehicle Condition")
    ax.set_ylabel("Price / Model Median Price")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=25, ha="right", fontsize=9)
    ax.legend()
    plt.tight_layout()
    return save_fig(fig, "fig_3_8_price_ratio_by_condition.png")


# ─────────────────────────────────────────────────────────────────────────────
# FIG 3.9 – Within-Model Feature Premium (Bar with error bars)
# ─────────────────────────────────────────────────────────────────────────────
def fig_3_9(df):
    model_med = df.groupby(["make", "model"])["price_aed"].transform("median")
    df2 = df.copy()
    df2["price_ratio"] = df2["price_aed"] / model_med.replace(0, np.nan)
    df2 = df2[df2["price_ratio"].between(0.1, 5)]
    flag_cols = ["has_sunroof", "has_leather_seats", "has_navigation",
                 "has_bluetooth", "has_rear_camera", "has_adaptive_cruise"]
    feature_labels = ["Sunroof", "Leather Seats", "Navigation",
                      "Bluetooth", "Rear Camera", "Adaptive Cruise"]
    premiums = []
    for col, label in zip(flag_cols, feature_labels):
        with_feat = df2[df2[col] == 1]["price_ratio"]
        without_feat = df2[df2[col] == 0]["price_ratio"]
        premiums.append({
            "Feature": label,
            "Premium": with_feat.mean() - without_feat.mean(),
            "SE": np.sqrt(with_feat.var()/len(with_feat) + without_feat.var()/len(without_feat))
        })
    prem_df = pd.DataFrame(premiums).sort_values("Premium", ascending=True)
    fig, ax = plt.subplots(figsize=(9, 5))
    colors = ["#2ca02c" if v > 0 else "#d62728" for v in prem_df["Premium"]]
    ax.barh(prem_df["Feature"], prem_df["Premium"] * 100, xerr=prem_df["SE"] * 100,
            color=colors, capsize=4)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_title("Figure 3.9. Within-Model Price Premium of Optional Features (%)",
                 fontweight="bold", fontsize=12)
    ax.set_xlabel("Premium vs. Same Model Without Feature (% of Model Median)")
    ax.set_ylabel("Optional Equipment")
    plt.tight_layout()
    return save_fig(fig, "fig_3_9_feature_premium_within_model.png")


# ─────────────────────────────────────────────────────────────────────────────
# FIG 3.10 – Interactive Treemap (Plotly) – Volume & Median Price
# ─────────────────────────────────────────────────────────────────────────────
def fig_3_10(df):
    agg = df.groupby(["make_tier", "make"]).agg(
        count=("price_aed", "count"),
        median_price=("price_aed", "median")
    ).reset_index()
    agg["make_title"] = agg["make"].str.title()
    fig = px.treemap(
        agg, path=["make_tier", "make_title"],
        values="count", color="median_price",
        color_continuous_scale="RdYlGn",
        title="Figure 3.10. UAE Used-Car Market – Volume (size) & Median Price (color) by Make",
        labels={"median_price": "Median Price (AED)", "count": "Listings"},
        color_continuous_midpoint=agg["median_price"].median()
    )
    fig.update_layout(font_size=11)
    html_path = FIGURES_DIR / "fig_3_10_treemap.html"
    fig.write_html(str(html_path))
    img_path = FIGURES_DIR / "fig_3_10_treemap.png"
    try:
        fig.write_image(str(img_path), width=1100, height=700)
        logger.info(f"Saved: {to_rel_path(img_path)}")
    except Exception as e:
        logger.warning(f"Kaleido PNG export failed ({e}); saving static fallback")
        # Fallback: static matplotlib treemap
        import matplotlib.pyplot as plt
        from matplotlib.patches import FancyBboxPatch
        fig2, ax = plt.subplots(figsize=(11, 7))
        tier_counts = agg.groupby("make_tier")["count"].sum().sort_values(ascending=False)
        colors_tier = list(TIER_COLORS.values())
        ax.bar(tier_counts.index, tier_counts.values, color=colors_tier[:len(tier_counts)])
        ax.set_title("Figure 3.10. Listings by Make Tier (Treemap Fallback)",
                     fontweight="bold", fontsize=12)
        ax.set_ylabel("Total Listings")
        plt.tight_layout()
        fig2.savefig(str(img_path), dpi=300, bbox_inches="tight")
        plt.close(fig2)
    logger.info(f"Saved: {to_rel_path(html_path)}")
    return img_path


# ─────────────────────────────────────────────────────────────────────────────
# FIG 3.11 – Interactive Scatter: Car Age vs Price Ratio (Plotly)
# ─────────────────────────────────────────────────────────────────────────────
def fig_3_11(df):
    model_med = df.groupby(["make", "model"])["price_aed"].transform("median")
    df2 = df.copy()
    df2["price_ratio"] = df2["price_aed"] / model_med.replace(0, np.nan)
    df2 = df2[df2["price_ratio"].between(0.05, 5)]
    sample = df2.sample(n=min(2000, len(df2)), random_state=42)
    sample["make_title"] = sample["make"].str.title()
    sample["model_title"] = sample["model"].str.title()

    fig = px.scatter(
        sample, x="car_age", y="price_ratio",
        color="make_tier", color_discrete_map=TIER_COLORS,
        hover_data={"make_title": True, "model_title": True,
                    "price_aed": ":,.0f", "car_age": True},
        title="Figure 3.11. Car Age vs. Price Ratio to Model Median",
        labels={"car_age": "Car Age (years)", "price_ratio": "Price / Model Median",
                "make_tier": "Make Tier"},
        opacity=0.5
    )
    fig.add_hline(y=1.0, line_dash="dash", line_color="gray",
                  annotation_text="Model Median = 1.0")
    html_path = FIGURES_DIR / "fig_3_11_scatter_age_vs_ratio.html"
    img_path = FIGURES_DIR / "fig_3_11_scatter_age_vs_ratio.png"
    fig.write_html(str(html_path))
    try:
        fig.write_image(str(img_path), width=1000, height=600)
    except Exception:
        # Static fallback
        fig2, ax = plt.subplots(figsize=(10, 6))
        for tier, grp in sample.groupby("make_tier"):
            ax.scatter(grp["car_age"], grp["price_ratio"],
                       alpha=0.3, s=12, label=tier,
                       color=TIER_COLORS.get(tier, "#999"))
        ax.axhline(1.0, color="gray", linestyle="--")
        ax.set_title("Figure 3.11. Car Age vs. Price Ratio to Model Median",
                     fontweight="bold", fontsize=12)
        ax.set_xlabel("Car Age (years)")
        ax.set_ylabel("Price / Model Median")
        ax.legend(title="Make Tier")
        plt.tight_layout()
        fig2.savefig(str(img_path), dpi=300, bbox_inches="tight")
        plt.close(fig2)
    logger.info(f"Saved: {to_rel_path(img_path)} + {to_rel_path(html_path)}")
    return img_path


# ─────────────────────────────────────────────────────────────────────────────
# FIG 3.12 – Median Price by Make Tier × Age Bucket (Heatmap)
# ─────────────────────────────────────────────────────────────────────────────
def fig_3_12(df):
    df2 = df.copy()
    bins = [0, 3, 6, 10, 15, 21]
    labels_b = ["0-3yr", "4-6yr", "7-10yr", "11-15yr", "16-20yr"]
    df2["age_bucket"] = pd.cut(df2["car_age"], bins=bins, labels=labels_b, right=True)
    pivot = df2.groupby(["make_tier", "age_bucket"])["price_aed"].median().unstack() / 1000
    tier_order = ["Budget", "Mid", "Premium", "Ultra-Luxury"]
    pivot = pivot.reindex([t for t in tier_order if t in pivot.index])
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.heatmap(pivot, annot=True, fmt=".0f", cmap="YlOrRd",
                linewidths=0.5, ax=ax, cbar_kws={"label": "Median Price (k AED)"})
    ax.set_title("Figure 3.12. Median Price (k AED) by Make Tier and Age Bracket",
                 fontweight="bold", fontsize=12)
    ax.set_xlabel("Vehicle Age Bracket")
    ax.set_ylabel("Make Tier")
    plt.tight_layout()
    return save_fig(fig, "fig_3_12_price_heatmap_tier_age.png")


# ─────────────────────────────────────────────────────────────────────────────
# ANALYSIS TABLES (for report)
# ─────────────────────────────────────────────────────────────────────────────
def compute_analysis_tables(df):
    results = {}

    # 3.3 Cross-tab: Body Type × Fuel Type
    ct_bt_fuel = pd.crosstab(df["body_type"], df["fuel_type"], margins=True)
    results["crosstab_body_fuel"] = ct_bt_fuel
    logger.info(f"Cross-tab Body×Fuel computed: {ct_bt_fuel.shape}")

    # 3.4 Multi-index groupby: Make Tier × Age Bucket
    bins = [0, 3, 6, 10, 15, 21]
    labels_b = ["0-3yr", "4-6yr", "7-10yr", "11-15yr", "16-20yr"]
    df["age_bucket"] = pd.cut(df["car_age"], bins=bins, labels=labels_b, right=True)
    tier_age = df.groupby(["make_tier", "age_bucket"]).agg(
        count=("price_aed", "count"),
        median_price_aed=("price_aed", "median"),
        median_mileage_km=("mileage_km", "median")
    ).round(0)
    results["tier_age_groupby"] = tier_age
    logger.info("Multi-index groupby Make Tier × Age Bucket computed")

    # 3.5 Spearman correlations
    num_cols = ["log_price", "car_age", "log_mileage", "cylinders", "n_features"]
    spearman_corr = df[num_cols].corr(method="spearman")
    results["spearman_corr"] = spearman_corr

    # Log key correlations
    rho_age = spearman_corr.loc["log_price", "car_age"]
    rho_mil = spearman_corr.loc["log_price", "log_mileage"]
    rho_cyl = spearman_corr.loc["log_price", "cylinders"]
    logger.info(f"Spearman ρ(log_price, car_age)={rho_age:.3f}, ρ(log_price, log_mileage)={rho_mil:.3f}, ρ(log_price, cylinders)={rho_cyl:.3f}")

    # 3.6 Within-model regression: price_ratio ~ car_age + log_mileage
    model_med = df.groupby(["make", "model"])["price_aed"].transform("median")
    df["price_ratio"] = df["price_aed"] / model_med.replace(0, np.nan)
    df_valid = df[df["price_ratio"].between(0.1, 5)].dropna(subset=["price_ratio", "car_age", "log_mileage"])
    slope_age, intercept_age, r_age, p_age, se_age = stats.linregress(df_valid["car_age"], df_valid["price_ratio"])
    slope_mil, intercept_mil, r_mil, p_mil, se_mil = stats.linregress(df_valid["log_mileage"], df_valid["price_ratio"])
    results["within_model_reg"] = {
        "car_age": {"slope": slope_age, "r": r_age, "p": p_age, "r2": r_age**2},
        "log_mileage": {"slope": slope_mil, "r": r_mil, "p": p_mil, "r2": r_mil**2}
    }
    logger.info(f"Within-model regression – car_age: slope={slope_age:.4f}, r²={r_age**2:.4f}, p={p_age:.4f}")
    logger.info(f"Within-model regression – log_mileage: slope={slope_mil:.4f}, r²={r_mil**2:.4f}, p={p_mil:.4f}")

    # 3.7 Kruskal-Wallis test for condition effect
    groups = [grp["price_ratio"].values for _, grp in df_valid.groupby("condition") if len(grp) > 5]
    kw_stat, kw_p = stats.kruskal(*groups)
    results["kruskal_wallis"] = {"stat": kw_stat, "p": kw_p}
    logger.info(f"Kruskal-Wallis condition test: H={kw_stat:.3f}, p={kw_p:.4f}")

    # Condition summary stats
    cond_stats = df_valid.groupby("condition")["price_ratio"].agg(["count", "mean", "median", "std"]).round(3)
    results["condition_stats"] = cond_stats
    logger.info(f"Condition stats:\n{cond_stats}")

    # Make tier summary
    tier_stats = df.groupby("make_tier").agg(
        count=("price_aed", "count"),
        median_price=("price_aed", "median"),
        mean_price=("price_aed", "mean")
    ).round(0)
    results["tier_stats"] = tier_stats
    logger.info(f"Make tier stats:\n{tier_stats}")

    return results


def run_phase3():
    logger.info("=" * 70)
    logger.info("STARTING PHASE 3: PYTHON ANALYSIS & VISUALIZATION")
    logger.info("=" * 70)

    df = load_data()

    # Generate all 12 figures
    logger.info("Generating figures...")
    paths = {}
    paths["fig_3_1"] = fig_3_1(df)
    paths["fig_3_2"] = fig_3_2(df)
    paths["fig_3_3"] = fig_3_3(df)
    paths["fig_3_4"] = fig_3_4(df)
    paths["fig_3_5"] = fig_3_5(df)
    paths["fig_3_6"] = fig_3_6(df)
    paths["fig_3_7"] = fig_3_7(df)
    paths["fig_3_8"] = fig_3_8(df)
    paths["fig_3_9"] = fig_3_9(df)
    paths["fig_3_10"] = fig_3_10(df)
    paths["fig_3_11"] = fig_3_11(df)
    paths["fig_3_12"] = fig_3_12(df)

    logger.info("Running analysis tables...")
    results = compute_analysis_tables(df)

    # Print summary for report
    logger.info("-" * 50)
    logger.info("CROSS-TAB Body Type × Fuel Type (top rows):")
    logger.info(str(results["crosstab_body_fuel"].head(5)))
    logger.info("\nMake Tier Summary:")
    logger.info(str(results["tier_stats"]))
    logger.info("\nWithin-Model Regression Summary:")
    for feat, vals in results["within_model_reg"].items():
        logger.info(f"  {feat}: slope={vals['slope']:.5f}, r²={vals['r2']:.4f}, p={vals['p']:.4f}")
    logger.info("\nKruskal-Wallis Condition Test:")
    logger.info(f"  H={results['kruskal_wallis']['stat']:.3f}, p={results['kruskal_wallis']['p']:.4f}")
    logger.info("=" * 70)
    logger.info("PHASE 3 COMPLETED SUCCESSFULLY.")
    logger.info("=" * 70)
    return paths, results


if __name__ == "__main__":
    run_phase3()
