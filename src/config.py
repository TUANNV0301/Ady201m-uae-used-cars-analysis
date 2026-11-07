"""
Project Configuration and Shared Constants
UAE Used-Car Market Analysis and Price Prediction (ADY201m)
"""

from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

# ==========================================
# 1. Project Directory Paths
# ==========================================
PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_PATH = DATA_DIR / "raw" / "uae_used_cars_10k.csv"
INTERIM_DATA_DIR = DATA_DIR / "interim"
CLEAN_LISTINGS_PATH = INTERIM_DATA_DIR / "clean_listings.csv"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODEL_FEATURES_PATH = PROCESSED_DATA_DIR / "model_features.csv"
DEAL_SCORES_PATH = PROCESSED_DATA_DIR / "deal_scores.csv"

DB_DIR = DATA_DIR / "db"
DB_PATH = DB_DIR / "uae_cars.sqlite"

SQL_DIR = PROJECT_ROOT / "sql"
MODELS_DIR = PROJECT_ROOT / "models"
MODEL_PIPELINE_PATH = MODELS_DIR / "price_model.joblib"
MODEL_METADATA_PATH = MODELS_DIR / "metadata.json"

FIGURES_DIR = PROJECT_ROOT / "figures"
SCREENSHOTS_DIR = PROJECT_ROOT / "screenshots"
REPORT_DIR = PROJECT_ROOT / "report"
LOGS_DIR = PROJECT_ROOT / "logs"

# Ensure all subdirectories exist
for d in [INTERIM_DATA_DIR, PROCESSED_DATA_DIR, DB_DIR, MODELS_DIR, FIGURES_DIR, SCREENSHOTS_DIR, REPORT_DIR, LOGS_DIR]:
    d.mkdir(parents=True, exist_ok=True)


def to_rel_path(path) -> str:
    """Returns a clean path relative to PROJECT_ROOT for logging and printing."""
    p = Path(path)
    try:
        if p.is_absolute():
            return str(p.relative_to(PROJECT_ROOT))
        return str(p)
    except ValueError:
        return str(p)


# ==========================================
# 2. Analytical Constants & Seeds
# ==========================================
RANDOM_SEED = 42
REFERENCE_YEAR = 2025  # Snapshot reference year (cars up to 2024; min age = 1)
AED_TO_VND = 7100.0    # Fixed conversion rate for presentation
FX_DATE = "2026-11-01" # Date rate was frozen

# Model and Outlier Fences
OUTLIER_CUTOFF_KM_PER_YEAR = 50_000 # Flag as implausible_usage
DEAL_THRESHOLD_LOG = 0.25           # Approx +28% overpriced / -22% undervalued


# ==========================================
# 3. Categorical Mappings & Features
# ==========================================
TARGET_FEATURES = [
    "Sunroof",
    "Leather seats",
    "Navigation",
    "Bluetooth",
    "Rear camera",
    "Adaptive cruise control"
]

FEATURE_FLAG_COLS = [
    "has_sunroof",
    "has_leather_seats",
    "has_navigation",
    "has_bluetooth",
    "has_rear_camera",
    "has_adaptive_cruise"
]

CONDITION_CATEGORIES = [
    "No damage",
    "Minor scratches",
    "Body repainted",
    "Accident history",
    "Engine repaired",
    "Gearbox issues"
]

REGION_MAPPING = {
    "Dubai": "Dubai",
    "Abu Dhabi": "Abu Dhabi",
    "Al Ain": "Abu Dhabi",
    "Sharjah": "Sharjah",
    "Ajman": "Other Northern Emirates",
    "Ras Al Khaimah": "Other Northern Emirates",
    "Umm Al Qawain": "Other Northern Emirates",
    "Fujeirah": "Other Northern Emirates"
}

MAKE_TIER_COLORS = {
    "Budget": "#4C72B0",
    "Mid": "#55A868",
    "Premium": "#E5A93B",
    "Ultra-Luxury": "#C44E52"
}


# ==========================================
# 4. Chart Aesthetics & Secondary Axis Helper
# ==========================================
def setup_chart_style():
    """Applies unified seaborn and matplotlib chart aesthetics."""
    sns.set_theme(style="whitegrid", palette="colorblind")
    plt.rcParams.update({
        "figure.dpi": 150,
        "axes.titlesize": 12,
        "axes.labelsize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "figure.autolayout": True
    })

def add_vnd_axis(ax, scale_unit="million"):
    """
    Adds a secondary y-axis on the right displaying values converted to Vietnamese Dong (VND).
    scale_unit: 'million' (1e6) or 'billion' (1e9)
    """
    divisor = 1e9 if scale_unit == "billion" else 1e6
    label_suffix = "billion VND" if scale_unit == "billion" else "million VND"
    
    to_vnd = lambda v: (v * AED_TO_VND) / divisor
    to_aed = lambda v: (v * divisor) / AED_TO_VND
    
    sec = ax.secondary_yaxis("right", functions=(to_vnd, to_aed))
    sec.set_ylabel(f"Price ({label_suffix})")
    return sec
