"""
Phase 4: Predictive Modeling and Valuation Engine
UAE Used-Car Market Analysis (ADY201m)

Executes:
1. 80/20 Stratified Train/Test Split (on deciles of log_price, seed 42)
2. Feature preprocessing pipeline (ColumnTransformer)
3. Model training: Baseline, OLS (HC3 robust SE), Random Forest, Gradient Boosting (GBDT)
4. 5-Fold Cross-Validation on training set
5. Test-set evaluation (Log-scale metrics, Duan smearing factor, AED metrics, MAPE by price band)
6. VIF multicollinearity assessment & Sensitivity analysis (Table 4.5)
7. Permutation feature importance & Deal flagging engine (DEAL_THRESHOLD_LOG = 0.25)
8. Generation of Figures 4.1 to 4.5 (publication-grade PNGs in figures/)
9. Exports models/price_model.joblib, models/metadata.json, data/processed/deal_scores.csv, logs/phase4_modeling.log
"""

import sys
import json
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import joblib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.dummy import DummyRegressor
from sklearn.model_selection import StratifiedKFold, cross_validate, cross_val_predict
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.inspection import permutation_importance, PartialDependenceDisplay
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import (
    MODEL_FEATURES_PATH, DEAL_SCORES_PATH,
    MODEL_PIPELINE_PATH, MODEL_METADATA_PATH,
    FIGURES_DIR, RANDOM_SEED, REFERENCE_YEAR,
    AED_TO_VND, DEAL_THRESHOLD_LOG, to_rel_path,
    setup_chart_style
)
from src.logger import get_logger

logger = get_logger("phase4", "phase4_modeling.log")
setup_chart_style()


def load_and_preprocess_data():
    df = pd.read_csv(MODEL_FEATURES_PATH)
    logger.info(f"Loaded feature dataset: {df.shape[0]} rows, {df.shape[1]} columns")

    # Group low-frequency makes: Top 20 makes by volume, rest as 'other'
    top20_makes = df["make"].value_counts().head(20).index.tolist()
    df["make_grouped"] = df["make"].apply(lambda m: m if m in top20_makes else "other")

    # Standardize category values
    df["fuel_type"] = df["fuel_type"].fillna("Gasoline")
    df["transmission"] = df["transmission"].fillna("Automatic Transmission")
    df["region_group"] = df["region_group"].fillna("Dubai")
    df["condition"] = df["condition"].fillna("No damage")
    df["body_type"] = df["body_type"].fillna("SUV")

    # Cast numeric and binary features to float64
    for c in ["car_age", "log_mileage", "cylinders", "n_features",
              "has_sunroof", "has_leather_seats", "has_navigation",
              "has_bluetooth", "has_rear_camera", "has_adaptive_cruise"]:
        df[c] = df[c].astype(float)

    return df, top20_makes


def get_stratified_split(df):
    bins = pd.qcut(df["log_price"], q=10, labels=False, duplicates="drop")
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
    train_idx, test_idx = next(skf.split(df, bins))
    train_df = df.iloc[train_idx].copy().reset_index(drop=True)
    test_df = df.iloc[test_idx].copy().reset_index(drop=True)
    logger.info(f"Train set: {len(train_df):,} rows | Test set: {len(test_df):,} rows")
    return train_df, test_df


def compute_vif(X_df):
    """Computes Variance Inflation Factor for numeric predictors."""
    vif_data = []
    numeric_cols = [c for c in X_df.columns if pd.api.types.is_numeric_dtype(X_df[c])]
    X_num = X_df[numeric_cols].dropna()
    X_const = sm.add_constant(X_num)
    for i, col in enumerate(X_const.columns):
        if col == "const":
            continue
        vif_val = variance_inflation_factor(X_const.values, i)
        vif_data.append({"Feature": col, "VIF": round(vif_val, 2)})
    return pd.DataFrame(vif_data).sort_values("VIF", ascending=False)


def run_ols_statsmodels(train_df, test_df, feature_cols, cat_cols, num_cols):
    """
    Fits OLS with HC3 robust standard errors via statsmodels.
    Reference categories:
    - make_grouped: 'toyota'
    - region_group: 'Dubai'
    - condition: 'No damage'
    - transmission: 'Automatic Transmission'
    - fuel_type: 'Gasoline'
    - body_type: 'SUV'
    """
    train_encoded = pd.get_dummies(train_df[feature_cols], columns=cat_cols, drop_first=False, dtype=float)
    test_encoded = pd.get_dummies(test_df[feature_cols], columns=cat_cols, drop_first=False, dtype=float)

    # Reference columns to drop for dummy encoding
    refs_to_drop = [
        "make_grouped_toyota",
        "region_group_Dubai",
        "condition_No damage",
        "transmission_Automatic Transmission",
        "fuel_type_Gasoline",
        "body_type_SUV"
    ]
    cols_to_use = [c for c in train_encoded.columns if c not in refs_to_drop]
    
    # Align test columns
    test_encoded = test_encoded.reindex(columns=cols_to_use, fill_value=0.0)

    X_train_c = sm.add_constant(train_encoded[cols_to_use])
    y_train = train_df["log_price"]
    X_test_c = sm.add_constant(test_encoded[cols_to_use])
    y_test = test_df["log_price"]

    model_ols = sm.OLS(y_train, X_train_c).fit(cov_type="HC3")
    logger.info(f"OLS fitted: R²={model_ols.rsquared:.4f}, Adj R²={model_ols.rsquared_adj:.4f}, AIC={model_ols.aic:.1f}")

    # Summary DataFrame
    coef_df = pd.DataFrame({
        "Feature": model_ols.params.index,
        "Coefficient": model_ols.params.values,
        "Std_Error": model_ols.bse.values,
        "t_stat": model_ols.tvalues.values,
        "p_value": model_ols.pvalues.values,
        "ci_lower": model_ols.conf_int()[0].values,
        "ci_upper": model_ols.conf_int()[1].values,
    })
    # Convert log-coefficients to percentage change: (exp(beta) - 1) * 100
    coef_df["pct_effect"] = (np.exp(coef_df["Coefficient"]) - 1.0) * 100.0

    # Predictions
    pred_train_log = model_ols.predict(X_train_c)
    pred_test_log = model_ols.predict(X_test_c)

    return model_ols, coef_df, pred_train_log, pred_test_log, cols_to_use


def evaluate_predictions(y_true_log, y_pred_log, actual_price_aed, smearing_factor=1.0):
    r2 = r2_score(y_true_log, y_pred_log)
    mse_log = mean_squared_error(y_true_log, y_pred_log)
    rmse_log = np.sqrt(mse_log)

    pred_aed = np.exp(y_pred_log) * smearing_factor
    mae_aed = mean_absolute_error(actual_price_aed, pred_aed)
    rmse_aed = np.sqrt(mean_squared_error(actual_price_aed, pred_aed))

    ape = np.abs((actual_price_aed - pred_aed) / actual_price_aed) * 100.0
    overall_mape = float(np.median(ape))

    b1_mask = actual_price_aed < 100_000
    b2_mask = (actual_price_aed >= 100_000) & (actual_price_aed <= 500_000)
    b3_mask = actual_price_aed > 500_000

    mape_under_100k = float(np.median(ape[b1_mask])) if np.any(b1_mask) else 0.0
    mape_100k_500k = float(np.median(ape[b2_mask])) if np.any(b2_mask) else 0.0
    mape_over_500k = float(np.median(ape[b3_mask])) if np.any(b3_mask) else 0.0

    return {
        "r2_log": float(round(r2, 4)),
        "rmse_log": float(round(rmse_log, 4)),
        "mse_log": float(round(mse_log, 4)),
        "mae_aed": float(round(mae_aed, 2)),
        "rmse_aed": float(round(rmse_aed, 2)),
        "median_ape_pct": float(round(overall_mape, 2)),
        "mape_under_100k": float(round(mape_under_100k, 2)),
        "mape_100k_500k": float(round(mape_100k_500k, 2)),
        "mape_over_500k": float(round(mape_over_500k, 2))
    }


def save_fig(fig, fname):
    path = FIGURES_DIR / fname
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved figure: {to_rel_path(path)}")
    return path


def generate_figures(test_df, y_test, pred_test_log, smearing_factor, gb_model, preprocessor, feature_cols):
    actual_aed = test_df["price_aed"].values
    pred_aed = np.exp(pred_test_log) * smearing_factor
    residuals_log = y_test - pred_test_log

    # ─────────────────────────────────────────────────────────────────────────
    # Fig 4.1: Predicted vs Actual (Dual Log and AED)
    # ─────────────────────────────────────────────────────────────────────────
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Log scale
    ax1.scatter(y_test, pred_test_log, alpha=0.3, s=15, color="#1f77b4")
    min_val = min(y_test.min(), pred_test_log.min())
    max_val = max(y_test.max(), pred_test_log.max())
    ax1.plot([min_val, max_val], [min_val, max_val], "r--", linewidth=1.5, label="Parity (y = x)")
    ax1.set_title("Figure 4.1a. Predicted vs. Actual Log Price (Test Set)", fontweight="bold")
    ax1.set_xlabel("Actual Log Price (ln AED)")
    ax1.set_ylabel("Predicted Log Price (ln AED)")
    ax1.legend()

    # Original AED scale (capped at 1.5M for visual clarity)
    mask = (actual_aed <= 1_500_000) & (pred_aed <= 1_500_000)
    ax2.scatter(actual_aed[mask] / 1000, pred_aed[mask] / 1000, alpha=0.3, s=15, color="#2ca02c")
    ax2.plot([0, 1500], [0, 1500], "r--", linewidth=1.5, label="Parity (y = x)")
    ax2.set_title("Figure 4.1b. Predicted vs. Actual Asking Price (AED)", fontweight="bold")
    ax2.set_xlabel("Actual Price (Thousands AED)")
    ax2.set_ylabel("Predicted Price (Thousands AED)")
    ax2.legend()
    plt.tight_layout()
    save_fig(fig, "fig_4_1_pred_vs_actual.png")

    # ─────────────────────────────────────────────────────────────────────────
    # Fig 4.2: Residuals vs Fitted
    # ─────────────────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.scatter(pred_test_log, residuals_log, alpha=0.3, s=15, color="#1f77b4")
    ax.axhline(0, color="red", linestyle="--", linewidth=1.2)
    ax.axhline(DEAL_THRESHOLD_LOG, color="orange", linestyle=":", label=f"Overpriced Cutoff (+{DEAL_THRESHOLD_LOG})")
    ax.axhline(-DEAL_THRESHOLD_LOG, color="green", linestyle=":", label=f"Undervalued Cutoff (-{DEAL_THRESHOLD_LOG})")
    ax.set_title("Figure 4.2. Residuals vs. Fitted Log Price (Homoscedasticity Assessment)", fontweight="bold")
    ax.set_xlabel("Fitted Log Price (ln AED)")
    ax.set_ylabel("Residual (Actual - Fitted ln AED)")
    ax.legend(loc="upper right")
    plt.tight_layout()
    save_fig(fig, "fig_4_2_residuals_vs_fitted.png")

    # ─────────────────────────────────────────────────────────────────────────
    # Fig 4.3: Residual Distribution and Q-Q Plot
    # ─────────────────────────────────────────────────────────────────────────
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    
    # Histogram + KDE
    sns.histplot(residuals_log, kde=True, ax=ax1, color="#1f77b4", bins=40, stat="density")
    norm_x = np.linspace(residuals_log.min(), residuals_log.max(), 100)
    norm_y = stats.norm.pdf(norm_x, loc=residuals_log.mean(), scale=residuals_log.std())
    ax1.plot(norm_x, norm_y, "r--", label="Normal Curve")
    ax1.set_title("Figure 4.3a. Residual Error Distribution", fontweight="bold")
    ax1.set_xlabel("Residual (ln AED)")
    ax1.set_ylabel("Density")
    ax1.legend()

    # Normal Q-Q Plot
    stats.probplot(residuals_log, dist="norm", plot=ax2)
    ax2.get_lines()[0].set_markersize(3.0)
    ax2.get_lines()[0].set_color("#1f77b4")
    ax2.get_lines()[1].set_color("red")
    ax2.set_title("Figure 4.3b. Normal Q-Q Plot of Residuals", fontweight="bold")
    plt.tight_layout()
    save_fig(fig, "fig_4_3_residual_qq_distribution.png")

    # ─────────────────────────────────────────────────────────────────────────
    # Fig 4.4: Permutation Feature Importance Bar Chart
    # ─────────────────────────────────────────────────────────────────────────
    pipe_for_pi = Pipeline([
        ("preprocessor", preprocessor),
        ("regressor", gb_model)
    ])
    perm_res = permutation_importance(pipe_for_pi, test_df[feature_cols], y_test, n_repeats=5, random_state=RANDOM_SEED, n_jobs=-1)
    
    # Process importance scores
    importances = pd.DataFrame({
        "Feature": feature_cols,
        "Importance": perm_res.importances_mean,
        "Std": perm_res.importances_std
    }).sort_values("Importance", ascending=True)

    fig, ax = plt.subplots(figsize=(10, 6))
    top_imp = importances.tail(12)
    ax.barh(top_imp["Feature"], top_imp["Importance"], xerr=top_imp["Std"], color="#1f77b4", capsize=3)
    ax.set_title("Figure 4.4. Permutation Feature Importance (GBDT Model, Test Set)", fontweight="bold")
    ax.set_xlabel("Drop in R² When Feature is Shuffled")
    ax.set_ylabel("Predictor Variable")
    plt.tight_layout()
    save_fig(fig, "fig_4_4_permutation_importance.png")

    # ─────────────────────────────────────────────────────────────────────────
    # Fig 4.5: Partial Dependence Plots for Key Drivers
    # ─────────────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    target_pdp = ["cylinders", "car_age", "log_mileage"]
    disp = PartialDependenceDisplay.from_estimator(
        pipe_for_pi, test_df[feature_cols], target_pdp,
        ax=axes, line_kw={"color": "#1f77b4", "linewidth": 2}
    )
    fig.suptitle("Figure 4.5. Partial Dependence of Asking Price on Primary Continuous Drivers", fontweight="bold", y=1.03)
    for ax in axes:
        ax.set_ylabel("Partial Dependence (ln Price)")
    plt.tight_layout()
    save_fig(fig, "fig_4_5_partial_dependence.png")

    return importances


def main():
    logger.info("=" * 70)
    logger.info("STARTING PHASE 4: PREDICTIVE MODELING & VALUATION ENGINE")
    logger.info("=" * 70)

    # 1. Load Data
    df, top20_makes = load_and_preprocess_data()
    train_df, test_df = get_stratified_split(df)

    # Define feature lists
    num_cols = ["car_age", "log_mileage", "cylinders", "n_features"]
    bin_cols = [
        "has_sunroof", "has_leather_seats", "has_navigation",
        "has_bluetooth", "has_rear_camera", "has_adaptive_cruise"
    ]
    cat_cols = ["make_grouped", "body_type", "transmission", "fuel_type", "region_group", "condition"]
    feature_cols = num_cols + bin_cols + cat_cols

    # VIF Check
    logger.info("Computing Variance Inflation Factors (VIF) on numeric predictors...")
    vif_df = compute_vif(train_df[num_cols])
    logger.info(f"VIF Results:\n{vif_df.to_string(index=False)}")

    # 2. Fit OLS with HC3 Standard Errors
    logger.info("Fitting Ordinary Least Squares (OLS) with HC3 robust covariance...")
    model_ols, coef_df, pred_ols_tr, pred_ols_te, encoded_cols = run_ols_statsmodels(
        train_df, test_df, feature_cols, cat_cols, num_cols
    )

    # Smearing factor for OLS
    residuals_train_ols = train_df["log_price"].values - pred_ols_tr
    smearing_ols = float(np.mean(np.exp(residuals_train_ols)))
    logger.info(f"Duan Smearing Factor (OLS): {smearing_ols:.4f}")

    # 3. Fit Machine Learning Models (scikit-learn Pipelines)
    logger.info("Building scikit-learn Preprocessing and Machine Learning Pipelines...")
    preprocessor = ColumnTransformer([
        ("num", StandardScaler(), num_cols),
        ("bin", "passthrough", bin_cols),
        ("cat", OneHotEncoder(drop="first", sparse_output=False, handle_unknown="ignore"), cat_cols)
    ])

    models = {
        "Baseline (Mean)": DummyRegressor(strategy="mean"),
        "OLS (Linear)": Pipeline([("prep", preprocessor), ("reg", DummyRegressor())]), # will be evaluated via statsmodels
        "Random Forest": RandomForestRegressor(n_estimators=100, max_depth=15, min_samples_split=10, random_state=RANDOM_SEED, n_jobs=-1),
        "Gradient Boosting (GBDT)": GradientBoostingRegressor(n_estimators=150, max_depth=5, learning_rate=0.08, subsample=0.8, random_state=RANDOM_SEED)
    }

    # 5-Fold Cross Validation on Train Set
    logger.info("Executing 5-Fold Cross-Validation on 8,000 training records...")
    cv_skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
    bins_train = pd.qcut(train_df["log_price"], q=10, labels=False, duplicates="drop")
    
    cv_results = {}
    
    # Baseline CV
    base_m = DummyRegressor(strategy="mean")
    cv_base = cross_validate(base_m, train_df[feature_cols], train_df["log_price"], cv=cv_skf.split(train_df, bins_train),
                             scoring=["r2", "neg_root_mean_squared_error", "neg_mean_absolute_error"])
    cv_results["Baseline (Mean)"] = {
        "cv_r2": float(round(np.mean(cv_base["test_r2"]), 4)),
        "cv_rmse": float(round(-np.mean(cv_base["test_neg_root_mean_squared_error"]), 4)),
        "cv_mae": float(round(-np.mean(cv_base["test_neg_mean_absolute_error"]), 4))
    }

    # OLS CV
    from sklearn.linear_model import LinearRegression
    ols_pipe = Pipeline([("prep", preprocessor), ("reg", LinearRegression())])
    cv_ols = cross_validate(ols_pipe, train_df[feature_cols], train_df["log_price"], cv=cv_skf.split(train_df, bins_train),
                            scoring=["r2", "neg_root_mean_squared_error", "neg_mean_absolute_error"])
    cv_results["OLS (Linear)"] = {
        "cv_r2": float(round(np.mean(cv_ols["test_r2"]), 4)),
        "cv_rmse": float(round(-np.mean(cv_ols["test_neg_root_mean_squared_error"]), 4)),
        "cv_mae": float(round(-np.mean(cv_ols["test_neg_mean_absolute_error"]), 4))
    }

    # RF CV
    rf_pipe = Pipeline([("prep", preprocessor), ("reg", models["Random Forest"])])
    cv_rf = cross_validate(rf_pipe, train_df[feature_cols], train_df["log_price"], cv=cv_skf.split(train_df, bins_train),
                           scoring=["r2", "neg_root_mean_squared_error", "neg_mean_absolute_error"])
    cv_results["Random Forest"] = {
        "cv_r2": float(round(np.mean(cv_rf["test_r2"]), 4)),
        "cv_rmse": float(round(-np.mean(cv_rf["test_neg_root_mean_squared_error"]), 4)),
        "cv_mae": float(round(-np.mean(cv_rf["test_neg_mean_absolute_error"]), 4))
    }

    # GBDT CV
    gb_pipe = Pipeline([("prep", preprocessor), ("reg", models["Gradient Boosting (GBDT)"])])
    cv_gb = cross_validate(gb_pipe, train_df[feature_cols], train_df["log_price"], cv=cv_skf.split(train_df, bins_train),
                           scoring=["r2", "neg_root_mean_squared_error", "neg_mean_absolute_error"])
    cv_results["Gradient Boosting (GBDT)"] = {
        "cv_r2": float(round(np.mean(cv_gb["test_r2"]), 4)),
        "cv_rmse": float(round(-np.mean(cv_gb["test_neg_root_mean_squared_error"]), 4)),
        "cv_mae": float(round(-np.mean(cv_gb["test_neg_mean_absolute_error"]), 4))
    }

    logger.info("5-Fold CV Performance Comparison:")
    for mname, mres in cv_results.items():
        logger.info(f"  {mname:25s}: CV R²={mres['cv_r2']:.4f}, CV RMSE={mres['cv_rmse']:.4f}, CV MAE={mres['cv_mae']:.4f}")

    # 4. Train final models on full train_df and score on 2,000 test_df
    logger.info("Training production GBDT pipeline on full training split...")
    gb_pipe.fit(train_df[feature_cols], train_df["log_price"])
    pred_gb_tr = gb_pipe.predict(train_df[feature_cols])
    pred_gb_te = gb_pipe.predict(test_df[feature_cols])

    rf_pipe.fit(train_df[feature_cols], train_df["log_price"])
    pred_rf_te = rf_pipe.predict(test_df[feature_cols])

    base_m.fit(train_df[feature_cols], train_df["log_price"])
    pred_base_te = base_m.predict(test_df[feature_cols])

    # Compute smearing factors
    smearing_gb = float(np.mean(np.exp(train_df["log_price"] - pred_gb_tr)))
    smearing_rf = float(np.mean(np.exp(train_df["log_price"] - rf_pipe.predict(train_df[feature_cols]))))
    smearing_base = float(np.mean(np.exp(train_df["log_price"] - base_m.predict(train_df[feature_cols]))))

    # Test Evaluations
    y_test = test_df["log_price"].values
    actual_aed = test_df["price_aed"].values

    test_metrics = {
        "Baseline (Mean)": evaluate_predictions(y_test, pred_base_te, actual_aed, smearing_base),
        "OLS (Linear)": evaluate_predictions(y_test, pred_ols_te, actual_aed, smearing_ols),
        "Random Forest": evaluate_predictions(y_test, pred_rf_te, actual_aed, smearing_rf),
        "Gradient Boosting (GBDT)": evaluate_predictions(y_test, pred_gb_te, actual_aed, smearing_gb)
    }

    logger.info("Test-Set Evaluation Results (2,000 records):")
    for mname, mets in test_metrics.items():
        logger.info(f"  {mname:25s}: R²={mets['r2_log']:.4f}, RMSE={mets['rmse_log']:.4f}, MedAPE={mets['median_ape_pct']:.1f}%, MAE={mets['mae_aed']:,.0f} AED")

    # 5. Sensitivity Analysis (Table 4.5)
    logger.info("Conducting Sensitivity Analysis on Test Set...")
    # Trim extreme residuals (> 3 sigma on log scale)
    res_gb_te = y_test - pred_gb_te
    std_res = np.std(res_gb_te)
    trim_mask = np.abs(res_gb_te) <= 3.0 * std_res
    sens_trimmed = evaluate_predictions(y_test[trim_mask], pred_gb_te[trim_mask], actual_aed[trim_mask], smearing_gb)

    # Remove implausible usage rows
    valid_usage_mask = (test_df["implausible_usage"] == 0).values
    sens_usage = evaluate_predictions(y_test[valid_usage_mask], pred_gb_te[valid_usage_mask], actual_aed[valid_usage_mask], smearing_gb)

    sensitivity_table = pd.DataFrame([
        {"Sample": "Full Test Set (N=2,000)", "R2_Log": test_metrics["Gradient Boosting (GBDT)"]["r2_log"],
         "RMSE_Log": test_metrics["Gradient Boosting (GBDT)"]["rmse_log"], "Median_APE": test_metrics["Gradient Boosting (GBDT)"]["median_ape_pct"],
         "MAE_AED": test_metrics["Gradient Boosting (GBDT)"]["mae_aed"]},
        {"Sample": "Trimmed Outliers (|e| <= 3σ) (N=" + str(np.sum(trim_mask)) + ")", "R2_Log": sens_trimmed["r2_log"],
         "RMSE_Log": sens_trimmed["rmse_log"], "Median_APE": sens_trimmed["median_ape_pct"],
         "MAE_AED": sens_trimmed["mae_aed"]},
        {"Sample": "Plausible Usage Only (N=" + str(np.sum(valid_usage_mask)) + ")", "R2_Log": sens_usage["r2_log"],
         "RMSE_Log": sens_usage["rmse_log"], "Median_APE": sens_usage["median_ape_pct"],
         "MAE_AED": sens_usage["mae_aed"]}
    ])
    logger.info(f"Sensitivity Analysis Table:\n{sensitivity_table.to_string(index=False)}")

    # 6. Figures Generation
    logger.info("Generating Phase 4 Publication Figures (Fig 4.1 to 4.5)...")
    perm_importances = generate_figures(
        test_df, y_test, pred_gb_te, smearing_gb, models["Gradient Boosting (GBDT)"], preprocessor, feature_cols
    )

    # 7. Deal Flagging Engine & Export deal_scores.csv
    logger.info("Generating Deal Flagging Engine and Saving deal_scores.csv...")
    df_all = df.copy()
    full_pred_log = gb_pipe.predict(df_all[feature_cols])
    df_all["pred_log_price"] = full_pred_log
    df_all["predicted_price_aed"] = (np.exp(full_pred_log) * smearing_gb).round(0)
    df_all["residual_log"] = df_all["log_price"] - full_pred_log
    df_all["pct_gap"] = ((df_all["price_aed"] - df_all["predicted_price_aed"]) / df_all["predicted_price_aed"] * 100.0).round(1)

    # Deal Flags
    conditions = [
        df_all["residual_log"] > DEAL_THRESHOLD_LOG,
        df_all["residual_log"] < -DEAL_THRESHOLD_LOG
    ]
    choices = ["Overpriced", "Undervalued"]
    df_all["deal_flag"] = np.select(conditions, choices, default="Fair Price")

    flag_counts = df_all["deal_flag"].value_counts()
    flag_pcts = (flag_counts / len(df_all) * 100.0).round(2)
    deal_summary = pd.DataFrame({"Count": flag_counts, "Share_Pct": flag_pcts})
    logger.info(f"Deal Flagging Summary across 10,000 listings:\n{deal_summary.to_string()}")

    # Export deal scores with columns compatible with Streamlit Deal Finder
    df_all["car_id"] = np.arange(1, len(df_all) + 1)
    df_all["make_name"] = df_all["make"]
    df_all["model_name"] = df_all["model"]
    df_all["emirate_name"] = df_all["location"]
    df_all["pred_price_aed"] = df_all["predicted_price_aed"]
    cols_deal = [
        "car_id", "make_name", "model_name", "make", "model", "model_year", 
        "price_aed", "pred_price_aed", "predicted_price_aed",
        "residual_log", "pct_gap", "deal_flag", "mileage_km", "body_type",
        "emirate_name", "location", "condition"
    ]
    df_all[cols_deal].to_csv(DEAL_SCORES_PATH, index=False)
    logger.info(f"Exported deal scores to: {to_rel_path(DEAL_SCORES_PATH)}")

    # 8. Save Model Pipeline and Metadata
    logger.info(f"Saving production model pipeline to: {to_rel_path(MODEL_PIPELINE_PATH)}")
    joblib.dump(gb_pipe, MODEL_PIPELINE_PATH)

    metadata = {
        "model_name": "Gradient Boosting Regressor (GBDT)",
        "features": feature_cols,
        "n_train": len(train_df),
        "n_test": len(test_df),
        "smearing_factor": round(smearing_gb, 4),
        "test_r2_log": test_metrics["Gradient Boosting (GBDT)"]["r2_log"],
        "test_rmse_log": test_metrics["Gradient Boosting (GBDT)"]["rmse_log"],
        "test_mae_aed": test_metrics["Gradient Boosting (GBDT)"]["mae_aed"],
        "test_median_ape_pct": test_metrics["Gradient Boosting (GBDT)"]["median_ape_pct"],
        "cv_results": cv_results,
        "test_metrics": test_metrics,
        "deal_flags": deal_summary.to_dict(),
        "ols_summary": {
            "r2": round(model_ols.rsquared, 4),
            "adj_r2": round(model_ols.rsquared_adj, 4),
            "aic": round(model_ols.aic, 1),
            "f_stat": round(model_ols.fvalue, 1) if model_ols.fvalue is not None else None,
            "f_pvalue": float(model_ols.f_pvalue) if model_ols.f_pvalue is not None else None
        }
    }

    with open(MODEL_METADATA_PATH, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    logger.info(f"Saved model metadata to: {to_rel_path(MODEL_METADATA_PATH)}")

    logger.info("=" * 70)
    logger.info("PHASE 4 COMPLETED SUCCESSFULLY.")
    logger.info("=" * 70)

    return {
        "model_ols": model_ols,
        "coef_df": coef_df,
        "vif_df": vif_df,
        "cv_results": cv_results,
        "test_metrics": test_metrics,
        "sensitivity_table": sensitivity_table,
        "deal_summary": deal_summary,
        "perm_importances": perm_importances
    }


if __name__ == "__main__":
    main()
