"""
Predictive Modeling and Valuation Engine
UAE Used-Car Market Analysis (ADY201m - Phase 4)
"""

import json
from typing import Dict, Tuple, Any, Optional
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

try:
    import statsmodels.api as sm
except ImportError:
    sm = None

from src.config import (
    RANDOM_SEED,
    DEAL_THRESHOLD_LOG,
    MODEL_PIPELINE_PATH,
    MODEL_METADATA_PATH
)


def create_stratified_split(
    df: pd.DataFrame, 
    target_col: str = "log_price", 
    n_splits: int = 5, 
    random_state: int = RANDOM_SEED
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Creates an 80/20 train/test split stratified on deciles of log_price.
    Guarantees balanced price distributions between train and test sets.
    """
    bins = pd.qcut(df[target_col], q=10, labels=False, duplicates="drop")
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    
    train_idx, test_idx = next(skf.split(df, bins))
    train_df = df.iloc[train_idx].copy().reset_index(drop=True)
    test_df = df.iloc[test_idx].copy().reset_index(drop=True)
    return train_df, test_df


def compute_smearing_factor(y_true_log: np.ndarray, y_pred_log: np.ndarray) -> float:
    """
    Duan's Smearing Factor to correct re-transformation bias from log to linear scale:
    smearing = mean(exp(residuals))
    """
    residuals = y_true_log - y_pred_log
    return float(np.mean(np.exp(residuals)))


def evaluate_predictions(
    y_true_log: np.ndarray,
    y_pred_log: np.ndarray,
    actual_price_aed: np.ndarray,
    smearing_factor: float = 1.0
) -> Dict[str, Any]:
    """
    Calculates comprehensive regression performance metrics on both
    log scale (R2, RMSE, MSE) and original AED scale (MAE, RMSE, MAPE by price band).
    """
    r2 = r2_score(y_true_log, y_pred_log)
    mse_log = mean_squared_error(y_true_log, y_pred_log)
    rmse_log = np.sqrt(mse_log)
    
    # Back-transformed predictions in AED
    pred_aed = np.exp(y_pred_log) * smearing_factor
    mae_aed = mean_absolute_error(actual_price_aed, pred_aed)
    rmse_aed = np.sqrt(mean_squared_error(actual_price_aed, pred_aed))
    
    # MAPE calculations by segment
    ape = np.abs((actual_price_aed - pred_aed) / actual_price_aed) * 100
    overall_mape = float(np.median(ape)) # Median APE resists extreme supercar outliers
    
    b1_mask = actual_price_aed < 100_000
    b2_mask = (actual_price_aed >= 100_000) & (actual_price_aed <= 500_000)
    b3_mask = actual_price_aed > 500_000
    
    mape_under_100k = float(np.median(ape[b1_mask])) if np.any(b1_mask) else 0.0
    mape_100k_500k = float(np.median(ape[b2_mask])) if np.any(b2_mask) else 0.0
    mape_over_500k = float(np.median(ape[b3_mask])) if np.any(b3_mask) else 0.0

    return {
        "r2_log": float(r2),
        "rmse_log": float(rmse_log),
        "mse_log": float(mse_log),
        "mae_aed": float(mae_aed),
        "rmse_aed": float(rmse_aed),
        "median_ape_pct": float(overall_mape),
        "mape_under_100k": float(mape_under_100k),
        "mape_100k_500k": float(mape_100k_500k),
        "mape_over_500k": float(mape_over_500k)
    }


def fit_ols_statsmodels(
    X_train: pd.DataFrame, 
    y_train: pd.Series
) -> Tuple[Any, pd.DataFrame]:
    """
    Fits Ordinary Least Squares (OLS) regression with heteroscedasticity-robust (HC3) standard errors.
    Returns: (fitted_model, coefficient_summary_dataframe)
    """
    if sm is None:
        raise ImportError("statsmodels is required for OLS regression. Install with: pip install statsmodels")
    X_with_const = sm.add_constant(X_train)
    model = sm.OLS(y_train, X_with_const).fit(cov_type="HC3")
    
    coef_df = pd.DataFrame({
        "feature": model.params.index,
        "coef": model.params.values,
        "std_err": model.bse.values,
        "p_value": model.pvalues.values,
        "ci_lower": model.conf_int()[0].values,
        "ci_upper": model.conf_int()[1].values
    })
    # Convert log coefficients to percentage effects: (exp(beta) - 1) * 100
    coef_df["pct_effect"] = (np.exp(coef_df["coef"]) - 1.0) * 100.0
    return model, coef_df


def assign_deal_flags(
    df: pd.DataFrame,
    y_true_log_col: str = "log_price",
    y_pred_log_col: str = "pred_log_price",
    threshold: float = DEAL_THRESHOLD_LOG
) -> pd.DataFrame:
    """
    Flags listings based on residual = actual - predicted:
    - residual > +threshold (+28% over fair market value): 'Overpriced'
    - residual < -threshold (-22% under fair market value): 'Undervalued'
    - otherwise: 'Fair Price'
    """
    df = df.copy()
    residual = df[y_true_log_col] - df[y_pred_log_col]
    df["log_residual"] = residual
    df["pct_gap"] = (np.exp(residual) - 1.0) * 100.0
    
    conditions = [
        residual > threshold,
        residual < -threshold
    ]
    choices = ["Overpriced", "Undervalued"]
    df["deal_flag"] = np.select(conditions, choices, default="Fair Price")
    return df
