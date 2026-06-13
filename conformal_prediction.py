import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler

def conformal_prediction_score(returns, macro_df, confidence=0.90, n_estimators=50, max_depth=10):
    """
    Compute the width of the conformal prediction interval for the next‑day return.
    Non‑conformity score is weighted by macro volatility (composite factor from all macros).
    """
    if len(returns) < 20 or macro_df is None or len(macro_df) < 20:
        return 0.0
    # Align lengths
    min_len = min(len(returns), len(macro_df))
    returns = returns[:min_len]
    macro_df = macro_df.iloc[:min_len]
    # Remove NaN
    mask = ~(np.isnan(returns) | np.isnan(macro_df).any(axis=1))
    returns = returns[mask]
    macro_df = macro_df[mask]
    if len(returns) < 20:
        return 0.0
    # Standardise macro
    scaler = StandardScaler()
    macro_scaled = scaler.fit_transform(macro_df)
    # Split into training (past) and calibration (recent)
    split = int(len(returns) * 0.8)
    X_train = macro_scaled[:split]
    y_train = returns[:split]
    X_cal = macro_scaled[split:]
    y_cal = returns[split:]
    if len(y_cal) < 5:
        return 0.0
    # Train a quantile regression forest (using random forest with quantile loss? simpler: use standard RF and then compute residuals)
    rf = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
    rf.fit(X_train, y_train)
    # Predict on calibration set
    y_pred_cal = rf.predict(X_cal)
    # Non‑conformity scores = absolute residuals, weighted by macro factor
    # Compute composite macro factor for calibration points
    from sklearn.linear_model import Ridge
    # Use ridge to estimate weights for macro variables
    ridge = Ridge(alpha=1.0)
    ridge.fit(X_train, y_train)
    weights = np.abs(ridge.coef_)
    weights = weights / (weights.sum() + 1e-8)
    # Composite macro factor for calibration points
    macro_factor_cal = X_cal @ weights
    # Weighted non‑conformity scores: residual / (1 + macro_factor) (higher macro -> smaller effective residual)
    resid = np.abs(y_cal - y_pred_cal)
    weighted_resid = resid / (1 + macro_factor_cal + 1e-8)
    # Compute quantile threshold
    q = np.quantile(weighted_resid, confidence)
    # For the last point (today), predict and compute interval width
    last_macro = macro_scaled[-1].reshape(1, -1)
    y_pred_last = rf.predict(last_macro)[0]
    last_macro_factor = (last_macro @ weights)[0]
    # Interval width = 2 * q * (1 + last_macro_factor)
    interval_width = 2 * q * (1 + last_macro_factor)
    # Score = -interval_width (lower width = more confident), but for ranking we want narrow intervals to have high score
    # We'll use 1 / (1 + interval_width) as the score
    score = 1.0 / (1.0 + interval_width)
    return float(score)
