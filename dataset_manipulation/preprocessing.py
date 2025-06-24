import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf
from scipy import stats
from scipy.stats import boxcox
from pykalman import KalmanFilter
from sklearn.preprocessing import StandardScaler

def apply_kalman_filter(series):
    kf = KalmanFilter(
        transition_matrices=[1],
        observation_matrices=[1],
        initial_state_mean=series.iloc[0],
        n_dim_state=1
    )
    kf = kf.em(series.values)
    state_means, _ = kf.smooth(series.values)
    return pd.Series(state_means.flatten(), index=series.index)

def inverse_boxcox_transform(data, lmbda):
    if lmbda == 0:
        return np.exp(data)
    else:
        return np.power(data * lmbda + 1, 1 / lmbda)

def preprocess_column(
    df, col_name,
    train_size=0.70, val_size=0.15, test_size=0.15,
    seasonal_lag=7,
    apply_boxcox=False,
    apply_kalman=False,
    plot=True,
    plot_acf_final=True
):
    """
    Flexible preprocessing pipeline for a single column in a time series DataFrame
    with options for Box-Cox, Kalman filter, and standardization.
    """
    # 1. Filtering, splitting the dataset and initial statistical tests
    dataset = df.loc[df[col_name] > 0, col_name].dropna()
    n = len(dataset)
    n_train = int(n * train_size)
    n_val = int(n * val_size)

    train = dataset.iloc[:n_train].copy()
    val = dataset.iloc[n_train:n_train + n_val].copy()
    test = dataset.iloc[n_train + n_val:].copy()
    print(f"\nDataset '{col_name}': train={len(train)}, val={len(val)}, test={len(test)}")
    
    adf_p = adfuller(train)[1]
    shapiro_p = stats.shapiro(train)[1]
    print(f"Initial ADF p-value: {adf_p:.4f} {'stationary' if adf_p<0.05 else 'non stationary'}")
    print(f"Initial Shapiro p-value: {shapiro_p:.4f} {'normal' if shapiro_p>=0.05 else 'non normal'}")

    # 2. Seasonal and simple differencing
    train_diff = train.diff(seasonal_lag).diff().dropna()
    val_diff = val.diff(seasonal_lag).diff().dropna()
    test_diff = test.diff(seasonal_lag).diff().dropna()

    # 3. Clipping outliers using IQR only on training set
    q1, q3 = np.percentile(train_diff, [25, 75])
    iqr = q3 - q1
    lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    print(f"IQR: {iqr:.2f}, lower: {lower:.2f}, upper: {upper:.2f}")
    train_clip = train_diff.clip(lower, upper)
    val_clip = val_diff.clip(lower, upper)
    test_clip = test_diff.clip(lower, upper)

    # 4. Optional Box-Cox transformation
    lmbda = None
    if apply_boxcox:
        # Box-Cox requires positive values, so we shift if necessary
        min_value = min(train_clip.min(), val_clip.min(), test_clip.min())
        shift = -min_value + 1e-6 if min_value <= 0 else 0
        print(f"Box-Cox: shift applied = {shift:.6f}")
        train_bc, lmbda = boxcox(train_clip + shift)
        val_bc = boxcox(val_clip + shift, lmbda=lmbda)
        test_bc = boxcox(test_clip + shift, lmbda=lmbda)
        train_proc = pd.Series(train_bc, index=train_clip.index)
        val_proc = pd.Series(val_bc, index=val_clip.index)
        test_proc = pd.Series(test_bc, index=test_clip.index)
        print(f"Box-Cox applied (lambda={lmbda:.4f})")
    else:
        shift = 0
        train_proc, val_proc, test_proc = train_clip, val_clip, test_clip


    # 5. Optional Kalman filter smoothing
    if apply_kalman:
        train_proc = apply_kalman_filter(train_proc)
        val_proc = apply_kalman_filter(val_proc)
        test_proc = apply_kalman_filter(test_proc)
        print("Kalman filter applied.")

    # 6. Standardization
    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(train_proc.values.reshape(-1,1)).flatten()
    val_scaled = scaler.transform(val_proc.values.reshape(-1,1)).flatten()
    test_scaled = scaler.transform(test_proc.values.reshape(-1,1)).flatten()

    train_scaled = pd.Series(train_scaled, index=train_proc.index)
    val_scaled = pd.Series(val_scaled, index=val_proc.index)
    test_scaled = pd.Series(test_scaled, index=test_proc.index)

    # 7. Final statistical tests
    adf_p = adfuller(train_scaled)[1]
    shapiro_p = stats.shapiro(train_scaled)[1]
    print(f"Final ADF p-value: {adf_p:.4f} {'stationary' if adf_p<0.05 else 'non stationary'}")
    print(f"Final Shapiro p-value: {shapiro_p:.4f} {'normal' if shapiro_p>=0.05 else 'non normal'}")

    # 8. Plotting
    if plot:
        plt.figure(figsize=(16, 6))
        plt.subplot(2,1,1)
        plt.plot(train.index, train, label='Train Original', color='lightblue')
        plt.plot(val.index, val, label='Val Original', color='gold')
        plt.plot(test.index, test, label='Test Original', color='lightgreen')
        plt.title(f"{col_name} — Original Series")
        plt.legend(); plt.grid(True)

        plt.subplot(2,1,2)
        plt.plot(train_scaled.index, train_scaled, label='Train Processed', color='blue')
        plt.plot(val_scaled.index, val_scaled, label='Val Processed', color='orange')
        plt.plot(test_scaled.index, test_scaled, label='Test Processed', color='green')
        extra = []
        if apply_boxcox: extra.append('Box-Cox')
        if apply_kalman: extra.append('Kalman')
        title_proc = ' | '.join(['Difference', 'Clipped', 'Standardized'] + extra)
        plt.title(f"{col_name} — {title_proc}")
        plt.legend(); plt.grid(True)
        plt.tight_layout()
        plt.show()

    if plot_acf_final:
        plt.figure(figsize=(8,4))
        plot_acf(train_scaled, lags=30)
        plt.title(f"ACF final {col_name} (train processed)")
        plt.tight_layout()
        plt.show()

    # 9. Return processed series and parameters
    preprocess_params = {
        "scaler": scaler,
        "seasonal_lag": seasonal_lag,
        "iqr_lower": lower,
        "iqr_upper": upper,
        "boxcox_lambda": lmbda,
        "boxcox_shift": shift,   # AGGIUNGI QUESTO!
        "boxcox_applied": apply_boxcox,
        "kalman_applied": apply_kalman
    }

    # Final series: align to original dataset index (after double differencing we lose initial values)
    final_series = pd.Series(index=dataset.index, dtype=float)
    final_series.loc[train_scaled.index] = train_scaled
    final_series.loc[val_scaled.index] = val_scaled
    final_series.loc[test_scaled.index] = test_scaled
    return final_series, preprocess_params

def inverse_transform_predictions(pred_scaled, original_series, preprocess_params):
    """
    Returns the predictions to the original scale by reversing standardization, Box-Cox, and differencing.
    pred_scaled: predictions on the processed series (STANDARDIZED, DOUBLE-DIFFERENCED)
    original_series: the original NON-differenced series (the same column used in preprocessing!)
    preprocess_params: dict
    """
    # Step 1: Inverse StandardScaler
    scaler = preprocess_params['scaler']
    pred_unscaled = scaler.inverse_transform(pred_scaled.values.reshape(-1,1)).flatten()
    pred_unscaled = pd.Series(pred_unscaled, index=pred_scaled.index)

    # Step 2: Inverse Box-Cox (if applied)
    if preprocess_params.get('boxcox_applied') and preprocess_params.get('boxcox_lambda') is not None:
        lmbda = preprocess_params['boxcox_lambda']
        shift = preprocess_params.get('boxcox_shift', 0)
        pred_unscaled = inverse_boxcox_transform(pred_unscaled, lmbda)
        pred_unscaled = pred_unscaled - shift   
        pred_unscaled = pd.Series(pred_unscaled, index=pred_scaled.index)


    # Step 3: Inverse differencing (integration)
    seasonal_lag = preprocess_params['seasonal_lag']
    rec_series = original_series.copy()
    rec_values = []
    for i, idx in enumerate(pred_unscaled.index):
        if idx - seasonal_lag in rec_series.index and idx - 1 in rec_series.index:
            val = pred_unscaled.loc[idx] + rec_series.loc[idx - 1] + (rec_series.loc[idx - seasonal_lag] - rec_series.loc[idx - seasonal_lag - 1])
        else:
            val = pred_unscaled.loc[idx] + rec_series.loc[idx - 1] if idx - 1 in rec_series.index else pred_unscaled.loc[idx]
        rec_series.loc[idx] = val
        rec_values.append(val)
    return pd.Series(rec_values, index=pred_unscaled.index)
