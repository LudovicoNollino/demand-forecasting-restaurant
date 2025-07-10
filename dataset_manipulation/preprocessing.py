import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
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
    apply_standardization=False,
    apply_boxcox=False,
    apply_kalman=False,
    plot=True,
):
    # 1. Filtering, splitting the dataset and initial statistical tests
    dataset = df[col_name].dropna().copy()
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

    # 2. Clipping outliers using IQR only on training set
    q1, q3 = np.percentile(train, [25, 75])
    iqr = q3 - q1
    lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    print(f"IQR: {iqr:.2f}, lower: {lower:.2f}, upper: {upper:.2f}")
    train_clip = train.clip(lower, upper)
    val_clip = val.clip(lower, upper)
    test_clip = test.clip(lower, upper)

    # 3. Optional Box-Cox transformation
    lmbda = None
    if apply_boxcox:
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

    # 4. Optional Kalman filter smoothing
    if apply_kalman:
        train_proc = apply_kalman_filter(train_proc)
        val_proc = apply_kalman_filter(val_proc)
        test_proc = apply_kalman_filter(test_proc)
        print("Kalman filter applied.")

    # 5. Standardization
    scaler = None
    if apply_standardization:
        scaler = StandardScaler()
        train_scaled = scaler.fit_transform(train_proc.values.reshape(-1,1)).flatten()
        val_scaled = scaler.transform(val_proc.values.reshape(-1,1)).flatten()
        test_scaled = scaler.transform(test_proc.values.reshape(-1,1)).flatten()
        train_scaled = pd.Series(train_scaled, index=train_proc.index)
        val_scaled = pd.Series(val_scaled, index=val_proc.index)
        test_scaled = pd.Series(test_scaled, index=test_proc.index)
    else:
        train_scaled = train_proc
        val_scaled = val_proc
        test_scaled = test_proc

    # 6. Final statistical tests
    if apply_boxcox or apply_kalman or apply_standardization:
        adf_p = adfuller(train_scaled)[1]
        shapiro_p = stats.shapiro(train_scaled)[1]
        print(f"Final ADF p-value: {adf_p:.4f} {'stationary' if adf_p<0.05 else 'non stationary'}")
        print(f"Final Shapiro p-value: {shapiro_p:.4f} {'normal' if shapiro_p>=0.05 else 'non normal'}")

    # 7. Plotting
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
        if apply_standardization: extra.append('Standardized')
        title_proc = ' | '.join(['Clipped'] + extra)
        plt.title(f"{col_name} — {title_proc}")
        plt.legend(); plt.grid(True)
        plt.tight_layout()
        plt.show()

    # 8. Return processed series and parameters
    preprocess_params = {
        "scaler": scaler,
        "seasonal_lag": seasonal_lag,
        "iqr_lower": lower,
        "iqr_upper": upper,
        "boxcox_lambda": lmbda,
        "boxcox_shift": shift, 
        "boxcox_applied": apply_boxcox,
        "kalman_applied": apply_kalman
    }

    # Final series: align to original dataset index 
    final_series = pd.Series(index=dataset.index, dtype=float)
    final_series.loc[train_scaled.index] = train_scaled
    final_series.loc[val_scaled.index] = val_scaled
    final_series.loc[test_scaled.index] = test_scaled
    return final_series, preprocess_params


def inverse_transform_predictions_forecast(pred_processed, preprocess_params):
    # Inverse scaling
    scaler = preprocess_params.get('scaler', None)
    if scaler is not None:
        pred_unscaled = scaler.inverse_transform(pred_processed.values.reshape(-1,1)).flatten()
        pred_unscaled = pd.Series(pred_unscaled, index=pred_processed.index)
    else:
        pred_unscaled = pred_processed.copy()

    # Inverse Box-Cox 
    if preprocess_params.get('boxcox_applied') and preprocess_params.get('boxcox_lambda') is not None:
        lmbda = preprocess_params['boxcox_lambda']
        shift = preprocess_params.get('boxcox_shift', 0)
        pred_unscaled = inverse_boxcox_transform(pred_unscaled, lmbda)
        pred_unscaled = pred_unscaled - shift
        pred_unscaled = pd.Series(pred_unscaled, index=pred_processed.index)
    return pred_unscaled
