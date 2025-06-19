import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from scipy import stats
from scipy.stats import boxcox
from pykalman import KalmanFilter
from statsmodels.tsa.seasonal import seasonal_decompose
from sklearn.preprocessing import StandardScaler


def kalman_smooth(x):
    """Simple univariate Kalman smoothing."""
    kf = KalmanFilter(
        transition_matrices=[1],
        observation_matrices=[1],
        initial_state_mean=x.iloc[0]
    )
    kf = kf.em(x.values, n_iter=5)
    smoothed, _ = kf.smooth(x.values)
    return pd.Series(smoothed.flatten(), index=x.index)

def preprocess_column(df, col_name):
    """Run full pipeline on a single column, return processed Series."""
    s = df[col_name].dropna()
    n = len(s)
    i1 = int(0.70 * n)
    i2 = i1 + int(0.15 * n)

    train, val, test = s.iloc[:i1], s.iloc[i1:i2], s.iloc[i2:]

    # Stationarity
    adf_p = adfuller(train)[1]
    print(f"{col_name} — ADF p-value: {adf_p:.4f}")

    # Normality
    sh_p = stats.shapiro(train)[1]
    print(f"{col_name} — Shapiro–Wilk p-value: {sh_p:.4f}")

    # Outlier bounds (IQR)
    q1, q3 = np.percentile(train, [25, 75])
    iqr = q3 - q1
    lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    print(f"{col_name} — outlier bounds: [{lower:.2f}, {upper:.2f}]")

    # Clip outliers
    train_c = train.clip(lower, upper)
    val_c, test_c = val.copy(), test.copy()

    # Box–Cox transform
    shift = 1e-6 if (train_c <= 0).any() else 0
    train_bc, lmbda = boxcox(train_c + shift)
    val_bc   = boxcox(val_c + shift,   lmbda)
    test_bc  = boxcox(test_c + shift,  lmbda)
    print(f"{col_name} — Box–Cox λ: {lmbda:.4f}")

    # Kalman smoothing
    train_k = kalman_smooth(pd.Series(train_bc, index=train_c.index))
    val_k   = kalman_smooth(pd.Series(val_bc,   index=val_c.index))
    test_k  = kalman_smooth(pd.Series(test_bc,  index=test_c.index))
    print(f"{col_name} — Kalman smoothing applied")

    # Reassemble full series
    processed = pd.concat([
        pd.Series(train_k, index=train.index),
        pd.Series(val_k,   index=val.index),
        pd.Series(test_k,  index=test.index),
    ]).sort_index()
    processed_full = pd.Series(index=df.index, dtype=float)
    processed_full.loc[processed.index] = processed   

    # Plot
    plt.figure(figsize=(10, 4))
    plt.plot(df['Data di chiusura'], df[col_name], label='Original', alpha=0.5)
    plt.plot(df['Data di chiusura'], processed, label='Processed', alpha=0.8)
    plt.title(f"{col_name} — Original vs Processed")
    plt.xlabel("Date")
    plt.ylabel(col_name)
    plt.legend()
    plt.tight_layout()
    plt.show()

    return processed_full