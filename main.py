import pandas as pd
from dataset_manipulation.preprocessing import preprocess_column, inverse_transform_predictions_forecast
from algorithms.SARIMAX import sarimax_grid_search, fit_sarimax_model
from algorithms.XGBoost import fit_xgboost_model, xgboost_grid_search
from algorithms.mlp_torch import fit_mlp_model

df = pd.read_csv(r"demand-forecasting-restaurant\dataset\chiusure_di giornata_autentiko_beach_estate_2024.csv", sep=';')
df.columns = df.columns.str.strip()
df['Data di chiusura'] = pd.to_datetime(df['Data di chiusura'], dayfirst=True, errors='raise')
df = df.sort_values('Data di chiusura').reset_index(drop=True)
df['day_of_week'] = df['Data di chiusura'].dt.day_name()
df['is_weekend']  = df['Data di chiusura'].dt.weekday >= 5
df = df.dropna(subset=['Numero ospiti', 'Chiusura di giornata (scalata in un intervallo)'])

seasonal_period = 7  

from statsmodels.tsa.stattools import adfuller
from scipy.stats import shapiro

def diagnostica_post_diff(series, d, D, s, col_name="Serie"):
    diffed = series.diff(d) if d > 0 else series.copy()
    if D > 0:
        diffed = diffed.diff(s * D)
    diffed = diffed.dropna()
    adf_p = adfuller(diffed)[1]
    adf_msg = "stazionaria" if adf_p < 0.05 else "NON stazionaria"
    shapiro_p = shapiro(diffed)[1]
    shapiro_msg = "normale" if shapiro_p >= 0.05 else "NON normale"
    print(f"\n--- Diagnostica su '{col_name}' dopo differenziazione d={d}, D={D}, s={s} ---")
    print(f"ADF p-value: {adf_p:.4f} → {adf_msg}")
    print(f"Shapiro p-value: {shapiro_p:.4f} → {shapiro_msg}")
    return diffed

def get_preprocessed_dict(df, col_name, **kwargs):
    proc, params = preprocess_column(df, col_name, **kwargs)
    n = len(proc)
    n_train = int(n * 0.70)
    n_val = int(n * 0.15)
    n_test = n - n_train - n_val
    train = proc.iloc[:n_train].copy()
    val = proc.iloc[n_train:n_train + n_val].copy()
    test = proc.iloc[n_train + n_val:].copy()
    orig = df.loc[proc.index, col_name]
    train_orig = orig.iloc[:n_train]
    val_orig = orig.iloc[n_train:n_train + n_val]
    test_orig = orig.iloc[n_train + n_val:]
    return {
        "train": train,
        "val": val,
        "test": test,
        "train_orig": train_orig,
        "val_orig": val_orig,
        "test_orig": test_orig,
        "preprocess_params": params
    }

# ------- SARIMAX -------

guests_data = get_preprocessed_dict(
    df,
    'Numero ospiti',
    apply_boxcox=False,
    apply_kalman=False,
    apply_standardization=False,
    seasonal_lag=seasonal_period,
)

model_results_guests, val_predictions_guests, test_predictions_guests = fit_sarimax_model(
    guests_data,
    order=(0, 1, 2),
    seasonal_order=(0, 1, 1, seasonal_period),
    col_name='Numero ospiti',
)

diagnostica_post_diff(
    df['Numero ospiti'].dropna(),
    d=1,
    D=1,
    s=seasonal_period,
    col_name='Numero ospiti'
)

closure_data = get_preprocessed_dict(
    df,
    'Chiusura di giornata (scalata in un intervallo)',
    apply_boxcox=False,
    apply_kalman=False,
    apply_standardization=False,
    seasonal_lag=seasonal_period,
)

model_results_closure, val_predictions_closure, test_predictions_closure = fit_sarimax_model(
    closure_data,
    order=(0, 1, 2),
    seasonal_order=(0, 1, 1, seasonal_period),
    col_name='Chiusura di giornata (scalata in un intervallo)',
)

diagnostica_post_diff(
    df['Chiusura di giornata (scalata in un intervallo)'].dropna(),
    d=1,
    D=1,
    s=seasonal_period,
    col_name='Chiusura di giornata (scalata in un intervallo)'
)

# ------- MLP TORCH --------

print("\n================= MLP TORCH SU 'Numero ospiti' =================")
from algorithms.mlp_torch import fit_mlp_model

mlp_data = get_preprocessed_dict(
    df,
    'Numero ospiti',
    apply_boxcox=False,
    apply_kalman=False,
    apply_standardization=True,
    seasonal_lag=seasonal_period,
)

# Parametri per grid search (facoltativo, puoi modificarli!)
grid_params = dict(
    hidden1_grid=[8],
    hidden2_grid=[8],
    lr_grid=[0.0005],
    activations=['relu'],
    n_epochs_grid=[500],
    batch_size=1,
    print_every=50
)

model, best_params, grid, val_pred, test_pred, future_pred = fit_mlp_model(
    mlp_data,
    window_size=7,
    hidden_dim1=16,
    hidden_dim2=8,
    lr=0.001,
    activation='relu',
    n_epochs=200,
    batch_size=1,
    print_every=20,
    future_steps=30,
    grid_search=True,
    grid_params=grid_params,
    col_name='Numero ospiti',
    verbose=True
)

print("\nBest MLP params:", best_params)
print(f"Validation predictions shape: {val_pred.shape}")
print(f"Test predictions shape: {test_pred.shape}")
print(f"Future forecast shape: {future_pred.shape}")

# ------- XGBOOST --------

print("\n================= XGBOOST SU 'Numero ospiti' =================")

xgb_data = get_preprocessed_dict(
    df,
    'Numero ospiti',
    apply_boxcox=False,
    apply_kalman=False,
    apply_standardization=True,
    seasonal_lag=seasonal_period,
)

# Grid search: puoi cambiare i valori!
grid_results, best_config, best_model = xgboost_grid_search(
    xgb_data,
    look_back_grid=[7, 14, 21, 30, 45, 60],
    n_estimators_grid=[10, 15, 25, 35, 40, 50, 75, 100],
)

print("\n--- XGBoost grid search results ---")
print(grid_results.head())

# Lancia il modello migliore trovato
model_xgb, val_pred_xgb, test_pred_xgb, future_pred_xgb = fit_xgboost_model(
    xgb_data,
    look_back=best_config[0],
    n_estimators=best_config[1],
    col_name='Numero ospiti',
)

print("\nBest XGBoost config:", best_config)
print(f"Validation predictions shape: {val_pred_xgb.shape}")
print(f"Test predictions shape: {test_pred_xgb.shape}")
print(f"Future forecast shape: {future_pred_xgb.shape}")


# --------- FINE MAIN ---------

