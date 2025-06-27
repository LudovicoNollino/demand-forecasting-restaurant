import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error
from dataset_manipulation.preprocessing import inverse_transform_predictions_forecast

def create_sliding_window(series, look_back):
    arr = np.array(series)
    X, y = [], []
    for i in range(len(arr) - look_back):
        X.append(arr[i:i+look_back])
        y.append(arr[i+look_back])
    return np.array(X), np.array(y)

def fit_xgboost_model(
    data_dict,
    col_name,
    look_back=7,
    n_estimators=100,
    future_steps=30,
    verbose=True,
):
    # --- Preprocessing & split ---
    series_proc = pd.concat([data_dict["train"], data_dict["val"], data_dict["test"]]).astype(float)
    n_train, n_val, n_test = len(data_dict["train"]), len(data_dict["val"]), len(data_dict["test"])
    params = data_dict['preprocess_params']

    # --- PRIMO MODELLO: solo train, per forecast su val (val_rmse)
    X_train, y_train = create_sliding_window(series_proc[:n_train+look_back], look_back)
    model_val = XGBRegressor(objective='reg:squarederror', n_estimators=n_estimators, random_state=42)
    model_val.fit(X_train, y_train)

    # --- Rolling/Recursive prediction per validation
    val_forecast = []
    input_seq = list(series_proc[n_train-look_back:n_train])
    for _ in range(n_val):
        x_input = np.array(input_seq[-look_back:]).reshape(1, -1)
        pred = model_val.predict(x_input)[0]
        val_forecast.append(pred)
        input_seq.append(pred)
    val_forecast = np.array(val_forecast)
    val_forecast_inv = inverse_transform_predictions_forecast(
        pd.Series(val_forecast, index=data_dict["val"].index), params
    )
    val_rmse = mean_squared_error(data_dict["val_orig"].values, val_forecast_inv.values) ** 0.5

    # --- MODELLO FINALE: train + val, per test e future ---
    X_trainval, y_trainval = create_sliding_window(series_proc[:n_train+n_val+look_back], look_back)
    model = XGBRegressor(objective='reg:squarederror', n_estimators=n_estimators, random_state=42)
    model.fit(X_trainval, y_trainval)

    # Test prediction (rolling)
    test_forecast = []
    input_seq = list(series_proc[n_train+n_val-look_back:n_train+n_val])
    for _ in range(n_test):
        x_input = np.array(input_seq[-look_back:]).reshape(1, -1)
        pred = model.predict(x_input)[0]
        test_forecast.append(pred)
        input_seq.append(pred)
    test_forecast = np.array(test_forecast)
    test_forecast_inv = inverse_transform_predictions_forecast(
        pd.Series(test_forecast, index=data_dict["test"].index), params
    )
    test_rmse = mean_squared_error(data_dict["test_orig"].values, test_forecast_inv.values) ** 0.5

    # --- Forecast futuro ---
    input_seq = list(series_proc[-look_back:])
    future_forecast = []
    for _ in range(future_steps):
        x_input = np.array(input_seq[-look_back:]).reshape(1, -1)
        pred = model.predict(x_input)[0]
        future_forecast.append(pred)
        input_seq.append(pred)
    # Future index
    last_idx = data_dict["test_orig"].index[-1]
    if isinstance(last_idx, pd.Timestamp):
        future_idx = pd.date_range(start=last_idx + pd.Timedelta(days=1), periods=future_steps)
    else:
        future_idx = np.arange(last_idx + 1, last_idx + 1 + future_steps)
    future_forecast_inv = inverse_transform_predictions_forecast(
        pd.Series(future_forecast, index=future_idx), params
    )

    if verbose:
        print(f"Val RMSE: {val_rmse:.4f} | Test RMSE (definitivo): {test_rmse:.4f}")

    # (tieni il plot uguale a prima)

    plt.figure(figsize=(16, 9))
    alpha_zone = 0.12

    # ORIGINAL SCALE
    plt.subplot(2,1,1)
    idx_train = data_dict["train_orig"].index
    idx_val = data_dict["val_orig"].index
    idx_test = data_dict["test_orig"].index

    if len(idx_val) > 0:
        plt.axvspan(idx_val[0], idx_val[-1], color='orange', alpha=alpha_zone, label='Validation Period')
    if len(idx_test) > 0:
        plt.axvspan(idx_test[0], idx_test[-1], color='green', alpha=alpha_zone, label='Test Period')
    if len(future_forecast_inv) > 0:
        plt.axvspan(future_forecast_inv.index[0], future_forecast_inv.index[-1], color='red', alpha=alpha_zone, label='Future Forecast Period')

    plt.plot(idx_train, data_dict["train_orig"], color='blue', label='Train Obs', linewidth=2)
    plt.plot(idx_val, data_dict["val_orig"], color='orange', label='Val Obs', linewidth=2)
    plt.plot(idx_test, data_dict["test_orig"], color='green', label='Test Obs', linewidth=2)
    plt.plot(idx_val, val_forecast_inv.values, '--', color='red', label='Val Forecast', linewidth=2.5)
    plt.plot(idx_test, test_forecast_inv.values, '--', color='purple', label='Test Forecast', linewidth=2.5)
    plt.plot(future_forecast_inv.index, future_forecast_inv.values, '-.', color='black', label='Future Forecast', linewidth=2.5)

    plt.axvline(idx_train[-1], color="black", linestyle=":", label="Split Train/Val")
    plt.axvline(idx_val[-1], color="black", linestyle=":", label="Split Val/Test")
    plt.axvline(idx_test[-1], color="black", linestyle=":", label="Split Test/Future")

    plt.title(f"{col_name} - Original Scale")
    plt.legend(loc='upper left', fontsize=10)
    plt.grid(True)

    # PROCESSED SCALE
    plt.subplot(2,1,2)
    idx_full = idx_train.append(idx_val).append(idx_test)
    plt.plot(idx_full, series_proc, color='black', label='Observed (proc)', linewidth=2)
    plt.title(f"{col_name} - Processed Scale")
    plt.legend(loc='upper left', fontsize=10)
    plt.grid(True)

    plt.tight_layout()
    plt.show()

    return model, val_forecast_inv, test_forecast_inv, future_forecast_inv

def xgboost_grid_search(
    data_dict,
    look_back_grid,
    n_estimators_grid,
):
    """
    Grid search su XGBoost su validation RMSE (più efficiente, senza chiamare fit_xgboost_model)
    """
    best_rmse = np.inf
    best_config = None
    best_model = None
    results = []

    # Prepara dati processati
    series_proc = pd.concat([data_dict["train"], data_dict["val"], data_dict["test"]]).astype(float)
    n_train, n_val = len(data_dict["train"]), len(data_dict["val"])
    preprocess_params = data_dict['preprocess_params']

    for look_back in look_back_grid:
        # Sliding windows su train
        X_train, y_train = create_sliding_window(series_proc[:n_train+look_back], look_back)

        for n_estimators in n_estimators_grid:
            try:
                # Train model
                model = XGBRegressor(objective='reg:squarederror', n_estimators=n_estimators, random_state=42)
                model.fit(X_train, y_train)

                # Recursive validation forecast
                val_forecast = []
                input_seq = list(series_proc[n_train-look_back:n_train])
                for _ in range(n_val):
                    x_input = np.array(input_seq[-look_back:]).reshape(1, -1)
                    pred = model.predict(x_input)[0]
                    val_forecast.append(pred)
                    input_seq.append(pred)
                val_forecast = pd.Series(val_forecast, index=data_dict["val"].index)

                # Inverse transform
                val_forecast_inv = inverse_transform_predictions_forecast(val_forecast, preprocess_params)

                # Calcola RMSE in scala originale
                val_rmse = mean_squared_error(data_dict["val_orig"].values, val_forecast_inv.values) ** 0.5

                results.append({
                    "look_back": look_back,
                    "n_estimators": n_estimators,
                    "val_rmse": val_rmse
                })

                # Aggiorna miglior modello
                if val_rmse < best_rmse:
                    best_rmse = val_rmse
                    best_config = (look_back, n_estimators)
                    best_model = model
            except Exception as e:
                print(f"Errore con look_back={look_back}, n_estimators={n_estimators}: {e}")

    results_df = pd.DataFrame(results)
    if not results_df.empty:
        results_df = results_df.sort_values('val_rmse').reset_index(drop=True)
        print("\nGrid search results:")
        print(results_df)
        print(f"\nBest config: {best_config} (Val RMSE={best_rmse:.4f})")
    else:
        print("\nNessun risultato valido dalla grid search.")

    return results_df, best_config, best_model
