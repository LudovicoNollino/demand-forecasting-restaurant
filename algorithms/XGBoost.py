import numpy as np
import pandas as pd
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error
from dataset_manipulation.preprocessing import inverse_transform_predictions_forecast

def create_sliding_window_with_features(series, features, look_back):
    arr = np.array(series)
    X, y = [], []
    for i in range(len(arr) - look_back):
        window = arr[i:i+look_back]
        feat = features.iloc[i+look_back].values if isinstance(features, pd.DataFrame) else features[i+look_back]
        X.append(np.concatenate([window, feat]))
        y.append(arr[i+look_back])
    return np.array(X), np.array(y)

def xgboost_grid_search(
    data_dict,
    look_back_grid,
    n_estimators_grid,
):
    best_rmse = np.inf
    best_config = None
    best_model = None
    results = []

    series_proc = pd.concat([data_dict["train"], data_dict["val"], data_dict["test"]]).astype(float)
    n_train, n_val = len(data_dict["train"]), len(data_dict["val"])
    preprocess_params = data_dict['preprocess_params']
    features = data_dict['features'].reset_index(drop=True).astype(np.float32)

    for look_back in look_back_grid:
        X_train, y_train = create_sliding_window_with_features(
            series_proc[:n_train+look_back], features.iloc[:n_train+look_back], look_back
        )

        for n_estimators in n_estimators_grid:
            try:
                model = XGBRegressor(objective='reg:squarederror', n_estimators=n_estimators, random_state=42)
                model.fit(X_train, y_train)

                # Rolling forecast validation
                val_forecast = []
                input_seq = list(series_proc[n_train-look_back:n_train])
                for t in range(n_val):
                    feat = features.iloc[n_train + t].values
                    x_input = np.concatenate([input_seq[-look_back:], feat]).reshape(1, -1)
                    pred = model.predict(x_input)[0]
                    val_forecast.append(pred)
                    input_seq.append(pred)
                val_forecast = pd.Series(val_forecast, index=data_dict["val"].index)

                val_forecast_inv = inverse_transform_predictions_forecast(val_forecast, preprocess_params)
                val_rmse = mean_squared_error(data_dict["val_orig"].values, val_forecast_inv.values) ** 0.5

                results.append({
                    "look_back": look_back,
                    "n_estimators": n_estimators,
                    "val_rmse": val_rmse
                })

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

def fit_xgboost_model(
    data_dict,
    col_name,
    look_back=7,
    n_estimators=100,
    future_steps=30,
):
    series_proc = pd.concat([data_dict["train"], data_dict["val"], data_dict["test"]]).astype(float)
    n_train, n_val, n_test = len(data_dict["train"]), len(data_dict["val"]), len(data_dict["test"])
    params = data_dict['preprocess_params']
    features = data_dict['features'].reset_index(drop=True).astype(np.float32)

    # TRAIN/VAL per grid search
    X_train, y_train = create_sliding_window_with_features(
        series_proc[:n_train+look_back], features.iloc[:n_train+look_back], look_back
    )
    model_val = XGBRegressor(objective='reg:squarederror', n_estimators=n_estimators, random_state=42)
    model_val.fit(X_train, y_train)

    # Validation rolling
    val_forecast = []
    input_seq = list(series_proc[n_train-look_back:n_train])
    for t in range(n_val):
        feat = features.iloc[n_train + t].values
        x_input = np.concatenate([input_seq[-look_back:], feat]).reshape(1, -1)
        pred = model_val.predict(x_input)[0]
        val_forecast.append(pred)
        input_seq.append(pred)
    val_forecast = pd.Series(val_forecast, index=data_dict["val"].index)
    val_forecast_inv = inverse_transform_predictions_forecast(val_forecast, params)

    # FINAL MODEL train + val
    X_trainval, y_trainval = create_sliding_window_with_features(
        series_proc[:n_train+n_val+look_back], features.iloc[:n_train+n_val+look_back], look_back
    )
    model = XGBRegressor(objective='reg:squarederror', n_estimators=n_estimators, random_state=42)
    model.fit(X_trainval, y_trainval)

    # TEST rolling
    test_forecast = []
    input_seq = list(series_proc[n_train+n_val-look_back:n_train+n_val])
    for t in range(n_test):
        feat = features.iloc[n_train + n_val + t].values
        x_input = np.concatenate([input_seq[-look_back:], feat]).reshape(1, -1)
        pred = model.predict(x_input)[0]
        test_forecast.append(pred)
        input_seq.append(pred)
    test_forecast = pd.Series(test_forecast, index=data_dict["test"].index)
    test_forecast_inv = inverse_transform_predictions_forecast(test_forecast, params)

    # FUTURE
    input_seq = list(series_proc[-look_back:])
    future_forecast = []
    for i in range(future_steps):
        idx_feat = len(series_proc) + i
        if idx_feat < len(features):
            feat = features.iloc[idx_feat].values
        else:
            feat = np.zeros(features.shape[1])
        x_input = np.concatenate([input_seq[-look_back:], feat]).reshape(1, -1)
        pred = model.predict(x_input)[0]
        future_forecast.append(pred)
        input_seq.append(pred)
    # Future index
    test_idx = data_dict["test_orig"].index
    last_idx = test_idx[-1]
    if isinstance(last_idx, pd.Timestamp):
        future_idx = pd.date_range(start=last_idx + pd.Timedelta(days=1), periods=future_steps)
    else:
        future_idx = np.arange(last_idx + 1, last_idx + 1 + future_steps)
    future_forecast = pd.Series(future_forecast, index=future_idx)
    future_forecast_inv = inverse_transform_predictions_forecast(future_forecast, params)

    from plotter import plot_forecasting

    plot_forecasting(
        col_name=col_name,
        y_train_orig=data_dict["train_orig"],
        y_val_orig=data_dict["val_orig"],
        y_test_orig=data_dict["test_orig"],
        val_pred_orig=val_forecast_inv,
        test_pred_orig=test_forecast_inv,
        future_pred_orig=future_forecast_inv
    )

    return model, val_forecast_inv, test_forecast_inv, future_forecast_inv