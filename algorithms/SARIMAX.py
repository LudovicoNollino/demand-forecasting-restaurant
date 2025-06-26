import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error
from dataset_manipulation.preprocessing import inverse_transform_predictions_forecast
import warnings

warnings.filterwarnings("ignore")

def calc_metrics(y_true, y_pred):
    bias = np.mean(y_pred - y_true)
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    mad = np.mean(np.abs(y_pred - y_true))
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    return {"BIAS": bias, "MAPE": mape, "MAD": mad, "RMSE": rmse}

def sarimax_grid_search(
    data_dict,
    seasonal_period=7,
    max_p=2, max_d=1, max_q=2,
    max_P=1, max_D=1, max_Q=1,
    verbose=True
):
    print(f"Grid Search SARIMAX - Periodo stagionale: {seasonal_period}")
    y_train = data_dict['train']
    best_aic = np.inf
    best_order = None
    best_seasonal = None
    grid = []
    total = (max_p+1)*(max_d+1)*(max_q+1)*(max_P+1)*(max_D+1)*(max_Q+1)
    count = 0
    for p in range(max_p+1):
        for d in range(max_d+1):
            for q in range(max_q+1):
                for P in range(max_P+1):
                    for D in range(max_D+1):
                        for Q in range(max_Q+1):
                            count += 1
                            order = (p, d, q)
                            seasonal = (P, D, Q, seasonal_period)
                            try:
                                model = SARIMAX(
                                    y_train,
                                    order=order,
                                    seasonal_order=seasonal,
                                    enforce_stationarity=False,
                                    enforce_invertibility=False
                                )
                                res = model.fit(disp=False)
                                aic = res.aic
                                grid.append({"order": order, "seasonal": seasonal, "aic": aic})
                                if aic < best_aic:
                                    best_aic = aic
                                    best_order = order
                                    best_seasonal = seasonal
                                if verbose and count % 10 == 0:
                                    print(f"Progress: {count}/{total}, Best AIC: {best_aic:.2f}")
                            except Exception:
                                continue
    df = pd.DataFrame(grid).sort_values('aic').reset_index(drop=True)
    print(f"Best order: {best_order}, Best seasonal: {best_seasonal}, Best AIC: {best_aic:.2f}")
    print(df.head(10))
    return best_order, best_seasonal, df

def fit_sarimax_model(
    data_dict,
    order,
    seasonal_order,
    col_name,
    future_steps=30
):
    y_train, y_val, y_test = data_dict['train'], data_dict['val'], data_dict['test']
    y_train_orig, y_val_orig, y_test_orig = data_dict['train_orig'], data_dict['val_orig'], data_dict['test_orig']
    preprocess_params = data_dict['preprocess_params']

    # Decide se serve l'inversione automatica
    use_inverse_transform = (
        preprocess_params.get("boxcox_applied", False)
        or preprocess_params.get("kalman_applied", False)
        or (preprocess_params.get("scaler", None) is not None)
    )

    def maybe_inverse(pred_proc):
        if use_inverse_transform:
            return inverse_transform_predictions_forecast(pred_proc, preprocess_params)
        else:
            return pred_proc

    # 1. Fit su train → previsioni validation
    model_train = SARIMAX(y_train, order=order, seasonal_order=seasonal_order, enforce_stationarity=False, enforce_invertibility=False)
    res_train = model_train.fit(disp=False)
    val_pred = res_train.get_forecast(steps=len(y_val)).predicted_mean
    val_pred_orig = maybe_inverse(val_pred)

    # 2. Fit su train+val → previsioni test
    y_trainval = pd.concat([y_train, y_val])
    model_tv = SARIMAX(y_trainval, order=order, seasonal_order=seasonal_order, enforce_stationarity=False, enforce_invertibility=False)
    res_tv = model_tv.fit(disp=False)
    test_pred = res_tv.get_forecast(steps=len(y_test)).predicted_mean
    test_pred_orig = maybe_inverse(test_pred)

    # 3. Fit su tutti i dati → forecast futuro
    y_full = pd.concat([y_train, y_val, y_test])
    model_full = SARIMAX(y_full, order=order, seasonal_order=seasonal_order, enforce_stationarity=False, enforce_invertibility=False)
    res_full = model_full.fit(disp=False)
    future_pred_proc = res_full.get_forecast(steps=future_steps).predicted_mean
    future_pred_orig = maybe_inverse(future_pred_proc)

       # --- GRAFICO MIGLIORATO ---
    plt.figure(figsize=(16, 9))
    alpha_zone = 0.12

    # --- ORIGINAL SCALE ---
    plt.subplot(2,1,1)

    # Zone di background per validation/test/future
    if len(y_val_orig) > 0:
        plt.axvspan(y_val_orig.index[0], y_val_orig.index[-1], color='orange', alpha=alpha_zone, label='Validation Period')
    if len(y_test_orig) > 0:
        plt.axvspan(y_test_orig.index[0], y_test_orig.index[-1], color='green', alpha=alpha_zone, label='Test Period')
    if len(future_pred_orig) > 0:
        plt.axvspan(future_pred_orig.index[0], future_pred_orig.index[-1], color='red', alpha=alpha_zone, label='Future Forecast Period')

    # Serie osservate
    plt.plot(y_train_orig.index, y_train_orig, color='blue', label='Train Obs', linewidth=2)
    plt.plot(y_val_orig.index, y_val_orig, color='orange', label='Val Obs', linewidth=2)
    plt.plot(y_test_orig.index, y_test_orig, color='green', label='Test Obs', linewidth=2)

    # Previsioni
    plt.plot(y_val_orig.index, val_pred_orig, '--', color='red', label='Val Forecast', linewidth=2.5)
    plt.plot(y_test_orig.index, test_pred_orig, '--', color='purple', label='Test Forecast', linewidth=2.5)
    plt.plot(future_pred_orig.index, future_pred_orig, '-.', color='black', label='Future Forecast', linewidth=2.5)

    # Split verticale tra storico e futuro
    plt.axvline(y_train_orig.index[-1], color="black", linestyle=":", label="Split Train/Val")
    plt.axvline(y_val_orig.index[-1], color="black", linestyle=":", label="Split Val/Test")
    plt.axvline(y_test_orig.index[-1], color="black", linestyle=":", label="Split Test/Future")

    plt.title(f"{col_name} - Original Scale")
    plt.legend(loc='upper left', fontsize=10)
    plt.grid(True)

    # --- PROCESSED SCALE ---
    plt.subplot(2,1,2)

    if len(y_val) > 0:
        plt.axvspan(y_val.index[0], y_val.index[-1], color='orange', alpha=alpha_zone)
    if len(y_test) > 0:
        plt.axvspan(y_test.index[0], y_test.index[-1], color='green', alpha=alpha_zone)
    if len(future_pred_proc) > 0:
        plt.axvspan(future_pred_proc.index[0], future_pred_proc.index[-1], color='red', alpha=alpha_zone)

    plt.plot(y_train.index, y_train, color='blue', label='Train Obs (proc)', linewidth=2)
    plt.plot(y_val.index, y_val, color='orange', label='Val Obs (proc)', linewidth=2)
    plt.plot(y_test.index, y_test, color='green', label='Test Obs (proc)', linewidth=2)

    plt.plot(y_val.index, val_pred, '--', color='red', label='Val Forecast (proc)', linewidth=2.5)
    plt.plot(y_test.index, test_pred, '--', color='purple', label='Test Forecast (proc)', linewidth=2.5)
    plt.plot(future_pred_proc.index, future_pred_proc, '-.', color='black', label='Future Forecast (proc)', linewidth=2.5)

    plt.axvline(y_train.index[-1], color="black", linestyle=":", label="Split Train/Val")
    plt.axvline(y_val.index[-1], color="black", linestyle=":", label="Split Val/Test")
    plt.axvline(y_test.index[-1], color="black", linestyle=":", label="Split Test/Future")

    plt.title(f"{col_name} - Processed Scale")
    plt.legend(loc='upper left', fontsize=10)
    plt.grid(True)

    plt.tight_layout()
    plt.show()


    # Restituisci i risultati del modello e le previsioni
    return res_full, val_pred_orig, test_pred_orig