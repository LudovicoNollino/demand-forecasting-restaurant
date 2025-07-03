import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX
from dataset_manipulation.preprocessing import inverse_transform_predictions_forecast
import warnings

warnings.filterwarnings("ignore")

def sarimax_grid_search(
    data_dict,
    seasonal_period=7,
    max_p=2, max_d=1, max_q=2,
    max_P=1, max_D=1, max_Q=1,
    verbose=True
):
    print(f"Grid Search SARIMAX - Periodo stagionale: {seasonal_period}")
    y_train = data_dict['train']
    exog = data_dict.get('features', None)
    if exog is not None:
        exog_train = exog.iloc[:len(y_train)]
    else:
        exog_train = None

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
                                    enforce_invertibility=False,
                                    exog=exog_train
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
    future_steps=30,
    exog=None
):
    y_train, y_val, y_test = data_dict['train'], data_dict['val'], data_dict['test']
    y_train_orig, y_val_orig, y_test_orig = data_dict['train_orig'], data_dict['val_orig'], data_dict['test_orig']
    preprocess_params = data_dict['preprocess_params']

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

    # Esogene
    if exog is not None:
        indices = y_train.index.tolist() + y_val.index.tolist() + y_test.index.tolist()
        exog = exog.loc[indices]
        exog_train = exog.loc[y_train.index]
        exog_val = exog.loc[y_val.index]
        exog_test = exog.loc[y_test.index]
        exog_trainval = pd.concat([exog_train, exog_val])
        exog_full = pd.concat([exog_train, exog_val, exog_test])
    else:
        exog_train = exog_val = exog_test = exog_trainval = exog_full = None

    # 1. Fit on train → validation
    model_train = SARIMAX(y_train, order=order, seasonal_order=seasonal_order, 
                          enforce_stationarity=False, enforce_invertibility=False,
                          exog=exog_train)
    res_train = model_train.fit(disp=False)
    val_pred = res_train.get_forecast(steps=len(y_val), exog=exog_val).predicted_mean
    val_pred_orig = maybe_inverse(val_pred)

    # 2. Fit on train+validation → test
    y_trainval = pd.concat([y_train, y_val])
    model_tv = SARIMAX(y_trainval, order=order, seasonal_order=seasonal_order,
                       enforce_stationarity=False, enforce_invertibility=False,
                       exog=exog_trainval)
    res_tv = model_tv.fit(disp=False)
    test_pred = res_tv.get_forecast(steps=len(y_test), exog=exog_test).predicted_mean
    test_pred_orig = maybe_inverse(test_pred)

    # 3. Fit on all data → future forecast
    y_full = pd.concat([y_train, y_val, y_test])
    model_full = SARIMAX(y_full, order=order, seasonal_order=seasonal_order,
                         enforce_stationarity=False, enforce_invertibility=False,
                         exog=exog_full)
    res_full = model_full.fit(disp=False)
    if exog_full is not None:
        # Prepare future exogenous data
        # Use the last row of exog_full repeated for future_steps
        last_exog = exog_full.iloc[[-1]]
        exog_future = pd.concat([last_exog]*future_steps, ignore_index=True)
    else:
        exog_future = None
    
    # Forecast future values using the exogenous data used above
    future_pred_proc = res_full.get_forecast(steps=future_steps, exog=exog_future).predicted_mean
    future_pred_orig = maybe_inverse(future_pred_proc)

    test_idx = data_dict["test_orig"].index
    last_idx = test_idx[-1]
    if isinstance(last_idx, pd.Timestamp):
        future_idx = pd.date_range(start=last_idx + pd.Timedelta(days=1), periods=future_steps)
    else:
        future_idx = np.arange(last_idx + 1, last_idx + 1 + future_steps)

    val_pred_orig = pd.Series(np.array(val_pred_orig), index=y_val_orig.index)
    test_pred_orig = pd.Series(np.array(test_pred_orig), index=y_test_orig.index)
    future_pred_orig = pd.Series(np.array(future_pred_orig), index=future_idx)

    from plotter import plot_forecasting
    plot_forecasting(
        col_name=col_name,
        y_train_orig=y_train_orig,
        y_val_orig=y_val_orig,
        y_test_orig=y_test_orig,
        val_pred_orig=val_pred_orig,
        test_pred_orig=test_pred_orig,
        future_pred_orig=future_pred_orig
    )

    return res_full, val_pred_orig, test_pred_orig
