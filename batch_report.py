import os
import pandas as pd
import numpy as np
# import matplotlib
# matplotlib.use('Agg') 
# import matplotlib.pyplot as plt
# plt.show = lambda *args, **kwargs: None  
from dataset_manipulation.preprocessing import preprocess_column
from algorithms.SARIMAX import sarimax_grid_search, fit_sarimax_model
from algorithms.mlp_torch import fit_mlp_model
from algorithms.XGBoost import fit_xgboost_model, xgboost_grid_search
from benchmark import create_global_report

def run_full_report():
    best_preproc_dict = {
        "Numero ospiti": {
            "MLP": dict(apply_boxcox=True, apply_kalman=True, apply_standardization=True),
            "SARIMAX": dict(apply_boxcox=False, apply_kalman=False, apply_standardization=False),
            "XGBoost": dict(apply_boxcox=True, apply_kalman=True, apply_standardization=False)
        },
        "Chiusura di giornata (scalata in un intervallo)": {
            "MLP": dict(apply_boxcox=False, apply_kalman=False, apply_standardization=True),
            "SARIMAX": dict(apply_boxcox=False, apply_kalman=False, apply_standardization=False),
            "XGBoost": dict(apply_boxcox=True, apply_kalman=True, apply_standardization=False)
        }
    }

    dataset_path = os.path.abspath(
        os.path.join(os.path.dirname(__file__), 'dataset', 'chiusure_di giornata_autentiko_beach_estate_2024.csv')
    )
    df = pd.read_csv(dataset_path, sep=';')
    df.columns = df.columns.str.strip()
    df['Data di chiusura'] = pd.to_datetime(df['Data di chiusura'], dayfirst=True, errors='raise')
    df = df.sort_values('Data di chiusura').reset_index(drop=True)
    df = df.drop_duplicates(subset='Data di chiusura', keep='first')
    df = df.set_index('Data di chiusura')
    df['is_weekend'] = df.index.weekday >= 5
    df['day_of_week'] = df.index.weekday
    df = pd.get_dummies(df, columns=['day_of_week'], prefix='dow')
    feature_cols = [c for c in df.columns if c.startswith('dow_')] + ['is_weekend']
    df = df.dropna(subset=['Numero ospiti', 'Chiusura di giornata (scalata in un intervallo)'])

    all_reports = []

    for col in ["Numero ospiti", "Chiusura di giornata (scalata in un intervallo)"]:
        print(f"\n===== ELABORATING: {col} =====")

        res_full = val_pred = test_pred = None
        model_mlp = val_pred_mlp = test_pred_mlp = None
        model_xgb = val_pred_xgb = test_pred_xgb = grid_results = best_config = None

        orig = df[col].dropna()
        n = len(orig)
        n_train = int(n * 0.70)
        n_val = int(n * 0.15)
        n_test = n - n_train - n_val

        features_all = df[feature_cols].astype(np.float32)

        # === SARIMAX ===
        preproc = best_preproc_dict[col]["SARIMAX"]
        proc, params = preprocess_column(
            df, col,
            apply_boxcox=preproc['apply_boxcox'],
            apply_kalman=preproc['apply_kalman'],
            apply_standardization=preproc['apply_standardization'],
            plot=False
        )
        train = proc.iloc[:n_train].copy()
        val = proc.iloc[n_train:n_train + n_val].copy()
        test = proc.iloc[n_train + n_val:].copy()
        orig = df.loc[proc.index, col]
        train_orig = orig.iloc[:n_train].copy()
        val_orig = orig.iloc[n_train:n_train + n_val].copy()
        test_orig = orig.iloc[n_train + n_val:].copy()
        features = features_all.loc[proc.index, :]
        data_dict = {
            "train": train,
            "val": val,
            "test": test,
            "train_orig": train_orig,
            "val_orig": val_orig,
            "test_orig": test_orig,
            "preprocess_params": params,
            "features": features
        }
        best_order, best_seasonal, _ = sarimax_grid_search(
            data_dict, seasonal_period=7, max_p=2, max_d=1, max_q=2, max_P=1, max_D=1, max_Q=1
        )
        res_full, val_pred, test_pred = fit_sarimax_model(
            data_dict,
            order=best_order,
            seasonal_order=best_seasonal,
            col_name=col,
            exog=features
        )

        # === MLP ===
        preproc = best_preproc_dict[col]["MLP"]
        proc, params = preprocess_column(
            df, col,
            apply_boxcox=preproc['apply_boxcox'],
            apply_kalman=preproc['apply_kalman'],
            apply_standardization=preproc['apply_standardization'],
            plot=False
        )
        train = proc.iloc[:n_train].copy()
        val = proc.iloc[n_train:n_train + n_val].copy()
        test = proc.iloc[n_train + n_val:].copy()
        orig = df.loc[proc.index, col]
        train_orig = orig.iloc[:n_train].copy()
        val_orig = orig.iloc[n_train:n_train + n_val].copy()
        test_orig = orig.iloc[n_train + n_val:].copy()
        features = features_all.loc[proc.index, :]
        data_dict_mlp = {
            "train": train,
            "val": val,
            "test": test,
            "train_orig": train_orig,
            "val_orig": val_orig,
            "test_orig": test_orig,
            "preprocess_params": params,
            "features": features
        }
        model_mlp, best_params, grid, val_pred_mlp, test_pred_mlp, future_pred_mlp = fit_mlp_model(
            data_dict_mlp,
            window_size=7,
            hidden_dim1=16, hidden_dim2=8,
            lr=0.001, activation='relu',
            n_epochs=300,
            batch_size=1,
            print_every=20,
            future_steps=30,
            grid_search=False,
            col_name=col,
            verbose=False
        )

        # === XGBoost ===
        preproc = best_preproc_dict[col]["XGBoost"]
        proc, params = preprocess_column(
            df, col,
            apply_boxcox=preproc['apply_boxcox'],
            apply_kalman=preproc['apply_kalman'],
            apply_standardization=preproc['apply_standardization'],
            plot=False
        )
        train = proc.iloc[:n_train].copy()
        val = proc.iloc[n_train:n_train + n_val].copy()
        test = proc.iloc[n_train + n_val:].copy()
        orig = df.loc[proc.index, col]
        train_orig = orig.iloc[:n_train].copy()
        val_orig = orig.iloc[n_train:n_train + n_val].copy()
        test_orig = orig.iloc[n_train + n_val:].copy()
        features = features_all.loc[proc.index, :]
        data_dict_xgb = {
            "train": train,
            "val": val,
            "test": test,
            "train_orig": train_orig,
            "val_orig": val_orig,
            "test_orig": test_orig,
            "preprocess_params": params,
            "features": features
        }
        grid_results, best_config, best_model = xgboost_grid_search(
            data_dict_xgb,
            look_back_grid=[60],
            n_estimators_grid=[40],
        )
        model_xgb, val_pred_xgb, test_pred_xgb, future_pred_xgb = fit_xgboost_model(
            data_dict_xgb,
            look_back=best_config[0],
            n_estimators=best_config[1],
            col_name=col,
        )

        # ACCUMULO RISULTATI
        all_reports.append({
            "target": col,
            "data_dict": data_dict,
            "res_full": res_full,
            "val_pred": val_pred,
            "test_pred": test_pred,
            "model_mlp": model_mlp,
            "val_pred_mlp": val_pred_mlp,
            "test_pred_mlp": test_pred_mlp,
            "model_xgb": model_xgb,
            "val_pred_xgb": val_pred_xgb,
            "test_pred_xgb": test_pred_xgb,
            "xgb_results_df": grid_results,
            "best_xgb_config": {"look_back": best_config[0], "n_estimators": best_config[1]}
        })

    # === GLOBAL REPORT ===
    create_global_report(all_reports)
    print("\nProcess finished. Report saved!")

