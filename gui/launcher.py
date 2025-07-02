import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QLabel, QPushButton,
    QComboBox, QMessageBox, QSizePolicy
)
import pandas as pd
import numpy as np
from dataset_manipulation.preprocessing import preprocess_column
from algorithms.SARIMAX import sarimax_grid_search, fit_sarimax_model
from algorithms.mlp_torch import fit_mlp_model
from algorithms.XGBoost import fit_xgboost_model, xgboost_grid_search
from statsmodels.tsa.stattools import adfuller
from scipy.stats import shapiro
from utils import get_metrics_table

class Launcher(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Launcher")
        self.setGeometry(100, 100, 520, 400)
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout()
        self.col_label = QLabel("Choose the column to analyze:")
        self.col_combo = QComboBox()
        self.col_combo.addItems([
            "Numero ospiti",
            "Chiusura di giornata (scalata in un intervallo)"
        ])
        self.alg_label = QLabel("Choose the algorithm:")
        self.alg_combo = QComboBox()
        self.alg_combo.addItems(["SARIMAX", "XGBoost", "MLP"])
        self.run_btn = QPushButton("Run Analysis")
        self.run_btn.clicked.connect(self.run_analysis)

        self.diagnostica_label = QLabel("")
        self.diagnostica_label.setWordWrap(True)
        self.diagnostica_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.diagnostica_label.setStyleSheet("font-size: 13px; color: #114488;")

        layout.addWidget(self.col_label)
        layout.addWidget(self.col_combo)
        layout.addWidget(self.alg_label)
        layout.addWidget(self.alg_combo)
        layout.addWidget(self.run_btn)
        layout.addWidget(QLabel("Step by Step Diagnostics:"))
        layout.addWidget(self.diagnostica_label)
        self.setLayout(layout)

    def get_best_preprocessing(self, col, algorithm):
        
        if col == "Chiusura di giornata (scalata in un intervallo)":
            if algorithm == "MLP":
                return dict(apply_boxcox=False, apply_kalman=False, apply_standardization=True)
            elif algorithm == "SARIMAX":
                return dict(apply_boxcox=False, apply_kalman=False, apply_standardization=False)
            elif algorithm == "XGBoost":
                return dict(apply_boxcox=True, apply_kalman=True, apply_standardization=False)
        elif col == "Numero ospiti":
            if algorithm == "MLP":
                return dict(apply_boxcox=False, apply_kalman=False, apply_standardization=True)
            elif algorithm == "SARIMAX":
                return dict(apply_boxcox=False, apply_kalman=False, apply_standardization=False)
            elif algorithm == "XGBoost":
                return dict(apply_boxcox=False, apply_kalman=False, apply_standardization=True)
        
        return dict(apply_boxcox=False, apply_kalman=False, apply_standardization=False)

    def run_analysis(self):
        col = self.col_combo.currentText()
        algorithm = self.alg_combo.currentText()

        dataset_path = os.path.abspath(
            os.path.join(os.path.dirname(__file__), '..', 'dataset', 'chiusure_di giornata_autentiko_beach_estate_2024.csv')
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

        # --- PREPROCESSING CONFIG ---
        best_preproc = self.get_best_preprocessing(col, algorithm)
        
        orig = df[col].dropna()
        adf_p = adfuller(orig)[1]
        adf_msg = "stationary" if adf_p < 0.05 else "NOT stationary"
        shapiro_p = shapiro(orig)[1]
        shapiro_msg = "normal" if shapiro_p >= 0.05 else "NOT normal"
        diagnostica_text = (
            f"<b>Original series:</b><br>"
            f"ADF p-value: <b>{adf_p:.4f}</b> &rarr; {adf_msg}<br>"
            f"Shapiro p-value: <b>{shapiro_p:.4f}</b> &rarr; {shapiro_msg}<br>"
        )

        proc, params = preprocess_column(
            df, col,
            apply_boxcox=best_preproc['apply_boxcox'],
            apply_kalman=best_preproc['apply_kalman'],
            apply_standardization=best_preproc['apply_standardization'],
            plot=True
        )
        
        adf_p2 = adfuller(proc)[1]
        adf_msg2 = "stationary" if adf_p2 < 0.05 else "NOT stationary"
        shapiro_p2 = shapiro(proc)[1]
        shapiro_msg2 = "normal" if shapiro_p2 >= 0.05 else "NOT normal"
        diagnostica_text += (
            "<hr><b>Preprocessed series:</b><br>"
            f"ADF p-value: <b>{adf_p2:.4f}</b> &rarr; {adf_msg2}<br>"
            f"Shapiro p-value: <b>{shapiro_p2:.4f}</b> &rarr; {shapiro_msg2}<br>"
            f"<br><i>Preprocessing applied: BoxCox={best_preproc['apply_boxcox']}, "
            f"Kalman={best_preproc['apply_kalman']}, Standard={best_preproc['apply_standardization']}</i>"
        )

        feature_cols = [c for c in df.columns if c.startswith('dow_')] + ['is_weekend']
        features = df.loc[proc.index, feature_cols].copy()
        features = features.astype(np.float32)

        reply = QMessageBox.question(
            self,
            "Preprocessing completed",
            f"Preprocessing completed, \nproceed with modeling?",
            QMessageBox.Yes | QMessageBox.No
        )
        if reply != QMessageBox.Yes:
            self.diagnostica_label.setText(diagnostica_text)
            return

        # --- MODELING ---
        n = len(proc)
        n_train = int(n * 0.70)
        n_val = int(n * 0.15)
        n_test = n - n_train - n_val
        train = proc.iloc[:n_train].copy()
        val = proc.iloc[n_train:n_train + n_val].copy()
        test = proc.iloc[n_train + n_val:].copy()
        orig = df.loc[proc.index, col]
        train_orig = orig.iloc[:n_train].copy()
        val_orig = orig.iloc[n_train:n_train + n_val].copy()
        test_orig = orig.iloc[n_train + n_val:].copy()
        
        print("MEDIA test:", np.mean(test_orig))
        print("DEVIAZIONE STANDARD test:", np.std(test_orig))

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

        if algorithm == "SARIMAX":
            print("Launching SARIMAX")
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
            # Calcola la tabella metriche unica
            metrics_html = get_metrics_table(
                data_dict["val_orig"][val_pred.index].values, val_pred.values,
                data_dict["test_orig"][test_pred.index].values, test_pred.values
            )
            # Opzionale: differencing diagnostico come avevi già
            d = best_order[1]
            D = best_seasonal[1]
            s = best_seasonal[3]
            original_series = df[col].dropna()
            diagnostica_text += "<br><hr><br>" + self.diagnostica_post_diff(
                original_series, d=d, D=D, s=s, col_name=col
            )
            self.diagnostica_label.setText(diagnostica_text + metrics_html)
            QMessageBox.information(self, "SARIMAX", "SARIMAX analysis completed!")

        elif algorithm == "MLP":
            print("Launching MLP")
            model, best_params, grid, val_pred, test_pred, future_pred = fit_mlp_model(
                data_dict,
                window_size=7,
                hidden_dim1=16, hidden_dim2=8,
                lr=0.001, activation='relu',
                n_epochs=300,
                batch_size=1,
                print_every=20,
                future_steps=30,
                grid_search=True,
                grid_params={
                    'hidden1_grid': [8],
                    'hidden2_grid': [8],
                    'lr_grid': [0.0005],
                    'activations': ['relu'],
                    'n_epochs_grid': [300],
                    'batch_size': 1,
                    'print_every': 100
                },
                col_name=col,
                verbose=False
            )
            grid_html = ""
            if grid and len(grid) > 0:
                grid_html = "<br><b>MLP Grid Search (first 5 tested parameters):</b><br><table border='1' cellpadding='3'><tr>"
                for k in grid[0].keys():
                    grid_html += f"<th>{k}</th>"
                grid_html += "</tr>"
                for row in grid[:5]:
                    grid_html += "<tr>"
                    for v in row.values():
                        grid_html += f"<td>{v}</td>"
                    grid_html += "</tr>"
                grid_html += "</table>"
            else:
                grid_html = "<br><b>No grid search performed (fixed parameters).</b>"
            
            metrics_html = get_metrics_table(
                data_dict["val_orig"][val_pred.index].values, val_pred.values,
                data_dict["test_orig"][test_pred.index].values, test_pred.values
            )
            self.diagnostica_label.setText(
                diagnostica_text +
                grid_html +
                f"<br><hr><br>Best MLP params: {best_params}" +
                metrics_html
            )
            QMessageBox.information(self, "MLP", "MLP analysis completed!")

        elif algorithm == "XGBoost":
            print("Launching XGBoost")
            best_config = xgboost_grid_search(
                data_dict,
                look_back_grid=[60],
                n_estimators_grid=[40],
            )
            model_xgb, val_pred_xgb, test_pred_xgb, future_pred_xgb = fit_xgboost_model(
                data_dict,
                look_back=best_config[0],
                n_estimators=best_config[1],
                col_name=col,
            )
            # grid_html = ""
            # if grid_results is not None and not grid_results.empty:
            #     grid_html = "<br><b>XGBoost Grid Search (first 5 tested parameters):</b><br><table border='1' cellpadding='3'><tr>"
            #     for k in grid_results.columns:
            #         grid_html += f"<th>{k}</th>"
            #     grid_html += "</tr>"
            #     for _, row in grid_results.head().iterrows():
            #         grid_html += "<tr>"
            #         for v in row.values:
            #             grid_html += f"<td>{v:.4f}" if isinstance(v, float) else f"<td>{v}"
            #             grid_html += "</td>"
            #         grid_html += "</tr>"
            #     grid_html += "</table>"
            # else:
            #     grid_html = "<br><b>No grid search performed (fixed parameters).</b>"
            
            metrics_html = get_metrics_table(
                data_dict["val_orig"][val_pred_xgb.index].values, val_pred_xgb.values,
                data_dict["test_orig"][test_pred_xgb.index].values, test_pred_xgb.values
            )

            self.diagnostica_label.setText(
                diagnostica_text +
                # grid_html +
                f"<br><hr><br>Best XGBoost config: {best_config}<br>"
                f"Validation predictions shape: {val_pred_xgb.shape}<br>"
                f"Test predictions shape: {test_pred_xgb.shape}<br>"
                f"Future forecast shape: {future_pred_xgb.shape}" +
                metrics_html
            )
            QMessageBox.information(self, "XGBoost", "XGBoost analysis completed!")

        else:
            self.diagnostica_label.setText("Invalid algorithm.")
            QMessageBox.warning(self, "Error", "Invalid algorithm.")

    @staticmethod
    def diagnostica_post_diff(series, d, D, s, col_name):
        diffed = series.diff(d) if d > 0 else series.copy()
        if D > 0:
            diffed = diffed.diff(s * D)
        diffed = diffed.dropna()
        adf_p = adfuller(diffed)[1]
        adf_msg = "stationary" if adf_p < 0.05 else "NOT stationary"
        shapiro_p = shapiro(diffed)[1]
        shapiro_msg = "normal" if shapiro_p >= 0.05 else "NOT normal"
        out = (
            f"<b>Diagnostic after differencing d={d}, D={D}, s={s}</b>:<br>"
            f"ADF p-value: <b>{adf_p:.4f}</b> &rarr; {adf_msg}<br>"
            f"Shapiro p-value: <b>{shapiro_p:.4f}</b> &rarr; {shapiro_msg}"
        )
        print(out.replace('<br>', '\n'))
        return out

if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = Launcher()
    win.show()
    sys.exit(app.exec_())
