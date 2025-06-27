import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QLabel, QPushButton,
    QComboBox, QCheckBox, QMessageBox, QSizePolicy
)
import pandas as pd
from dataset_manipulation.preprocessing import preprocess_column, inverse_boxcox_transform, apply_kalman_filter
from algorithms.SARIMAX import sarimax_grid_search, fit_sarimax_model
from algorithms.mlp_torch import fit_mlp_model
from algorithms.XGBoost import fit_xgboost_model, xgboost_grid_search
from statsmodels.tsa.stattools import adfuller
from scipy.stats import shapiro
from sklearn.preprocessing import StandardScaler

class Launcher(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Operational Analytics Launcher")
        self.setGeometry(100, 100, 520, 400)
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout()
        self.col_label = QLabel("Scegli la colonna da analizzare:")
        self.col_combo = QComboBox()
        self.col_combo.addItems([
            "Numero ospiti",
            "Chiusura di giornata (scalata in un intervallo)"
        ])
        self.boxcox_cb = QCheckBox("Applica Box-Cox")
        self.kalman_cb = QCheckBox("Applica Kalman smoothing")
        self.std_cb = QCheckBox("Applica Standardization")
        self.alg_label = QLabel("Scegli l'algoritmo:")
        self.alg_combo = QComboBox()
        self.alg_combo.addItems(["SARIMAX", "XGBoost", "MLP"])
        self.run_btn = QPushButton("Lancia Analisi")
        self.run_btn.clicked.connect(self.run_analysis)

        # QLabel per diagnostica
        self.diagnostica_label = QLabel("")
        self.diagnostica_label.setWordWrap(True)
        self.diagnostica_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.diagnostica_label.setStyleSheet("font-size: 13px; color: #114488;")

        layout.addWidget(self.col_label)
        layout.addWidget(self.col_combo)
        layout.addWidget(self.boxcox_cb)
        layout.addWidget(self.kalman_cb)
        layout.addWidget(self.std_cb)
        layout.addWidget(self.alg_label)
        layout.addWidget(self.alg_combo)
        layout.addWidget(self.run_btn)
        layout.addWidget(QLabel("Diagnostica statistica (step by step):"))
        layout.addWidget(self.diagnostica_label)
        self.setLayout(layout)

    def run_analysis(self):
        col = self.col_combo.currentText()
        apply_boxcox = self.boxcox_cb.isChecked()
        apply_kalman = self.kalman_cb.isChecked()
        apply_standardization = self.std_cb.isChecked()
        algorithm = self.alg_combo.currentText()

        # Carica il dataset
        dataset_path = os.path.abspath(
            os.path.join(os.path.dirname(__file__), '..', 'dataset', 'chiusure_di giornata_autentiko_beach_estate_2024.csv')
        )
        df = pd.read_csv(dataset_path, sep=';')
        df.columns = df.columns.str.strip()
        df['Data di chiusura'] = pd.to_datetime(df['Data di chiusura'], dayfirst=True, errors='raise')
        df = df.sort_values('Data di chiusura').reset_index(drop=True)
        df['day_of_week'] = df['Data di chiusura'].dt.day_name()
        df['is_weekend']  = df['Data di chiusura'].dt.weekday >= 5
        df = df.dropna(subset=['Numero ospiti', 'Chiusura di giornata (scalata in un intervallo)'])

        # Diagnostica step by step
        diagnostica_text = self.diagnostica_multistep(df, col, apply_boxcox, apply_kalman, apply_standardization)

        # --- PREPROCESSING (e grafici) ---
        import matplotlib.pyplot as plt
        proc, params = preprocess_column(
            df, col,
            apply_boxcox=apply_boxcox,
            apply_kalman=apply_kalman,
            apply_standardization=apply_standardization,
            plot=True
        )

        reply = QMessageBox.question(
            self,
            "Preprocessing completato",
            "Preprocessing completato e grafici mostrati.\nVuoi procedere con la modellazione?",
            QMessageBox.Yes | QMessageBox.No
        )
        if reply != QMessageBox.Yes:
            self.diagnostica_label.setText(diagnostica_text)
            return

        # --- MODELLAZIONE ---
        n = len(proc)
        n_train = int(n * 0.70)
        n_val = int(n * 0.15)
        n_test = n - n_train - n_val
        train = proc.iloc[:n_train].copy()
        val = proc.iloc[n_train:n_train + n_val].copy()
        test = proc.iloc[n_train + n_val:].copy()
        orig = df.loc[proc.index, col]
        train_orig = orig.iloc[:n_train]
        val_orig = orig.iloc[n_train:n_train + n_val]
        test_orig = orig.iloc[n_train + n_val:]
        data_dict = {
            "train": train,
            "val": val,
            "test": test,
            "train_orig": train_orig,
            "val_orig": val_orig,
            "test_orig": test_orig,
            "preprocess_params": params
        }

        if algorithm == "SARIMAX":
            print("Sto lanciando SARIMAX")
            best_order, best_seasonal, _ = sarimax_grid_search(
                data_dict, seasonal_period=7, max_p=2, max_d=1, max_q=2, max_P=1, max_D=1, max_Q=1
            )
            fit_sarimax_model(
                data_dict,
                order=best_order,
                seasonal_order=best_seasonal,
                col_name=col
            )
            # Diagnostica differenziazione
            d = best_order[1]
            D = best_seasonal[1]
            s = best_seasonal[3]
            original_series = df[col].dropna()
            diagnostica_text += "<br><hr><br>" + self.diagnostica_post_diff(
                original_series, d=d, D=D, s=s, col_name=col
            )
            self.diagnostica_label.setText(diagnostica_text)
            QMessageBox.information(self, "SARIMAX", "Analisi SARIMAX completata!")

        elif algorithm == "MLP":
            print("Sto lanciando MLP")
            model, best_params, grid, val_pred, test_pred, future_pred = fit_mlp_model(
                data_dict,
                window_size=7,
                hidden_dim1=16, hidden_dim2=8,
                lr=0.001, activation='relu',
                n_epochs=300,
                batch_size=1,
                print_every=20,
                future_steps=30,
                grid_search=True,  # Attiva grid search!
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
                verbose=False  # Setta True per stampa dettagliata!
            )
            # Mostra risultati della grid search in HTML
            grid_html = ""
            if grid and len(grid) > 0:
                grid_html = "<br><b>MLP Grid Search (primi 5 parametri testati):</b><br><table border='1' cellpadding='3'><tr>"
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
                grid_html = "<br><b>Nessuna grid search eseguita (parametri fissi).</b>"

            self.diagnostica_label.setText(
                diagnostica_text +
                grid_html +
                f"<br><hr><br>Best MLP params: {best_params}"
            )
            QMessageBox.information(self, "MLP", "Analisi MLP completata!")
        elif algorithm == "XGBoost":
            print("Sto lanciando XGBoost")
            # Grid search
            grid_results, best_config, best_model = xgboost_grid_search(
                data_dict,
                look_back_grid=[7, 14, 21, 30, 45, 60],
                n_estimators_grid=[10, 15, 25, 35, 40, 50, 75, 100],
            )
            print("\n--- XGBoost grid search results ---")
            print(grid_results.head())

            # Fit finale sul best config (come nel main)
            model_xgb, val_pred_xgb, test_pred_xgb, future_pred_xgb = fit_xgboost_model(
                data_dict,
                look_back=best_config[0],
                n_estimators=best_config[1],
                col_name=col,
            )

            # Mostra i risultati nella GUI (puoi personalizzare!)
            grid_html = ""
            if grid_results is not None and not grid_results.empty:
                grid_html = "<br><b>XGBoost Grid Search (primi 5 parametri testati):</b><br><table border='1' cellpadding='3'><tr>"
                for k in grid_results.columns:
                    grid_html += f"<th>{k}</th>"
                grid_html += "</tr>"
                for _, row in grid_results.head().iterrows():
                    grid_html += "<tr>"
                    for v in row.values:
                        grid_html += f"<td>{v:.4f}" if isinstance(v, float) else f"<td>{v}"
                        grid_html += "</td>"
                    grid_html += "</tr>"
                grid_html += "</table>"
            else:
                grid_html = "<br><b>Nessuna grid search eseguita (parametri fissi).</b>"

            self.diagnostica_label.setText(
                diagnostica_text +
                grid_html +
                f"<br><hr><br>Best XGBoost config: {best_config}<br>"
                f"Validation predictions shape: {val_pred_xgb.shape}<br>"
                f"Test predictions shape: {test_pred_xgb.shape}<br>"
                f"Future forecast shape: {future_pred_xgb.shape}"
            )
            QMessageBox.information(self, "XGBoost", "Analisi XGBoost completata!")

        else:
            self.diagnostica_label.setText("Algoritmo non valido.")
            QMessageBox.warning(self, "Errore", "Algoritmo non valido.")


    @staticmethod
    def diagnostica_multistep(df, col, apply_boxcox, apply_kalman, apply_standardization):
        def stat_txt(label, series):
            adf_p = adfuller(series)[1]
            adf_msg = "stazionaria" if adf_p < 0.05 else "NON stazionaria"
            shapiro_p = shapiro(series)[1]
            shapiro_msg = "normale" if shapiro_p >= 0.05 else "NON normale"
            return (f"<b>{label}</b>:<br>"
                    f"ADF p-value: <b>{adf_p:.4f}</b> &rarr; {adf_msg}<br>"
                    f"Shapiro p-value: <b>{shapiro_p:.4f}</b> &rarr; {shapiro_msg}<br>")

        # Step iniziale
        orig = df[col].dropna()
        out = stat_txt("Serie originale", orig)

        # Box-Cox (se richiesto)
        shift = 0
        if apply_boxcox:
            from scipy.stats import boxcox
            minval = orig.min()
            if minval <= 0:
                shift = -minval + 1e-6
            bc, lmbda = boxcox(orig + shift)
            orig = pd.Series(bc, index=orig.index)
            out += stat_txt("Dopo Box-Cox", orig)

        # Kalman (se richiesto)
        if apply_kalman:
            orig = apply_kalman_filter(orig)
            out += stat_txt("Dopo Kalman", orig)

        # Standardization (se richiesto)
        if apply_standardization:
            scaler = StandardScaler()
            arr = scaler.fit_transform(orig.values.reshape(-1,1)).flatten()
            orig = pd.Series(arr, index=orig.index)
            out += stat_txt("Dopo Standardizzazione", orig)

        return out

    @staticmethod
    def diagnostica_post_diff(series, d, D, s, col_name="Serie"):
        diffed = series.diff(d) if d > 0 else series.copy()
        if D > 0:
            diffed = diffed.diff(s * D)
        diffed = diffed.dropna()
        adf_p = adfuller(diffed)[1]
        adf_msg = "stazionaria" if adf_p < 0.05 else "NON stazionaria"
        shapiro_p = shapiro(diffed)[1]
        shapiro_msg = "normale" if shapiro_p >= 0.05 else "NON normale"
        out = (
            f"<b>Diagnostica dopo differenziazione d={d}, D={D}, s={s}</b>:<br>"
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
