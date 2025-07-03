# 🍽️ Demand Forecasting for Restaurant

**Demand Forecasting for Restaurant** is a machine learning project aimed at predicting daily demand—specifically, the number of guests and daily revenue—for a restaurant. Developed for the Operational Analytics university course, this project leverages both statistical and machine learning models on real-world historical data.

---

## 🚀 Overview

This project forecasts:
- **Daily number of guests**
- **Daily revenue** (daily closure amount)

It utilizes several models, including:
- **SARIMAX** (statistical time series)
- **MLP** (Multi-Layer Perceptron, neural network)
- **XGBoost** (tree ensemble)

All models are automatically evaluated, compared, and documented in a Markdown report.

---


## 📁 Repository Structure

```text
demand-forecasting-restaurant/
├── algorithms/               # Model implementations (SARIMAX, MLP, XGBoost)
│   ├── SARIMAX.py
│   ├── mlp_torch.py
│   └── XGBoost.py
├── dataset/                  # Original CSV dataset
│   └── chiusure_di_giornata_autentiko_beach_estate_2024.csv
├── dataset_manipulation/     # Preprocessing and feature engineering
│   └── preprocessing.py
├── gui/                      # PyQt5 interactive application
│   └── launcher.py
├── benchmark.py              # Report generation (Markdown)
├── batch_report.py           # Batch pipeline for automatic reporting
├── utils.py                  # Utility functions (metrics, tables, etc.)
├── requirements.txt          # Python dependencies
├── README.md                 # This file!
```

---

## ⚙️ Installation

**Requirements:**
- Python 3.9+

**1. Clone the repository**

```bash
git clone https://github.com/LudovicoNollino/demand-forecasting-restaurant.git
cd demand-forecasting-restaurant
```

**2. (Recommended) Create a virtual environment**

```bash
python -m venv venv
# On Linux/Mac:
source venv/bin/activate
# On Windows:
venv\Scripts\activate
```

**3. Install the dependencies**

```bash
pip install -r requirements.txt
```

---

## 🖥️ Usage

### Interactive GUI

Launch the PyQt5 GUI for step-by-step analysis and diagnostics:

```bash
python gui/launcher.py
```

**Choose the time series:**
- `Numero ospiti` = Number of guests
- `Chiusura di giornata` = Daily closure

Select the model to test (SARIMAX, MLP, XGBoost), view diagnostics, results, plots, and comparisons. Generate the full report with the dedicated button.

### Batch Pipeline & Automatic Report

To run the entire pipeline on both time series and generate a complete report:

```bash
python batch_report.py
```

This will create `global_report.md` with a full comparison of all models and series.

---

## 📊 Models & Preprocessing

Each model uses an optimized preprocessing pipeline:

- **SARIMAX:** raw data, no transformation
- **MLP:** standardization, Box-Cox/Kalman (depending on series)
- **XGBoost:** Box-Cox and Kalman configuration, engineered features for weekday/weekend

Model performance is evaluated using:

- **RMSE** (Root Mean Squared Error)
- **MAE** (Mean Absolute Error)
- **MAPE** (Mean Absolute Percentage Error)

All results, metrics, and comparisons are saved in Markdown (`global_report.md`) for easy sharing and consultation.

---

## 🧑‍💻 Credits

Ludovico Nollino  
University Project – Operational Analytics, 2024  
