# DEMAND FORECASTING RESULTS


---

## Target: **Numero ospiti**
- **Train:** 70
- **Validation:** 15
- **Test:** 11
- **Total:** 96

| Set | Model | RMSE | MAE | MAPE (%) |
|-----|-------|------|-----|----------|
| Validation | SARIMAX | 40.9213 | 31.2812 | 27.04 |
| Validation | MLP | 13.2022 | 11.0545 | 10.29 |
| Validation | XGBoost | 8.1450 | 6.5827 | 5.95 |
| Test | SARIMAX | 41.9760 | 31.7692 | 42.16 |
| Test | MLP | 18.4358 | 13.1702 | 24.65 |
| Test | XGBoost | 31.0155 | 21.3559 | 19.12 |

**Best model (test RMSE):** MLP with RMSE=18.4358


---

## Target: **Chiusura di giornata (scalata in un intervallo)**
- **Train:** 70
- **Validation:** 15
- **Test:** 11
- **Total:** 96

| Set | Model | RMSE | MAE | MAPE (%) |
|-----|-------|------|-----|----------|
| Validation | SARIMAX | 436.0000 | 301.4236 | 18.67 |
| Validation | MLP | 559.3948 | 442.7350 | 27.25 |
| Validation | XGBoost | 156.2661 | 117.0151 | 7.47 |
| Test | SARIMAX | 657.8805 | 601.5173 | 51.25 |
| Test | MLP | 683.8828 | 658.2415 | 67.54 |
| Test | XGBoost | 148.3120 | 107.6900 | 7.13 |

**Best model (test RMSE):** XGBoost with RMSE=148.3120
