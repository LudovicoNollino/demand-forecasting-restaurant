# DEMAND FORECASTING RESULTS
**Generated on:** 2025-07-03 02:44:39



---

## Target: **Numero ospiti**
- **Train:** 70
- **Validation:** 15
- **Test:** 11
- **Total:** 96

| Set | Model | RMSE | MAE | MAPE (%) |
|-----|-------|------|-----|----------|
| Validation | SARIMAX | 40.9213 | 31.2812 | 27.04 |
| Validation | MLP | 16.9113 | 13.8393 | 13.04 |
| Validation | XGBoost | 0.0145 | 0.0128 | 0.01 |
| Test | SARIMAX | 41.9760 | 31.7692 | 42.16 |
| Test | MLP | 20.2067 | 12.7269 | 27.35 |
| Test | XGBoost | 4.9502 | 1.5088 | 0.67 |

**Best model (test RMSE):** XGBoost with RMSE=4.9502


---

## Target: **Chiusura di giornata (scalata in un intervallo)**
- **Train:** 70
- **Validation:** 15
- **Test:** 11
- **Total:** 96

| Set | Model | RMSE | MAE | MAPE (%) |
|-----|-------|------|-----|----------|
| Validation | SARIMAX | 436.0000 | 301.4236 | 18.67 |
| Validation | MLP | 383.0043 | 336.3021 | 22.22 |
| Validation | XGBoost | 156.2661 | 117.0151 | 7.47 |
| Test | SARIMAX | 657.8805 | 601.5173 | 51.25 |
| Test | MLP | 498.2479 | 375.7080 | 40.72 |
| Test | XGBoost | 148.3120 | 107.6900 | 7.13 |

**Best model (test RMSE):** XGBoost with RMSE=148.3120
