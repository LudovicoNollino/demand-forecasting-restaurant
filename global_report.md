# DEMAND FORECASTING RESULTS


---

## Target: **Numero ospiti**
- **Train:** 70
- **Validation:** 15
- **Test:** 15
- **Total:** 100

| Set | Model | RMSE | MAE | MAPE (%) |
|-----|-------|------|-----|----------|
| Validation | SARIMAX | 42.3663 | 31.2011 | 31.55 |
| Validation | MLP | 33.5126 | 22.6202 | 22.83 |
| Validation | XGBoost | 16.5990 | 11.7899 | 12.31 |
| Test | SARIMAX | 36.5730 | 26.9732 | 34.24 |
| Test | MLP | 44.4923 | 32.9533 | 35.55 |
| Test | XGBoost | 32.5464 | 21.9571 | 20.21 |

**Best model (test RMSE):** XGBoost with RMSE=32.5464


---

## Target: **Chiusura di giornata (scalata in un intervallo)**
- **Train:** 70
- **Validation:** 15
- **Test:** 15
- **Total:** 100

| Set | Model | RMSE | MAE | MAPE (%) |
|-----|-------|------|-----|----------|
| Validation | SARIMAX | 303.1121 | 209.7522 | 14.53 |
| Validation | MLP | 430.3266 | 352.4747 | 26.55 |
| Validation | XGBoost | 220.3090 | 172.0674 | 12.71 |
| Test | SARIMAX | 465.0945 | 370.6395 | 29.21 |
| Test | MLP | 381.8148 | 341.4259 | 30.34 |
| Test | XGBoost | 137.8145 | 102.2806 | 6.71 |

**Best model (test RMSE):** XGBoost with RMSE=137.8145
