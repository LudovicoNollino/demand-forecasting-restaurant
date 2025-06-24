import pandas as pd
from dataset_manipulation.preprocessing import preprocess_column

df = pd.read_csv(r"demand-forecasting-restaurant\dataset\chiusure_di giornata_autentiko_beach_estate_2024.csv", sep=';')
df.columns = df.columns.str.strip()
df['Data di chiusura'] = pd.to_datetime(df['Data di chiusura'], dayfirst=True, errors='raise')
df = df.sort_values('Data di chiusura').reset_index(drop=True)

# 2. Time features
df['day_of_week'] = df['Data di chiusura'].dt.day_name()
df['is_weekend']  = df['Data di chiusura'].dt.weekday >= 5

# 3. Drop missing
df = df.dropna(subset=['Numero ospiti', 'Chiusura di giornata (scalata in un intervallo)'])

# 4–10. Process each series
# Pipeline base, solo differenze
proc_guests, params_guests = preprocess_column(
    df, 'Numero ospiti', apply_boxcox=False, apply_kalman=False)

# Pipeline con Box-Cox
proc_guests_boxcox, _ = preprocess_column(
    df, 'Numero ospiti', apply_boxcox=True, apply_kalman=False)

# Pipeline con Kalman
proc_guests_kalman, _ = preprocess_column(
    df, 'Numero ospiti', apply_boxcox=False, apply_kalman=True)

# Pipeline con Box-Cox + Kalman
proc_guests_all, _ = preprocess_column(
    df, 'Numero ospiti', apply_boxcox=True, apply_kalman=True)

proc_closure, params_closure = preprocess_column(
    df, 'Chiusura di giornata (scalata in un intervallo)',
    apply_boxcox=False, apply_kalman=False)
# Pipeline con Box-Cox
proc_closure_boxcox, _ = preprocess_column(
    df, 'Chiusura di giornata (scalata in un intervallo)',
    apply_boxcox=True, apply_kalman=False)
# Pipeline con Kalman
proc_closure_kalman, _ = preprocess_column(
    df, 'Chiusura di giornata (scalata in un intervallo)',
    apply_boxcox=False, apply_kalman=True)
# Pipeline con Box-Cox + Kalman
proc_closure_all, _ = preprocess_column(
    df, 'Chiusura di giornata (scalata in un intervallo)',
    apply_boxcox=True, apply_kalman=True)


