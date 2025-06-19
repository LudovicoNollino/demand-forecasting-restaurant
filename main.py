import pandas as pd
from dataset_manipulation.preprocessing import preprocess_column

df = pd.read_csv(r"C:\Users\USER\Desktop\Magistrale\Final_Project_Operational_Analytics\demand-forecasting-restaurant\chiusure_di giornata_autentiko_beach_estate_2024.csv", sep=';')
df.columns = df.columns.str.strip()
df['Data di chiusura'] = pd.to_datetime(df['Data di chiusura'], dayfirst=True, errors='raise')
df = df.sort_values('Data di chiusura').reset_index(drop=True)

# 2. Time features
df['day_of_week'] = df['Data di chiusura'].dt.day_name()
df['is_weekend']  = df['Data di chiusura'].dt.weekday >= 5

# 3. Drop missing
df = df.dropna(subset=['Numero ospiti', 'Chiusura di giornata (scalata in un intervallo)'])

# 4–10. Process each series
df['Numero ospiti']  = preprocess_column(df, 'Numero ospiti')
df['Chiusura di giornata (scalata in un intervallo)'] = preprocess_column(df, 'Chiusura di giornata (scalata in un intervallo)')

# 11. Save
df.to_csv('./processed_data.csv', index=False)
