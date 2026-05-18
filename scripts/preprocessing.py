import pandas as pd

# LOAD DATA
df = pd.read_csv('data/inflation_raw.csv')

# UBAH NAMA KOLOM
df.columns = [
    'Year',
    'Country',
    'Code',
    'Currency',
    'Region',
    'Decade',
    'Inflation',
    'CPI',
    'Exchange_Rate',
    'Interest_Rate'
]

# PILIH KOLOM PENTING
df = df[
    [
        'Year',
        'CPI',
        'Exchange_Rate',
        'Interest_Rate',
        'Inflation'
    ]
]

# HAPUS NULL
df = df.dropna()

# HAPUS DUPLIKAT
df = df.drop_duplicates()

# SAVE CLEAN DATA
df.to_csv(
    'data/inflation_clean.csv',
    index=False
)

print(df.head())

print("PREPROCESSING BERHASIL")