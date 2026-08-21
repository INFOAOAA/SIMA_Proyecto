import pandas as pd

data = pd.read_csv('BD_2022_test_imputed.csv', sep = ';')
data_prev = pd.read_csv('BD 2022.csv',sep=';')
print(data.isna().sum())

def medidas_centrales(df: pd.DataFrame):
    print(
    'Media',df.mean(numeric_only=True, skipna=True),
    'Mediana',df.median(numeric_only=True,skipna= True))
    max = df.max(numeric_only=True, skipna=True)
    min = df.min(numeric_only=True, skipna=True)
    print('Rango medio',(max-min)/2)

medidas_centrales(data_prev)
medidas_centrales(data)