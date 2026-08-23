import pandas as pd
import os

path = os.path.abspath('BasesDeDatosCsv/Limpia/BD_Completa.csv')
datos = pd.read_csv(path)
print(datos.isna().sum())