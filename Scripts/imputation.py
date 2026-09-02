import pandas as pd
import numpy as np

df = pd.read_csv('SIMA_Diario_2020_2025.csv')
df['Fecha'] = pd.to_datetime(df['Fecha'])

# Seleccionar exclusivamente variables numéricas para evitar errores TypeError con texto
cols_numericas = df.select_dtypes(include=[np.number]).columns
df[cols_numericas] = df[cols_numericas].mask(df[cols_numericas] < -100, np.nan)

vars_positivas = ['CO', 'NO', 'NO2', 'NOX', 'O3', 'PM10', 'PM2.5', 'RH', 'SO2', 'SR', 'WSR', 'RAINF']
for col in vars_positivas:
    if col in df.columns:
        df.loc[df[col] < 0, col] = np.nan

vars_contaminantes_clima = ['CO', 'NO', 'NO2', 'NOX', 'O3', 'PM10', 'PM2.5', 'PRS', 'RH', 'SO2', 'SR', 'TOUT', 'WDR', 'WSR']

# Función de imputación
def imputar_series_tiempo(grupo):
    # 1. Capturar el identificador de la estación desde los metadatos del grupo
    nombre_estacion = grupo.name 
    
    grupo = grupo.sort_values('Fecha').set_index('Fecha')
    # Interpolación respetando la distancia de tiempo
    grupo[vars_contaminantes_clima] = grupo[vars_contaminantes_clima].interpolate(method='time')
    # Relleno de extremos sin datos
    grupo[vars_contaminantes_clima] = grupo[vars_contaminantes_clima].ffill().bfill()
    # La lluvia nula
    if 'RAINF' in grupo.columns:
        grupo['RAINF'] = grupo['RAINF'].fillna(0)
        
    grupo = grupo.reset_index()
    # 2. Reinyectar la columna de la estación explícitamente
    grupo['Estacion'] = nombre_estacion
    
    return grupo

df_imputado = df.groupby('Estacion', group_keys=False).apply(imputar_series_tiempo)
df_imputado = df_imputado.sort_values(['Estacion', 'Fecha']).reset_index(drop=True)

df_imputado.to_parquet('SIMA_Diario_Imputado.parquet', engine='pyarrow', index=False)