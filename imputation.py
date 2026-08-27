import pandas as pd
import numpy as np

print("Cargando el dataset maestro...")
# 1. Cargar el dataset maestro diario
df = pd.read_csv('SIMA_Diario_2020_2025.csv')
df['Fecha'] = pd.to_datetime(df['Fecha'])

# Definir las variables ambientales continuas (excepto lluvia)
vars_contaminantes_clima = ['CO', 'NO', 'NO2', 'NOX', 'O3', 'PM10', 'PM2.5', 'PRS', 'RH', 'SO2', 'SR', 'TOUT', 'WDR', 'WSR']

# 2. Función de imputación temporal
def imputar_series_tiempo(grupo):
    # Asegurar el orden cronológico estricto
    grupo = grupo.sort_values('Fecha')
    
    # Establecer la fecha como índice (requerido para interpolación 'time')
    grupo = grupo.set_index('Fecha')
    
    # A) Interpolar variables de contaminantes y clima basado en la distancia de los días
    grupo[vars_contaminantes_clima] = grupo[vars_contaminantes_clima].interpolate(method='time')
    
    # B) Si quedan nulos al principio o final extremo, usar el valor más cercano (forward-fill y backward-fill)
    grupo[vars_contaminantes_clima] = grupo[vars_contaminantes_clima].ffill().bfill()
    
    # C) Para la lluvia (RAINF), rellenamos los nulos con 0 (asumiendo que no hubo precipitación)
    if 'RAINF' in grupo.columns:
        grupo['RAINF'] = grupo['RAINF'].fillna(0)
        
    return grupo.reset_index()

print("Aplicando imputación cronológica independiente por estación...")
# Aplicar la función agrupando por estación para evitar Data Leakage espacial
df_imputado = df.groupby('Estacion', group_keys=False).apply(imputar_series_tiempo)

# Ordenar el dataset final por Estación y Fecha para que quede estético
df_imputado = df_imputado.sort_values(['Estacion', 'Fecha']).reset_index(drop=True)

# 3. Guardar el dataset imputado y final
archivo_final = 'SIMA_Diario_Imputado.csv'
df_imputado.to_csv(archivo_final, index=False)

# Validación final
nulos_restantes_pm10 = df_imputado['PM10'].isna().sum()
nulos_totales = df_imputado[vars_contaminantes_clima].isna().sum().sum()

print(f"¡Proceso completado! Archivo guardado como '{archivo_final}'")
print(f"Nulos restantes en PM10: {nulos_restantes_pm10}")
print(f"Nulos totales en el dataset: {nulos_totales}")