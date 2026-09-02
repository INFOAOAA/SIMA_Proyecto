import pandas as pd
import numpy as np
import os

# 1. Definir los años, variables ambientales y carpeta
years = [2020, 2021, 2022, 2023, 2024, 2025]
carpeta = 'BasesDeDatosXlsx'
dfs = []

vars_mean = ['CO', 'NO', 'NO2', 'NOX', 'O3', 'PM10', 'PM2.5', 'PRS', 'RH', 'SO2', 'SR', 'TOUT', 'WDR', 'WSR']
vars_sum = ['RAINF']
all_vars = vars_mean + vars_sum

for yr in years:
    # Ajuste de ruta para buscar dentro de la carpeta
    file_path = os.path.join(carpeta, f"BD {yr}.xlsx")
    xls = pd.ExcelFile(file_path)
    
    for sheet in xls.sheet_names:
        df = pd.read_excel(xls, sheet_name=sheet)
        
        date_col = next((c for c in df.columns if 'fecha' in str(c).lower() or 'date' in str(c).lower()), df.columns[0])
        df = df.rename(columns={date_col: 'Fecha_Hora'})
        df['Fecha_Hora'] = pd.to_datetime(df['Fecha_Hora'], errors='coerce')
        df = df.dropna(subset=['Fecha_Hora'])
        
        df['Estacion'] = sheet.strip()
        
        for col in all_vars:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            else:
                df[col] = np.nan
                
        dfs.append(df[['Fecha_Hora', 'Estacion'] + all_vars])

# 2. Concatenación total
df_horario = pd.concat(dfs, ignore_index=True)
df_horario['Fecha'] = df_horario['Fecha_Hora'].dt.date

# 3. Agregación diaria (Regla >= 18 horas válidas)
def agregar_diario_sima(group, min_horas=18):
    res = {'Horas_Registradas': len(group)}
    
    for v in vars_mean:
        valid_cnt = group[v].count()
        res[v] = group[v].mean() if valid_cnt >= min_horas else np.nan
        
    for v in vars_sum:
        valid_cnt = group[v].count()
        res[v] = group[v].sum(min_count=min_horas) if valid_cnt >= min_horas else np.nan
        
    return pd.Series(res)

df_diario = df_horario.groupby(['Estacion', 'Fecha']).apply(agregar_diario_sima).reset_index()
df_diario['Fecha'] = pd.to_datetime(df_diario['Fecha'])

# 4. Guardar archivo final
df_diario.to_csv('SIMA_Diario_2020_2025.csv', index=False)
