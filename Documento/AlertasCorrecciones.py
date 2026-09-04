import os
import numpy as np
import pandas as pd

def aplicar_filtros_fisicos(df: pd.DataFrame) -> pd.DataFrame:
    df_clean = df.copy()
    
    # 1. Identificar valores fuera de los límites físicos esperados para Monterrey
    # TOUT: Rango histórico realista entre -5°C y 45°C
    mask_tout = (df_clean['TOUT'] < -15.0) | (df_clean['TOUT'] > 45.0)
    
    # WSR: Rango de viento promedio diario realista (0 a 100 km/h)
    mask_wsr = (df_clean['WSR'] < 0.0) | (df_clean['WSR'] > 100.0)
    
    # 2. Reemplazar anomalías con NaN
    df_clean.loc[mask_tout, 'TOUT'] = np.nan
    df_clean.loc[mask_wsr, 'WSR'] = np.nan
    
    # 3. Imputar usando el último valor válido (ffill) por cada estación individual
    df_clean['TOUT'] = df_clean.groupby('Estacion')['TOUT'].ffill().bfill()
    df_clean['WSR'] = df_clean.groupby('Estacion')['WSR'].ffill().bfill()
    
    anomalias_tout = mask_tout.sum()
    anomalias_wsr = mask_wsr.sum()
    
    print(f"Filtro aplicado: Se corrigieron {anomalias_tout} errores de Temperatura.")
    print(f"Filtro aplicado: Se corrigieron {anomalias_wsr} errores de Viento.")
    
    return df_clean

def main():
    # Cargar datos
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    path_in = os.path.join(SCRIPT_DIR, "..", "BasesDeDatosParquet", "SIMA_Diario_Imputado.parquet")
    df = pd.read_parquet(path_in)
    
    # Aplicar limpieza física
    df_filtrado = aplicar_filtros_fisicos(df)
    
    # Sobrescribir o guardar como nueva versión
    df_filtrado.to_parquet(path_in.replace("Imputado", "Limpio"), index=False)
    print("Datos listos para análisis espacial.")

if __name__ == "__main__":
    main()