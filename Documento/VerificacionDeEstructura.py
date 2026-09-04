import pandas as pd
import numpy as np

def cargar_y_verificar_estructura(filepath: str) -> tuple[pd.DataFrame, dict]:
    # 1. Carga del conjunto de datos
    if filepath.endswith('.parquet'):
        df = pd.read_parquet(filepath)
    elif filepath.endswith('.csv'):
        df = pd.read_csv(filepath)
    else:
        raise ValueError("Formato no soportado. Debe ser .parquet o .csv")

    # Asegurar formato datetime en la columna Fecha
    df['Fecha'] = pd.to_datetime(df['Fecha'])
    
    # 2. Resumen de dimensiones y memoria
    info_general = {
        'total_registros': len(df),
        'total_columnas': len(df.columns),
        'estaciones_unicas': df['Estacion'].nunique() if 'Estacion' in df.columns else 0,
        'memoria_mb': round(df.memory_usage(deep=True).sum() / (1024 ** 2), 2)
    }
    
    # 3. Verificación de continuidad temporal por estación
    continuidad_estaciones = {}
    if 'Estacion' in df.columns:
        for estacion, group in df.groupby('Estacion'):
            fecha_min = group['Fecha'].min()
            fecha_max = group['Fecha'].max()
            dias_esperados = (fecha_max - fecha_min).days + 1
            dias_reales = group['Fecha'].nunique()
            dias_faltantes = dias_esperados - dias_reales
            
            continuidad_estaciones[estacion] = {
                'fecha_inicio': fecha_min.strftime('%Y-%m-%d'),
                'fecha_fin': fecha_max.strftime('%Y-%m-%d'),
                'dias_esperados': dias_esperados,
                'dias_registrados': dias_reales,
                'dias_faltantes': dias_faltantes,
                'continuo': dias_faltantes == 0
            }

    # 4. Conteo de valores nulos y tipos de datos
    reporte_nulos = pd.DataFrame({
        'Tipo_Dato': df.dtypes,
        'Valores_Nulos': df.isnull().sum(),
        'Porcentaje_Nulos (%)': (df.isnull().sum() / len(df) * 100).round(2)
    })

    # 5. Rango y coherencia física para variables principales
    rangos_esperados = {
        'PM10': (0, 600),
        'PM2.5': (0, 500),
        'TOUT': (-10, 50),   # Temperatura en °C
        'RH': (0, 100),       # Humedad Relativa %
        'WSR': (0, 100),      # Velocidad de viento km/h
        'PRS': (800, 1100)    # Presión atmosférica
    }
    
    alertas_fisiscas = []
    for col, (vmin, vmax) in rangos_esperados.items():
        if col in df.columns:
            fuerade_rango = df[(df[col] < vmin) | (df[col] > vmax)][col].count()
            if fuerade_rango > 0:
                alertas_fisiscas.append({
                    'Variable': col,
                    'Min_Detectado': df[col].min(),
                    'Max_Detectado': df[col].max(),
                    'Valores_Anomalos': fuerade_rango
                })

    reporte_completo = {
        'info_general': info_general,
        'continuidad': pd.DataFrame(continuidad_estaciones).T,
        'calidad_columnas': reporte_nulos,
        'alertas_rangos': pd.DataFrame(alertas_fisiscas)
    }

    return df, reporte_completo


# Ejemplo de ejecución
if __name__ == "__main__":
    df, reporte = cargar_y_verificar_estructura('Documento\\SIMA_Diario_Imputado.parquet')

    print("=== INFORMACIÓN GENERAL ===")
    print(reporte['info_general'])

    print("\n=== VERIFICACIÓN DE CONTINUIDAD TEMPORAL ===")
    print(reporte['continuidad'])

    print("\n=== ESTADO DE VALORES NULOS Y TIPOS DE DATO ===")
    print(reporte['calidad_columnas'])

    print("\n=== ALERTAS DE COHERENCIA FÍSICA ===")
    print(reporte['alertas_rangos'] if not reporte['alertas_rangos'].empty else "Sin anomalías detectadas.")