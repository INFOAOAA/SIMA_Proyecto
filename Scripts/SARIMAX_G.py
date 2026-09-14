# ==============================================================================
# 0. INSTALACIÓN DE LIBRERÍAS (Ejecutar primera celda en Google Colab con GPU)
# ==============================================================================
# quita nombre
import itertools
import os

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
from statsmodels.tsa.seasonal import seasonal_decompose

# Importación de cuML con fallback informativo
try:
    from cuml.tsa.arima import ARIMA as cumlARIMA
    HAS_CUML = True
    print("✅ NVIDIA cuML detectado. La aceleración GPU está lista.")
except ImportError:
    HAS_CUML = False
    print("⚠️ cuML no disponible. Asegúrate de estar en un entorno GPU e instalar cuml-cu12.")


# ==============================================================================
# 1. FUNCIÓN DE BÚSQUEDA GRIDSEARCH EN GPU (cuML)
# ==============================================================================
def gpu_sarimax_gridsearch(
    y_train: pd.Series,
    X_train: pd.DataFrame = None,
    seasonal_period: int = 7,
    max_p: int = 2,
    max_q: int = 2,
    max_P: int = 1,
    max_Q: int = 1,
    d_list: list = [0, 1],
    D_list: list = [0, 1]
):
    """Ejecuta la búsqueda en grilla (GridSearch) en GPU con cuML de forma robusta."""
    if not HAS_CUML:
        raise RuntimeError("Se requiere NVIDIA cuML para ejecutar el GridSearch en GPU.")

    # 1. Generar espacio de hiperparámetros
    p_vals = list(range(0, max_p + 1))
    q_vals = list(range(0, max_q + 1))
    P_vals = list(range(0, max_P + 1))
    Q_vals = list(range(0, max_Q + 1))

    grid = list(itertools.product(p_vals, d_list, q_vals, P_vals, D_list, Q_vals))
    num_combos = len(grid)
    print(f"🚀 Evaluando {num_combos} combinaciones SARIMAX en la GPU...")

    # Formatear datos a float32 para la GPU
    y_32 = y_train.values.astype(np.float32)
    X_32 = X_train.values.astype(np.float32) if X_train is not None else None

    best_aic = float("inf")
    best_order = None
    best_seasonal_order = None

    # Iterar sobre las combinaciones usando tuplas explícitas en cuML
    for g in grid:
        order_tuple = (int(g[0]), int(g[1]), int(g[2]))
        seasonal_tuple = (int(g[3]), int(g[4]), int(g[5]), int(seasonal_period))

        try:
            model = cumlARIMA(
                y_32,
                exog=X_32,
                order=order_tuple,
                seasonal_order=seasonal_tuple,
                fit_intercept=True
            )
            model_fit = model.fit()
            
            # Obtener AIC (extraer valor escalar si viene como array)
            current_aic = model_fit.aic
            if isinstance(current_aic, (np.ndarray, list)):
                current_aic = current_aic[0]

            if np.isfinite(current_aic) and current_aic < best_aic:
                best_aic = current_aic
                best_order = order_tuple
                best_seasonal_order = seasonal_tuple

        except Exception:
            # Ignora combinaciones no convergentes
            continue

    if best_order is None:
        raise RuntimeError("Ninguna combinación convergió en la GPU. Revisa los datos de entrada.")

    print(f"✅ Mejor Orden SARIMAX: {best_order}")
    print(f"✅ Mejor Orden Estacional: {best_seasonal_order}")
    print(f"✅ Menor AIC Obtenido: {best_aic:.3f}")

    return best_order, best_seasonal_order

# ==============================================================================
# 2. PIPELINE DE ENTRENAMIENTO Y EVALUACIÓN SARIMAX
# ==============================================================================
def fit_and_evaluate_daily_sarimax(
    df: pd.DataFrame,
    target_col: str = "PM10",
    date_col: str = "ds",
    exog_cols: list[str] = None,
    seasonal_period: int = 7,
    station: str = "SE3",
    test_days: int = 7,
    output_dir: str = "./models"
):
    if exog_cols is None:
        exog_cols = []

    # 1. Limpieza y Agregación Diaria
    df_clean = df.copy()
    df_clean[date_col] = pd.to_datetime(df_clean[date_col])

    cols_to_select = [target_col] + exog_cols
    df_daily = (
        df_clean.set_index(date_col)[cols_to_select]
        .resample("D")
        .mean()
        .dropna()
    )

    y_daily = df_daily[target_col]
    X_daily = df_daily[exog_cols] if exog_cols else None

    print(f"\n=== Dataset Diario Preparado: {len(y_daily)} días ({station} - {target_col}) ===")

    # 2. Descomposición de la serie
    decomp = seasonal_decompose(y_daily, model="additive", period=seasonal_period)

    # 3. División Train / Test
    train_df = df_daily.iloc[:-test_days]
    test_df = df_daily.iloc[-test_days:]

    y_train = train_df[target_col]
    X_train = train_df[exog_cols] if exog_cols else None
    
    y_test = test_df[target_col]
    X_test = test_df[exog_cols] if exog_cols else None

    # 4. Búsqueda de Hiperparámetros con GPU
    best_order, best_seasonal_order = gpu_sarimax_gridsearch(
        y_train=y_train,
        X_train=X_train,
        seasonal_period=seasonal_period,
        max_p=2,
        max_q=2,
        max_P=1,
        max_Q=1
    )

    # 5. Evaluación Fuera de Muestra (Out-Of-Sample) con cuML
    y_train_32 = y_train.values.astype(np.float32)
    X_train_32 = X_train.values.astype(np.float32) if X_train is not None else None
    X_test_32 = X_test.values.astype(np.float32) if X_test is not None else None

    eval_model = cumlARIMA(
        y_train_32,
        exog=X_train_32,
        order=best_order,
        seasonal_order=best_seasonal_order
    ).fit()

    predictions = eval_model.forecast(steps=test_days, exog=X_test_32)

    # Métricas de error
    mae = mean_absolute_error(y_test, predictions)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    mape = np.mean(np.abs((y_test.values - predictions) / y_test.values)) * 100

    print("\n==================================================================")
    print(f"       EVALUACIÓN HOLDOUT ({test_days} DÍAS) - {station}         ")
    print("==================================================================")
    print(f"MAE:  {mae:.3f} µg/m³")
    print(f"RMSE: {rmse:.3f} µg/m³")
    print(f"MAPE: {mape:.2f}%")

    # 6. Modelo Final de Producción (Full Dataset)
    print("\n=== Ajustando Modelo Final con Dataset Completo en GPU ===")
    y_full_32 = y_daily.values.astype(np.float32)
    X_full_32 = X_daily.values.astype(np.float32) if X_daily is not None else None

    final_model = cumlARIMA(
        y_full_32,
        exog=X_full_32,
        order=best_order,
        seasonal_order=best_seasonal_order
    ).fit()

    # 7. Guardado de Artefactos (Compatible con Colab)
    os.makedirs(output_dir, exist_ok=True)
    model_path = os.path.join(output_dir, f"{station}_sarimax_{target_col}.cuml")
    
    # Guardar paquete ligero con los mejores hiperparámetros e historial
    artifact = {
        "station": station,
        "target": target_col,
        "order": best_order,
        "seasonal_order": best_seasonal_order,
        "metrics": {"MAE": mae, "RMSE": rmse, "MAPE": mape}
    }
    pd.to_pickle(artifact, model_path)
    print(f"✅ Artefacto guardado exitosamente en: {model_path}")

    eval_df = pd.DataFrame(
        {"Actual": y_test, "Forecast": predictions}, index=test_df.index
    )

    return final_model, df_daily, decomp, eval_df, artifact["metrics"]


# ==============================================================================
# 3. EJECUCIÓN (Adaptada a Google Colab / Entorno local)
# ==============================================================================
import os
import re

import pandas as pd


def run_pipeline(data_path: str):
    if not os.path.exists(data_path):
        print(f"⚠️ El archivo no fue encontrado en: {data_path}")
        return

    data = pd.read_parquet(data_path)
    filename = os.path.basename(data_path)

    # 1. Extraer la estación desde el nombre del archivo (ej. NE3, NO2, SE3)
    station_match = re.search(r'(SE3|NO2|NE3|CE|NLE|Obispado)', filename, re.IGNORECASE)
    station_from_file = station_match.group(1).upper() if station_match else "Estacion"

    # 2. Identificar la columna objetivo (Target)
    target_col = "PM10"
    if target_col not in data.columns:
        # Buscar alternativas comunes si PM10 no está
        candidates = [c for c in data.columns if "PM10" in c.upper() or "TARGET" in c.upper()]
        if candidates:
            target_col = candidates[0]
        else:
            raise KeyError(f"No se encontró la columna objetivo '{target_col}' en el dataset.")

    # 3. Detectar variables exógenas automáticamente (excluyendo Fecha, Estacion y Target)
    ignore_cols = ["ds", "Date", "Estacion", target_col]
    exog_cols = [c for c in data.columns if c not in ignore_cols]

    print(f"📊 Dataset cargado: {filename}")
    print(f"📌 Estación identificada: {station_from_file}")
    print(f"🎯 Variable Objetivo: {target_col}")
    print(f"🧪 Variables Exógenas ({len(exog_cols)}): {exog_cols}")

    # 4. Si existe la columna 'Estacion' dentro del DataFrame, procesamos por grupo o filtro
    if 'Estacion' in data.columns:
        stations = data['Estacion'].unique()
        for st in stations:
            st_df = data[data['Estacion'] == st].copy()
            fit_and_evaluate_daily_sarimax(
                st_df,
                target_col=target_col,
                exog_cols=exog_cols,
                station=st
            )
    else:
        # Si NO existe la columna 'Estacion', el DataFrame ya está filtrado por estación
        fit_and_evaluate_daily_sarimax(
            data,
            target_col=target_col,
            exog_cols=exog_cols,
            station=station_from_file
        )

# Para ejecutar en Colab:
# 1. Asegúrate de seleccionar el entorno de ejecución: Entorno de ejecución > Cambiar tipo > T4 GPU
# 2. Sube tus Parquets o monta Google Drive
if __name__ == "__main__":
    # 1. Directorio base en Google Drive donde están tus carpetas y archivos Parquet
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

    # 2. Lista de datasets Parquet a procesar en el ciclo
    datasets = [
        "factor_analysis_NE3_factor_scores.parquet",
        "factor_analysis_NO2_factor_scores.parquet",
        "factor_analysis_SE3_factor_scores.parquet"
    ]

    # 3. Ciclo para ejecutar la canalización en cada dataset
    print(f"\n================ STARTING BATCH PROCESSING ({len(datasets)} DATASETS) ================")
    
    for filename in datasets:
        file_path = os.path.join(SCRIPT_DIR,"..", 'BasesDeDatosParquet', filename)
        
        print(f"\n------------------------------------------------------------------")
        print(f"🔄 Processing dataset: {filename}")
        print(f"📍 Full Path: {file_path}")
        print(f"------------------------------------------------------------------")

        if os.path.exists(file_path):
            try:
                run_pipeline(file_path)
                print(f"✅ Successfully completed pipeline for: {filename}")
            except Exception as e:
                print(f"❌ Error processing {filename}: {e}")
        else:
            print(f"⚠️ Warning: File not found at '{file_path}'. Skipping...")

    print("\n================ ALL DATASETS PROCESSED ================")
