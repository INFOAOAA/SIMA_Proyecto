# ==============================================================================
# PIPELINE SARIMAX ADAPTADO A CPU LOCAL (PC Ryzen 7730U, sin GPU NVIDIA/cuML)
# Sustituye cuML por statsmodels.tsa.statespace.sarimax.SARIMAX
# Granularidad: HORARIA (periodo estacional = 24h), holdout de última semana (7 horas)
# Datos de entrada: BasesDeDatosParquet/factor_analysis_*_factor_scores.parquet
# ==============================================================================

import os
import re
import itertools
import warnings

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.seasonal import seasonal_decompose

# ==============================================================================
# Parámetros de la búsqueda en grilla (ajustables en CPU)
# ==============================================================================
GRID_MAX_P = 2
GRID_MAX_Q = 2
GRID_MAX_P_SEAS = 1
GRID_MAX_Q_SEAS = 1
GRID_D = [0, 1]
GRID_D_SEAS = [0, 1]


def sarimax_gridsearch(
    y_train: pd.Series,
    X_train: pd.DataFrame = None,
    seasonal_period: int = 24,
    max_p: int = GRID_MAX_P,
    max_q: int = GRID_MAX_Q,
    max_P: int = GRID_MAX_P_SEAS,
    max_Q: int = GRID_MAX_Q_SEAS,
    d_list: list = None,
    D_list: list = None,
):
    """Búsqueda en grilla secuencial en CPU sobre statsmodels SARIMAX (criterio AIC)."""
    if d_list is None:
        d_list = GRID_D
    if D_list is None:
        D_list = GRID_D_SEAS

    p_vals = list(range(0, max_p + 1))
    q_vals = list(range(0, max_q + 1))
    P_vals = list(range(0, max_P + 1))
    Q_vals = list(range(0, max_Q + 1))

    grid = list(itertools.product(p_vals, d_list, q_vals, P_vals, D_list, Q_vals))
    num_combos = len(grid)
    print(f"Evaluando {num_combos} modelos SARIMAX en CPU...")

    best_aic = float("inf")
    best_order = None
    best_seasonal_order = None

    for i, g in enumerate(grid, 1):
        order_tuple = (int(g[0]), int(g[1]), int(g[2]))
        seasonal_tuple = (int(g[3]), int(g[4]), int(g[5]), int(seasonal_period))

        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model = SARIMAX(
                    y_train,
                    exog=X_train,
                    order=order_tuple,
                    seasonal_order=seasonal_tuple,
                    enforce_stationarity=True,
                    enforce_invertibility=True,
                )
                model_fit = model.fit(disp=False)
        except Exception:
            continue

        if not hasattr(model_fit, "aic") or not np.isfinite(model_fit.aic):
            continue

        if model_fit.aic < best_aic:
            best_aic = model_fit.aic
            best_order = order_tuple
            best_seasonal_order = seasonal_tuple

        if i % 25 == 0 or i == num_combos:
            print(f"  ... {i}/{num_combos} combinaciones evaluadas")

    if best_order is None:
        raise RuntimeError(
            "Ninguna combinación SARIMAX convergió. Revisa los datos de entrada."
        )

    print(f"Mejor orden SARIMAX: {best_order}")
    print(f"Mejor orden estacional: {best_seasonal_order}")
    print(f"Menor AIC: {best_aic:.3f}")

    return best_order, best_seasonal_order


# ==============================================================================
# PIPELINE DE ENTRENAMIENTO Y EVALUACIÓN SARIMAX (CPU / statsmodels) - HORARIO
# ==============================================================================
def fit_and_evaluate_hourly_sarimax(
    df: pd.DataFrame,
    target_col: str = "PM10",
    date_col: str = "ds",
    exog_cols: list[str] = None,
    seasonal_period: int = 24,
    station: str = "SE3",
    test_hours: int = 7,
    output_dir: str = "./models",
):
    if exog_cols is None:
        exog_cols = []

    # 1. Limpieza y preparación de la serie horaria
    df_clean = df.copy()
    df_clean[date_col] = pd.to_datetime(df_clean[date_col])

    cols_to_select = [target_col] + exog_cols
    df_hourly = (
        df_clean.set_index(date_col)[cols_to_select]
        .resample("h")
        .mean()
        .dropna()
    )

    y_hourly = df_hourly[target_col]
    X_hourly = df_hourly[exog_cols] if exog_cols else None

    print(f"\n=== Dataset Horario Preparado: {len(y_hourly)} registros ({station} - {target_col}) ===")

    # 2. Descomposición de la serie
    decomp = seasonal_decompose(y_hourly, model="additive", period=seasonal_period)

    # 3. División Train / Test (últimas `test_hours` horas)
    train_df = df_hourly.iloc[:-test_hours]
    test_df = df_hourly.iloc[-test_hours:]

    y_train = train_df[target_col]
    X_train = train_df[exog_cols] if exog_cols else None

    y_test = test_df[target_col]
    X_test = test_df[exog_cols] if exog_cols else None

    # 4. Búsqueda de hiperparámetros en CPU
    best_order, best_seasonal_order = sarimax_gridsearch(
        y_train=y_train,
        X_train=X_train,
        seasonal_period=seasonal_period,
        max_p=GRID_MAX_P,
        max_q=GRID_MAX_Q,
        max_P=GRID_MAX_P_SEAS,
        max_Q=GRID_MAX_Q_SEAS,
    )

    # 5. Evaluación fuera de muestra (out-of-sample) con statsmodels
    eval_model = SARIMAX(
        y_train,
        exog=X_train,
        order=best_order,
        seasonal_order=best_seasonal_order,
        enforce_stationarity=True,
        enforce_invertibility=True,
    ).fit(disp=False)

    forecasts = eval_model.get_forecast(steps=test_hours, exog=X_test)
    predictions = forecasts.predicted_mean

    # Métricas de error
    mae = mean_absolute_error(y_test, predictions)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    denom = np.where(y_test.values == 0, np.nan, y_test.values)
    mape = np.nanmean(np.abs((y_test.values - predictions.values) / denom)) * 100

    print("\n==================================================================")
    print(f"       EVALUACIÓN HOLDOUT ({test_hours} HORAS) - {station}         ")
    print("==================================================================")
    print(f"MAE:  {mae:.3f} µg/m³")
    print(f"RMSE: {rmse:.3f} µg/m³")
    print(f"MAPE: {mape:.2f}%")

    # 6. Modelo final de producción (full dataset)
    print(f"\n=== Ajustando Modelo Final con Dataset Completo ({station}) ===")
    final_model = SARIMAX(
        y_hourly,
        exog=X_hourly,
        order=best_order,
        seasonal_order=best_seasonal_order,
        enforce_stationarity=True,
        enforce_invertibility=True,
    ).fit(disp=False)

    # 7. Guardado de artefactos
    os.makedirs(output_dir, exist_ok=True)
    model_path = os.path.join(output_dir, f"{station}_sarimax_{target_col}.pkl")

    artifact = {
        "station": station,
        "target": target_col,
        "freq": "H",
        "order": best_order,
        "seasonal_order": best_seasonal_order,
        "metrics": {"MAE": mae, "RMSE": rmse, "MAPE": mape},
    }
    pd.to_pickle(artifact, model_path)
    print(f"Artefacto guardado exitosamente en: {model_path}")

    eval_df = pd.DataFrame(
        {"Actual": y_test, "Forecast": predictions}, index=test_df.index
    )

    return final_model, df_hourly, decomp, eval_df, artifact["metrics"]


# ==============================================================================
# EJECUCIÓN (adaptada a entorno local)
# ==============================================================================
def run_pipeline(data_path: str):
    if not os.path.exists(data_path):
        print(f"El archivo no fue encontrado en: {data_path}")
        return

    data = pd.read_parquet(data_path)
    filename = os.path.basename(data_path)

    # 1. Extraer la estación desde el nombre del archivo (ej. NE3, NO2, SE3)
    station_match = re.search(r"(SE3|NO2|NE3|CE|NLE|Obispado)", filename, re.IGNORECASE)
    station_from_file = station_match.group(1).upper() if station_match else "Estacion"

    # 2. Identificar la columna objetivo (Target)
    target_col = "PM10"
    if target_col not in data.columns:
        candidates = [c for c in data.columns if "PM10" in c.upper() or "TARGET" in c.upper()]
        if candidates:
            target_col = candidates[0]
        else:
            raise KeyError(f"No se encontró la columna objetivo '{target_col}' en el dataset.")

    # 3. Detectar variables exógenas automáticamente (excluyendo Fecha, Estación y Target)
    ignore_cols = ["ds", "Date", "Estacion", target_col]
    exog_cols = [c for c in data.columns if c not in ignore_cols]

    print(f"Dataset cargado: {filename}")
    print(f"Estación identificada: {station_from_file}")
    print(f"Variable Objetivo: {target_col}")
    print(f"Variables Exógenas ({len(exog_cols)}): {exog_cols}")

    # 4. Si existe la columna 'Estacion', procesamos por grupo; si no, el DataFrame
    #    ya está filtrado por estación
    if "Estacion" in data.columns:
        for st in data["Estacion"].unique():
            st_df = data[data["Estacion"] == st].copy()
            fit_and_evaluate_hourly_sarimax(
                st_df,
                target_col=target_col,
                exog_cols=exog_cols,
                station=st,
            )
    else:
        fit_and_evaluate_hourly_sarimax(
            data,
            target_col=target_col,
            exog_cols=exog_cols,
            station=station_from_file,
        )


if __name__ == "__main__":
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(SCRIPT_DIR, "..", "BasesDeDatosParquet")

    datasets = [
        "factor_analysis_NE3_factor_scores.parquet",
        "factor_analysis_NO2_factor_scores.parquet",
        "factor_analysis_SE3_factor_scores.parquet",
    ]

    print(f"\n================ STARTING BATCH PROCESSING ({len(datasets)} DATASETS) ================")

    for filename in datasets:
        file_path = os.path.join(data_dir, filename)
        print(f"\n------------------------------------------------------------------")
        print(f"Processing dataset: {filename}")
        print(f"Full Path: {file_path}")
        print(f"------------------------------------------------------------------")

        if os.path.exists(file_path):
            try:
                run_pipeline(file_path)
                print(f"Successfully completed pipeline for: {filename}")
            except Exception as e:
                print(f"Error processing {filename}: {e}")
        else:
            print(f"Warning: File not found at '{file_path}'. Skipping...")

    print("\n================ ALL DATASETS PROCESSED ================")