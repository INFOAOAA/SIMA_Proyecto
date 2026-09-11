# ==============================================================================
# PIPELINE SARIMAX ÓPTIMO (CPU / statsmodels)
# Usa las combinaciones (p,d,q) x (P,D,Q,s) ya encontradas en el gridsearch.
# Granularidad: HORARIA (periodo estacional = 24h), holdout de última semana (7 horas)
# Datos de entrada: BasesDeDatosParquet/factor_analysis_*_factor_scores.parquet
#
# Salidas:
#   - Evaluación holdout (MAE, RMSE, MAPE) en la escala original (µg/m³)
#   - Gráfica de residuales del modelo final con bandas de confianza
#   - Gráfica Actual vs Ajustado/Pronóstico con línea de regresión por estación
#   - Guardado del modelo final (formato ligero, recargable) + artefacto
#   - Pesos (betas) de cada regresor exógeno en un reporte de texto
#
# Nota: antes de ajustar, la variable objetivo PM10 se transforma con
# Yeo-Johnson (maneja valores negativos/cero). Las métricas y gráficas se
# devuelven a la escala original invertiendo la transformación.
# ==============================================================================

import os
import re
import pickle
import warnings

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import PowerTransformer
from statsmodels.tsa.statespace.sarimax import SARIMAX

# ==============================================================================
# MEJORES COMBINACIONES ENCONTRADAS (gridsearch previo)
# order = (p, d, q) | seasonal_order = (P, D, Q, s) con s = 24 horas
# ==============================================================================
OPTIMAL_CONFIGS = {
    "NE3": {"order": (2, 1, 0), "seasonal_order": (0, 1, 1, 24)},
    "NO2": {"order": (2, 0, 2), "seasonal_order": (0, 1, 1, 24)},
    "SE3": {"order": (2, 0, 2), "seasonal_order": (0, 1, 1, 24)},
}

# Máximo de horas recientes usadas para entrenar (evita OOM/lentitud)
TRAIN_HOURS = 4380  # 6 meses de datos horarios
TEST_HOURS = 7
SEASONAL_PERIOD = 24


# ==============================================================================
# PIPELINE DE ENTRENAMIENTO Y EVALUACIÓN SARIMAX (órdenes fijos) - HORARIO
# ==============================================================================
def fit_optimal_hourly_sarimax(
    df: pd.DataFrame,
    target_col: str = "PM10",
    date_col: str = "ds",
    exog_cols: list[str] = None,
    order: tuple = None,
    seasonal_order: tuple = None,
    station: str = "SE3",
    test_hours: int = TEST_HOURS,
    output_dir: str = None,
    plots_dir: str = None,
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
        .interpolate(method="linear", limit_area="inside")
        .ffill()
        .bfill()
        .asfreq("h")
    )
    df_hourly.index.freq = "h"

    # Usar solo las últimas TRAIN_HOURS horas para limitar costo de cómputo
    if len(df_hourly) > TRAIN_HOURS:
        df_hourly = df_hourly.iloc[-TRAIN_HOURS:]

    y_hourly = df_hourly[target_col]
    X_hourly = df_hourly[exog_cols] if exog_cols else None

    print(f"\n=== Dataset Horario Preparado: {len(y_hourly)} registros ({station} - {target_col}) ===")
    print(f"✅ Órdenes fijos: SARIMAX{order} x {seasonal_order}")

    # 2. Transformación Yeo-Johnson de la variable objetivo (escala modelo)
    pt = PowerTransformer(method="yeo-johnson")
    y_tr_values = pt.fit_transform(y_hourly.values.reshape(-1, 1)).ravel()
    y_hourly_tr = pd.Series(y_tr_values, index=y_hourly.index, name=target_col)
    yeo_lambda = float(pt.lambdas_[0])
    print(
        f"🔄 Transformación Yeo-Johnson aplicada a {target_col} "
        f"(λ = {yeo_lambda:.4f})"
    )

    # 3. División Train / Test (últimas `test_hours` horas)
    #    y_test se conserva en escala original para evaluar en µg/m³
    y_train_tr = y_hourly_tr.iloc[:-test_hours]
    y_test = y_hourly.iloc[-test_hours:]
    y_train_orig = y_hourly.iloc[:-test_hours]

    X_train = X_hourly.iloc[:-test_hours] if X_hourly is not None else None
    X_test = X_hourly.iloc[-test_hours:] if X_hourly is not None else None

    # 4. Evaluación fuera de muestra (out-of-sample) con statsmodels
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        eval_model = SARIMAX(
            y_train_tr,
            exog=X_train,
            order=order,
            seasonal_order=seasonal_order,
            enforce_stationarity=True,
            enforce_invertibility=True,
        ).fit(disp=False)

        forecasts = eval_model.get_forecast(steps=test_hours, exog=X_test)
        predictions_tr = forecasts.predicted_mean

    # Regresar los pronósticos a la escala original (µg/m³)
    predictions = pd.Series(
        pt.inverse_transform(predictions_tr.values.reshape(-1, 1)).ravel(),
        index=predictions_tr.index,
        name=target_col,
    )

    # 5. Modelo final de producción (dataset completo, escala transformada)
    print(f"\n=== Ajustando Modelo Final con Dataset Completo ({station}) ===")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        final_model = SARIMAX(
            y_hourly_tr,
            exog=X_hourly,
            order=order,
            seasonal_order=seasonal_order,
            enforce_stationarity=True,
            enforce_invertibility=True,
        ).fit(disp=False)

    # 6. Métricas de error (escala original)
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

    metrics = {"MAE": mae, "RMSE": rmse, "MAPE": mape}

    # 7. Pesos (betas) de cada regresor exógeno (escala transformada)
    std_x = X_hourly[exog_cols].std().values if X_hourly is not None else None
    df_betas = extract_betas(final_model, exog_cols, std_x=std_x)

    print("\n------------------------------------------------------------------")
    print("   PESOS DE LOS REGRESORES EXÓGENOS (BETAS, escala Yeo-Johnson)")
    print("------------------------------------------------------------------")
    print(df_betas.to_string(index=False))

    # 8. Gráficas
    if plots_dir is not None:
        os.makedirs(plots_dir, exist_ok=True)

        plot_residuals(
            final_model,
            station=station,
            save_path=os.path.join(plots_dir, f"residuales_{station}.png"),
        )

        # Ajustados en escala original para el gráfico de regresión
        fitted_tr = final_model.fittedvalues.dropna()
        fitted_orig = pd.Series(
            pt.inverse_transform(fitted_tr.values.reshape(-1, 1)).ravel(),
            index=fitted_tr.index,
            name=target_col,
        )
        plot_regression_lines(
            y_train=y_train_orig,
            y_test=y_test,
            predictions=predictions,
            fitted=fitted_orig,
            station=station,
            order=order,
            seasonal_order=seasonal_order,
            save_path=os.path.join(plots_dir, f"regresion_{station}.png"),
        )

    # 9. Guardado de artefactos + modelo
    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)
        model_path = os.path.join(output_dir, f"{station}_sarimax_{target_col}.pkl")
        model_full_path = os.path.join(output_dir, f"{station}_sarimax_{target_col}_modelo.pkl")

        artifact = {
            "station": station,
            "target": target_col,
            "freq": "H",
            "order": order,
            "seasonal_order": seasonal_order,
            "exog_cols": exog_cols,
            "y_transform": {"method": "yeo-johnson", "lambda": yeo_lambda},
            "betas": df_betas.to_dict(orient="records"),
            "metrics": metrics,
        }
        pd.to_pickle(artifact, model_path)
        save_model(
            final_model,
            y_hourly=y_hourly_tr,
            X_hourly=X_hourly,
            order=order,
            seasonal_order=seasonal_order,
            yeo_lambda=yeo_lambda,
            pt=pt,
            path=model_full_path,
        )
        print(f"\n✅ Modelo guardado: {model_path}")
        print(f"✅ Modelo completo guardado: {model_full_path}")

    if plots_dir is not None:
        report_path = os.path.join(plots_dir, f"betas_{station}.txt")
        save_betas_report(
            report_path, station, order, seasonal_order, metrics, df_betas,
            yeo_lambda=yeo_lambda,
        )

    eval_df = pd.DataFrame(
        {"Actual": y_test, "Forecast": predictions}, index=y_test.index
    )

    return final_model, df_hourly, eval_df, metrics, df_betas, pt


# ==============================================================================
# EXTRACCIÓN DE PESOS (BETAS) DE LOS REGRESORES EXÓGENOS
# ==============================================================================
def extract_betas(model_fit, exog_cols: list, std_x=None):
    params = model_fit.params
    pvalues = model_fit.pvalues
    bse = model_fit.bse

    rows = []
    for i, col in enumerate(exog_cols):
        if col in params.index:
            std = std_x[i] if (std_x is not None and i < len(std_x)) else np.nan
            rows.append({
                "Regresor": col,
                "Beta": round(float(params[col]), 6),
                "Error Est.": round(float(bse[col]), 6),
                "p-value": round(float(pvalues[col]), 6),
                "Impacto Abs. (1 DE)": round(abs(float(params[col]) * std), 6),
                "Significativo": bool(pvalues[col] < 0.05),
            })
        else:
            rows.append({
                "Regresor": col,
                "Beta": np.nan,
                "Error Est.": np.nan,
                "p-value": np.nan,
                "Impacto Abs. (1 DE)": np.nan,
                "Significativo": False,
            })

    df_betas = pd.DataFrame(rows)
    if not df_betas.empty:
        df_betas = df_betas.sort_values(by="Impacto Abs. (1 DE)", ascending=False)
    return df_betas


def save_betas_report(path, station, order, seasonal_order, metrics, df_betas,
                      yeo_lambda=None):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"REPORTE SARIMAX ÓPTIMO - ESTACIÓN {station}\n")
        f.write(f"order = {order} | seasonal_order = {seasonal_order}\n")
        if yeo_lambda is not None:
            f.write(f"Transformación objetivo: Yeo-Johnson (λ = {yeo_lambda:.4f})\n")
        f.write("-" * 60 + "\n")
        f.write("MÉTRICAS HOLDOUT (7 HORAS, escala µg/m³)\n")
        for k, v in metrics.items():
            f.write(f"  {k}: {v:.3f}\n")
        f.write("-" * 60 + "\n")
        f.write("PESOS (BETAS) DE LOS REGRESORES (escala Yeo-Johnson)\n")
        f.write(df_betas.to_string(index=False) + "\n")
    print(f"✅ Reporte de betas guardado en: {path}")


# ==============================================================================
# GUARDADO / CARGA LIGERA DEL MODELO
# Guarda solo especificación + datos + parámetros (en lugar de todo el objeto
# de resultados, que incluye las matrices del suavizado de Kalman y ocupa ~1.6 GB).
# Al cargar se reconstruye el SARIMAX y se evalúa en los parámetros guardados
# con model.filter(params), sin volver a optimizar.
# La endog guardada está en escala Yeo-Johnson; se guarda λ para poder
# invertir los pronósticos a µg/m³ al cargar.
# ==============================================================================
def save_model(model_fit, y_hourly, X_hourly, order, seasonal_order, path,
               yeo_lambda=None, pt=None):
    blob = {
        "order": order,
        "seasonal_order": seasonal_order,
        "params": np.asarray(model_fit.params, dtype=float),
        "endog": np.asarray(y_hourly.values, dtype=float),
        "exog": np.asarray(X_hourly.values, dtype=float) if X_hourly is not None else None,
        "index": y_hourly.index,
        "exog_cols": list(X_hourly.columns) if X_hourly is not None else [],
        "y_transform": {
            "method": "yeo-johnson",
            "lambda": yeo_lambda,
            "mean": float(pt._scaler.mean_[0]) if pt is not None else None,
            "scale": float(pt._scaler.scale_[0]) if pt is not None else None,
        },
    }
    with open(path, "wb") as f:
        pickle.dump(blob, f)


def inverse_yeo_johnson_scaled(z, lam, mean, scale):
    """Inversa de la transformación de sklearn PowerTransformer:
    primero des-estandariza (z = z*scale + mean) y luego invierte Yeo-Johnson,
    devolviendo el valor en la escala original del target."""
    eps = 1e-12
    z = np.asarray(z, dtype=float) * scale + mean
    out = np.empty_like(z)
    pos = z >= 0
    neg = ~pos

    def _inv_pos(x):
        if abs(lam) < eps:
            return np.exp(x) - 1.0
        return np.power(x * lam + 1.0, 1.0 / lam) - 1.0

    def _inv_neg(x):
        if abs(lam - 2.0) < eps:
            return 1.0 - np.exp(x)
        return 1.0 - np.power(-x * (2.0 - lam) + 1.0, 1.0 / (2.0 - lam))

    out[pos] = _inv_pos(z[pos])
    out[neg] = _inv_neg(z[neg])
    return out


def load_model(path):
    with open(path, "rb") as f:
        blob = pickle.load(f)

    index = pd.DatetimeIndex(blob["index"])
    y = pd.Series(blob["endog"], index=index)
    X = (
        pd.DataFrame(blob["exog"], index=index, columns=blob["exog_cols"])
        if blob["exog"] is not None
        else None
    )

    model = SARIMAX(
        y,
        exog=X,
        order=blob["order"],
        seasonal_order=blob["seasonal_order"],
        enforce_stationarity=True,
        enforce_invertibility=True,
    )
    return model.filter(blob["params"])


# ==============================================================================
# GRÁFICA DE RESIDUALES DEL MODELO FINAL
# ==============================================================================
def plot_residuals(model_fit, station, save_path):
    resid = model_fit.resid.dropna()
    sigma = resid.std()

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    fig.suptitle(f"Residuales SARIMAX({model_fit.model.order}) x "
                 f"{model_fit.model.seasonal_order} - {station}", fontsize=13)

    # Serie temporal de residuales con bandas de confianza
    ax = axes[0, 0]
    ax.plot(resid.index, resid.values, lw=0.7, color="steelblue", label="Residual")
    ax.axhline(0, color="black", lw=0.8, linestyle="--")
    # Bandas de confianza al 95% y 99%
    ax.fill_between(
        resid.index, -1.96 * sigma, 1.96 * sigma,
        color="steelblue", alpha=0.15, label="95% (±1.96σ)",
    )
    ax.fill_between(
        resid.index, -2.576 * sigma, 2.576 * sigma,
        color="crimson", alpha=0.08, label="99% (±2.576σ)",
    )
    ax.axhline(1.96 * sigma, color="steelblue", ls=":", lw=1)
    ax.axhline(-1.96 * sigma, color="steelblue", ls=":", lw=1)
    ax.axhline(2.576 * sigma, color="crimson", ls=":", lw=1)
    ax.axhline(-2.576 * sigma, color="crimson", ls=":", lw=1)
    ax.set_title("Residuales en el tiempo (bandas de confianza)")
    ax.set_xlabel("Fecha")
    ax.set_ylabel("Residual")
    ax.legend(loc="best")

    # Histograma + KDE
    axes[0, 1].hist(resid.values, bins=60, density=True, alpha=0.7, color="steelblue")
    if sigma > 0:
        x = np.linspace(resid.min(), resid.max(), 300)
        axes[0, 1].plot(
            x,
            (1 / (resid.std() * np.sqrt(2 * np.pi)))
            * np.exp(-((x - resid.mean()) ** 2) / (2 * resid.std() ** 2)),
            color="crimson", lw=1.5, label="Normal teórica",
        )
    axes[0, 1].set_title("Histograma de residuales")
    axes[0, 1].legend()

    # QQ-plot
    from scipy import stats as scipy_stats
    scipy_stats.probplot(resid.values, dist="norm", plot=axes[1, 0])
    axes[1, 0].set_title("Q-Q Plot")

    # ACF de residuales
    from statsmodels.graphics.tsaplots import plot_acf
    plot_acf(resid.values, lags=48, ax=axes[1, 1])
    axes[1, 1].set_title("ACF de residuales")

    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"✅ Gráfica de residuales guardada en: {save_path}")


# ==============================================================================
# GRÁFICA ACTUAL vs AJUSTADO/PRONÓSTICO CON LÍNEA DE REGRESIÓN POR ESTACIÓN
# ==============================================================================
def plot_regression_lines(y_train, y_test, predictions, fitted, station,
                          order, seasonal_order, save_path):
    fitted = fitted.dropna().reindex(y_train.index)
    actual_fit = y_train.loc[fitted.index]
    actual_test = y_test.values.astype(float)
    forecast = np.asarray(predictions.values, dtype=float)

    def add_regression_line(ax, x, y, label):
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        m, b = np.polyfit(x, y, 1)
        r2 = 1 - np.sum((y - (m * x + b)) ** 2) / max(np.sum((y - np.mean(y)) ** 2), 1e-12)
        xs = np.linspace(x.min(), x.max(), 100)
        ax.plot(xs, m * xs + b, color="crimson", lw=1.8,
                label=f"Línea de regresión (R²={r2:.3f})")

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(f"Actual vs Ajustado/Pronóstico - {station}\n"
                 f"SARIMAX{order} x {seasonal_order}", fontsize=13)

    # In-sample: ajustados vs actuales
    ax = axes[0]
    ax.scatter(actual_fit, fitted, s=12, alpha=0.4, color="steelblue", label="In-sample")
    add_regression_line(ax, fitted, actual_fit, "in-sample")
    lo, hi = min(actual_fit.min(), fitted.min()), max(actual_fit.max(), fitted.max())
    ax.plot([lo, hi], [lo, hi], "k--", lw=1, label="y = x (perfecto)")
    ax.set_xlabel("Ajustados (fitted)")
    ax.set_ylabel("Actuales")
    ax.set_title("Regresión Actual vs Ajustado (in-sample)")
    ax.legend()

    # Out-of-sample: pronóstico vs actuales (holdout)
    ax = axes[1]
    ax.scatter(forecast, actual_test, s=60, alpha=0.8, color="seagreen", label="Holdout (7h)")
    add_regression_line(ax, forecast, actual_test, "holdout")
    lo = min(actual_test.min(), forecast.min())
    hi = max(actual_test.max(), forecast.max())
    ax.plot([lo, hi], [lo, hi], "k--", lw=1, label="y = x (perfecto)")
    ax.set_xlabel("Pronóstico")
    ax.set_ylabel("Actual")
    ax.set_title("Regresión Actual vs Pronóstico (holdout)")
    ax.legend()

    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"✅ Gráfica de regresión guardada en: {save_path}")


# ==============================================================================
# EJECUCIÓN
# ==============================================================================
def run_pipeline(data_path: str):
    if not os.path.exists(data_path):
        print(f"El archivo no fue encontrado en: {data_path}")
        return

    data = pd.read_parquet(data_path)
    filename = os.path.basename(data_path)

    # 1. Extraer la estación desde el nombre del archivo
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

    # 3. Detectar variables exógenas automáticamente
    ignore_cols = ["ds", "Date", "Estacion", target_col]
    exog_cols = [c for c in data.columns if c not in ignore_cols]

    config = OPTIMAL_CONFIGS.get(station_from_file)
    if config is None:
        print(f"⚠️ No hay configuración óptima definida para {station_from_file}. Saltando...")
        return

    print(f"\nDataset cargado: {filename}")
    print(f"Estación identificada: {station_from_file}")
    print(f"Variable Objetivo: {target_col}")
    print(f"Variables Exógenas ({len(exog_cols)}): {exog_cols}")

    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(SCRIPT_DIR, "..", "models")
    plots_dir = os.path.join(SCRIPT_DIR, "..", "texts")

    fit_optimal_hourly_sarimax(
        data,
        target_col=target_col,
        exog_cols=exog_cols,
        order=config["order"],
        seasonal_order=config["seasonal_order"],
        station=station_from_file,
        test_hours=TEST_HOURS,
        output_dir=output_dir,
        plots_dir=plots_dir,
    )


if __name__ == "__main__":
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(SCRIPT_DIR, "..", "BasesDeDatosParquet")

    datasets = [
        "factor_analysis_NE3_factor_scores.parquet",
        "factor_analysis_NO2_factor_scores.parquet",
        "factor_analysis_SE3_factor_scores.parquet",
    ]

    print(f"\n================ STARTING OPTIMAL BATCH PROCESSING ({len(datasets)} DATASETS) ================")

    for filename in datasets:
        file_path = os.path.join(data_dir, filename)
        print(f"\n------------------------------------------------------------------")
        print(f"Processing dataset: {filename}")
        print(f"Full Path: {file_path}")
        print(f"------------------------------------------------------------------")

        if os.path.exists(file_path):
            try:
                run_pipeline(file_path)
                print(f"\n✅ Successfully completed pipeline for: {filename}")
            except Exception as e:
                print(f"❌ Error processing {filename}: {e}")
                import traceback
                traceback.print_exc()
        else:
            print(f"⚠️ Warning: File not found at '{file_path}'. Skipping...")

    print("\n================ ALL DATASETS PROCESSED ================")