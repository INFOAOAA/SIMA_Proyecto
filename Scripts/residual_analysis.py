"""
Analisis de residuales del modelo SARIMAX producido por SARIMAX_CPU.py.

Ejecuta el pipeline de SARIMAX_CPU (SARIMAX_CPU.fit_and_evaluate_hourly_sarimax),
toma el modelo final (final_model) y genera los graficos de diagnostico de
residuales: serie temporal, histograma, Q-Q, ACF, PACF y residuales vs ajuste.

Uso:
    python residual_analysis.py
    python residual_analysis.py --dataset factor_analysis_NE3_factor_scores.parquet
    python residual_analysis.py --dataset SIMA_Diario_Imputado.parquet --station SE3
"""

import os
import re
import sys
import warnings

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from scipy import stats
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.stats.stattools import durbin_watson, jarque_bera
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

from SARIMAX_CPU import fit_and_evaluate_hourly_sarimax

STATION_PATTERN = re.compile(r"(SE3|NO2|NE3|CE|NLE|Obispado)", re.IGNORECASE)


def build_residual_graphs(
    result,
    station: str,
    target_col: str = "PM10",
    seasonal_period: int = 24,
    output_dir: str = "./residuals",
):
    """Genera los graficos de diagnostico a partir del modelo ajustado."""
    resid = pd.Series(result.resid).dropna()
    resid.name = "Residual"
    fitted = pd.Series(result.fittedvalues).dropna()

    os.makedirs(output_dir, exist_ok=True)
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.15)
    base = os.path.join(output_dir, f"{station}_{target_col}_residuals")

    residuals_ts = os.path.join(output_dir, f"{station}_{target_col}_residuals_serie.png")
    plt.figure(figsize=(13, 5))
    plt.plot(resid.index, resid.values, color="steelblue", linewidth=0.9, label="Residuo")
    plt.axhline(0, color="red", linestyle="--", linewidth=1)
    plt.title(f"Residuos del Modelo SARIMAX - {station} ({target_col})")
    plt.xlabel("Tiempo")
    plt.ylabel("Residuo")
    plt.legend()
    plt.tight_layout()
    plt.savefig(residuals_ts, dpi=300)
    plt.close()

    counts, bin_edges = np.histogram(resid, bins=40)
    bin_width = np.diff(bin_edges).mean()
    plt.figure(figsize=(13, 5))
    sns.histplot(resid, kde=False, color="teal", bins=40, label="Residuos")
    x_norm = np.linspace(resid.min(), resid.max(), 300)
    plt.plot(
        x_norm,
        stats.norm.pdf(x_norm, resid.mean(), resid.std(ddof=1)) * len(resid) * bin_width,
        color="black", linestyle="--", label="Normal"
    )
    plt.axvline(0, color="red", linestyle="--", linewidth=1)
    plt.title(f"Distribucion de Residuos vs Normal - {station} ({target_col})")
    plt.xlabel("Residuo")
    plt.ylabel("Frecuencia")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{station}_{target_col}_residuals_hist.png"), dpi=300)
    plt.close()

    plt.figure(figsize=(6.5, 6))
    stats.probplot(resid, dist="norm", plot=plt)
    plt.title(f"Q-Q Plot de Residuos - {station} ({target_col})")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{station}_{target_col}_residuals_qq.png"), dpi=300)
    plt.close()

    n_lags = min(seasonal_period * 2, len(resid) // 2 - 1)
    fig, axes = plt.subplots(1, 2, figsize=(16, 5))
    plot_acf(resid, ax=axes[0], lags=n_lags, alpha=0.05,
             title=f"ACF de Residuos - {station} (s={seasonal_period})")
    plot_pacf(resid, ax=axes[1], lags=min(seasonal_period, len(resid) // 2 - 1),
              alpha=0.05, method="ywm",
              title=f"PACF de Residuos - {station} (s={seasonal_period})")
    fig.tight_layout()
    acf_path = os.path.join(output_dir, f"{station}_{target_col}_residuals_acf_pacf.png")
    fig.savefig(acf_path, dpi=300)
    plt.close(fig)

    plt.figure(figsize=(13, 5))
    plt.scatter(fitted, resid, s=12, alpha=0.5, color="mediumpurple")
    plt.axhline(0, color="red", linestyle="--", linewidth=1)
    plt.title(f"Residuos vs Valores Ajustados - {station} ({target_col})")
    plt.xlabel("Valor Ajustado (fitted)")
    plt.ylabel("Residuo")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{station}_{target_col}_residuals_vs_fitted.png"), dpi=300)
    plt.close()

    return {
        "serie": residuals_ts,
        "hist": os.path.join(output_dir, f"{station}_{target_col}_residuals_hist.png"),
        "qq": os.path.join(output_dir, f"{station}_{target_col}_residuals_qq.png"),
        "acf_pacf": acf_path,
        "vs_fitted": os.path.join(output_dir, f"{station}_{target_col}_residuals_vs_fitted.png"),
    }


def residual_diagnostics_table(result, seasonal_period: int = 24) -> pd.DataFrame:
    """Calcula estadisticos y pruebas de diagnostico sobre los residuales."""
    resid = pd.Series(result.resid).dropna()
    n = len(resid)

    lags = sorted({10, min(seasonal_period, max(1, n // 5)), min(20, max(1, n // 5))})
    lb = acorr_ljungbox(resid, lags=lags, return_df=True)

    sw_stat, sw_p = stats.shapiro(resid) if n <= 5000 else (np.nan, np.nan)
    jb_stat, jb_p, skew, kurt = jarque_bera(resid)
    t_stat, t_p = stats.ttest_1samp(resid, 0.0)
    dw = durbin_watson(resid)

    rows = [
        ("Observaciones (n)", n),
        ("Media", resid.mean()),
        ("Desviacion estandar", resid.std(ddof=1)),
        ("Minimo", resid.min()),
        ("Maximo", resid.max()),
        ("Asimetria (skewness)", skew),
        ("Curtosis", kurt),
        ("Durbin-Watson", dw),
        ("t-test media=0 (p-valor)", t_p),
        ("Shapiro-Wilk (p-valor)", sw_p),
        ("Jarque-Bera (p-valor)", jb_p),
    ]
    for lag in lb.index:
        rows.append((f"Ljung-Box lag {int(lag)} (p-valor)", lb.loc[lag, "lb_pvalue"]))

    tabla = pd.DataFrame(rows, columns=["Metrica", "Valor"])
    tabla["Valor"] = tabla["Valor"].round(4)
    return tabla


def analyze_dataset(data_path: str, datas: pd.DataFrame, station: str,
                    target_col: str = "PM10", seasonal_period: int = 24,
                    test_hours: int = 7, output_dir: str = "./residuals"):
    """Ajusta el SARIMAX via SARIMAX_CPU y genera el analisis de residuales."""
    print(f"\n{'='*62}")
    print(f"  ANALISIS DE RESIDUALES - {station} ({target_col})")
    print(f"{'='*62}")

    final_model, df_hourly, decomp, eval_df, metrics = fit_and_evaluate_hourly_sarimax(
        df=datas,
        target_col=target_col,
        exog_cols=[c for c in datas.columns if c not in ["ds", "Date", "Estacion", target_col]],
        seasonal_period=seasonal_period,
        station=station,
        test_hours=test_hours,
        output_dir=output_dir.replace("residuals", "models"),
    )

    table = residual_diagnostics_table(final_model, seasonal_period=seasonal_period)
    print("\n--- PRUEBAS DE DIAGNOSTICO SOBRE RESIDUALES ---")
    print(table.to_string(index=False))

    graphs = build_residual_graphs(
        final_model, station, target_col, seasonal_period, output_dir
    )
    print("\n--- GRAFICOS GENERADOS ---")
    for key, path in graphs.items():
        print(f"  {key:<12}: {path}")

    table.to_csv(os.path.join(output_dir, f"{station}_{target_col}_diagnostics.csv"), index=False)
    pd.DataFrame(dict(result=final_model.resid)).to_csv(
        os.path.join(output_dir, f"{station}_{target_col}_residuals.csv"), header=True
    )
    print(f"\nTabla de diagnostico: {output_dir}/{station}_{target_col}_diagnostics.csv")
    return final_model


def run_pipeline_residuals(dataset_name: str, station_override: str = None):
    data_dir = os.path.join(SCRIPT_DIR, "..", "BasesDeDatosParquet")
    data_path = os.path.join(data_dir, dataset_name)
    if not os.path.exists(data_path):
        print(f"Archivo no encontrado: {data_path}")
        return

    data = pd.read_parquet(data_path)
    match = STATION_PATTERN.search(os.path.basename(data_path))
    station_from_file = match.group(1).upper() if match else "Estacion"

    target_col = "PM10"
    if target_col not in data.columns:
        candidates = [c for c in data.columns if "PM10" in c.upper() or "TARGET" in c.upper()]
        if candidates:
            target_col = candidates[0]
        else:
            raise KeyError(f"Columna objetivo '{target_col}' ausente.")

    output_dir = os.path.join(SCRIPT_DIR, "residuals")
    os.makedirs(output_dir, exist_ok=True)

    print(f"Dataset: {dataset_name} | Estacion: {station_from_file} | Target: {target_col}")

    if "Estacion" in data.columns and not station_override:
        for st in data["Estacion"].unique():
            st_df = data[data["Estacion"] == st].copy()
            analyze_dataset(dataset_name, st_df, st, target_col=target_col, output_dir=output_dir)
    else:
        st = station_override or station_from_file
        analyze_dataset(dataset_name, data, st, target_col=target_col, output_dir=output_dir)


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Analisis de residuales del output de SARIMAX_CPU.")
    parser.add_argument("--dataset", type=str, default="factor_analysis_NO2_factor_scores.parquet",
                        help="Nombre del parquet en BasesDeDatosParquet.")
    parser.add_argument("--station", type=str, default=None,
                        help="Filtrar por estacion si el dataset la incluye.")
    args = parser.parse_args()

    run_pipeline_residuals(args.dataset, args.station)


if __name__ == "__main__":
    main()