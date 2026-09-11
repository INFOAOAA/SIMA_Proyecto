import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from factor_analyzer import calculate_kmo
from statsmodels.multivariate.factor import Factor

STATIONS = ["SE3", "NO2", "NE3"]
KMO_THRESHOLD = 0.5
MIN_EIGENVALUE = 1.0
MIN_REL_DROP = 0.01
ROTATION = "varimax"
MAX_UNIQUENESS = 0.90
EXCLUDE_COLS = ["Mes", "Hora", "PM10", "NOX"]

FACTOR_ROLE = {
    "CO": "Combustion",
    "NO": "Combustion",
    "NO2": "Combustion",
    "PM2.5": "Combustion",
    "O3": "Photochemical",
    "RH": "Moisture",
    "PRS": "Pressure",
    "SR": "Radiation",
    "TOUT": "Temperature",
    "WSR": "Wind",
    "WDR": "Wind",
    "SO2": "Sulfur",
}


def numeric_features(df: pd.DataFrame) -> pd.DataFrame:
    df_num = df.select_dtypes(include=[np.number]).dropna().copy()
    drop = [c for c in EXCLUDE_COLS if c in df_num.columns]
    return df_num.drop(columns=drop)


def select_kmo_vars(
    df: pd.DataFrame, threshold: float = KMO_THRESHOLD
) -> tuple[list[str], list[tuple[float, pd.Series]]]:
    current = df.copy()
    history: list[tuple[float, pd.Series]] = []

    while True:
        kmo_per, kmo_model = calculate_kmo(current)
        series = pd.Series(kmo_per, index=current.columns, name="KMO_Score").sort_values()
        history.append((kmo_model, series.copy()))

        low = series[series < threshold]
        if low.empty:
            break

        if current.shape[1] <= 3:
            break

        worst = low.index[0]
        current = current.drop(columns=[worst])

    return current.columns.tolist(), history


def select_n_factors(
    evals: np.ndarray,
    min_eig: float = MIN_EIGENVALUE,
    min_drop: float = MIN_REL_DROP,
) -> int:
    n = 0
    for k, eig in enumerate(evals):
        if eig < min_eig:
            break
        if k > 0:
            prev = evals[k - 1]
            rel_drop = (prev - eig) / prev if prev > 0 else 0.0
            if rel_drop < min_drop:
                break
        n = k + 1
    return n


def run_factor_analysis(df: pd.DataFrame, n_factors: int, rotation: str = ROTATION):
    fa = Factor(df, n_factor=n_factors, method="pa")
    res = fa.fit()
    res.rotate(method=rotation)
    return res


def interpret_factors(loadings: pd.DataFrame) -> dict[str, str]:
    labels = {}
    for col in loadings.columns:
        top = loadings[col].abs().sort_values(ascending=False).head(3)
        weights: dict[str, float] = {}
        for var, val in top.items():
            role = FACTOR_ROLE.get(var)
            if role is not None:
                weights[role] = weights.get(role, 0.0) + float(val)
        if not weights:
            labels[col] = "Factor_Ambiental"
            continue
        best = max(weights, key=weights.get)
        ordered = sorted(weights.values(), reverse=True)
        if len(ordered) >= 2 and ordered[0] - ordered[1] < 1e-9:
            best = "Ambiental"
        labels[col] = f"Factor_{best}"
    return labels


def describe_factors(loadings: pd.DataFrame, labels: dict[str, str]) -> list[str]:
    desc = []
    for col in loadings.columns:
        top = loadings[col].abs().sort_values(ascending=False).head(3)
        top_str = ", ".join(f"{var} ({loadings.loc[var, col]:.3f})" for var in top.index)
        desc.append(f"{col} = {labels[col]}: {top_str}")
    return desc


def compute_factor_scores(
    df: pd.DataFrame, res, n_factors: int
) -> pd.DataFrame:
    Z = (df - df.mean()) / df.std(ddof=0)
    R_inv = np.linalg.pinv(Z.corr().values)
    L = np.real(res.loadings)
    W = np.dot(R_inv, L)
    scores = np.dot(Z.values, W)
    return pd.DataFrame(
        scores,
        index=df.index,
        columns=[f"Factor_{i+1}" for i in range(n_factors)],
    )


def save_dataset(df: pd.DataFrame, station: str, parquet_dir: str):
    os.makedirs(parquet_dir, exist_ok=True)
    path = os.path.join(parquet_dir, f"factor_analysis_{station}.parquet")
    df.to_parquet(path, engine="pyarrow", index=False)


def save_fa_diagram(
    loadings: pd.DataFrame,
    labels: dict[str, str],
    uniqueness: pd.Series,
    station: str,
    outpath: str,
):
    n_vars = loadings.shape[0]
    n_factors = loadings.shape[1]
    var_x, factor_x = 2.0, 8.0
    span = max(n_vars, n_factors) - 1

    def positions(count: int) -> list[float]:
        if count == 1:
            return [span / 2]
        return [span - i * span / (count - 1) for i in range(count)]

    var_ys = positions(n_vars)
    fac_ys = positions(n_factors)

    fig, ax = plt.subplots(figsize=(11, max(6, span + 2)))
    ax.set_xlim(0, 10)
    ax.set_ylim(-0.5, span + 0.5)
    ax.axis("off")
    ax.set_title(f"Factor Analysis Diagram - Estación {station}", fontsize=13, fontweight="bold")

    for i, var in enumerate(loadings.index):
        unq = uniqueness.get(var, float("nan"))
        ax.text(
            var_x,
            var_ys[i],
            f"{var}\nu²={unq:.2f}",
            ha="center",
            va="center",
            fontsize=10,
            bbox=dict(boxstyle="round,pad=0.35", facecolor="#EFEFEF", edgecolor="#888888"),
        )

    for j, fac in enumerate(loadings.columns):
        label_short = labels.get(fac, fac).replace("Factor_", "")
        ax.text(
            factor_x,
            fac_ys[j],
            f"{fac}\n{label_short}",
            ha="center",
            va="center",
            fontsize=10,
            bbox=dict(boxstyle="round,pad=0.45", facecolor="#D9EAF7", edgecolor="#4A72A8"),
        )

    strongest = loadings.abs().idxmax(axis=1)
    for i, var in enumerate(loadings.index):
        fac = strongest[var]
        j = loadings.columns.get_loc(fac)
        w = loadings.loc[var, fac]
        if not np.isfinite(w):
            continue
        color = "#B02318" if w < 0 else "#333333"
        ax.annotate(
            "",
            xy=(var_x + 0.6, var_ys[i]),
            xytext=(factor_x - 0.6, fac_ys[j]),
            arrowprops=dict(arrowstyle="->", color=color, lw=max(0.8, min(3.0, abs(w)))),
        )
        ax.text(
            (var_x + factor_x) / 2,
            (var_ys[i] + fac_ys[j]) / 2,
            f"{w:.2f}",
            fontsize=8,
            color=color,
            ha="center",
            va="center",
            bbox=dict(facecolor="white", alpha=0.75, edgecolor="none", pad=1),
        )

    plt.tight_layout()
    plt.savefig(outpath, dpi=120, bbox_inches="tight")
    plt.close()


def save_scree_plot(evals, n_factors: int, station: str, outpath: str):
    num = len(evals)
    x = np.arange(1, num + 1)
    plt.figure(figsize=(8, 5))
    plt.scatter(x, evals, color="red", zorder=3)
    plt.plot(x, evals, color="blue", linestyle="--", zorder=2)
    plt.axhline(y=1, color="grey", linestyle=":", linewidth=1.5, label="Kaiser (λ=1)")
    plt.axvline(x=n_factors + 0.5, color="green", linestyle="--", label=f"{n_factors} factores")
    plt.title(f"Scree Plot - Estación {station}", fontsize=12, fontweight="bold")
    plt.xlabel("Número de Factor")
    plt.ylabel("Eigenvalue")
    plt.xticks(x)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.savefig(outpath, dpi=120)
    plt.close()


def analyze_station(df: pd.DataFrame, station: str, texts_dir: str, parquet_dir: str) -> str:
    sep = "=" * 70
    lines = [sep, f" ANÁLISIS FACTORIAL INDIVIDUAL - ESTACIÓN {station}", sep]

    df_num = numeric_features(df)
    lines.append(f"\nNúmero de observaciones: {len(df_num)}")
    lines.append(f"Variables iniciales: {list(df_num.columns)}")

    retained, history = select_kmo_vars(df_num)
    lines.append(f"\n--- Selección de variables por KMO (>= {KMO_THRESHOLD}) ---")
    for kmo_model, series in history:
        lines.append(f"\nKMO global: {kmo_model:.4f}")
        lines.append(series.round(4).to_string())
        low = series[series < KMO_THRESHOLD]
        if not low.empty:
            lines.append(f"  -> Eliminando variable con menor KMO: {low.index[0]} ({low.iloc[0]:.4f})")
    lines.append(f"\nVariables retenidas: {retained}")

    df_fa = df_num[retained]

    removed_by_uniqueness: list[str] = []
    iteration = 0
    while True:
        iteration += 1
        corr_matrix = df_fa.corr().values
        evals = np.sort(np.linalg.eigvalsh(corr_matrix))[::-1]
        n_factors = select_n_factors(evals)
        n_factors = max(1, min(n_factors, len(df_fa.columns) - 1))

        res = run_factor_analysis(df_fa, n_factors)

        uniqueness = pd.Series(res.uniqueness, index=df_fa.columns, name="Uniqueness")
        high = uniqueness[uniqueness >= MAX_UNIQUENESS]

        if high.empty:
            break

        if len(df_fa.columns) - len(high) < 3:
            lines.append(
                f"\n⚠️ No se pueden eliminar más variables (mínimo 3 para factorizar). "
                f"Se mantienen con uniqueness >= {MAX_UNIQUENESS}."
            )
            break

        lines.append(
            f"\n--- Iteración {iteration}: eliminando {len(high)} variables con uniqueness >= {MAX_UNIQUENESS} ---"
        )
        lines.append(uniqueness.round(3).to_string())
        removed_by_uniqueness.extend(high.index.tolist())
        df_fa = df_fa.drop(columns=high.index)

    lines.append(
        f"\nVariables eliminadas por uniqueness >= {MAX_UNIQUENESS}: "
        f"{removed_by_uniqueness or 'ninguna'}"
    )

    save_dataset(df_fa, station, parquet_dir)

    lines.append(f"\n--- Eigenvalues (Matriz de correlación) ---")
    prev = None
    for i, eig in enumerate(evals, 1):
        if prev is None:
            drop = None
        else:
            drop = (prev - eig) / prev if prev > 0 else 0.0
        prev = eig
        tag = f"  (drop vs anterior: {drop:.1%})" if drop is not None else ""
        lines.append(f"Factor {i}: {eig:.4f}{tag}")

    lines.append(
        f"\n--- Decisión número de factores (Kaiser >= 1 y caída relativa >= {MIN_REL_DROP:.0%}) ---"
    )
    lines.append(f"Factores seleccionados: {n_factors}")

    lines.append(f"\n--- Resultados del análisis factorial ({ROTATION.upper()}) ---")
    lines.append(str(res.summary()))

    loadings = pd.DataFrame(
        res.loadings,
        index=df_fa.columns,
        columns=[f"Factor_{i+1}" for i in range(n_factors)],
    )
    uniqueness = pd.Series(res.uniqueness, index=df_fa.columns, name="Uniqueness")

    scores = compute_factor_scores(df_fa, res, n_factors)
    scores.insert(0, "ds", df.loc[scores.index, "ds"])
    scores.insert(1, "PM10", df.loc[scores.index, "PM10"])
    save_dataset(scores, f"{station}_factor_scores", parquet_dir)

    lines.append(f"\n--- PUNTAJES FACTORIALES (resumen) ---")
    lines.append(f"Shape: {scores.shape[0]} obs x {scores.shape[1]} factores")
    lines.append(scores.describe().round(4).to_string())

    lines.append(f"\n--- LOADINGS ROTADOS ({ROTATION.upper()}) ---")
    lines.append(loadings.round(3).to_string())

    lines.append("\n--- UNIQUENESSES ---")
    lines.append(uniqueness.round(3).to_string())

    lines.append("\n--- Interpretación (3 variables con mayor carga por factor) ---")
    labels = interpret_factors(loadings)
    lines.extend(describe_factors(loadings, labels))

    diagram_path = os.path.join(texts_dir, f"factor_analysis_{station}_diagram.png")
    save_fa_diagram(loadings, labels, uniqueness, station, diagram_path)

    txt_path = os.path.join(texts_dir, f"factor_analysis_{station}.txt")
    png_path = os.path.join(texts_dir, f"factor_analysis_{station}_scree.png")

    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    save_scree_plot(evals, n_factors, station, png_path)

    return "\n".join(lines)


def main():
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(SCRIPT_DIR, "..", "BasesDeDatosParquet", "hourly_database.parquet")
    texts_dir = os.path.join(SCRIPT_DIR, "..", "texts")
    os.makedirs(texts_dir, exist_ok=True)

    df = pd.read_parquet(data_path)
    df = df[df["Estacion"].isin(STATIONS)]

    parquet_dir = os.path.join(SCRIPT_DIR, "..", "BasesDeDatosParquet")

    for station in STATIONS:
        station_df = df[df["Estacion"] == station]
        report = analyze_station(station_df, station, texts_dir, parquet_dir)
        print(report)
        print("\n")


if __name__ == "__main__":
    main()