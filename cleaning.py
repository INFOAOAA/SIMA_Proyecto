import os
import numpy as np
import pandas as pd
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.preprocessing import StandardScaler
import pyarrow


def clean_database():
    # -------------------------------------------------------------------------
    # 1. READ & CONCATENATE DATA
    # -------------------------------------------------------------------------
    datas = [
        pd.read_csv(os.path.abspath(f"BasesDeDatosCsv/BD_202{i}_Unido.csv"))
        for i in range(6)
    ]
    datas[5].rename(columns={"date": "Fecha y hora"}, inplace=True)

    df_raw = pd.concat(datas, axis=0, ignore_index=True)

    # -------------------------------------------------------------------------
    # 2. BASE CLEANING & FEATURE EXTRACTION
    # -------------------------------------------------------------------------
    df_raw["Fecha y hora"] = pd.to_datetime(
        df_raw["Fecha y hora"], yearfirst=True
    )
    df_raw = df_raw.drop_duplicates().drop(columns=["RAINF"], errors="ignore")

    # Add temporal features & season binning
    months = df_raw["Fecha y hora"].dt.month
    bins = [0, 2, 5, 8, 11, 12]
    labels = ["Winter", "Spring", "Summer", "Fall", "Winter"]

    df_raw["season"] = pd.cut(
        months, bins=bins, labels=labels, ordered=False, include_lowest=True
    )

    # Convert to Categorical with 'Fall' first so drop_first=True drops Fall as baseline
    season_order = ["Fall", "Winter", "Spring", "Summer"]
    df_raw["season"] = pd.Categorical(
        df_raw["season"], categories=season_order, ordered=True
    )

    # Convert season to dummy variables (season_Fall will be dropped)

    df_raw["Mes"] = df_raw["Fecha y hora"].dt.month
    df_raw["Hora"] = df_raw["Fecha y hora"].dt.hour

    # Add Categorical PM10 Air Quality Binning
    bins_pm10 = [0, 50, 100, np.inf]
    etiquetas = ["Buena", "Regular", "Mala"]
    df_raw["Calidad_Aire_PM10"] = pd.cut(
        df_raw["PM10"], bins=bins_pm10, labels=etiquetas
    )

    # -------------------------------------------------------------------------
    # 3. FEATURE SCALING (Applied first; ignores NaNs naturally)
    # -------------------------------------------------------------------------
    cols_numericas = [
        "CO",
        "NO",
        "NO2",
        "NOX",
        "O3",
        "PM10",
        "PM2.5",
        "PRS",
        "RH",
        "SR",
        "TOUT",
        "WSR",
        "WDR",
    ]
    scaler = StandardScaler()

    os.makedirs("BasesDeDatosParquet", exist_ok=True)

    path4 = os.path.abspath('BasesDeDatosParquet/BD_no_standarization.parquet')
    df_raw.to_parquet(path4, engine='pyarrow', index=False)

    df_raw[cols_numericas] = scaler.fit_transform(df_raw[cols_numericas])

    # Ensure output directory exists

    path1 = os.path.abspath("BasesDeDatosParquet/BD_Completa.parquet")
    path2 = os.path.abspath("BasesDeDatosParquet/BD_no_dummies.parquet")
    path3 = os.path.abspath("BasesDeDatosParquet/BD_no_imputation.parquet")
    

    # -------------------------------------------------------------------------
    # PATH 3: Scaled, NO Imputation, NO 'Estacion' Dummies
    # -------------------------------------------------------------------------
    df_raw.to_parquet(path3, engine="pyarrow", index=False)
    print(f"Saved Path 3 (Scaled, Unimputed) -> {path3}")

    # -------------------------------------------------------------------------
    # 4. IMPUTATION & OUTLIER CAPPING
    # -------------------------------------------------------------------------
    df_imp = df_raw.copy()

    # Separate metadata and non-numeric columns from imputation matrix
    non_impute_cols = ["Fecha y hora", "Estacion", "Calidad_Aire_PM10", 'season']
    meta_df = df_imp[non_impute_cols].reset_index(drop=True)
    features_df = df_imp.drop(columns=non_impute_cols)

    # Iterative Imputation on scaled features
    print("Iniciando imputación iterativa...")
    imp = IterativeImputer(max_iter=10, random_state=42)
    imputed_array = imp.fit_transform(features_df)

    df_clean = pd.DataFrame(imputed_array, columns=features_df.columns)

    # Reattach metadata columns
    for col in reversed(non_impute_cols):
        df_clean.insert(0, col, meta_df[col])

    # Outlier Capping (Clipping via IQR)
    cols_outliers = ["PM10", "PM2.5", "O3", "CO"]
    for col in cols_outliers:
        Q1 = df_clean[col].quantile(0.25)
        Q3 = df_clean[col].quantile(0.75)
        IQR = Q3 - Q1
        limite_inf = Q1 - 1.5 * IQR
        limite_sup = Q3 + 1.5 * IQR
        df_clean[col] = np.clip(df_clean[col], limite_inf, limite_sup)

    # -------------------------------------------------------------------------
    # PATH 2: Scaled, Imputed, NO 'Estacion' Dummies
    # -------------------------------------------------------------------------
    df_clean.to_parquet(path2, engine="pyarrow", index=False)
    print(f"Saved Path 2 (Scaled & Imputed) -> {path2}")

    # -------------------------------------------------------------------------
    # PATH 1: Complete (Scaled, Imputed, WITH 'Estacion' Dummies)
    # -------------------------------------------------------------------------
    df_complete = pd.get_dummies(df_clean, columns=["Estacion"])
    df_complete.to_parquet(path1, engine="pyarrow", index=False)
    print(f"Saved Path 1 (Complete) -> {path1}")


if __name__ == "__main__":
    clean_database()