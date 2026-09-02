import numpy as np
import pandas as pd
import os


def impute_all_stations_hourly(
    df: pd.DataFrame,
    date_col: str = 'Fecha y hora',
    station_col: str = 'Estacion',
) -> pd.DataFrame:
    """Imputes missing values in hourly air quality data independently for every station

    in the dataset while preserving each station's unique diurnal cycles.
    """
    df_work = df.copy()
    df_work[date_col] = pd.to_datetime(df_work[date_col])

    # Identify numeric features to impute
    exclude_cols = [
        date_col,
        station_col,
        'season',
        'Mes',
        'Hora',
        'Calidad_Aire_PM10',
    ]
    numeric_cols = [
        c
        for c in df_work.select_dtypes(include=[np.number]).columns
        if c not in exclude_cols
    ]

    imputed_station_dfs = []
    stations = df_work[station_col].dropna().unique()

    print(
        f'Processing time-aware imputation for {len(stations)} stations...\n'
    )

    for st in stations:
        # 1. Filter station data and remove duplicate timestamps
        st_df = (
            df_work[df_work[station_col] == st]
            .sort_values(date_col)
            .drop_duplicates(subset=[date_col])
            .copy()
        )

        # 2. Enforce continuous 1-hour frequency grid for this station
        full_time_grid = pd.date_range(
            start=st_df[date_col].min(), end=st_df[date_col].max(), freq='h'
        )

        st_df = (
            st_df.set_index(date_col)
            .reindex(full_time_grid)
            .rename_axis('ds')
            .reset_index()
        )

        # Re-assign station label to filled empty grid rows
        st_df[station_col] = st

        # Set 'ds' as index temporarily for time-based interpolation
        st_df = st_df.set_index('ds')

        # STEP 1: Linear time-based interpolation for short gaps (<= 3 hours)
        st_df[numeric_cols] = st_df[numeric_cols].interpolate(
            method='time', limit=3
        )

        st_df = st_df.reset_index()

        # STEP 2: Station-specific Diurnal Group Mean (Month x Hour) for long gaps (> 3 hours)
        st_df['Month'] = st_df['ds'].dt.month
        st_df['Hour'] = st_df['ds'].dt.hour

        for col in numeric_cols:
            if st_df[col].isna().sum() > 0:
                diurnal_mean = st_df.groupby(['Month', 'Hour'])[col].transform(
                    'mean'
                )
                st_df[col] = st_df[col].fillna(diurnal_mean)

        st_df = st_df.drop(columns=['Month', 'Hour'])

        # STEP 3: Global Column Mean for the specific station (safety net)
        for col in numeric_cols:
            if st_df[col].isna().sum() > 0:
                st_df[col] = st_df[col].fillna(st_df[col].mean())

        imputed_station_dfs.append(st_df)

    # Recombine all clean station dataframes into a single dataset
    full_imputed_df = pd.concat(imputed_station_dfs, ignore_index=True)

    print('Imputation complete across all stations.')
    return full_imputed_df


def main():
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(
    SCRIPT_DIR, "..", "BasesDeDatosParquet", "BD_no_standarization.parquet"
    )
    df = pd.read_parquet(path=path)
    full_impute= impute_all_stations_hourly(df=df)
    path = os.path.join(
        SCRIPT_DIR, "..", "BasesDeDatosParquet", "hourly_database.parquet"
    )
    full_impute.to_parquet(engine='pyarrow', index=False, path=path)

if __name__ == '__main__':
    main()