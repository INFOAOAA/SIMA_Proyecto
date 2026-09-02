import pandas as pd
import numpy as np
import itertools
from scipy.spatial.distance import mahalanobis
import os



def mahalanobis_selection(df: pd.DataFrame):
    cols_to_exclude = [
        'Estacion',
        'Station',
        'Fecha y hora',
        'Fecha',
        'hora',
        'Mes',
        'Hora',
    ]
    numeric_vars = [
        c
        for c in df.select_dtypes(include=[np.number]).columns
        if c not in cols_to_exclude
    ]

    stations = df['Estacion'].unique()
    means = df.groupby('Estacion')[numeric_vars].mean()

    # Calculate Pooled Covariance Matrix
    n_total = len(df)
    k_groups = len(stations)
    pooled_cov = np.zeros((len(numeric_vars), len(numeric_vars)))

    for s in stations:
        sub = df[df['Estacion'] == s][numeric_vars]
        n_s = len(sub)
        pooled_cov += (n_s - 1) * sub.cov().values

    pooled_cov /= n_total - k_groups

    # Invert the pooled covariance matrix
    inv_pooled_cov = np.linalg.inv(pooled_cov)

    # Compute Pairwise Mahalanobis Distances
    dist_df = pd.DataFrame(index=stations, columns=stations, dtype=float)

    for s1, s2 in itertools.combinations(stations, 2):
        u = means.loc[s1].values
        v = means.loc[s2].values
        d = mahalanobis(u, v, inv_pooled_cov)
        dist_df.loc[s1, s2] = d
        dist_df.loc[s2, s1] = d

    #np.fill_diagonal(dist_df.values, 0)
    for st in stations:
        dist_df.loc[st, st] = 0.0

    # Find the triplet that maximizes pairwise sum of Mahalanobis distances
    best_triplet = None
    max_distance = -1

    for triplet in itertools.combinations(stations, 3):
        # Cast loc lookups explicitly to Python floats or use .item() to pull scalar floats
        d1 = float(dist_df.loc[triplet[0], triplet[1]])
        d2 = float(dist_df.loc[triplet[0], triplet[2]])
        d3 = float(dist_df.loc[triplet[1], triplet[2]])

        total_d = d1 + d2 + d3

        if total_d > max_distance:
            max_distance = total_d
            best_triplet = triplet
    with pd.option_context(
    'display.max_columns',
    None,
    'display.max_rows',
    None,
    'display.width',
    1000,
    ):
        print('=== Pairwise Mahalanobis Distance Matrix ===')
        print(dist_df.round(2))
        print('\n=== Selection Results ===')
        print(f'Top 3 Representative Stations: {best_triplet}')
        print(f'Combined Mahalanobis Separation: {max_distance:.4f}')

    return best_triplet, dist_df

def main():
    SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(
        SCRIPTS_DIR, "..", "BasesDeDatosParquet", "hourly_database.parquet"
    )
    df = pd.read_parquet(path=path)
    mahalanobis_selection(df=df)

if __name__ == "__main__":
    main()