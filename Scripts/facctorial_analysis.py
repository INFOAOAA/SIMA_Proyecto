import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from factor_analyzer import calculate_kmo
from statsmodels.multivariate.factor import Factor

# Fit ML Factor Analysis


import pandas as pd
from factor_analyzer.factor_analyzer import calculate_kmo


import pandas as pd
from factor_analyzer.factor_analyzer import calculate_kmo


def kmo(df: pd.DataFrame, threshold: float = 0.5) -> list[str]:
  # Filter float columns and drop non-analytical features upfront


  not_yet = True

  while not_yet:
    # Calculate KMO
    kmo_per_variable, kmo_model = calculate_kmo(df)

    # Map individual scores to column names
    kmo_series = pd.Series(
        kmo_per_variable, index=df.columns, name='KMO_Score'
    ).sort_values(ascending=True)  # Sort ascending to identify lowest easily

    print(f'\n=== Modelo KMO Overall Score: {kmo_model:.4f} ===')
    print('=== Coeficientes KMO por Variable ===')
    print(kmo_series.round(4).to_string())

    # Identify variables below threshold (< 0.5)
    low_kmo = kmo_series[kmo_series < threshold]

    if not low_kmo.empty:
      # Drop ONLY the single lowest variable to re-evaluate remaining interactions
      worst_var = low_kmo.index[0]
      worst_val = low_kmo.iloc[0]

      print(
          f'\n⚠️ Eliminando variable con menor KMO: {worst_var} ({worst_val:.4f})'
      )

      # FIX: Assign directly without inplace=True
      df = df.drop(columns=[worst_var])
    else:
      print(f'\n✅ Todas las variables restantes tienen KMO >= {threshold}')
      not_yet = False

  return df.columns

import matplotlib.pyplot as plt
import pandas as pd
from factor_analyzer import FactorAnalyzer


def scree_plot(df: pd.DataFrame):
  # 1. Ensure clean numeric dataframe without NaNs
  df_clean = df.select_dtypes(include=[np.number]).dropna()

  # 2. Compute correlation matrix and its eigenvalues directly via NumPy
  corr_matrix = df_clean.corr().values
  eigenvalues = np.linalg.eigvalsh(corr_matrix)

  # Sort eigenvalues in descending order
  eigenvalues = np.sort(eigenvalues)[::-1]

  print('\n=== Eigenvalues (Correlation Matrix) ===')
  for i, val in enumerate(eigenvalues, 1):
    print(f'Factor {i}: {val:.4f}')

  # 3. Plot Scree Plot
  num_vars = len(eigenvalues)
  plt.figure(figsize=(8, 5))
  plt.scatter(range(1, num_vars + 1), eigenvalues, color='red', zorder=3)
  plt.plot(
      range(1, num_vars + 1),
      eigenvalues,
      color='blue',
      linestyle='--',
      zorder=2,
  )

  # Kaiser Criterion reference line (Eigenvalue = 1)
  plt.axhline(
      y=1,
      color='grey',
      linestyle=':',
      linewidth=1.5,
      label='Kaiser Criterion (λ=1)',
  )

  plt.title('Scree Plot - Selected SIMA Variables', fontsize=12, fontweight='bold')
  plt.xlabel('Factor Number')
  plt.ylabel('Eigenvalue')
  plt.xticks(range(1, num_vars + 1))
  plt.grid(True, linestyle='--', alpha=0.6)
  plt.legend()
  plt.tight_layout()
  plt.show()

def factorial_analysis(
    df: pd.DataFrame, n_factors: int = 3, rotation: str = "quartimax"
):
  # Clean numeric data (ensure no NaNs)
  df_clean = df.select_dtypes(include=[np.number]).dropna().copy()

  # 1. Fit Maximum Likelihood Factor Analysis
  fa_ml = Factor(df_clean, n_factor=n_factors, method="pa")
  res = fa_ml.fit()

  # 2. Print full statistical report
  print("==================================================================")
  print("               MAXIMUM LIKELIHOOD FACTOR ANALYSIS                 ")
  print("==================================================================")
  print(res.summary())

  # 3. FIX: Apply rotation IN-PLACE (do not assign to a variable)
  res.rotate(method=rotation)

  # 4. Extract rotated loadings directly from res.loadings
  loadings_df = pd.DataFrame(
      res.loadings,
      index=df_clean.columns,
      columns=[f"Factor_{i+1}" for i in range(n_factors)],
  )

  # 5. Extract Uniquenesses
  uniqueness_series = pd.Series(
      res.uniqueness, index=df_clean.columns, name="Uniqueness"
  )

  print(
      f"\n================ ROTATED LOADINGS ({rotation.upper()}) ================"
  )
  print(loadings_df.round(3))

  print("\n================ VARIABLE UNIQUENESSES ================")
  print(uniqueness_series.round(3))

  return res, loadings_df

def export_fa_dataset(
    df: pd.DataFrame,
    timestamp_col: str = "Fecha",
    station_col: str = "Estacion",
    target_col: str = "PM10",
) -> pd.DataFrame:
  """Fits a 3-factor model on air quality features and returns a DataFrame containing

  ['Fecha', 'Estacion', 'PM10', 'Factor_Combustion', 'Factor_Photochemical',
  'Factor_Moisture'].
  """
  # 1. Target features for the 3-factor model
  features = ["CO", "NO", "NO2", "O3", "PM2.5", "RH", "SO2", "WSR", "WDR"]

  # 2. Subset data including metadata, target variable, and FA features
  meta_cols = [timestamp_col, station_col, target_col]
  all_cols = meta_cols + features

  df_clean = df[all_cols].dropna().copy()
  X = df_clean[features].astype(float)

  # 3. Fit 3-factor Principal Axis model
  fa = Factor(X, n_factor=3, method="pa")
  res = fa.fit()
  res.rotate(method="quartimax")

  # 4. Compute factor scores via standard regression method (Z * R^-1 * L)
  Z = (X - X.mean()) / X.std(ddof=0)
  R_inv = np.linalg.pinv(Z.corr().values)
  L = np.real(res.loadings)
  W = np.dot(R_inv, L)

  scores = np.dot(Z.values, W)

  # 5. Construct final DataFrame with metadata, PM10, and factor scores
  df_export = pd.DataFrame(
      scores,
      index=df_clean.index,
      columns=[
          "Factor_Combustion",
          "Factor_Photochemical",
          "Factor_Moisture",
      ],
  )

  # Insert metadata and target variable at the beginning
  df_export.insert(0, target_col, df_clean[target_col])
  df_export.insert(0, station_col, df_clean[station_col])
  df_export.insert(0, timestamp_col, df_clean[timestamp_col])

  return df_export

def main():
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(
        SCRIPT_DIR, "..", "BasesDeDatosParquet", "SIMA_Diario_Imputado.parquet"
    )

    df = pd.read_parquet(path=path)
    dfcopy = df
    df = df.select_dtypes(include=['float64', 'float32']).copy()
    df.drop(columns=['Mes', 'Hora', 'PM10', 'NOX'], errors='ignore', inplace=True)
    selected_columns = kmo(df=df, threshold=0.6)
    print(f"""
    Variables seleccionadas 
    {selected_columns}
    """)
    scree_plot(df=df)
    factorial_analysis(df = df,n_factors = 4)
    df = df.drop(columns=['PRS', 'TOUT', 'SR'])
    scree_plot(df=df)
    factorial_analysis(df = df, n_factors = 3)
    fa = export_fa_dataset(df=dfcopy)
    path = os.path.join(
      SCRIPT_DIR, "..", "BasesDeDatosParquet", "factor_analysisi.parquet"
    )
    fa.to_parquet(path=path, engine='pyarrow', index=False)

if __name__ == "__main__":
    main()