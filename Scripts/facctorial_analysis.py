import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from factor_analyzer import FactorAnalyzer, calculate_kmo


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




def main():
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(
        SCRIPT_DIR, "..", "BasesDeDatosParquet", "hourly_database.parquet"
    )
    df = pd.read_parquet(path=path)
    df = df.select_dtypes(include=['float64', 'float32']).copy()
    df.drop(columns=['Mes', 'Hora', 'NOX'], errors='ignore', inplace=True)
    selected_columns = kmo(df=df, threshold=0.6)
    print(f"""
    Variables seleccionadas 
    {selected_columns}
    """)
    scree_plot(df=df)
    factorial_analysis(df = df)

if __name__ == "__main__":
    main()