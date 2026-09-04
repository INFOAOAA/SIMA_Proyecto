import numpy as np
import pandas as pd
import pmdarima as pm
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.statespace.sarimax import SARIMAX
import os

def fit_daily_sarimax(
    df: pd.DataFrame,
    target_col: str = "PM10",
    date_col: str = "Fecha",
    exog_cols: list[str] = 
    [],
    seasonal_period: int = 7,  # 7 for daily data (weekly cycle)
    ):
  """Resamples SIMA data to daily means, runs auto_arima to identify optimal orders,

  and fits a SARIMAX model with exogenous factor scores.
  """
  # 1. Ensure datetime index and resample to Daily Means
  df_clean = df.copy()
  df_clean[date_col] = pd.to_datetime(df_clean[date_col])
  df_daily = (
      df_clean.set_index(date_col)
      .resample("D")[[target_col] + exog_cols]
      .mean()
      .dropna()
  )

  y_daily = df_daily[target_col]
  X_daily = df_daily[exog_cols]

  print(f"=== Daily Dataset Prepared: {len(y_daily)} days ===")

  # 2. Decomposition Check (Optional Diagnostic)
  decomp = seasonal_decompose(y_daily, model="additive", period=seasonal_period)
  print(
      f"Decomposition complete for {target_col} (Period = {seasonal_period} days)."
  )

  # 3. Stepwise Auto-ARIMA Search for Optimal (p,d,q)x(P,D,Q)s
  print("\n=== Running pmdarima Auto-ARIMA Search ===")
  auto_model = pm.auto_arima(
      y=y_daily,
      X=X_daily,
      seasonal=True,
      m=seasonal_period,  # Weekly seasonality on daily data
      stepwise=True,
      suppress_warnings=True,
      error_action="ignore",
      trace=True,
  )

  print(f"\n✅ Optimal SARIMAX Order: {auto_model.order}")
  print(f"✅ Optimal Seasonal Order: {auto_model.seasonal_order}")

  # 4. Fit Final SARIMAX via statsmodels using Auto-ARIMA Order
  sarimax_model = SARIMAX(
      endog=y_daily,
      exog=X_daily,
      order=auto_model.order,
      seasonal_order=auto_model.seasonal_order,
      enforce_stationarity=False,
      enforce_invertibility=False,
  )
  results = sarimax_model.fit(disp=False)

  # 5. Print Full Statistical Report
  print("\n==================================================================")
  print(f"         SARIMAX DAILY MODEL REPORT ({target_col})               ")
  print("==================================================================")
  print(results.summary())

  return results, df_daily, decomp

def main():
  SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
  path = os.path.join(
    SCRIPT_DIR, "..","BasesDeDatosParquet", "SIMA_Diario_Imputado.parquet"
  )
  data = pd.read_parquet(path=path)
  SE3 = data[data['Estacion'] == 'SE3'].copy()
  SE3 = SE3[['Fecha','PM10','NOX', 'RH', 'NO2', 'CO','SR','PRS']]
  fit_daily_sarimax(SE3,exog_cols= ['NOX', 'RH', 'NO2', 'CO','SR','PRS'])

if __name__ == "__main__":
  main()