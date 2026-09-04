import numpy as np
import pandas as pd
import pmdarima as pm
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.statespace.sarimax import SARIMAX, SARIMAXResults
from sklearn.metrics import mean_absolute_error, mean_squared_error
from statsmodels.tsa.seasonal import seasonal_decompose

import os
def fit_and_evaluate_daily_sarimax(
    df: pd.DataFrame,
    target_col: str = "PM10",
    date_col: str = "Fecha",
    exog_cols: list[str] = None,
    seasonal_period: int = 7,
    station: str = "SE3",
    test_days: int = 7,
):
  """Resamples data to daily means, runs auto_arima, evaluates out-of-sample

  performance on a holdout test set, fits the full model, and saves it.
  """
  if exog_cols is None:
    exog_cols = []

  # 1. Datetime Index & Daily Aggregation
  df_clean = df.copy()
  df_clean[date_col] = pd.to_datetime(df_clean[date_col])

  cols_to_select = [target_col] + exog_cols
  df_daily = (
      df_clean.set_index(date_col)[cols_to_select]
      .resample("D")
      .mean()
      .dropna()
  )

  y_daily = df_daily[target_col]
  X_daily = df_daily[exog_cols] if exog_cols else None

  print(f"=== Daily Dataset Prepared: {len(y_daily)} days ===")

  # 2. Decomposition
  decomp = seasonal_decompose(y_daily, model="additive", period=seasonal_period)
  print(
      f"Decomposition complete for {target_col} (Period = {seasonal_period} days)."
  )

  # 3. Train/Test Split for Evaluation
  train_df = df_daily.iloc[:-test_days]
  test_df = df_daily.iloc[-test_days:]

  y_train, X_train = (
      train_df[target_col],
      (train_df[exog_cols] if exog_cols else None),
  )
  y_test, X_test = (
      test_df[target_col],
      (test_df[exog_cols] if exog_cols else None),
  )

  # 4. Stepwise Auto-ARIMA Search (on Training Set)
  print(
      f"\n=== Running pmdarima Search (Train Period: {len(y_train)} days) ==="
  )
  auto_model = pm.auto_arima(
        y=y_train,
        X=X_train,
        seasonal=True,
        m=seasonal_period,
        max_p=2,  # Limit non-seasonal orders
        max_q=2,
        max_P=1,  # Restrict seasonal AR to max lag 1 (7 days)
        max_Q=1,  # Restrict seasonal MA to max lag 1 (7 days)
        stepwise=True,
        suppress_warnings=True,
        error_action="ignore",
        trace=True,
    )

  print(f"\n✅ Optimal SARIMAX Order: {auto_model.order}")
  print(f"✅ Optimal Seasonal Order: {auto_model.seasonal_order}")

  # 5. Out-of-Sample Evaluation
  eval_model = SARIMAX(
      endog=y_train,
      exog=X_train,
      order=auto_model.order,
      seasonal_order=auto_model.seasonal_order,
      enforce_stationarity=False,
      enforce_invertibility=False,
  ).fit(maxiter=200, disp=False)

  predictions = eval_model.forecast(steps=test_days, exog=X_test)

  mae = mean_absolute_error(y_test, predictions)
  rmse = np.sqrt(mean_squared_error(y_test, predictions))
  mape = np.mean(np.abs((y_test - predictions) / y_test)) * 100

  print("\n==================================================================")
  print(f"       OUT-OF-SAMPLE EVALUATION ({test_days} DAYS HOLDOUT)        ")
  print("==================================================================")
  print(f"MAE:  {mae:.3f} µg/m³")
  print(f"RMSE: {rmse:.3f} µg/m³")
  print(f"MAPE: {mape:.2f}%")

  # 6. Fit Final Model on Entire Dataset (Train + Test)
  print("\n=== Fitting Final Production Model on Full Dataset ===")
  final_model = SARIMAX(
      endog=y_daily,
      exog=X_daily,
      order=auto_model.order,
      seasonal_order=auto_model.seasonal_order,
      enforce_stationarity=False,
      enforce_invertibility=False,
  )
  results = final_model.fit(maxiter=200, disp=False)

  print("\n==================================================================")
  print(f"     FINAL SARIMAX DAILY MODEL REPORT ({station} - {target_col})   ")
  print("==================================================================")
  print(results.summary())

  # 7. Save Model Artifact
  SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
  models_dir = os.path.join(SCRIPT_DIR, "..", "models")
  os.makedirs(models_dir, exist_ok=True)

  model_path = os.path.join(
      models_dir, f"{station}_sarimax_{target_col}.pickle"
  )
  results.save(model_path)
  print(f"\n✅ Model saved to {model_path}")

  # Prepare evaluation comparison table
  eval_df = pd.DataFrame(
      {"Actual": y_test, "Forecast": predictions}, index=test_df.index
  )

  metrics = {"MAE": mae, "RMSE": rmse, "MAPE": mape}

  return results, df_daily, decomp, eval_df, metrics

def no_fact():
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(
    SCRIPT_DIR, "..","BasesDeDatosParquet", "SIMA_Diario_Imputado.parquet"
    )
    data = pd.read_parquet(path=path)
    SE3 = data[data['Estacion'] == 'SE3'].copy()
    SE3 = SE3[['Fecha','PM10', 'RH', 'NO2', 'CO','SR','PRS']]
    fit_and_evaluate_daily_sarimax(SE3,exog_cols= ['RH', 'NO2', 'CO','SR','PRS'])
    NO2 = data[data['Estacion'] == 'NO2']
    NO2 = NO2[['']]

def fact():
  SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

  # Correct file path relative to SIMA script location
  path = os.path.join(
      SCRIPT_DIR, "..", "BasesDeDatosParquet", "factor_analysisi.parquet"
  )
  data = pd.read_parquet(path=path)

  cols = [
      "Factor_Combustion",
      "Factor_Photochemical",
      "Factor_Moisture",
  ]

  # Top 3 representative SIMA stations
  target_stations = ["SE3", "NO2", "NE3"]

  for station in target_stations:
    print(f"\n================ Running Model for Station: {station} ================")
    station_df = data[data["Estacion"] == station].copy()

    if not station_df.empty:
      fit_and_evaluate_daily_sarimax(
          station_df,
          exog_cols=cols,
          station=station,
      )
    else:
      print(f"⚠️ Warning: No records found for station {station} in parquet dataset.")


def main():
    no_fact()
    


if __name__ == "__main__":
  main()