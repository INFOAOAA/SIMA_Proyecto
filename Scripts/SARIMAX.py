import os
import numpy as np
import pandas as pd
import pmdarima as pm
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def fit_daily_sarimax_ma(
    df: pd.DataFrame,
    target_col: str = "PM10",
    date_col: str = "Fecha",
    exog_cols: list[str] = None,
    seasonal_period: int = 7,
    test_size: int = 30,
    rolling_window: int = 3
):
    if exog_cols is None:
        exog_cols = []

    # 1. Preprocesamiento y serie continua
    df_clean = df.copy()
    df_clean[date_col] = pd.to_datetime(df_clean[date_col])
    
    cols_a_usar = [target_col] + exog_cols
    df_daily = df_clean.set_index(date_col)[cols_a_usar].resample("D").mean()
    df_daily = df_daily.interpolate(method='linear').ffill().bfill()
    df_daily.index.freq = 'D'

    # 2. Aplicar Promedios Móviles a variables exógenas
    exog_cols_ma = []
    if exog_cols:
        for col in exog_cols:
            col_ma = f"{col}_MA{rolling_window}"
            df_daily[col_ma] = df_daily[col].rolling(window=rolling_window).mean()
            exog_cols_ma.append(col_ma)
        
        # Rellenar los primeros días vacíos generados por la ventana móvil
        df_daily[exog_cols_ma] = df_daily[exog_cols_ma].bfill()

    y = df_daily[target_col]
    X = df_daily[exog_cols_ma] if exog_cols_ma else None

    # 3. División Train / Test
    y_train, y_test = y.iloc[:-test_size], y.iloc[-test_size:]
    X_train = X.iloc[:-test_size] if X is not None else None
    X_test = X.iloc[-test_size:] if X is not None else None

    # 4. Búsqueda con Auto-ARIMA
    auto_model = pm.auto_arima(
        y=y_train,
        X=X_train,
        seasonal=True,
        m=seasonal_period,
        stepwise=True,
        suppress_warnings=True,
        error_action="ignore",
        trace=False
    )

    # 5. Ajuste del modelo SARIMAX
    sarimax_model = SARIMAX(
        endog=y_train,
        exog=X_train,
        order=auto_model.order,
        seasonal_order=auto_model.seasonal_order,
        enforce_stationarity=True,
        enforce_invertibility=True
    )
    results = sarimax_model.fit(disp=False)

    # 6. Predicción Out-of-Sample
    predictions = results.predict(
        start=len(y_train),
        end=len(y_train) + len(y_test) - 1,
        exog=X_test
    )

    # 7. Cálculo de Métricas
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    mae = mean_absolute_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)
    mape = np.mean(np.abs((y_test.values - predictions.values) / np.maximum(np.abs(y_test.values), 1e-8))) * 100

    df_metricas = pd.DataFrame([
        {'Métrica': 'RMSE', 'Valor': round(rmse, 4)},
        {'Métrica': 'MAE', 'Valor': round(mae, 4)},
        {'Métrica': 'MAPE (%)', 'Valor': round(mape, 4)},
        {'Métrica': 'R²', 'Valor': round(r2, 4)},
        {'Métrica': 'AIC', 'Valor': round(results.aic, 4)},
        {'Métrica': 'BIC', 'Valor': round(results.bic, 4)}
    ])

    # 8. Importancia de Exógenas
    params = results.params
    pvalues = results.pvalues
    bse = results.bse

    exog_params = [col for col in exog_cols_ma if col in params.index]
    
    if exog_params and X_train is not None:
        std_x = X_train[exog_params].std().values
        df_imp = pd.DataFrame({
            'Variable': exog_params,
            'Coeficiente (Beta)': [round(params[c], 4) for c in exog_params],
            'Error Est.': [round(bse[c], 4) for c in exog_params],
            'p-value': [round(pvalues[c], 4) for c in exog_params],
            'Impacto Abs. (1 DE)': [round(abs(params[c] * std_x[i]), 4) for i, c in enumerate(exog_params)],
            'Significativo': [pvalues[c] < 0.05 for c in exog_params]
        }).sort_values(by='Impacto Abs. (1 DE)', ascending=False)
    else:
        df_imp = pd.DataFrame(columns=['Variable', 'Coeficiente (Beta)', 'p-value', 'Significativo'])

    print("\n==================================================================")
    print(f"   SARIMAX {auto_model.order} x {auto_model.seasonal_order} (Ventana MA: {rolling_window} días)")
    print("==================================================================")
    print("1. MÉTRICAS DE EVALUACIÓN (OUT-OF-SAMPLE)")
    print("------------------------------------------------------------------")
    print(df_metricas.to_string(index=False))
    
    print("\n2. IMPORTANCIA DE VARIABLES EXÓGENAS (ACUMULADAS)")
    print("------------------------------------------------------------------")
    if not df_imp.empty:
        print(df_imp.to_string(index=False))
    print("==================================================================\n")

    return results, df_metricas, df_imp

def main():
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(SCRIPT_DIR, "..", "BasesDeDatosParquet", "SIMA_Diario_Imputado.parquet")
    
    data = pd.read_parquet(path=path)
    SE3 = data[data['Estacion'] == 'SE3'].copy()
    
    # Variables sin la Radiación Solar (SR)
    exog_variables = ['RH', 'NO2', 'CO', 'PRS']
    
    # Prueba ajustando el parámetro rolling_window (ej. 3 o 7 días)
    results, df_m, df_i = fit_daily_sarimax_ma(
        df=SE3,
        target_col='PM10',
        exog_cols=exog_variables,
        seasonal_period=7,
        test_size=30,
        rolling_window=3
    )

if __name__ == "__main__":
    main()