import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def modelar_random_forest(df: pd.DataFrame, estacion: str, target_var: str = 'PM10', lags: list = [1, 2, 3, 7]):
    # Filtrar por estación y ordenar cronológicamente
    sub_df = df[df['Estacion'] == estacion].sort_values('Fecha').copy()

    # Generar retardos temporales (lags)
    for lag in lags:
        sub_df[f'{target_var}_lag_{lag}'] = sub_df[target_var].shift(lag)

    sub_df = sub_df.dropna().reset_index(drop=True)

    # Excluir metadatos e identificadores
    cols_excluir = ['Fecha', 'Estacion', 'Calidad_Aire_PM10', target_var]
    features = [c for c in sub_df.columns if c not in cols_excluir]

    # División temporal 80% train / 20% test (respetando la secuencia de tiempo)
    split_idx = int(len(sub_df) * 0.8)
    train_df, test_df = sub_df.iloc[:split_idx], sub_df.iloc[split_idx:]

    X_train, y_train = train_df[features], train_df[target_var]
    X_test, y_test = test_df[features], test_df[target_var]

    # Entrenamiento del modelo
    rf = RandomForestRegressor(n_estimators=150, max_depth=12, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)

    # Evaluación
    preds = rf.predict(X_test)
    metrics = {
        'RMSE': np.sqrt(mean_squared_error(y_test, preds)),
        'MAE': mean_absolute_error(y_test, preds),
        'R2': r2_score(y_test, preds)
    }

    importances = pd.Series(rf.feature_importances_, index=features).sort_values(ascending=False)
    return rf, importances, metrics, test_df['Fecha'], y_test, preds



import pandas as pd
import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import adfuller
from sklearn.metrics import mean_squared_error, mean_absolute_error

def verificar_estacionariedad(serie: pd.Series):
    p_value = adfuller(serie.dropna())[1]
    return p_value, p_value < 0.05

def modelar_sarimax(df: pd.DataFrame, estacion: str, target_var: str = 'PM10', exog_vars: list = ['TOUT', 'RH', 'WSR', 'NO2'], order=(1, 1, 1), seasonal_order=(1, 0, 1, 7)):
    sub_df = df[df['Estacion'] == estacion].sort_values('Fecha').set_index('Fecha').copy()

    y = sub_df[target_var]
    X = sub_df[exog_vars]

    # División train / test cronológica
    split_idx = int(len(sub_df) * 0.8)
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]

    # Ajuste del modelo SARIMAX
    model = SARIMAX(
        endog=y_train,
        exog=X_train,
        order=order,
        seasonal_order=seasonal_order,
        enforce_stationarity=False,
        enforce_invertibility=False
    )
    res = model.fit(disp=False)

    # Predicción fuera de muestra (out-of-sample)
    preds = res.predict(start=len(y_train), end=len(y_train) + len(y_test) - 1, exog=X_test)

    metrics = {
        'RMSE': np.sqrt(mean_squared_error(y_test, preds)),
        'MAE': mean_absolute_error(y_test, preds)
    }

    return res, preds, y_test, metrics


if __name__ == "__main__":
    df = pd.read_parquet('Documento\\SIMA_Diario_Imputado.parquet')

    # Estaciones clave elegidas (ejemplo proveniente del script de Mahalanobis)
    top_estaciones = ['Cadereyta', 'Obispado', 'San Nicolas']

    resultados = []
    for est in top_estaciones:
        # 1. Random Forest
        rf_model, importances, rf_metrics, fe, y_real, rf_preds = modelar_random_forest(df, estacion=est)

        # Tomar las mejores 4 exógenas según Random Forest para SARIMAX
        top_exog = [f for f in importances.index if 'lag' not in f][:4]

        # 2. SARIMAX
        sarimax_res, sarimax_preds, y_test, sarimax_metrics = modelar_sarimax(
            df, estacion=est, exog_vars=top_exog
        )

        resultados.append({
            'Estacion': est,
            'RF_RMSE': rf_metrics['RMSE'],
            'RF_R2': rf_metrics['R2'],
            'SARIMAX_RMSE': sarimax_metrics['RMSE'],
            'Top_Exog_Vars': top_exog
        })

    df_resultados = pd.DataFrame(resultados)
    print(df_resultados)