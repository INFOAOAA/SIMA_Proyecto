import pandas as pd
import pyarrow
import os
import numpy as np

def clean_database():
    datas = [pd.read_csv(os.path.abspath(f'BasesDeDatosCsv/BD_202{i}_Unido.csv')) for i in range(0,6)]
    datas[5].rename(columns={'date': 'Fecha y hora'}, inplace=True)

    complete_database = pd.concat(datas, axis=0)

    complete_database["Fecha y hora"] = pd.to_datetime(complete_database['Fecha y hora'], yearfirst=True)

    months = complete_database["Fecha y hora"].dt.month

    bins = [0, 2, 5, 8, 11, 12]
    labels = ["Winter", "Spring", "Summer", "Fall", "Winter"]

    complete_database["season"] = pd.cut(
        months, bins=bins, labels=labels, ordered=False, include_lowest=True
    )




    print(f'{(complete_database.isna().sum()/complete_database.shape[0])*100} %')

    print("--- Información de las variables ---")
    complete_database.info()

    # c) Calidad de los datos: Valores nulos y duplicados
    print("\n--- Porcentaje de Valores Nulos por Variable ---")
    nulos_pct = (complete_database.isna().sum() / len(complete_database)) * 100
    print(nulos_pct.round(2).astype(str) + ' %')

    print(f"\nCantidad de registros duplicados: {complete_database.duplicated().sum()}")

    from sklearn.experimental import enable_iterative_imputer
    from sklearn.impute import IterativeImputer

    # Primero, debemos crear df_prep a partir de df_2022
    df_prep = complete_database.drop_duplicates().copy()

    # Eliminar variable RAINF por carecer de varianza
    df_prep = df_prep.drop(columns=['RAINF'])

    # Tratamiento de Fechas y preparación para imputar
    df_prep['Fecha y hora'] = pd.to_datetime(df_prep['Fecha y hora'])
    df_prep['Mes'] = df_prep['Fecha y hora'].dt.month
    df_prep['Hora'] = df_prep['Fecha y hora'].dt.hour

    # Extraemos la fecha temporalmente ya que el imputador no acepta Datetime
    fechas = df_prep['Fecha y hora']
    season = df_prep['season']
    df_prep = df_prep.drop(columns=['Fecha y hora', 'season'])

    # Manejo de Datos Categóricos (Variables Dummy)
    df_prep = pd.get_dummies(df_prep, columns=['Estacion'], drop_first=True)

    # Imputación de datos faltantes (IterativeImputer)
    print("Iniciando imputación iterativa (esto puede tomar un momento)...")
    imp = IterativeImputer(max_iter=10, random_state=42)

    # Guardamos el nombre de las columnas
    columnas = df_prep.columns

    df_imputado_array = imp.fit_transform(df_prep)

    # Reconstruimos el DataFrame
    df_clean = pd.DataFrame(df_imputado_array, columns=columnas)

    # Reintegramos la columna de Fecha y hora
    df_clean.insert(0, 'Fecha y hora', fechas.values)

    df_clean.insert(0, 'season', season.values)

    print("Imputación finalizada sin nulos restantes")

    cols_outliers = ['PM10', 'PM2.5', 'O3', 'CO']

    for col in cols_outliers:
        Q1 = df_clean[col].quantile(0.25)
        Q3 = df_clean[col].quantile(0.75)
        IQR = Q3 - Q1
        limite_inf = Q1 - 1.5 * IQR
        limite_sup = Q3 + 1.5 * IQR
        
        # Aplicar Capping (Clipping): limitar los valores a los rangos calculadaos
        df_clean[col] = np.where(df_clean[col] > limite_sup, limite_sup, df_clean[col])
        df_clean[col] = np.where(df_clean[col] < limite_inf, limite_inf, df_clean[col])

    print("Outliers tratados mediante método de Clipping (IQR).")

    from sklearn.preprocessing import StandardScaler

    complete_database = df_clean
    # Discretizar datos (Binning) - Creando categorías de Calidad del Aire para PM10
    bins_pm10 = [0, 50, 100, np.inf]
    etiquetas = ['Buena', 'Regular', 'Mala']
    complete_database['Calidad_Aire_PM10'] = pd.cut(complete_database['PM10'], bins=bins_pm10, labels=etiquetas)

    complete_database = pd.get_dummies(complete_database, columns=['season'], drop_first=True)

    # Escalar y normalizar los datos (StandardScaler)
    cols_numericas = ['CO', 'NO', 'NO2', 'NOX', 'O3', 'PM10', 'PM2.5', 'PRS', 'RH', 'SR', 'TOUT', 'WSR', 'WDR']

    scaler = StandardScaler()
    # Creamos copias escaladas para mantener la legibilidad de la base principal si se desea, 
    complete_database[cols_numericas] = scaler.fit_transform(complete_database[cols_numericas])

    # Guardado de la base final reestructurada
    path = os.path.abspath('BasesDeDatosParquet/BD_Completa.parquet')

    import polars as pl

    complete_database.to_parquet(path, engine = 'pyarrow', index = False)



    print("Base de datos limpia y transformada guardada como 'BD_Completa.parquet'")


if __name__ == '__main__':
    clean_database()