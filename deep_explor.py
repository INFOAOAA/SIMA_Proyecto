
# en un principio, se usan todos los datos, durante este script se ejecutan observaciones generales de todas las variables
import pandas as pd
import numpy as np
import seaborn as sns
import pyarrow
import os
import plotly.figure_factory as ff
import matplotlib.pyplot as plt
from plotly.subplots import make_subplots
from scipy.stats import f_oneway


def standard_measurements(df: pd.DataFrame):
    means = df.mean(axis=0, numeric_only=True)
    median = df.median(axis=0, numeric_only=True)
    maximum = df.max(axis=0, numeric_only=True)
    minimum = df.min(axis=0, numeric_only=True)
    medium_range = (maximum + minimum)/2
    variances = df.var(axis=0, numeric_only=True)
    std = np.sqrt(variances)


    print('---------------------------------- \n',
          f'Medias de las variables: \n{means} \n',
          f'Medianas de las variables: \n{median} \n',
          f'Rango medio de las variables: \n{medium_range} \n',
          f'Varianzas de las variables: \n{variances} \n',
          f'Desviaciones estándar de las variables: \n{std} \n'
          '---------------------------------- \n')


def distribution_graph(df: pd.DataFrame):
    fig = make_subplots(rows=7, cols=2, subplot_titles=[f'Distribución de {str(var)}' for var in df.columns])

    for i,variable in enumerate(df.columns):
        row = (i // 2) +1
        column = (i %2) +1
        fig.add_histogram(x=df[variable], name = f'Distribución de {str(variable)}', row=row, col=column)

    fig.update_layout(height=2100, width=900, showlegend=False)
    fig.show()

def pie_chart(df: pd.DataFrame):
    counts = df.value_counts()
    fig, ax = plt.subplots()
    wedges, texts, autotexts = ax.pie(
        x=counts.values,
        labels=counts.index,
        colors=sns.color_palette('pastel'),
        startangle=90,
        autopct='%.0f%%',
        wedgeprops=dict(width=0.5)
    )
    ax.legend(wedges, counts.index, title="Calidad del aire", loc="center left", bbox_to_anchor=(1, 0, 0.5, 1))
    plt.show()

def residual_graph(df: pd.DataFrame):
    fig = make_subplots()

def anova_by_stations(df: pd.DataFrame):
    f_statistic, p_value = f_oneway()

def visualize_options(data: pd.DataFrame):
    print('''
    Visualizar:
    1. Pie chart (Calidad de aire)
    2. Medidas estándar
    3. Gráfica de distribución
    4. Anova por estación
    q) quit
    ''')
    opt = input().strip()
    nums = data[['CO','NO','NO2','NOX','O3','PM10','PM2.5','PRS','RH','SO2','SR','TOUT','WSR','WDR']]

    match opt:
        case '1':
            pie_chart(data['Calidad_Aire_PM10'])
            visualize_options(data=data)
        case '2': 
            standard_measurements(nums)
            visualize_options(data=data)
        case '3':
            distribution_graph(nums)
            visualize_options(data=data)
        case '4':
            anova_by_stations(nums)
        case 'q':
            pass
        case _: 
            visualize_options(data=data)

        
def main():
    path = os.path.abspath('BasesDeDatosParquet/BD_no_imputation.parquet')
    data = pd.read_parquet(path=path)
    visualize_options(data=data)
    #standard_measurements(data)    
    #distribution_graph(data)       ## Las distribuciones siguen una similar a la normal,
                                    ## Aun así, se cree que el resultado de valores tan grandes y menores uno detras del otro
                                    ## podría ser la consecuencia de la imputación del dataset completo
                                    ## sugerencia: imputar por estación, confirmación: prueba anova





if __name__ == '__main__':
    main()


    