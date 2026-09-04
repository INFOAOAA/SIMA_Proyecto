import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def generar_graficos_eda():
    # 1. Cargar datos limpios
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(SCRIPT_DIR, "..", "Documento", "SIMA_Diario_Limpio.parquet")
    df = pd.read_parquet(path)
    
    # Filtrar el triplete representativo de Mahalanobis
    estaciones_top = ['NO2', 'NE3', 'SE3']
    df_top = df[df['Estacion'].isin(estaciones_top)].copy()
    
    # Extraer mes para el análisis estacional
    df_top['Mes'] = df_top['Fecha'].dt.month
    
    # Configurar estilo visual académico
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)
    
    # ---------------------------------------------------------
    # FIGURA 1: Estacionalidad Invernal vs Estival (Boxplot)
    # ---------------------------------------------------------
    plt.figure(figsize=(12, 6))
    sns.boxplot(data=df_top, x='Mes', y='PM10', hue='Estacion', palette='Set2', fliersize=2)
    plt.title('Distribución Mensual de PM10 (Efecto de Inversión Térmica Invernal)')
    plt.xlabel('Mes del Año')
    plt.ylabel('Concentración de PM10 (µg/m³)')
    plt.legend(title='Estación')
    plt.tight_layout()
    plt.savefig(os.path.join(SCRIPT_DIR, 'Figura_1_Estacionalidad.png'), dpi=300)
    plt.close()

    # ---------------------------------------------------------
    # FIGURA 2: Matriz de Correlación Termodinámica
    # ---------------------------------------------------------
    plt.figure(figsize=(8, 6))
    vars_corr = ['PM10', 'PM2.5', 'TOUT', 'RH', 'WSR', 'PRS']
    matriz_corr = df_top[vars_corr].corr()
    
    # Crear un mapa de calor (heatmap)
    sns.heatmap(matriz_corr, annot=True, fmt=".2f", cmap="coolwarm", 
                vmin=-1, vmax=1, square=True, linewidths=.5)
    plt.title('Matriz de Correlación: Contaminantes y Meteorología')
    plt.tight_layout()
    plt.savefig(os.path.join(SCRIPT_DIR, 'Figura_2_Correlacion.png'), dpi=300)
    plt.close()

    # ---------------------------------------------------------
    # FIGURA 3: Efecto Dual del Viento (Scatter + Tendencia)
    # ---------------------------------------------------------
    plt.figure(figsize=(10, 6))
    # Dibujar puntos
    sns.scatterplot(data=df_top, x='WSR', y='PM10', hue='Estacion', alpha=0.4, palette='Set2')
    # Añadir línea de tendencia no lineal (LOWESS) para ver la resuspensión
    sns.regplot(data=df_top, x='WSR', y='PM10', scatter=False, color='black', 
                lowess=True, line_kws={'linestyle':'--'})
    
    plt.title('Relación Velocidad del Viento (WSR) vs PM10')
    plt.xlabel('Velocidad del Viento (km/h)')
    plt.ylabel('Concentración de PM10 (µg/m³)')
    plt.tight_layout()
    plt.savefig(os.path.join(SCRIPT_DIR, 'Figura_3_EfectoViento.png'), dpi=300)
    plt.close()

    print("Gráficos generados con éxito. Búscalos en la misma carpeta de este script.")

if __name__ == "__main__":
    generar_graficos_eda()