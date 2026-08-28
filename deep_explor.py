
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
import statsmodels.api as sm
from statsmodels.formula.api import ols
from scipy import stats
import pingouin as pg
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis, LinearDiscriminantAnalysis
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from scipy.cluster.hierarchy import dendrogram, fcluster, linkage
from sklearn.metrics import calinski_harabasz_score, silhouette_score
from sklearn.ensemble import RandomForestRegressor


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


def anova_by_stations(df: pd.DataFrame):
    # 1. Select numeric variables only and explicitly exclude Station/Datetime/ID columns
    cols_to_exclude = ['Estacion', 'Station', 'Fecha y hora', 'Fecha', 'hora']

    # Select columns that are floats/ints and NOT in our exclude list
    numeric_vars = [
        col
        for col in df.select_dtypes(include=[np.number]).columns
        if col not in cols_to_exclude
    ]

    results = []

    for var in numeric_vars:
        # Drop NAs specifically for the current variable and Station
        clean_df = df[[var, 'Estacion']].dropna()

        # Extract non-empty arrays for each station
        groups = [
            group[var].values
            for name, group in clean_df.groupby('Estacion')
            if len(group[var]) > 1
        ]

        # Ensure we have at least 2 station groups with valid data to test
        if len(groups) < 2:
            continue

        # Run Levene's Test for Homoscedasticity
        levene_stat, levene_p = stats.levene(*groups)

        # Check condition: If variances are equal (p > 0.05), use standard ANOVA
        if levene_p > 0.05:
            # Q() wraps variable names safely in case they contain spaces or special characters
            model = ols(f'Q("{var}") ~ C(Estacion)', data=clean_df).fit()
            anova_tbl = sm.stats.anova_lm(model, typ=2)
            p_val = anova_tbl.loc['C(Estacion)', 'PR(>F)']
            method = 'Standard ANOVA'
        else:
            # If variances are NOT equal (p <= 0.05), use Welch's ANOVA
            welch_res = stats.alexandergovern(*groups)
            p_val = welch_res.pvalue
            method = "Welch's ANOVA (Alexander-Govern)"

        results.append({
            'Variable': var,
            'Levene_p_value': round(float(levene_p), 4),
            'Homoscedastic': levene_p > 0.05,
            'Test_Used': method,
            'ANOVA_p_value': (
                round(float(p_val), 5) if not np.isnan(p_val) else np.nan
            ),
            'Significant_Diff': p_val < 0.05,
        })

    summary_df = pd.DataFrame(results)
    print(summary_df)
    return summary_df

def gaussian_discriminant_per_station(df: pd.DataFrame):
    cols_to_exclude = ['Estacion', 'Station', 'Fecha y hora', 'Fecha', 'hora', 'Mes', 'Hora']

    # Select columns that are floats/ints and NOT in our exclude list
    numeric_vars = [
        col
        for col in df.select_dtypes(include=[np.number]).columns
        if col not in cols_to_exclude
    ]
    X = df[numeric_vars].dropna()
    y = df.loc[X.index, 'Estacion']

    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.3, random_state=42
    )

    # Fit QDA model
    regulation_parameters = np.arange(0,1,0.1)
    results = []
    for par in regulation_parameters:
        try:
            qda = QuadraticDiscriminantAnalysis(reg_param=par)
            qda.fit(X_train, y_train)
            print(f'QDA Classification Accuracy: {qda.score(X_test, y_test):.4f} with {par}')

        except Exception as e:
            print(f'Regulation parameter {par} no fue posible \nException {e}')

            

    print(f'QDA Classification Accuracy: {max(results):.4f}')

def linear_discriminant_analysis_per_station(df: pd.DataFrame):
    cols_to_exclude = ['Estacion', 'Station', 'Fecha y hora', 'Fecha', 'hora', 'Mes', 'Hora']

    # Select columns that are floats/ints and NOT in our exclude list
    numeric_vars = [
        col
        for col in df.select_dtypes(include=[np.number]).columns
        if col not in cols_to_exclude
    ]
    X = df[numeric_vars].dropna()
    y = df.loc[X.index, 'Estacion']

    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    lda = LinearDiscriminantAnalysis(n_components=2)
    X_lda = lda.fit_transform(X_scaled, y)

    # Plot the 2D LDA projection
    plt.figure(figsize=(10, 6))
    for station in y.unique():
        mask = y == station
        plt.scatter(
            X_lda[mask, 0], X_lda[mask, 1], label=station, alpha=0.6, edgecolors='k'
        )

    plt.xlabel('Linear Discriminant 1')
    plt.ylabel('Linear Discriminant 2')
    plt.title('Station Separation via LDA')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.show()

def unsupervised_learning(df: pd.DataFrame):
    # 1. Define pollutant columns only (exclude meteorological noise)
    pollutant_cols = [
        'CO',
        'NO',
        'NO2',
        'NOX',
        'O3',
        'PM10',
        'PM2.5',
        'SO2',
    ]

    # Filter to only include pollutants present in the DataFrame
    available_pollutants = [col for col in pollutant_cols if col in df.columns]

    # 2. Compute means and stds for pollutants only
    station_means = (
        df.groupby('Estacion')[available_pollutants]
        .mean()
        .add_suffix('_mean')
    )
    station_stds = (
        df.groupby('Estacion')[available_pollutants]
        .std()
        .fillna(0)
        .add_suffix('_std')
    )

    # Combine profiles (only pollutant features)
    station_profiles = pd.concat([station_means, station_stds], axis=1).dropna()

    # 3. Scale feature matrix
    scaler = StandardScaler()
    scaled_profiles = scaler.fit_transform(station_profiles)

    # 4. Compute Ward's Hierarchical Linkage on pollutants
    Z = linkage(scaled_profiles, method='ward')

    clusters = range(3, 7)
    results = []

    for n_clusters in clusters:
        cluster_labels = fcluster(Z, t=n_clusters, criterion='maxclust')

        # Compute validation metrics
        sil_score = silhouette_score(scaled_profiles, cluster_labels)
        ch_score = calinski_harabasz_score(scaled_profiles, cluster_labels)

        # Build assignment mapping
        cluster_df = pd.DataFrame({
            'Estacion': station_profiles.index,
            'Cluster': [f'Group_{label}' for label in cluster_labels],
        })

        print(f'\n--- Station Cluster Assignments (k={n_clusters}) ---')
        print(cluster_df.sort_values(by='Cluster'))
        print(f'Silhouette Score: {sil_score:.4f}')
        print(f'Calinski-Harabasz Index: {ch_score:.2f}')

        # Plot Dendrogram with cut line
        plt.figure(figsize=(10, 6))
        dendrogram(
            Z,
            labels=station_profiles.index.tolist(),
            leaf_rotation=45,
            leaf_font_size=11,
        )
        plt.title(
            f'Hierarchical Station Clustering - Pollutants Only (k={n_clusters})'
        )
        plt.xlabel('Station')
        plt.ylabel('Euclidean Distance')
        plt.axhline(
            y=Z[-n_clusters + 1, 2],
            color='r',
            linestyle='--',
            label=f'Cut Threshold (k={n_clusters})',
        )
        plt.legend()
        plt.tight_layout()
        plt.show()

        results.append({
            'n_clusters': n_clusters,
            'silhouette': sil_score,
            'calinski_harabasz': ch_score,
            'assignments': cluster_df,
        })

    return results

def select_features_per_station(df: pd.DataFrame, target_var: str = 'PM10', top_n: int = 7):
    """Calculates Random Forest feature importance for PM10 per individual station.

    Returns:
    - summary_df: DataFrame mapping each station to its top recommended
    regressors.
    - all_importances: Dictionary of feature importance Series for each
    station.
    """
    candidate_features = [
        'TOUT',
        'RH',
        'WSR',
        'WDR',
        'PRS',
        'SR',  # Weather
        'CO',
        'NO',
        'NO2',
        'NOX',
        'O3',
        'SO2',  # Co-pollutants
    ]

    valid_features = [c for c in candidate_features if c in df.columns]

    stations = df['Estacion'].unique()
    all_importances = {}
    summary_data = []

    # Calculate grid size for subplots
    n_stations = len(stations)
    cols = 3
    rows = (n_stations + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(15, 4 * rows), sharex=True)
    axes = axes.flatten()

    for idx, station in enumerate(stations):
        station_data = df[df['Estacion'] == station].copy()

        # Clean missing values for this specific station
        clean_data = station_data[[target_var] + valid_features].dropna()

        # Skip station if not enough clean samples
        if len(clean_data) < 100:
            print(
                f'Skipping station {station}: Insufficient clean rows ({len(clean_data)}).'
            )
            continue

        X = clean_data[valid_features]
        # Log-transform target variable to stabilize PM10 variance
        y = np.log1p(clean_data[target_var])

        # Fit Random Forest Regressor
        rf = RandomForestRegressor(
            n_estimators=100, max_depth=10, random_state=42, n_jobs=-1
        )
        rf.fit(X, y)

        importances = pd.Series(
            rf.feature_importances_, index=X.columns
        ).sort_values(ascending=False)
        all_importances[station] = importances

        top_features = importances.head(top_n).index.tolist()

        summary_data.append({
            'Station': station,
            'Top_1_Regressor': top_features[0] if len(top_features) > 0 else None,
            'Top_2_Regressor': top_features[1] if len(top_features) > 1 else None,
            'Top_3_Regressor': top_features[2] if len(top_features) > 2 else None,
            'Samples_Used': len(clean_data),
        })

        # Subplot visualization
        ax = axes[idx]
        importances.head(6).plot(kind='barh', ax=ax, color='teal')
        ax.invert_yaxis()
        ax.set_title(f'Station: {station}', fontsize=12, fontweight='bold')
        ax.grid(axis='x', linestyle='--', alpha=0.6)

    # Turn off unused subplots
    for j in range(idx + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.suptitle(
        f'Top Feature Importances for log1p({target_var}) by SIMA Station',
        fontsize=16,
        y=1.02,
    )
    plt.tight_layout()
    plt.show()

    summary_df = pd.DataFrame(summary_data)

    print('\n==================================================')
    print(f'   PROPHET REGRESSOR RECOMMENDATIONS PER STATION')
    print('==================================================')
    print(summary_df.to_string(index=False))

    return summary_df, all_importances


def visualize_options(data: pd.DataFrame):
    print('''
    Visualizar:
    1. Pie chart (Calidad de aire)
    2. Medidas estándar
    3. Gráfica de distribución
    4. Anova por estación
    5. Gaussian Discriminant Analysis (Por estación)
    6. Linear Discriminant Analysis (Por estación)
    7. Unsupervised para estación
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
            anova_by_stations(df=data)
            visualize_options(data=data)
        case '5': 
            gaussian_discriminant_per_station(df=data)
            visualize_options(data=data)
        case '6':
            linear_discriminant_analysis_per_station(df=data)
            visualize_options(data=data)
        case '7':
            unsupervised_learning(df=data)
            visualize_options(data=data)
        case '8':
            select_features_per_station(df = data)
            visualize_options(data=data)
        case 'q':
            pass
        case _: 
            visualize_options(data=data)

        
def main():
    path = os.path.abspath('BasesDeDatosParquet/BD_no_standarization.parquet')
    data = pd.read_parquet(path=path)
    visualize_options(data=data)
    #standard_measurements(data)    
    #distribution_graph(data)       ## Las distribuciones siguen una similar a la normal,
                                    ## Aun así, se cree que el resultado de valores tan grandes y menores uno detras del otro
                                    ## podría ser la consecuencia de la imputación del dataset completo
                                    ## sugerencia: imputar por estación, confirmación: prueba anova





if __name__ == '__main__':
    main()


    