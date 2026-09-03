import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def seasonal_behaviour(df: pd.DataFrame):
    # Set visual style
    sns.set_theme(style="whitegrid")

    # 1. Boxplot comparison for PM10 across all stations
    plt.figure(figsize=(14, 6))
    sns.boxplot(
        data=df,
        x="Estacion",
        y="PM10",
        palette="viridis",
        showfliers=False,  # Hide extreme outliers for clearer scaling
    )
    plt.title(
        "PM10 Distribution by Monitoring Station", fontsize=14, fontweight="bold"
    )
    plt.xlabel("Station")
    plt.ylabel("PM10 (µg/m³)")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

    # 2. Compare Multiple Variables Across Stations via Faceted Density Plots
    vars_to_compare = ["PM10", "WSR", "RH", "TOUT"]
    df_melted = df.melt(
        id_vars=["Estacion"], value_vars=vars_to_compare, var_name="Variable"
    )

    g = sns.FacetGrid(
        df_melted,
        col="Variable",
        hue="Estacion",
        palette="tab20",
        col_wrap=2,
        height=4,
        aspect=1.5,
        sharey=False,
    )
    g.map(sns.kdeplot, "value", alpha=0.4)
    g.add_legend(title="Station")
    g.fig.suptitle(
        "Variable Densities Across Stations", y=1.02, fontsize=14, fontweight="bold"
    )
    plt.show()

def main():
    pd.read_parquet("hourly_database.parquet")
    