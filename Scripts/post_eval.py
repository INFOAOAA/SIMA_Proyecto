import pandas as pd
import numpy as np
from scipy import stats
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.stats.diagnostic import acorr_ljungbox
from pmdarima import auto_arima
import os


def compute_smape(y_true, y_pred):
    """Calculates Symmetric Mean Absolute Percentage Error (sMAPE)."""
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2.0
    mask = denominator != 0
    if not np.any(mask):
        return 0.0
    return np.mean(np.abs(y_pred[mask] - y_true[mask]) / denominator[mask]) * 100


def prepare_lagged_factors(df, factor_cols, max_lag=1):
    """Creates lagged features for exogenous latent factors."""
    df_lagged = df.copy()
    for col in factor_cols:
        for lag in range(1, max_lag + 1):
            df_lagged[f"{col}_lag{lag}"] = df_lagged[col].shift(lag)
    return df_lagged.dropna()


def evaluate_station_sarimax(
    df,
    target_col,
    factor_cols,
    d_override=None,
    n_splits=3,
    test_horizon=14,
):
    """
    Fits SARIMAX with lagged factor features, runs expanding window CV,
    and returns MAE, MAPE, sMAPE, and Pearson correlation metrics.
    """
    # 1. Feature Engineering: Create Lagged Latent Factors
    data = prepare_lagged_factors(df, factor_cols, max_lag=1)
    exog_cols = factor_cols + [f"{c}_lag1" for c in factor_cols]

    y = data[target_col]
    X = data[exog_cols]

    # 2. Automatically select optimal ARIMA orders
    print(f"\n==========================================")
    print(f"  Optimizing Model for Station: {target_col}")
    print(f"==========================================")

    stepwise_model = auto_arima(
        y,
        X=X,
        d=d_override,  # Forces d=1 for NE3 to clear residual autocorrelation
        seasonal=True,
        m=7,  # Weekly cycle
        stepwise=True,
        suppress_warnings=True,
        error_action="ignore",
    )

    best_order = stepwise_model.order
    best_seasonal = stepwise_model.seasonal_order
    print(
        f"Selected Order: ARIMA{best_order} x {best_seasonal}_7 (d_override={d_override})"
    )

    # 3. Expanding Window Time-Series Cross-Validation
    maes, mapes, smapes = [], [], []
    all_actuals, all_preds = [], []

    total_len = len(data)

    for fold in range(n_splits):
        split_idx = total_len - (n_splits - fold) * test_horizon
        train_y, test_y = (
            y.iloc[:split_idx],
            y.iloc[split_idx : split_idx + test_horizon],
        )
        train_X, test_X = (
            X.iloc[:split_idx],
            X.iloc[split_idx : split_idx + test_horizon],
        )

        model = SARIMAX(
            train_y,
            exog=train_X,
            order=best_order,
            seasonal_order=best_seasonal,
            enforce_stationarity=False,
            enforce_invertibility=False,
        )
        fit_res = model.fit(disp=False)
        preds = fit_res.forecast(steps=test_horizon, exog=test_X)

        # Store predictions for holdout evaluation
        maes.append(np.mean(np.abs(test_y - preds)))
        mapes.append(np.mean(np.abs((test_y - preds) / test_y)) * 100)
        smapes.append(compute_smape(test_y, preds))

        all_actuals.extend(test_y.values)
        all_preds.extend(preds.values)

    # 4. Compute Aggregate Pearson Correlation across Cross-Validation
    pearson_r, p_value = stats.pearsonr(all_actuals, all_preds)

    # 5. Fit full dataset to inspect residual diagnostics (Ljung-Box)
    final_model = SARIMAX(
        y, exog=X, order=best_order, seasonal_order=best_seasonal
    ).fit(disp=False)
    lb_test = acorr_ljungbox(final_model.resid, lags=[10], return_df=True)
    lb_pvalue = lb_test["lb_pvalue"].values[0]

    return {
        "Station": target_col,
        "Order": f"{best_order}x{best_seasonal}",
        "CV MAE": round(np.mean(maes), 3),
        "CV MAPE (%)": round(np.mean(mapes), 2),
        "CV sMAPE (%)": round(np.mean(smapes), 2),
        "Pearson r": round(pearson_r, 4),
        "Pearson p-val": f"{p_value:.4e}",
        "R-Squared": round(pearson_r**2, 4),
        "Ljung-Box p-val": round(lb_pvalue, 4),
        "AIC": round(final_model.aic, 2),
    }


def main():
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(
        SCRIPT_DIR, "..", "BasesDeDatosParquet", 'factor_analysisi.parquet'
    )
    
    # 1. Load Parquet
    df_raw = pd.read_parquet(data_path)
    
    # Ensure Fecha is datetime and set as index
    df_raw['Fecha'] = pd.to_datetime(df_raw['Fecha'])
    df_raw = df_raw.set_index('Fecha').sort_index()

    factor_columns = [
        "Factor_Combustion",
        "Factor_Photochemical",
        "Factor_Moisture",
    ]

    # 2. Pivot station PM10 values into wide columns (SE3, NE3, etc.)
    df_wide = df_raw.pivot_table(
        index='Fecha', 
        columns='Estacion', 
        values='PM10'
    )

    # 3. Aggregate latent factor features (taking first/mean per date)
    df_factors = df_raw[factor_columns].groupby(level='Fecha').mean()

    # 4. Merge back into a single unified DataFrame
    df = pd.concat([df_wide, df_factors], axis=1).dropna()

    # Verify target columns exist now
    print("Reshaped Columns:", df.columns.tolist())

    # 5. Run Evaluations
    station_configs = [
        {"target": "SE3", "d_override": None},
        {"target": "NO2", "d_override": None},  # Note: ensure 'NO2' exists in df_raw['Estacion'].unique()
        {"target": "NE3", "d_override": 1},
    ]

    results = []
    for config in station_configs:
        if config["target"] in df.columns:
            res = evaluate_station_sarimax(
                df=df,
                target_col=config["target"],
                factor_cols=factor_columns,
                d_override=config["d_override"],
                n_splits=3,
                test_horizon=14,
            )
            results.append(res)
        else:
            print(f"Warning: Station '{config['target']}' not found in dataset. Skipping.")

    # Display Results Matrix
    if results:
        results_df = pd.DataFrame(results)
        print("\n==========================================")
        print("        FINAL MODEL EVALUATION MATRIX      ")
        print("==========================================")
        print(results_df.to_string(index=False))


if __name__ == "__main__":
    main()