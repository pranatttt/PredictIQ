import pandas as pd
import numpy as np
import xgboost as xgb
import os
import joblib

# === Feature Engineering Function ===
def add_features(df):
    df['Date'] = pd.to_datetime(df['Date'], dayfirst=True)
    df['Year'] = df['Date'].dt.year
    df['Month'] = df['Date'].dt.month
    df['Week'] = df['Date'].dt.isocalendar().week
    df['Quarter'] = df['Date'].dt.quarter

    df = df.sort_values(['Store', 'Date']).reset_index(drop=True)

    # Lags and rolling
    lags = [1, 2, 3]
    rolling_windows = [2, 3]
    for lag in lags:
        df[f'Sales_lag_{lag}'] = df.groupby('Store')['Weekly_Sales'].shift(lag)
    for window in rolling_windows:
        df[f'Sales_roll_{window}_mean'] = (
            df.groupby('Store')['Weekly_Sales']
            .shift(1).rolling(window).mean().reset_index(level=0, drop=True)
        )
        df[f'Sales_roll_{window}_std'] = (
            df.groupby('Store')['Weekly_Sales']
            .shift(1).rolling(window).std().reset_index(level=0, drop=True)
        )

    df['Sales_pct_change_1'] = df.groupby('Store')['Weekly_Sales'].pct_change(periods=1)
    df = df.dropna().reset_index(drop=True)
    return df

# === Monthly Update and Forecast Function ===
def monthly_update_and_forecast(
    monthly_csv="Data/monthly_data.csv",
    master_csv="Data/base_data.csv",
    model_path="best_xgb_model_init.ubj",
    params_path="best_xgb_params.pkl",
    output_dir="Data",
    future_months=24
):
    feature_cols = [
        'Holiday_Flag', 'Temperature', 'Fuel_Price', 'CPI', 'Unemployment',
        'Year', 'Month', 'Week', 'Quarter',
        'Sales_lag_1', 'Sales_lag_2', 'Sales_lag_3',
        'Sales_roll_2_mean', 'Sales_roll_2_std', 'Sales_roll_3_mean', 'Sales_roll_3_std',
        'Sales_pct_change_1', 'Store'
    ]

    # Load master and new monthly data
    master_df = pd.read_csv(master_csv)
    monthly_df = pd.read_csv(monthly_csv)
    combined_df = pd.concat([master_df, monthly_df], ignore_index=True)
    combined_df = add_features(combined_df)

    # Train on known data
    train_df = combined_df[combined_df['Weekly_Sales'].notna()]
    X_train = train_df[feature_cols]
    y_train = train_df['Weekly_Sales']
    dtrain = xgb.DMatrix(X_train, label=y_train)

    # Load booster and parameters
    booster = xgb.Booster()
    booster.load_model(model_path)
    best_model = joblib.load(params_path)
    xgb_native_params = best_model.get_xgb_params()
    xgb_native_params['objective'] = 'reg:squarederror'
    xgb_native_params.pop('n_estimators', None)

    # Retrain incrementally
    booster = xgb.train(xgb_native_params, dtrain, num_boost_round=50, xgb_model=booster)

    # Generate future dates
    future_dates = pd.date_range(
        start=combined_df['Date'].max() + pd.Timedelta(weeks=1),
        periods=future_months * 4,  # 4 weeks per month
        freq='W'
    )
    last_known = combined_df[combined_df['Weekly_Sales'].notna()]
    base_rows = last_known.groupby('Store').tail(1)

    future_records = []
    for date in future_dates:
        for _, row in base_rows.iterrows():
            future_records.append({**row.to_dict(), 'Date': date, 'Weekly_Sales': np.nan})

    future_df = pd.DataFrame(future_records)
    forecast_df = pd.concat([combined_df, future_df], ignore_index=True)
    forecast_df = add_features(forecast_df)

    # Predict and update
    to_predict = forecast_df[forecast_df['Weekly_Sales'].isna()]
    dpredict = xgb.DMatrix(to_predict[feature_cols])
    preds = booster.predict(dpredict)

    forecast_df.loc[forecast_df['Weekly_Sales'].isna(), 'Predicted_Sales'] = preds

    # Save updated master CSV
    forecast_df.to_csv(master_csv, index=False)

    # Save new model
    latest_month = pd.to_datetime(monthly_df['Date']).dt.month.iloc[0]
    latest_year = pd.to_datetime(monthly_df['Date']).dt.year.iloc[0]
    new_model_path = os.path.join(output_dir, f"xgb_model_{latest_year}_{latest_month:02d}.ubj")
    booster.save_model(new_model_path)

    print(f"✅ Updated master CSV and saved model to: {new_model_path}")
    return forecast_df
