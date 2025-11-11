import numpy as np
import pandas as pd
import os
import joblib
import xgboost as xgb
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import mean_squared_error
 
# Load data
df = pd.read_csv("Data/base_data.csv")

# Preprocessing
df['Date'] = pd.to_datetime(df['Date'], dayfirst=True)
df['Year'] = df['Date'].dt.year
df['Month'] = df['Date'].dt.month
df['Week'] = df['Date'].dt.isocalendar().week
df['Quarter'] = df['Date'].dt.quarter
 
# Feature engineering
df = df.sort_values(['Store', 'Date']).reset_index(drop=True)
lags = [1, 2, 3]
rolling_windows = [2, 3]
for lag in lags:
    df[f'Sales_lag_{lag}'] = df.groupby('Store')['Weekly_Sales'].shift(lag)
for window in rolling_windows:
    df[f'Sales_roll_{window}_mean'] = (
        df.groupby('Store')['Weekly_Sales']
        .shift(1)
        .rolling(window)
        .mean()
        .reset_index(level=0, drop=True)
    )
    df[f'Sales_roll_{window}_std'] = (
        df.groupby('Store')['Weekly_Sales']
        .shift(1)
        .rolling(window)
        .std()
        .reset_index(level=0, drop=True)
    )
df['Sales_pct_change_1'] = df.groupby('Store')['Weekly_Sales'].pct_change(periods=1)
df = df.dropna().reset_index(drop=True)
 
feature_cols = [
    'Holiday_Flag', 'Temperature', 'Fuel_Price', 'CPI', 'Unemployment',
    'Year', 'Month', 'Week', 'Quarter',
    'Sales_lag_1', 'Sales_lag_2', 'Sales_lag_3',
    'Sales_roll_2_mean', 'Sales_roll_2_std', 'Sales_roll_3_mean', 'Sales_roll_3_std',
    'Sales_pct_change_1',
    'Store'
]
target = 'Weekly_Sales'
 
# Split data: use all data before 2012 for initial training
train_mask = (df['Year'] < 2012)
X_train = df.loc[train_mask, feature_cols]
y_train = df.loc[train_mask, target]
 
# Hyperparameter tuning (one-time)
param_dist = {
    'n_estimators': [300, 500, 700, 1000],            # more trees for better fit
    'learning_rate': [0.001, 0.01, 0.03, 0.05],      # slower but better learning
    'max_depth': [6, 8, 10, 12],                      # deeper trees for complexity
    'min_child_weight': [1, 3, 5],                    # control overfitting
    'gamma': [0, 0.1, 0.3, 0.5],                      # complexity control
    'subsample': [0.7, 0.8, 1.0],                     # row sampling
    'colsample_bytree': [0.7, 0.8, 1.0],              # feature sampling
    'reg_alpha': [0, 0.01, 0.1, 1],                   # L1 regularization
    'reg_lambda': [1, 1.5, 2, 3],                      # L2 regularization
}
 
xgb_reg = xgb.XGBRegressor(random_state=42, n_jobs=-1)
random_search = RandomizedSearchCV(
    estimator=xgb_reg,
    param_distributions=param_dist,
    n_iter=20,
    scoring='neg_mean_squared_error',
    cv=3,
    verbose=2,
    random_state=42,
    n_jobs=-1
)
print("Starting hyperparameter tuning...")
random_search.fit(X_train, y_train)
print("Best parameters:", random_search.best_params_)
 
# Save best params and model
best_params = random_search.best_params_

joblib.dump(best_params, "best_xgb_params.pkl")
best_model = random_search.best_estimator_
best_model.save_model("best_xgb_model_init.ubj")
print("Saved initial model and best parameters.")
 