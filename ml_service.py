import pandas as pd
import numpy as np
import os
from prophet import Prophet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

def calculate_rmse(actual, predicted):
    actual, predicted = np.array(actual), np.array(predicted)
    mask = ~np.isnan(actual) & ~np.isnan(predicted)
    if np.sum(mask) == 0: return float('inf')
    return np.sqrt(mean_squared_error(actual[mask], predicted[mask]))

def preprocess_data(series):
    # IQR Outlier Treatment
    Q1 = series.quantile(0.25)
    Q3 = series.quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    series_cleaned = series.clip(lower=lower_bound, upper=upper_bound)

    scaler = MinMaxScaler()
    scaled_values = scaler.fit_transform(series_cleaned.values.reshape(-1, 1)).flatten()
    
    return scaled_values, scaler

def create_features(df):
    df = df.copy()
    df['month'] = df['ds'].dt.month
    df['year'] = df['ds'].dt.year
    df['time_index'] = np.arange(len(df))
    return df[['month', 'year', 'time_index']]

def train_and_select_best(df, target_index):
    ml_df = df[['date', target_index]].dropna().rename(columns={'date': 'ds', target_index: 'y'})
    
    if len(ml_df) < 12:
        return {"error": f"Insufficient data for {target_index}"}
    y_cleaned, scaler = preprocess_data(ml_df['y'])
    ml_df['y_scaled'] = y_cleaned
    train_size = len(ml_df) - 6
    train_df = ml_df.iloc[:train_size]
    test_df = ml_df.iloc[train_size:]

    X = create_features(ml_df)
    X_train, X_test = X.iloc[:train_size], X.iloc[train_size:]
    y_train, y_test = ml_df['y_scaled'].iloc[:train_size], ml_df['y_scaled'].iloc[train_size:]

    results = {}

    # --- 1. Prophet ---
    try:
        m_prophet = Prophet(yearly_seasonality=True, weekly_seasonality=False, daily_seasonality=False)
        prophet_train = train_df[['ds', 'y_scaled']].rename(columns={'y_scaled': 'y'})
        m_prophet.fit(prophet_train)
        future_prophet = m_prophet.make_future_dataframe(periods=6, freq='MS')
        forecast_prophet = m_prophet.predict(future_prophet).iloc[train_size:]['yhat']
        results['Prophet'] = (calculate_rmse(y_test, forecast_prophet), m_prophet)
    except: results['Prophet'] = (float('inf'), None)

    # --- 2. Random Forest ---
    try:
        m_rf = RandomForestRegressor(n_estimators=100, random_state=42)
        m_rf.fit(X_train, y_train)
        results['Random Forest'] = (calculate_rmse(y_test, m_rf.predict(X_test)), m_rf)
    except: results['Random Forest'] = (float('inf'), None)

    # --- 3. XGBoost ---
    try:
        m_xgb = XGBRegressor(n_estimators=100, learning_rate=0.05, max_depth=5)
        m_xgb.fit(X_train, y_train)
        results['XGBoost'] = (calculate_rmse(y_test, m_xgb.predict(X_test)), m_xgb)
    except: results['XGBoost'] = (float('inf'), None)

    # --- 4. Gradient Boosting ---
    try:
        m_gb = GradientBoostingRegressor(n_estimators=100, learning_rate=0.05)
        m_gb.fit(X_train, y_train)
        results['Gradient Boost'] = (calculate_rmse(y_test, m_gb.predict(X_test)), m_gb)
    except: results['Gradient Boost'] = (float('inf'), None)

    best_model_name = min(results, key=lambda k: results[k][0])
    best_rmse, _ = results[best_model_name]

    history_list = ml_df[['ds', 'y']].rename(columns={'ds': 'date', 'y': 'value'})
    history_list['date'] = history_list['date'].dt.strftime('%Y-%m-%d')

    future_dates = pd.date_range(ml_df['ds'].iloc[-1], periods=13, freq='MS')[1:]
    future_df = pd.DataFrame({'ds': future_dates})
    X_future = create_features(pd.concat([ml_df, future_df]).reset_index()).tail(12)

    if best_model_name == "Prophet":
        m_final = Prophet(yearly_seasonality=True).fit(ml_df[['ds', 'y_scaled']].rename(columns={'y_scaled': 'y'}))
        f_future = m_final.make_future_dataframe(periods=12, freq='MS')
        preds_scaled = m_final.predict(f_future).tail(12)['yhat']
    else:
        m_final = results[best_model_name][1]
        m_final.fit(X, ml_df['y_scaled'])
        preds_scaled = m_final.predict(X_future)
    preds_final = scaler.inverse_transform(np.array(preds_scaled).reshape(-1, 1)).flatten()
    
    future_values = pd.DataFrame({
        'date': future_dates.strftime('%Y-%m-%d'),
        'value': np.round(preds_final, 4)
    })

    return {
        "model_used": best_model_name,
        "accuracy_score": round(best_rmse, 4),
        "history": history_list.to_dict(orient='records'),
        "future": future_values.to_dict(orient='records')
    }

def predict_all_trends():
    csv_path = "environment_dataset.csv"
    if not os.path.exists(csv_path): return {"error": "Dataset not found."}
    try:
        df = pd.read_csv(csv_path)
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date')
        return {idx: train_and_select_best(df, idx) for idx in ['NDVI', 'NDWI', 'NDBI', 'LST']}
    except Exception as e:
        return {"error": f"ML Service Error: {str(e)}"}