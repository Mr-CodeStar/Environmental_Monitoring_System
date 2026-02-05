import pandas as pd
from prophet import Prophet
import os
import numpy as np

def predict_trends(target_index='NDVI'):
    csv_path = "environment_dataset.csv"
    if not os.path.exists(csv_path):
        return {"error": "Dataset not found. Please generate a dataset first."}
    try:
        # 1. Load Data
        df = pd.read_csv(csv_path)
        df['date'] = pd.to_datetime(df['date'])
        
        # 2. Prepare for Prophet (columns must be 'ds' and 'y')
        data = df[['date', target_index]].rename(columns={'date': 'ds', target_index: 'y'})
        
        # 3. Data Cleaning (Remove outliers/clouds)
        if target_index == 'LST':
            data = data[data['y'] > 0]
        else:
            data = data[data['y'] > 0.05]

        if len(data) < 10:
            return {"error": "Insufficient clean data points for prediction."}

        # 4. Train Model
        # Using a flatter trend and 0.95 confidence interval
        model = Prophet(
            yearly_seasonality=True,
            weekly_seasonality=False,
            daily_seasonality=False,
            changepoint_prior_scale=0.05 
        )
        model.fit(data)

        # 5. Forecast Future (12 Months)
        future_df = model.make_future_dataframe(periods=12, freq='MS') # MS = Month Start
        forecast = model.predict(future_df)

        # 6. Formatting for FastAPI/JSON
        # We must convert Timestamps to Strings and ensure no NaN values
        
        # Format Historical Data
        history_clean = data.copy()
        history_clean['date'] = history_clean['ds'].dt.strftime('%Y-%m-%d')
        history_list = history_clean[['date', 'y']].rename(columns={'y': 'value'}).to_dict(orient='records')

        # Format Future Data (Tail 12 records)
        future_clean = forecast.tail(12)[['ds', 'yhat']].copy()
        future_clean['date'] = future_clean['ds'].dt.strftime('%Y-%m-%d')
        # Replace any potential NaN or Infinity with 0 to prevent JSON crash
        future_clean['yhat'] = future_clean['yhat'].replace([np.inf, -np.inf], 0).fillna(0)
        future_list = future_clean[['date', 'yhat']].rename(columns={'yhat': 'value'}).to_dict(orient='records')
        return {
            "history": history_list,
            "future": future_list
        }
    except Exception as e:
        print(f"ML SERVICE CRASH: {str(e)}")
        return {"error": f"Model error: {str(e)}"}
