import pandas as pd
import numpy as np
import joblib
import os
import xarray as xr
from datetime import datetime
import database
import data_collector

# Path to the trained model
MODEL_PATH = os.path.join(os.path.dirname(__file__), "rf_pm25_model.joblib")
NC_OUTPUT = os.path.join(os.path.dirname(__file__), "latest_predictions.nc")

def load_model():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model not found at {MODEL_PATH}")
    return joblib.load(MODEL_PATH)

def run_pipeline():
    print(f"[{datetime.now()}] Starting pipeline...")
    
    # Ensure DB is ready
    database.init_db()
    
    # 1. Collect Data
    try:
        sat_csv, weather_csv = data_collector.collect_data()
    except Exception as e:
        print(f"Data collection failed: {e}")
        return

    print("Loading collected data...")
    sat_df = pd.read_csv(sat_csv)
    weather_df = pd.read_csv(weather_csv)
    
    # 2. Merge Data
    print("Merging data...")
    # Ensure lat/lon are rounded similarly to avoid mismatch
    sat_df['latitude'] = sat_df['latitude'].round(1)
    sat_df['longitude'] = sat_df['longitude'].round(1)
    weather_df['latitude'] = weather_df['latitude'].round(1)
    weather_df['longitude'] = weather_df['longitude'].round(1)
    
    merged_df = pd.merge(
        sat_df,
        weather_df,
        how="left",
        on=["latitude", "longitude"]
    )
    
    # 3. Feature Engineering
    now = datetime.now()
    # If date/time columns exist in sat_df, use them, otherwise use current time
    # sat.py output has 'date', 'time', 'Hour', 'Month', 'Day', 'Weekday'
    
    # Add Lag Features from MongoDB
    print("Fetching lag features from MongoDB...")
    lags_list = []
    for index, row in merged_df.iterrows():
        # Use the timestamp from the data if available, else now
        # sat.py produces 'date' (DD-MM-YYYY) and 'time' (HH:MM)
        # We need a comparable timestamp for DB
        try:
            row_time = pd.to_datetime(f"{row['date']} {row['time']}", dayfirst=True)
        except:
            row_time = now
            
        lags = database.get_lag_features(row_time, row['latitude'], row['longitude'])
        lags_list.append(lags)
    
    lags_df = pd.DataFrame(lags_list)
    df = pd.concat([merged_df, lags_df], axis=1)
    
    # Compute Rolling Averages (FIXED: matching the model's training logic)
    # Using the history_6 sequence fetched from DB
    df['PM_avg3'] = [np.mean(hist[:3]) for hist in df['history_6']]
    df['PM_avg6'] = [np.mean(hist[:6]) for hist in df['history_6']]
    
    # Drop the history_6 column as it's not needed for prediction
    df = df.drop(columns=['history_6'])
    
    # Optional: If you want to force specific distinct defaults for avgs even when lags are present 
    # (which you likely don't, you want them derived), keep as above.
    # But if the user implies "use these hardcoded values for avg3/avg6 IF not available":
    # Since we derive them from lags, and lags now have hardcoded defaults, result is safe.
    # 24.5 and 23.66 respectively with the user's defaults.
    
    # Ensure all features required by model are present
    # Model features from ml_model.py:
    # "tp","blh","RH","WS","AT","PCA1","PCA2","PCA3",
    # "Hour","Month","Day","Weekday",
    # "PM_lag1","PM_lag3","PM_lag6","PM_avg3","PM_avg6"
    
    # Map columns if names differ
    # weather_avg_by_grid.csv has: tp, AT, RH, blh, WS
    # sat.py has: PCA1, PCA2, PCA3, Hour, Month, Day, Weekday
    
    features = [
        "tp","blh","RH","WS","AT","PCA1","PCA2","PCA3",
        "Hour","Month","Day","Weekday",
        "PM_lag1","PM_lag3","PM_lag6","PM_avg3","PM_avg6"
    ]
    
    # Fill missing columns with 0 (e.g. if weather data missing for some points)
    for col in features:
        if col not in df.columns:
            df[col] = 0
            
    X = df[features]
    
    # Save the final feature dataset for verification/debugging
    print("Saving final feature dataset...")
    feature_output = os.path.join(os.path.dirname(__file__), "deployment_features.csv")
    # Include metadata columns + all 17 features
    save_cols = ['date', 'time', 'latitude', 'longitude'] + features
    df[save_cols].to_csv(feature_output, index=False)
    print(f"✅ Saved final features to: {feature_output}")
    
    # 4. Predict
    print("Predicting...")
    model = load_model()
    # Model predicts log(PM2.5 + 1)
    preds_log = model.predict(X)
    preds = np.expm1(preds_log)
    
    df['pm25'] = preds
    df['timestamp'] = now # Store prediction time
    
    # Save predictions with features for complete transparency
    print("Saving predictions with features...")
    pred_output = os.path.join(os.path.dirname(__file__), "deployment_predictions.csv")
    pred_cols = ['timestamp', 'date', 'time', 'latitude', 'longitude', 'pm25'] + features
    df[pred_cols].to_csv(pred_output, index=False)
    print(f"✅ Saved predictions to: {pred_output}")
    
    # 5. Save to Database
    print("Saving to MongoDB...")
    # IMPORTANT: Deduplicate by lat/lon before saving to avoid duplicate entries
    # If multiple satellite passes exist for same location, take the mean prediction
    save_df = df.groupby(['latitude', 'longitude'], as_index=False).agg({
        'pm25': 'mean'
    })
    save_df['timestamp'] = now  # Add timestamp after grouping
    database.save_predictions(save_df)
    
    # 6. Export to NetCDF
    print(f"Exporting to {NC_OUTPUT}...")
    try:
        # Deduplicate: If multiple rows exist for same lat/lon (e.g. from multiple timestamps in same batch),
        # take the mean or last one. For 'latest' map, we usually want the most recent or average.
        # Let's group by lat, lon and take the mean of predictions.
        ds_df = df.groupby(['latitude', 'longitude'])[['pm25']].mean()
        
        ds = ds_df.to_xarray()
        ds.to_netcdf(NC_OUTPUT)
        print("✅ Pipeline completed successfully.")
    except Exception as e:
        print(f"❌ Error saving NetCDF: {e}")

if __name__ == "__main__":
    run_pipeline()
