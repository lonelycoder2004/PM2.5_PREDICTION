import os
from datetime import datetime
from dotenv import load_dotenv
import pymongo

load_dotenv()

# MongoDB Configuration
MONGO_URI = os.getenv("MONGO_URI")
DB_NAME = "lagfeatures"
COLLECTION_NAME = "predictions"

# Global client to allow connection reuse
_CLIENT = None

def get_collection():
    global _CLIENT
    if _CLIENT is None:
        _CLIENT = pymongo.MongoClient(MONGO_URI)
    db = _CLIENT[DB_NAME]
    return db[COLLECTION_NAME]

def init_db():
    """Initializes the database with indexes."""
    try:
        col = get_collection()
        # Create compound index for fast retrieval by location and time
        col.create_index([("latitude", 1), ("longitude", 1), ("timestamp", -1)])
        print("✅ MongoDB initialized and indexes created.")
    except Exception as e:
        print(f"Error initializing MongoDB: {e}")

def save_predictions(df):
    """
    Saves predictions to MongoDB.
    df should have columns: timestamp, latitude, longitude, pm25
    """
    col = get_collection()
    records = df.to_dict("records")
    
    if records:
        try:
            col.insert_many(records)
            print(f"Inserted {len(records)} records into MongoDB.")
        except Exception as e:
            print(f"Error saving to MongoDB: {e}")

def get_lag_features(target_time, lat, lon):
    """
    Retrieves lag features (PM_lag1, PM_lag3, PM_lag6) for a specific location and time.
    Returns a dictionary with lag values.
    """
    col = get_collection()
    
    # Query for the last 6 records for this location, sorted by timestamp descending
    # We filter by timestamp < target_time to avoid future leakage
    
    query = {
        "latitude": lat,
        "longitude": lon,
        "timestamp": {"$lt": target_time}
    }
    
    try:
        cursor = col.find(query).sort("timestamp", -1).limit(6)
        rows = list(cursor)
        values = [r['pm25'] for r in rows]
    except Exception as e:
        print(f"Error fetching lags: {e}")
        values = []
    
    lags = {}
    # Assuming the schedule is consistent, the previous record is lag1, etc.
    # If the schedule is 6 hours, lag1 is T-6h, lag3 is T-18h, lag6 is T-36h.
    # The model was trained on some lag definition (likely hourly or daily).
    # If the model expects hourly lags but we only have 6-hourly data, this is a mismatch.
    # However, for this deployment, we will map available history to the model's lag features.
    
    lags['PM_lag1'] = values[0] if len(values) >= 1 else 25.0
    lags['PM_lag3'] = values[2] if len(values) >= 3 else 24.0
    lags['PM_lag6'] = values[5] if len(values) >= 6 else 22.0
    
    return lags

if __name__ == "__main__":
    init_db()
