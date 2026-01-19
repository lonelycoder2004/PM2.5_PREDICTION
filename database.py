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
    # Lag 1, 3, 6 (1st, 3rd, and 6th previous values)
    lags['PM_lag1'] = values[0] if len(values) >= 1 else 25.0
    lags['PM_lag3'] = values[2] if len(values) >= 3 else 24.0
    lags['PM_lag6'] = values[5] if len(values) >= 6 else 22.0
    
    # Values for rolling averages (need the full sequence for PM_avg3 and PM_avg6)
    # PM_avg3 needs last 3: [v0, v1, v2]
    # PM_avg6 needs last 6: [v0, v1, v2, v3, v4, v5]
    # Default sequence that matches lag defaults: [25.0, 25.0, 24.0, 24.0, 22.0, 22.0]
    default_history = [25.0, 25.0, 24.0, 24.0, 22.0, 22.0]
    
    if len(values) == 0:
        lags['history_6'] = default_history
    elif len(values) < 6:
        # Pad with defaults from the end
        lags['history_6'] = values + default_history[len(values):]
    else:
        lags['history_6'] = values[:6]
    
    return lags

if __name__ == "__main__":
    init_db()
