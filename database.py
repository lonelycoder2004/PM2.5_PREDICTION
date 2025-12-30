import os
from datetime import datetime
from dotenv import load_dotenv
import pymongo

load_dotenv()

# MongoDB Configuration
MONGO_URI = os.getenv("MONGO_URI")
DB_NAME = "lagfeatures"
COLLECTION_NAME = "predictions"  # For model outputs
LAG_COLLECTION_NAME = "newpredictions"  # For historical/observed PM2.5 values

# Global client to allow connection reuse
_CLIENT = None

def get_collection():
    """Returns the predictions collection (for model outputs)."""
    global _CLIENT
    if _CLIENT is None:
        _CLIENT = pymongo.MongoClient(MONGO_URI)
    db = _CLIENT[DB_NAME]
    return db[COLLECTION_NAME]

def get_lag_collection():
    """Returns the newpredictions collection (for historical/observed lag features)."""
    global _CLIENT
    if _CLIENT is None:
        _CLIENT = pymongo.MongoClient(MONGO_URI)
    db = _CLIENT[DB_NAME]
    return db[LAG_COLLECTION_NAME]

def init_db():
    """Initializes the database with indexes for both collections."""
    try:
        # Initialize predictions collection (model outputs)
        col = get_collection()
        # Drop old indexes that may conflict
        try:
            col.drop_index("unique_location_time")
        except:
            pass
        col.create_index([("latitude", 1), ("longitude", 1), ("timestamp", -1)])
        col.create_index(
            [("latitude", 1), ("longitude", 1), ("observation_time", 1)],
            name="unique_location_time_pred"
        )
        
        # Initialize newpredictions collection (historical/observed values for lags)
        lag_col = get_lag_collection()
        # Drop old indexes that may conflict
        try:
            lag_col.drop_index("unique_location_time_lag")
        except:
            pass
        lag_col.create_index([("latitude", 1), ("longitude", 1), ("observation_time", -1)])
        lag_col.create_index(
            [("latitude", 1), ("longitude", 1), ("observation_time", 1)],
            name="unique_location_time_lag"
        )
        
        print("✅ MongoDB initialized with indexes for 'predictions' and 'newpredictions' collections.")
    except Exception as e:
        print(f"Error initializing MongoDB: {e}")

def save_predictions(df, source="model"):
    """
    Saves predictions to MongoDB using upsert to avoid duplicates.
    df should have columns: timestamp (prediction time), observation_time, latitude, longitude, pm25
    source: 'model' (default) or other tag
    """
    col = get_collection()
    records = df.to_dict("records")
    
    if records:
        try:
            upserted_count = 0
            for record in records:
                # Ensure observation_time is a datetime object
                if isinstance(record.get("observation_time"), str):
                    try:
                        record["observation_time"] = datetime.fromisoformat(record["observation_time"])
                    except Exception:
                        pass
                # Attach source tag
                record["source"] = source
                # Use observation_time as the unique identifier along with location
                col.update_one(
                    {
                        "latitude": round(record["latitude"], 1),
                        "longitude": round(record["longitude"], 1),
                        "observation_time": record["observation_time"]
                    },
                    {"$set": record},
                    upsert=True
                )
                upserted_count += 1
            print(f"Upserted {upserted_count} records into MongoDB (collection: {COLLECTION_NAME}).")
        except Exception as e:
            print(f"Error saving to MongoDB: {e}")

def get_lag_features(target_time, lat, lon):
    """
    Retrieves lag features (PM_lag1, PM_lag3, PM_lag6) from the newpredictions collection.
    This collection contains actual historical/observed PM2.5 values.
    Uses 1h, 3h and 6h lags to match the training (shift 1,3,6 hours).
    Returns a dictionary with lag values.
    """
    from datetime import timedelta
    col = get_lag_collection()  # Use lag collection for observed values
    
    # Define time intervals for lags to match training shifts: 1h, 3h, 6h
    lag_intervals = {
        'PM_lag1': timedelta(hours=1),
        'PM_lag3': timedelta(hours=3),
        'PM_lag6': timedelta(hours=6)
    }
    
    lags = {}
    
    # Climatological fallback values for Kerala (used only when no observed value exists)
    climatological_fallback = {
        'PM_lag1': 32.0,
        'PM_lag3': 33.0,
        'PM_lag6': 35.0
    }
    
    try:
        # Round coordinates to the same precision used when saving (1 decimal)
        lat_r = round(float(lat), 1)
        lon_r = round(float(lon), 1)

        for lag_name, time_delta in lag_intervals.items():
            lag_time = target_time - time_delta
            
            # Query for records within ±1.5 hours of the target lag time (tolerance)
            query = {
                "latitude": lat_r,
                "longitude": lon_r,
                "observation_time": {
                    "$gte": lag_time - timedelta(hours=1.5),
                    "$lte": lag_time + timedelta(hours=1.5)
                }
            }
            
            # Find the most recent matching record (observed values only expected in this collection)
            cursor = col.find(query).sort("observation_time", -1).limit(1)
            rows = list(cursor)
            
            if rows:
                lags[lag_name] = rows[0]['pm25']
            else:
                # Use climatological fallback if no data found
                lags[lag_name] = climatological_fallback[lag_name]
                
    except Exception as e:
        print(f"Error fetching lags for ({lat}, {lon}): {e}")
        # Use climatological fallbacks on error
        lags = climatological_fallback.copy()
    
    return lags

def save_historical_data(df):
    """
    Saves historical/observed PM2.5 data to the newpredictions collection.
    This data will be used for lag features.
    df should have columns: observation_time, latitude, longitude, pm25
    """
    col = get_lag_collection()
    records = df.to_dict("records")
    
    if records:
        try:
            upserted_count = 0
            for record in records:
                # Ensure observation_time is a datetime
                if isinstance(record.get("observation_time"), str):
                    try:
                        record["observation_time"] = datetime.fromisoformat(record["observation_time"])
                    except Exception:
                        pass
                # Round lat/lon to grid precision
                record["latitude"] = round(float(record["latitude"]), 1)
                record["longitude"] = round(float(record["longitude"]), 1)
                # Tag as observed
                record["source"] = record.get("source", "observed")
                col.update_one(
                    {
                        "latitude": record["latitude"],
                        "longitude": record["longitude"],
                        "observation_time": record["observation_time"]
                    },
                    {"$set": record},
                    upsert=True
                )
                upserted_count += 1
            print(f"✅ Upserted {upserted_count} historical records into 'newpredictions' collection.")
        except Exception as e:
            print(f"❌ Error saving historical data: {e}")

def count_records():
    """Returns count of records in both collections."""
    pred_count = get_collection().count_documents({})
    lag_count = get_lag_collection().count_documents({})
    return {
        "predictions": pred_count,
        "newpredictions": lag_count
    }

def get_lag_collection_stats():
    """Returns statistics about the newpredictions collection."""
    col = get_lag_collection()
    total = col.count_documents({})
    
    if total == 0:
        return {"total": 0, "message": "No historical data in newpredictions collection"}
    
    # Get PM2.5 statistics
    pipeline = [
        {"$group": {
            "_id": None,
            "min_pm": {"$min": "$pm25"},
            "max_pm": {"$max": "$pm25"},
            "avg_pm": {"$avg": "$pm25"}
        }}
    ]
    stats = list(col.aggregate(pipeline))
    
    # Get unique location pairs (latitude, longitude)
    pipeline_unique = [
        {"$group": {"_id": {"lat": "$latitude", "lon": "$longitude"}}},
        {"$count": "unique_pairs"}
    ]
    unique_res = list(col.aggregate(pipeline_unique))
    unique_locations = unique_res[0]["unique_pairs"] if unique_res else 0
    
    return {
        "total": total,
        "unique_locations": unique_locations,
        "pm25_min": stats[0]["min_pm"] if stats else None,
        "pm25_max": stats[0]["max_pm"] if stats else None,
        "pm25_avg": stats[0]["avg_pm"] if stats else None
    }

if __name__ == "__main__":
    init_db()
    print("\nCollection counts:", count_records())
    print("\nLag collection stats:", get_lag_collection_stats())
