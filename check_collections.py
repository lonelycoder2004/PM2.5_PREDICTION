"""
Check MongoDB collections status and verify lag feature retrieval.
"""

from database import count_records, get_lag_collection_stats, get_lag_features, init_db
from datetime import datetime, timedelta

def check_collections():
    """Check the status of both collections."""
    print("\n=== MongoDB Collections Status ===\n")
    
    counts = count_records()
    print(f"📦 predictions (model outputs): {counts['predictions']} records")
    print(f"📦 newpredictions (historical lag data): {counts['newpredictions']} records")
    
    if counts['newpredictions'] > 0:
        print("\n📊 Historical Data Statistics:")
        stats = get_lag_collection_stats()
        print(f"   Unique locations: {stats.get('unique_locations', 'N/A')}")
        print(f"   PM2.5 range: {stats.get('pm25_min', 0):.1f} - {stats.get('pm25_max', 0):.1f} µg/m³")
        print(f"   PM2.5 average: {stats.get('pm25_avg', 0):.1f} µg/m³")
    else:
        print("\n⚠️  newpredictions collection is EMPTY!")
        print("   Run: python populate_historical_data.py sample")
        print("   Or: python populate_historical_data.py csv <your_csv_file>")

def test_lag_features():
    """Test lag feature retrieval for a sample location."""
    print("\n=== Testing Lag Feature Retrieval ===\n")
    
    # Test with Trivandrum coordinates
    test_lat = 8.5
    test_lon = 76.9
    test_time = datetime.now()
    
    print(f"Testing for location: ({test_lat}, {test_lon})")
    print(f"Target time: {test_time}")
    
    lags = get_lag_features(test_time, test_lat, test_lon)
    
    print("\nRetrieved lag features:")
    for key, value in lags.items():
        print(f"  {key}: {value:.2f} µg/m³")
    
    # Check if using fallback values
    if lags['PM_lag1'] == 35.0 and lags['PM_lag3'] == 33.0:
        print("\n⚠️  Using climatological fallback values (no historical data found)")
        print("   This means the newpredictions collection doesn't have data for this location/time")
    else:
        print("\n✅ Successfully retrieved historical lag features from database")

def compare_before_after():
    """Compare predictions collection vs newpredictions collection."""
    from database import get_collection, get_lag_collection
    
    print("\n=== Collection Comparison ===\n")
    
    pred_col = get_collection()
    lag_col = get_lag_collection()
    
    # Sample from predictions
    pred_sample = list(pred_col.find().limit(5))
    lag_sample = list(lag_col.find().limit(5))
    
    print("Sample from 'predictions' (model outputs):")
    if pred_sample:
        for i, rec in enumerate(pred_sample[:3], 1):
            print(f"  {i}. Time: {rec.get('observation_time')}, "
                  f"Lat: {rec.get('latitude')}, Lon: {rec.get('longitude')}, "
                  f"PM2.5: {rec.get('pm25', 0):.2f}")
    else:
        print("  (empty)")
    
    print("\nSample from 'newpredictions' (historical lag data):")
    if lag_sample:
        for i, rec in enumerate(lag_sample[:3], 1):
            print(f"  {i}. Time: {rec.get('observation_time')}, "
                  f"Lat: {rec.get('latitude')}, Lon: {rec.get('longitude')}, "
                  f"PM2.5: {rec.get('pm25', 0):.2f}")
    else:
        print("  (empty)")

if __name__ == "__main__":
    init_db()
    
    check_collections()
    test_lag_features()
    compare_before_after()
    
    print("\n" + "="*50)
    print("\n✅ Next steps:")
    counts = count_records()
    if counts['newpredictions'] == 0:
        print("1. Populate historical data:")
        print("   python populate_historical_data.py csv dataset_with_pca.csv")
        print("   OR")
        print("   python populate_historical_data.py sample")
    else:
        print("1. Historical data is populated ✓")
    
    print("2. Run inference to generate new predictions:")
    print("   python inference.py")
    print("3. Check API output:")
    print("   python api.py")
    print("   Then visit: http://localhost:5000/pm25?lat=8.5&lon=76.9")
