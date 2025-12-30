"""
Script to populate the newpredictions collection with actual historical PM2.5 values.
You can use this to import data from CSV files, external APIs, or other sources.
"""

import pandas as pd
from datetime import datetime
from database import save_historical_data, get_lag_collection_stats, init_db

def populate_from_csv(csv_path):
    """
    Populate historical data from a CSV file.
    CSV should have columns: Date, Time, latitude, longitude, PM2.5
    """
    try:
        df = pd.read_csv(csv_path)
        
        # Convert Date and Time to observation_time datetime
        if 'Date' in df.columns and 'Time' in df.columns:
            df['observation_time'] = pd.to_datetime(
                df['Date'] + ' ' + df['Time'], 
                dayfirst=True
            )
        elif 'observation_time' in df.columns:
            df['observation_time'] = pd.to_datetime(df['observation_time'])
        else:
            print("❌ CSV must have 'Date' and 'Time' columns or 'observation_time' column")
            return
        
        # Rename PM2.5 to pm25 if needed
        if 'PM2.5' in df.columns:
            df['pm25'] = df['PM2.5']
        
        # Select required columns
        required_cols = ['observation_time', 'latitude', 'longitude', 'pm25']
        missing_cols = [col for col in required_cols if col not in df.columns]
        
        if missing_cols:
            print(f"❌ Missing required columns: {missing_cols}")
            return
        
        historical_df = df[required_cols].copy()
        
        # Remove any NaN values
        historical_df = historical_df.dropna()
        
        print(f"📊 Loaded {len(historical_df)} records from {csv_path}")
        print(f"   Date range: {historical_df['observation_time'].min()} to {historical_df['observation_time'].max()}")
        print(f"   PM2.5 range: {historical_df['pm25'].min():.1f} to {historical_df['pm25'].max():.1f}")
        
        # Save to database
        save_historical_data(historical_df)
        
        # Show stats
        print("\n📈 Updated collection stats:")
        stats = get_lag_collection_stats()
        print(f"   Total records: {stats['total']}")
        print(f"   Unique locations: {stats.get('unique_locations', 'N/A')}")
        print(f"   PM2.5 avg: {stats.get('pm25_avg', 0):.1f} µg/m³")
        
    except Exception as e:
        print(f"❌ Error loading CSV: {e}")

def populate_sample_data():
    """
    Populate with sample data for testing (if you don't have historical CSV).
    This creates dummy data for the past 7 days.
    """
    from datetime import datetime, timedelta
    import numpy as np
    
    print("⚠️  Creating sample historical data for testing...")
    
    # Kerala grid (simplified)
    lats = np.arange(8.0, 12.2, 0.2)
    lons = np.arange(74.0, 77.8, 0.2)
    
    records = []
    now = datetime.now()
    
    # Generate data for past 7 days, every 6 hours
    for days_ago in range(7):
        for hour in [0, 6, 12, 18]:
            obs_time = now - timedelta(days=days_ago, hours=(24-hour))
            
            for lat in lats:
                for lon in lons:
                    # Generate realistic PM2.5 values (25-60 with some variation)
                    base_pm = 35.0
                    urban_boost = 10.0 if (10.0 <= lat <= 10.5 and 76.0 <= lon <= 76.5) else 0
                    time_variation = 5.0 * np.sin(hour * np.pi / 12)  # Higher during day
                    noise = np.random.normal(0, 3)
                    
                    pm25 = base_pm + urban_boost + time_variation + noise
                    pm25 = max(15.0, min(80.0, pm25))  # Clamp between 15-80
                    
                    records.append({
                        'observation_time': obs_time,
                        'latitude': round(lat, 1),
                        'longitude': round(lon, 1),
                        'pm25': round(pm25, 2)
                    })
    
    df = pd.DataFrame(records)
    print(f"📊 Generated {len(df)} sample records")
    print(f"   Date range: {df['observation_time'].min()} to {df['observation_time'].max()}")
    print(f"   PM2.5 range: {df['pm25'].min():.1f} to {df['pm25'].max():.1f}")
    
    save_historical_data(df)
    
    print("\n📈 Collection stats:")
    stats = get_lag_collection_stats()
    print(f"   Total records: {stats['total']}")
    print(f"   PM2.5 avg: {stats.get('pm25_avg', 0):.1f} µg/m³")

if __name__ == "__main__":
    import sys
    
    # Initialize database
    init_db()
    
    print("\n=== Populate Historical PM2.5 Data ===\n")
    print("Options:")
    print("1. Load from CSV file")
    print("2. Generate sample data for testing")
    print("3. Show current statistics only")
    
    if len(sys.argv) > 1:
        if sys.argv[1] == "csv" and len(sys.argv) > 2:
            csv_path = sys.argv[2]
            populate_from_csv(csv_path)
        elif sys.argv[1] == "sample":
            populate_sample_data()
        elif sys.argv[1] == "stats":
            stats = get_lag_collection_stats()
            print("\n📈 Current collection stats:")
            print(f"   Total records: {stats.get('total', 0)}")
            if stats.get('total', 0) > 0:
                print(f"   Unique locations: {stats.get('unique_locations', 'N/A')}")
                print(f"   PM2.5 range: {stats.get('pm25_min', 0):.1f} - {stats.get('pm25_max', 0):.1f} µg/m³")
                print(f"   PM2.5 avg: {stats.get('pm25_avg', 0):.1f} µg/m³")
    else:
        print("\nUsage:")
        print("  python populate_historical_data.py csv <path_to_csv>")
        print("  python populate_historical_data.py sample")
        print("  python populate_historical_data.py stats")
        print("\nExample:")
        print("  python populate_historical_data.py csv dataset_with_pca.csv")
