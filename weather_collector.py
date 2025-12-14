import openmeteo_requests
import pandas as pd
import numpy as np
import requests_cache
from retry_requests import retry

# Output CSV
output_csv = "weather_avg_by_grid.csv"

# Open-Meteo client
cache_session = requests_cache.CachedSession('.cache', expire_after=3600)
retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
openmeteo = openmeteo_requests.Client(session=retry_session)

# Kerala bounding box (0.2° resolution)
lat_min, lat_max = 8.0, 12.0
lon_min, lon_max = 74.0, 77.5
step = 0.2   # (change to 0.1 if you want more resolution)

latitudes = np.round(np.arange(lat_min, lat_max + step, step), 1)
longitudes = np.round(np.arange(lon_min, lon_max + step, step), 1)

rows = []

print("Total grid points:", len(latitudes) * len(longitudes))

for lat in latitudes:
    for lon in longitudes:

        params = {
            "latitude": float(lat),
            "longitude": float(lon),
            "hourly": [
                "precipitation",     # tp
                "temperature_2m",          # at
                "relative_humidity_2m",
                "boundary_layer_height",
                "wind_speed_10m"
            ],
            "forecast_days": 1
        }

        # Fetch weather data
        try:
            responses = openmeteo.weather_api(
                "https://api.open-meteo.com/v1/forecast",
                params=params
            )
        except Exception as e:
            print("Error:", lat, lon, e)
            continue

        for response in responses:
            hourly = response.Hourly()

            df = pd.DataFrame({
                "datetime": pd.date_range(
                    pd.to_datetime(hourly.Time(), unit="s", utc=True),
                    pd.to_datetime(hourly.TimeEnd(), unit="s", utc=True),
                    freq=pd.Timedelta(seconds=hourly.Interval()),
                    inclusive="left"
                ),
                "tp": hourly.Variables(0).ValuesAsNumpy(),  # total precipitation
                "at": hourly.Variables(1).ValuesAsNumpy(),  # air temp
                "rh": hourly.Variables(2).ValuesAsNumpy(),
                "blh": hourly.Variables(3).ValuesAsNumpy(),
                "ws": hourly.Variables(4).ValuesAsNumpy(),
            })

            # Keep hours 0–11 to match satellite window
            df["hour"] = df["datetime"].dt.hour
            df = df[df["hour"] <= 11]

            # Compute averages
            avg_tp = df["tp"].mean()
            avg_at = df["at"].mean()
            avg_rh = df["rh"].mean()
            avg_blh = df["blh"].mean()
            avg_ws = df["ws"].mean()

            # Add row
            rows.append({
                "latitude": lat,
                "longitude": lon,
                "tp": avg_tp,
                "AT": avg_at,
                "RH": avg_rh,
                "blh": avg_blh,
                "WS": avg_ws
            })

            print("Processed:", lat, lon)

# Final CSV
final_df = pd.DataFrame(rows)
final_df.to_csv(output_csv, index=False)

print("\n DONE — Created file:", output_csv)
