from flask import Flask, request, jsonify
import xarray as xr
import os
import pandas as pd

app = Flask(__name__)

NC_PATH = os.path.join(os.path.dirname(__file__), "latest_predictions.nc")

@app.route('/')
def home():
    return "PM2.5 Estimation API is running."

@app.route('/pm25', methods=['GET'])
def get_pm25():
    """
    Get PM2.5 prediction for a specific latitude and longitude.
    Query params: lat, lon
    """
    lat = request.args.get('lat', type=float)
    lon = request.args.get('lon', type=float)
    
    if lat is None or lon is None:
        return jsonify({"error": "Please provide 'lat' and 'lon' parameters."}), 400
    
    if not os.path.exists(NC_PATH):
        return jsonify({"error": "Prediction data not available yet."}), 503
        
    try:
        # Open the NetCDF file
        ds = xr.open_dataset(NC_PATH)
        
        # Find nearest point
        # method='nearest' selects the closest grid point
        val = ds.sel(latitude=lat, longitude=lon, method="nearest")['pm25'].values
        
        # Get the actual lat/lon of the grid point found
        grid_lat = float(ds.sel(latitude=lat, longitude=lon, method="nearest")['latitude'])
        grid_lon = float(ds.sel(latitude=lat, longitude=lon, method="nearest")['longitude'])
        
        return jsonify({
            "requested_latitude": lat,
            "requested_longitude": lon,
            "grid_latitude": grid_lat,
            "grid_longitude": grid_lon,
            "pm25": float(val),
            "unit": "µg/m³"
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
