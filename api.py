from flask import Flask, request, jsonify
import xarray as xr
import os
import pandas as pd
from datetime import datetime
from dotenv import load_dotenv
from google import genai

# Load environment variables
load_dotenv()

app = Flask(__name__)

NC_PATH = os.path.join(os.path.dirname(__file__), "latest_predictions.nc")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")



genai_client = genai.Client(api_key=GEMINI_API_KEY)



def validate_with_gemini(lat, lon, pm25_value):
    """
    Use Gemini AI to validate and potentially correct PM2.5 predictions.
    
    Args:
        lat: Latitude
        lon: Longitude
        pm25_value: Predicted PM2.5 value
        
    Returns:
        dict with 'corrected_value', 'is_corrected', 'reason'
    """

    
    try:
        # Get current time context
        now = datetime.now()
        hour = now.hour
        month = now.month
        
        # Create a detailed prompt for Gemini
        prompt = f"""You are an air quality expert analyzing PM2.5 predictions for Kerala, India.

Location: Latitude {lat}, Longitude {lon}
Current Time: {now.strftime('%Y-%m-%d %H:%M')} (Hour: {hour}, Month: {month})
Predicted PM2.5: {pm25_value} µg/m³

IMPORTANT: First search Google for current air quality conditions in Kerala, India, especially near this location.
Look for:
- Current PM2.5 levels in Kerala cities
- Recent air quality reports
- Weather conditions affecting air quality
- Any pollution events or advisories

Context:
- Kerala typical PM2.5 range: 15-60 µg/m³
- Urban areas (Thiruvananthapuram ~8.5°N,76.9°E, Kochi ~9.9°N,76.3°E, Kozhikode ~11.2°N,75.8°E): 30-50 µg/m³
- Rural areas: 15-30 µg/m³
- Morning rush (6-9 AM): +20% higher
- Evening rush (5-8 PM): +15% higher
- Night (12-5 AM): -15% lower
- Monsoon months (Jun-Sep): -20% lower
- Winter months (Dec-Feb): +15% higher

Task: Compare the predicted value with current real-world data from your search, then analyze if it's realistic.

Respond in EXACTLY this JSON format (no markdown, no extra text):
{{"corrected_value": <number>, "is_corrected": <true/false>, "reason": "<brief explanation with source if possible>"}}

Rules:
1. Use Google Search results to validate the prediction
2. If value matches real-world data (±30%), return is_corrected: false
3. If value differs significantly from current conditions, return is_corrected: true with corrected value
4. Keep reason under 100 characters, mention if based on search results
5. Corrected value must be between 10 and 150 µg/m³"""

        # Generate content without Google Search
        response = genai_client.models.generate_content(
            model="gemini-3-flash-preview",
            contents=prompt
        )
        
        # Parse Gemini response
        import json
        response_text = response.text.strip()
        
        # Remove markdown code blocks if present
        if response_text.startswith("```"):
            response_text = response_text.split("```")[1]
            if response_text.startswith("json"):
                response_text = response_text[4:]
            response_text = response_text.strip()
        
        result = json.loads(response_text)
        
        return {
            'corrected_value': float(result.get('corrected_value', pm25_value)),
            'is_corrected': bool(result.get('is_corrected', False)),
            'reason': str(result.get('reason', 'No correction needed'))
        }
        
    except Exception as e:
        # If Gemini fails, return original value
        print(f"Gemini validation error: {e}")
        return {
            'corrected_value': pm25_value,
            'is_corrected': False,
            'reason': f'Validation skipped: {str(e)[:50]}'
        }

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
        
        original_pm25 = float(val)
        
        # Validate with Gemini AI
        validation = validate_with_gemini(grid_lat, grid_lon, original_pm25)
        
        # Prepare response
        response = {
            "requested_latitude": lat,
            "requested_longitude": lon,
            "grid_latitude": grid_lat,
            "grid_longitude": grid_lon,
            "pm25": validation['corrected_value'],  # Use AI-corrected value
            "unit": "µg/m³",
            "ai_validation": {
                "original_value": original_pm25,
                "corrected": validation['is_corrected'],
                "reason": validation['reason']
            }
        }
        
        return jsonify(response)
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
