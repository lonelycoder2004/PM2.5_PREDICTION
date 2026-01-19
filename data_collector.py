import os
import json
import subprocess
import pandas as pd
from datetime import datetime
from dotenv import load_dotenv
load_dotenv()

# Paths - All relative to this script now
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.path.join(BASE_DIR, "config.json")
MDAPI_SCRIPT = os.path.join(BASE_DIR, "mdapi.py")
SAT_SCRIPT = os.path.join(BASE_DIR, "sat_processor.py")
WEATHER_SCRIPT = os.path.join(BASE_DIR, "weather_collector.py")

# Output files expected
SAT_OUTPUT = os.path.join(BASE_DIR, "kerala_deployment_ready_with_pca.csv")
WEATHER_OUTPUT = os.path.join(BASE_DIR, "weather_avg_by_grid.csv")

def update_config_date():
    """Updates the config.json with today's date."""
    try:
        with open(CONFIG_PATH, 'r') as f:
            config = json.load(f)
        
        #update user name
        config['user_credentials']['username/email'] = os.getenv("MOSDAC_USER")
        config['user_credentials']['password'] = os.getenv("MOSDAC_PASS")

        today = datetime.now().strftime("%Y-%m-%d")
        config['search_parameters']['startTime'] = today
        config['search_parameters']['endTime'] = today
        
        # Ensure skip_user_input is True
        config['download_settings']['skip_user_input'] = True
        
        with open(CONFIG_PATH, 'w') as f:
            json.dump(config, f, indent=4)
        
        print(f"✅ Updated config.json with date: {today}")
    except Exception as e:
        print(f"❌ Error updating config.json: {e}")

import sys

def run_script(script_path):
    """Runs a python script using subprocess with real-time output."""
    print(f"🚀 Running {os.path.basename(script_path)}...")
    try:
        # Run process with real-time output (no capture_output)
        # This allows user to see progress bars and status messages
        result = subprocess.run(
            [sys.executable, script_path],
            cwd=BASE_DIR,  # Set CWD to ensure relative paths work
            check=False    # Don't raise exception automatically
        )
        
        # Check return code manually
        if result.returncode != 0:
            raise subprocess.CalledProcessError(result.returncode, script_path)

        print(f"✅ {os.path.basename(script_path)} completed.")
    except subprocess.CalledProcessError as e:
        print(f"❌ Error running {os.path.basename(script_path)}: Exit code {e.returncode}")
        raise

def collect_data():
    """
    Orchestrates the data collection pipeline.
    """
    print("--- Starting Data Collection ---")
    
    # 0. Cleanup Old Files
    if os.path.exists(SAT_OUTPUT):
        os.remove(SAT_OUTPUT)
        print(f"🗑️ Removed old file: {SAT_OUTPUT}")
    if os.path.exists(WEATHER_OUTPUT):
        os.remove(WEATHER_OUTPUT)
        print(f"🗑️ Removed old file: {WEATHER_OUTPUT}")
    
    # 1. Update Config
    update_config_date()
    
    # 2. Download Satellite Data
    try:
        run_script(MDAPI_SCRIPT)
    except Exception as e:
        print("⚠️ Satellite download failed or interrupted (maybe no data yet).")
    
    # 3. Process Satellite Data
    run_script(SAT_SCRIPT)
    
    # 4. Fetch Weather Data
    run_script(WEATHER_SCRIPT)
    
    # Verify outputs
    if os.path.exists(SAT_OUTPUT) and os.path.exists(WEATHER_OUTPUT):
        return SAT_OUTPUT, WEATHER_OUTPUT
    else:
        # Check what is missing
        if not os.path.exists(SAT_OUTPUT):
            print(f"❌ Missing Satellite Output: {SAT_OUTPUT}")
        if not os.path.exists(WEATHER_OUTPUT):
            print(f"❌ Missing Weather Output: {WEATHER_OUTPUT}")
        
        # Allow partial return if one failed? No, pipeline needs both.
        raise FileNotFoundError("One or more output CSVs were not generated.")

if __name__ == "__main__":
    collect_data()
