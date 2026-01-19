import os
import math
import h5py
import re
import numpy as np
import pandas as pd
import joblib
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# Paths
BASE_DIR = os.path.dirname(__file__)
ASSETS_DIR = os.path.join(BASE_DIR, "assets")
UNPROCESSED_DIR = os.path.join(BASE_DIR, "unprocessed-data")
OUTPUT_CSV = os.path.join(BASE_DIR, "kerala_deployment_ready_with_pca.csv")

# -------------------------------
# Step 1: Define projection and grid
# -------------------------------

a = 6378137.0
b = 6356752.3142
lon0 = 75.0

x_ul = -6122571.993630046
y_ul = 6413524.594094472

dx = 3999.067272129357
dy = 3999.703519859353

nrows, ncols = 3207, 3062
ee = math.sqrt(1 - (b**2 / a**2))

def latlong_to_pixel(lat, lon):
    phi = math.radians(lat)
    lam = math.radians(lon)
    lam0 = math.radians(lon0)
    x = a * (lam - lam0)
    y = a * math.log(
        math.tan(math.pi/4 + phi/2) *
        ((1 - ee * math.sin(phi)) / (1 + ee * math.sin(phi)))**(ee/2)
    )
    col = (x - x_ul) / dx
    row = (y_ul - y) / dy
    return min(max(int(row), 0), nrows - 1), min(max(int(col), 0), ncols - 1)


# -------------------------------
# Step 2: Kerala grid points
# -------------------------------

lat_min, lat_max = 8.0, 12.0
lon_min, lon_max = 74.0, 77.5
grid_res = 0.1  # ~11 km resolution - better for PM2.5 prediction

lat_points = np.arange(lat_min, lat_max + grid_res, grid_res)
lon_points = np.arange(lon_min, lon_max + grid_res, grid_res)

grid_points = [(lat, lon) for lat in lat_points for lon in lon_points]


# -------------------------------
# Step 3: Extract datetime from filename
# -------------------------------

def extract_datetime(fname):
    match = re.search(r'_(\d{2})([A-Z]{3})(\d{4})_(\d{2})(\d{2})_', fname)
    if match:
        day = match.group(1)
        month_str = match.group(2)
        year = match.group(3)
        hour = match.group(4)
        minute = match.group(5)
        months = {"JAN":"01","FEB":"02","MAR":"03","APR":"04","MAY":"05","JUN":"06",
                  "JUL":"07","AUG":"08","SEP":"09","OCT":"10","NOV":"11","DEC":"12"}
        month = months.get(month_str.upper(), "01")
        return f"{day}-{month}-{year}", f"{hour}:{minute}"
    return "", ""


# -------------------------------
# Step 4: Extract satellite data from all HDF5 files
# -------------------------------

def process_satellite_data():
    folder = UNPROCESSED_DIR
    if not os.path.exists(folder):
        print(f"❌ Directory not found: {folder}")
        return

    records = []

    ds_dict = {
        "WV_RADIANCE": ("IMG_WV", "IMG_WV_RADIANCE"),
        "VIS_ALBEDO": ("IMG_VIS", "IMG_VIS_ALBEDO"),
        "VIS_RADIANCE": ("IMG_VIS", "IMG_VIS_RADIANCE"),
        "TIR1_TEMP": ("IMG_TIR1", "IMG_TIR1_TEMP"),
        "TIR1_RADIANCE": ("IMG_TIR1", "IMG_TIR1_RADIANCE"),
        "TIR2_TEMP": ("IMG_TIR2", "IMG_TIR2_TEMP"),
        "TIR2_RADIANCE": ("IMG_TIR2", "IMG_TIR2_RADIANCE"),
        "MIR_RADIANCE": ("IMG_MIR", "IMG_MIR_RADIANCE"),
        "SWIR_RADIANCE": ("IMG_SWIR", "IMG_SWIR_RADIANCE"),
        "SAT_AZIMUTH": ("Sat_Azimuth", None),
        "SAT_ELEVATION": ("Sat_Elevation", None),
        "SUN_AZIMUTH": ("Sun_Azimuth", None),
        "SUN_ELEVATION": ("Sun_Elevation", None)
    }

    for fname in os.listdir(folder):
        if fname.endswith(".h5"):
            print(f"Processing: {fname}")
            file_path = os.path.join(folder, fname)
            date_str, time_str = extract_datetime(fname)

            try:
                with h5py.File(file_path, "r") as f:
                    for lat, lon in grid_points:
                        row, col = latlong_to_pixel(lat, lon)
                        row_data = {"date": date_str, "time": time_str, "latitude": lat, "longitude": lon}

                        for key, (ds_name, ds_radiance_name) in ds_dict.items():
                            ds = f[ds_name]
                            val_index = ds[0, row, col]

                            if ds_radiance_name:
                                ds_radiance = f[ds_radiance_name]
                                val = ds_radiance[val_index]
                            else:
                                val = ds[0, row, col]

                            row_data[key] = val

                        records.append(row_data)

            except Exception as e:
                print(f"Error reading {fname}: {e}")

    if not records:
        print("No records extracted.")
        return

    df = pd.DataFrame(records)
    print(f"Total rows extracted = {len(df)}")


    # -------------------------------
    # Step 5: Feature engineering
    # -------------------------------

    df['vis_mir_index'] = (df['VIS_RADIANCE'] - df['MIR_RADIANCE']) / \
                          (df['VIS_RADIANCE'] + df['MIR_RADIANCE'])

    df['temp_diff_TIR'] = df['TIR1_TEMP'] - df['TIR2_TEMP']

    zenith_deg = 90 - (df['SUN_ELEVATION'] / 100.0)
    df['solar_angle_correction'] = np.cos(np.deg2rad(zenith_deg))

    keep_cols = [
        'date','time','latitude','longitude',
        'vis_mir_index','temp_diff_TIR','WV_RADIANCE','TIR1_TEMP',
        'solar_angle_correction','SAT_ELEVATION','VIS_ALBEDO'
    ]

    df = df[[c for c in keep_cols if c in df.columns]]


    # -------------------------------
    # Step 6: Aggregate (per grid, per hour)
    # -------------------------------

    df['time'] = df['time'].apply(
        lambda t: f"{int(t.split(':')[0]):02d}:00" if pd.notnull(t) else t
    )

    agg_dict = {
        'vis_mir_index': 'mean',
        'temp_diff_TIR': 'median',
        'WV_RADIANCE': 'mean',
        'TIR1_TEMP': 'mean',
        'solar_angle_correction': 'mean',
        'SAT_ELEVATION': 'median',
        'VIS_ALBEDO': 'mean'
    }

    df_grouped = df.groupby(
        ['latitude','longitude','date','time'],
        as_index=False
    ).agg(agg_dict)


    # -------------------------------
    # Step 7: KNN Imputation & PCA (Using Saved Models)
    # -------------------------------

    impute_cols = [
        'vis_mir_index','temp_diff_TIR','WV_RADIANCE','TIR1_TEMP',
        'solar_angle_correction','SAT_ELEVATION','VIS_ALBEDO'
    ]

    present_cols = [c for c in impute_cols if c in df_grouped.columns]
    
    try:
        # Load Preprocessors
        imputer_path = os.path.join(ASSETS_DIR, "imputer.joblib")
        scaler_path = os.path.join(ASSETS_DIR, "scaler.joblib")
        pca_path = os.path.join(ASSETS_DIR, "pca.joblib")
        
        if os.path.exists(imputer_path) and os.path.exists(scaler_path) and os.path.exists(pca_path):
            print("[OK] Loading saved preprocessors...")
            imputer = joblib.load(imputer_path)
            scaler = joblib.load(scaler_path)
            pca = joblib.load(pca_path)
            
            # Transform
            # We must ensure we transform ONLY the columns expected by the imputer/scaler
            # KNNImputer and Scaler were trained on 'satellite_features' in 1.py.
            # satellite_features = ["vis_mir_index","temp_diff_TIR","WV_RADIANCE","TIR1_TEMP","solar_angle_correction","SAT_ELEVATION","VIS_ALBEDO"]
            # These match 'impute_cols'.
            
            # Ensure correct column order as expected by the saved models
            X_input = df_grouped[impute_cols]
            
            X_imputed = imputer.transform(X_input)
            X_scaled = scaler.transform(X_imputed)
            sat_pca = pca.transform(X_scaled)
        else:
            print("[WARN] Saved preprocessors not found! Falling back to fresh fit.")
            imputer = KNNImputer(n_neighbors=3)
            X_imputed = imputer.fit_transform(df_grouped[impute_cols])
            
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X_imputed)
            
            pca = PCA(n_components=3)
            sat_pca = pca.fit_transform(X_scaled)

        df_grouped["PCA1"] = sat_pca[:, 0]
        df_grouped["PCA2"] = sat_pca[:, 1]
        df_grouped["PCA3"] = sat_pca[:, 2]

    except Exception as e:
        print(f"[ERROR] CRITICAL ERROR during transformation: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


    # ----------------------------------------
    # Step 8.2: Add Date, Hour, Month, Day, Weekday
    # ----------------------------------------

    df_grouped["Date"] = pd.to_datetime(df_grouped["date"], dayfirst=True)
    df_grouped["Hour"] = pd.to_datetime(df_grouped["time"], format="%H:%M").dt.hour

    df_grouped["Month"] = df_grouped["Date"].dt.month
    df_grouped["Day"] = df_grouped["Date"].dt.day
    df_grouped["Weekday"] = df_grouped["Date"].dt.weekday


    df_grouped = df_grouped[
        [
            'date','time','Hour','Month','Day','Weekday',
            'latitude','longitude','PCA1','PCA2','PCA3'
        ]
    ]

    # -------------------------------
    # Step 9: Format floats
    # -------------------------------

    for col in df_grouped.select_dtypes(include=['float', 'float64']).columns:
        df_grouped[col] = df_grouped[col].map(
            lambda x: ('%.10f' % x).rstrip('0').rstrip('.') if pd.notnull(x) else ''
        )


    # -------------------------------
    # Step 10: Save final deployment-ready CSV
    # -------------------------------

    df_grouped.to_csv(OUTPUT_CSV, index=False)
    if os.path.exists(OUTPUT_CSV):
        print(f"[OK] Final CSV created: {OUTPUT_CSV}")
        import sys
        sys.exit(0)
    else:
        print("[ERROR] CSV not created.")
        import sys
        sys.exit(1)

if __name__ == "__main__":
    process_satellite_data()
