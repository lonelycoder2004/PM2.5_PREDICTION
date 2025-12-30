import os
import json
import joblib
import numpy as np
import pandas as pd
from datetime import datetime
from database import get_collection  # adjust if your DB helper is different

# Load model & metadata
meta = json.load(open("rf_pm25_model_meta.json"))
features = meta["features"]
model = joblib.load("rf_pm25_model.joblib")

# Load input data
sat = pd.read_csv("kerala_deployment_ready_with_pca.csv")
weather = pd.read_csv("weather_avg_by_grid.csv")
df = sat.merge(weather, on=["latitude", "longitude"], how="inner")
print("Rows merged:", len(df))
print("\nPCA describe:\n", df[["PCA1","PCA2","PCA3"]].describe())
print("\nWeather describe:\n", df[["tp","blh","RH","WS","AT"]].describe())

# Inspect DB lag distributions for a sample of grid points
col = get_collection()
sample = df.sample(min(10, len(df)))
lag_vals = {"PM_lag1": [], "PM_lag3": [], "PM_lag6": []}
for _, r in sample.iterrows():
    lat, lon = r["latitude"], r["longitude"]
    # Query DB for recent records near this point (match your DB shape)
    rec = col.find_one({"latitude": round(lat,1), "longitude": round(lon,1)}, sort=[("observation_time",-1)])
    if rec:
        lag_vals["PM_lag1"].append(rec.get("pm25"))
        lag_vals["PM_lag3"].append(rec.get("pm25"))
        lag_vals["PM_lag6"].append(rec.get("pm25"))
    else:
        lag_vals["PM_lag1"].append(None)
print("\nSample lag values (may be None):")
for k,v in lag_vals.items():
    print(k, "unique non-null:", sorted(set([x for x in v if x is not None])))

# Ensure required lag cols exist (fill fallback to NaN if missing)
for c in ["PM_lag1","PM_lag3","PM_lag6","PM_avg3","PM_avg6"]:
    if c not in df.columns:
        df[c] = np.nan

print("\nLag columns describe (before fill):\n", df[["PM_lag1","PM_lag3","PM_lag6","PM_avg3","PM_avg6"]].describe())

# Fill missing lags conservatively (do NOT overwrite real DB values)
# Use NaN->column mean if you want to test, or leave NaNs to see effect
df_f = df.copy()
for c in ["PM_lag1","PM_lag3","PM_lag6"]:
    if df_f[c].isna().all():
        df_f[c] = 32.0  # temporary to let model run; we will see effect below

df_f["PM_avg3"] = df_f[["PM_lag1","PM_lag3"]].mean(axis=1)
df_f["PM_avg6"] = df_f[["PM_lag1","PM_lag3","PM_lag6"]].mean(axis=1)

# Build X using meta order and check for constant columns
X = df_f[features]
print("\nFeature uniqueness (small number = nearly constant):")
for colname in X.columns:
    print(colname, "unique:", X[colname].nunique(), "min,max,mean:", X[colname].min(), X[colname].max(), X[colname].mean())

# Predict and invert correctly (model trained on log1p)
pred_log = model.predict(X)
pred_pm = np.expm1(pred_log)  # correct inverse
print("\nPrediction stats (expm1): min,max,mean:", pred_pm.min(), pred_pm.max(), pred_pm.mean())
print("Some predictions sample:\n", pd.Series(pred_pm).describe())

# Feature importances and correlation with predictions
if hasattr(model, "feature_importances_"):
    fi = pd.Series(model.feature_importances_, index=features).sort_values(ascending=False)
    print("\nTop feature importances:\n", fi.head(10))

corrs = []
for colname in X.columns:
    try:
        corr = np.corrcoef(X[colname].fillna(X[colname].mean()), pred_pm)[0,1]
    except Exception:
        corr = np.nan
    corrs.append((colname, corr))
print("\nFeature -> prediction corr (abs sorted):")
print(sorted(corrs, key=lambda x: -abs(x[1]))[:10])

# Show top/bottom 5 predictions with coords
out = df[["latitude","longitude"]].copy()
out["pred_pm"] = pred_pm
print("\nTop 5 preds:\n", out.sort_values("pred_pm", ascending=False).head(5))
print("\nBottom 5 preds:\n", out.sort_values("pred_pm", ascending=True).head(5))