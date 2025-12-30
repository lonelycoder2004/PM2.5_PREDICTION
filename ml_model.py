import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor
import joblib
import json
import matplotlib.pyplot as plt

# -----------------------------------------------------
# 1. Load the dataset
# -----------------------------------------------------
df = pd.read_csv("dataset_with_pca.csv")

df["Date"] = pd.to_datetime(df["Date"], dayfirst=True)
df["Hour"] = pd.to_datetime(df["Time"], format="%H:%M").dt.hour
df["Month"] = df["Date"].dt.month
df["Day"] = df["Date"].dt.day
df["Weekday"] = df["Date"].dt.weekday

# Lag features
df["PM_lag1"] = df["PM2.5"].shift(1)
df["PM_lag3"] = df["PM2.5"].shift(3)
df["PM_lag6"] = df["PM2.5"].shift(6)



# Rolling averages (NOTE: future-leak – fixed version available)
df["PM_avg3"] = df["PM2.5"].rolling(window=3).mean()
df["PM_avg6"] = df["PM2.5"].rolling(window=6).mean()

df = df.dropna()


# -----------------------------------------------------
# 1A. Handle Outliers (recommended: cap at 200 µg/m³)
# -----------------------------------------------------
df["PM2.5"] = np.where(df["PM2.5"] > 200, 200, df["PM2.5"])


# -----------------------------------------------------
# 2. Log-transform target
# -----------------------------------------------------
df["PM25_log"] = np.log1p(df["PM2.5"])

# -----------------------------------------------------
# 3. Feature selection
# -----------------------------------------------------
features = [
    "tp","blh","RH","WS","AT","PCA1","PCA2","PCA3",
    "Hour","Month","Day","Weekday",
    "PM_lag1","PM_lag3","PM_lag6","PM_avg3","PM_avg6"
]

X = df[features]
y = df["PM25_log"]

# -----------------------------------------------------
# 4. Train–Validation–Test Split
# -----------------------------------------------------
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.30, random_state=42
)

X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.50, random_state=42
)

print("Train:", X_train.shape, "Val:", X_val.shape, "Test:", X_test.shape)

# -----------------------------------------------------
# 5. Random Forest Model
# -----------------------------------------------------
rf = RandomForestRegressor(
    n_estimators=500,
    max_depth=20,
    min_samples_split=4,
    min_samples_leaf=2,
    max_features="sqrt",
    bootstrap=True,
    random_state=42,
    n_jobs=-1
)

# Train
rf.fit(X_train, y_train)

# -----------------------------------------------------
# 6. Evaluate Function
# -----------------------------------------------------
def evaluate(model, X, y):
    preds = model.predict(X)
    rmse = np.sqrt(mean_squared_error(y, preds))
    r2 = r2_score(y, preds)
    return rmse, r2

rmse_val, r2_val = evaluate(rf, X_val, y_val)
rmse_test, r2_test = evaluate(rf, X_test, y_test)

print("\n=== Validation ===")
print("RMSE:", rmse_val)
print("R²:", r2_val)

print("\n=== Test ===")
print("RMSE:", rmse_test)
print("R²:", r2_test)

# -----------------------------------------------------
# 7. Plot actual vs predicted on the test set
# -----------------------------------------------------
# Predict on test set (log scale)
preds_test_log = rf.predict(X_test)

# Convert back to original PM2.5 scale
preds_test = np.expm1(preds_test_log)
actual_test = np.expm1(y_test)

# Build dataframe for plotting, reset index so x-axis is sample order
df_plot = pd.DataFrame({"Actual": actual_test, "Predicted": preds_test}, index=X_test.index)
df_plot = df_plot.sort_index().reset_index(drop=True)

plt.figure(figsize=(12, 6))
plt.plot(df_plot.index, df_plot["Actual"], label="Actual PM2.5", marker="o", markersize=3, linewidth=1)
plt.plot(df_plot.index, df_plot["Predicted"], label="Predicted PM2.5", marker="x", markersize=3, linewidth=1, linestyle="--")
plt.xlabel("Test sample index (sorted by original index)")
plt.ylabel("PM2.5")
plt.title("Actual vs Predicted PM2.5 (Test set)")
plt.legend()
plt.tight_layout()
plt.savefig("actual_vs_predicted_test.png", dpi=300)
plt.show()

# Save trained model
joblib.dump(rf, "rf_pm25_model.joblib")

# Save feature list and basic metrics for reproducibility
meta = {
    "features": features,
    "validation_rmse": float(rmse_val),
    "validation_r2": float(r2_val),
    "test_rmse": float(rmse_test),
    "test_r2": float(r2_test)
}
with open("rf_pm25_model_meta.json", "w") as f:
    json.dump(meta, f)

print("Model saved to rf_pm25_model.joblib and metadata to rf_pm25_model_meta.json")