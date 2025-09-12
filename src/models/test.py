import sys
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, ParameterGrid
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import joblib

# === Make repo root importable no matter where we run this ===
repo_root = Path(__file__).resolve().parents[2]
if str(repo_root) not in sys.path:
    sys.path.append(str(repo_root))

# Define model directory inside src/models
model_dir = Path(__file__).parent / "src" / "models"
model_dir.mkdir(parents=True, exist_ok=True)

# Functions from your teammate's code
def clean_drone_data(file_path: Path, sigma_threshold: float = 3.0, dropna: bool = True):
    # ... (same content as before)
    return df

def _grid_search_cleaning(file_path: Path, sigma_values=[2.0, 2.5, 3.0, 3.5, 4.0], dropna_values=[True]):
    # ... (same content as before)
    return best_params

def get_cleaned_data(sigma_threshold: float = 3.0, dropna: bool = True, use_grid_search: bool = False):
    # ... (same content as before)
    return clean_drone_data(file_path, sigma_threshold=sigma_threshold, dropna=dropna)

# Sliding window
def create_sliding_window(X, y, window_size=5):
    Xs, ys = [], []
    for i in range(len(X) - window_size):
        Xs.append(X[i:i+window_size].flatten())
        ys.append(y[i+window_size])
    return np.array(Xs), np.array(ys)

# Get cleaned data
df = get_cleaned_data(use_grid_search=True)
print("✅ Cleaned dataset shape:", df.shape)

# 1. Features and targets
X = df[['accel_x', 'accel_y', 'accel_z',
        'gyro_x', 'gyro_y', 'gyro_z',
        'mag_x', 'mag_y', 'mag_z']].values
y = df[['pos_x', 'pos_y', 'pos_z', 'roll', 'pitch', 'yaw']].values

# 2. Create sliding windows
window_size = 5
X_windowed, y_windowed = create_sliding_window(X, y, window_size=window_size)

# 3. Train/test split
X_train, X_test, y_train, y_test = train_test_split(X_windowed, y_windowed, test_size=0.2, random_state=42)

# 4. Normalize
scaler_X = StandardScaler()
scaler_y = StandardScaler()
X_train_scaled = scaler_X.fit_transform(X_train)
X_test_scaled = scaler_X.transform(X_test)
y_train_scaled = scaler_y.fit_transform(y_train)
y_test_scaled = scaler_y.transform(y_test)

# 5. Train model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train_scaled, y_train_scaled)

# 6. Predict
y_pred_scaled = model.predict(X_test_scaled)
y_pred = scaler_y.inverse_transform(y_pred_scaled)

# 7. Evaluate
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"Test MSE: {mse:.4f}")
print(f"Test R2: {r2:.4f}")

# 8. Save model and scalers
joblib.dump(scaler_X, model_dir / "scaler_X.pkl")
joblib.dump(scaler_y, model_dir / "scaler_y.pkl")
joblib.dump(model, model_dir / "drone_model.pkl")
print("✅ Model and scalers saved successfully in src/models/")
