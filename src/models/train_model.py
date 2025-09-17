import sys
from pathlib import Path

# === Make repo root importable no matter where we run this ===
repo_root = Path(__file__).resolve().parents[2]
if str(repo_root) not in sys.path:
    sys.path.append(str(repo_root))

# =============================================================

import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.neural_network import MLPRegressor
import joblib

# Import cleaning function (works now because repo root is in sys.path)
from src.data.clean_data import get_cleaned_data

# ======================
# 1. Load cleaned dataset
# ======================
df = get_cleaned_data()   # <--- CLEANED DATA HERE

# ======================
# 2. Select features and targets
# ======================
X = df[['accel_x', 'accel_y', 'accel_z',
        'gyro_x', 'gyro_y', 'gyro_z',
        'mag_x', 'mag_y', 'mag_z']].values

y = df[['pos_x', 'pos_y', 'pos_z',
        'roll', 'pitch', 'yaw']].values

# ======================
# 3. Normalize
# ======================
scaler_X = StandardScaler()
scaler_y = StandardScaler()

X = scaler_X.fit_transform(X)
y = scaler_y.fit_transform(y)

# Save scalers in src/models
model_dir = Path(__file__).parent
joblib.dump(scaler_X, model_dir / "scaler_X_1.pkl")
joblib.dump(scaler_y, model_dir / "scaler_y.pkl")

# ======================
# 4. Sliding window creation
# ======================
def create_windows(X, y, window_size=50):
    Xs, ys = [], []
    for i in range(len(X) - window_size):
        Xs.append(X[i:(i+window_size)].flatten())  # flatten for sklearn
        ys.append(y[i+window_size])
    return np.array(Xs), np.array(ys)

window_size = 50
X_seq, y_seq = create_windows(X, y, window_size)

# ======================
# 5. Train/test split
# ======================
split_idx = int(len(X_seq) * 0.8)
X_train, X_test = X_seq[:split_idx], X_seq[split_idx:]
y_train, y_test = y_seq[:split_idx], y_seq[split_idx:]

# ======================
# 6. Define and train model
# ======================
model = MLPRegressor(hidden_layer_sizes=(128, 64),
                     activation='relu',
                     solver='adam',
                     max_iter=50,
                     batch_size=64,
                     random_state=42,
                     verbose=True)

model.fit(X_train, y_train)

# ======================
# 7. Evaluate
# ======================
y_pred = model.predict(X_test)

# Undo normalization
y_true = scaler_y.inverse_transform(y_test)
y_pred = scaler_y.inverse_transform(y_pred)

print("R² per output:", [r2_score(y_true[:, i], y_pred[:, i]) for i in range(y_true.shape[1])])
print("MAE per output:", [mean_absolute_error(y_true[:, i], y_pred[:, i]) for i in range(y_true.shape[1])])
print("RMSE per output:", [np.sqrt(mean_squared_error(y_true[:, i], y_pred[:, i])) for i in range(y_true.shape[1])])

# ======================
# 8. Save trained model
# ======================
joblib.dump(model, model_dir / "drone_model_1.pkl")
print("✅ Model and scalers saved successfully in src/models/")
