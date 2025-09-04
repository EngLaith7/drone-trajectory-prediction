import sys
from pathlib import Path

# === Ensure repo root is in sys.path ===
repo_root = Path(__file__).resolve().parents[2]
if str(repo_root) not in sys.path:
    sys.path.append(str(repo_root))

# =================================================
import numpy as np
import joblib
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

# Import the cleaned data
from src.data.clean_data import get_cleaned_data

# ======================
# 1. Load cleaned dataset
# ======================
df = get_cleaned_data()

# Features & targets
X = df[['accel_x', 'accel_y', 'accel_z',
        'gyro_x', 'gyro_y', 'gyro_z',
        'mag_x', 'mag_y', 'mag_z']].values

y = df[['pos_x', 'pos_y', 'pos_z',
        'roll', 'pitch', 'yaw']].values

# ======================
# 2. Load scalers + model
# ======================
model_dir = Path(__file__).parent
scaler_X = joblib.load(model_dir / "scaler_X.pkl")
scaler_y = joblib.load(model_dir / "scaler_y.pkl")
model = joblib.load(model_dir / "drone_model.pkl")

# Apply normalization
X = scaler_X.transform(X)
y = scaler_y.transform(y)

# ======================
# 3. Recreate sliding windows (same as in training)
# ======================
def create_windows(X, y, window_size=50):
    Xs, ys = [], []
    for i in range(len(X) - window_size):
        Xs.append(X[i:(i+window_size)].flatten())
        ys.append(y[i+window_size])
    return np.array(Xs), np.array(ys)

window_size = 50
X_seq, y_seq = create_windows(X, y, window_size)

# ======================
# 4. Run predictions
# ======================
y_pred = model.predict(X_seq)

# Undo normalization
y_true = scaler_y.inverse_transform(y_seq)
y_pred = scaler_y.inverse_transform(y_pred)

# ======================
# 5. Report performance
# ======================
print("\n📊 Model Evaluation on Test Dataset")
print("R² per output:", [r2_score(y_true[:, i], y_pred[:, i]) for i in range(y_true.shape[1])])
print("MAE per output:", [mean_absolute_error(y_true[:, i], y_pred[:, i]) for i in range(y_true.shape[1])])
print("RMSE per output:", [np.sqrt(mean_squared_error(y_true[:, i], y_pred[:, i])) for i in range(y_true.shape[1])])

# ======================
# 6. Example predictions
# ======================
print("\n🔮 Example predictions vs actual:")
for i in range(5):  # show 5 samples
    print(f"True: {y_true[i]}, Pred: {y_pred[i]}")
