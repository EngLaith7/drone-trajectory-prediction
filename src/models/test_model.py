import sys
from pathlib import Path

# === Ensure repo root is in sys.path ===
repo_root = Path(__file__).resolve().parents[2]
if str(repo_root) not in sys.path:
    sys.path.append(str(repo_root))

# =================================================
# Imports
# =================================================
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import joblib
from src.data.clean_data import get_cleaned_data

# =================================================
# 1. Load cleaned dataset
# =================================================
print("📂 Loading dataset...")
df = get_cleaned_data()

# Features & targets
X = df[['accel_x', 'accel_y', 'accel_z',
        'gyro_x', 'gyro_y', 'gyro_z',
        'mag_x', 'mag_y', 'mag_z']].values

y = df[['pos_x', 'pos_y', 'pos_z',
        'roll', 'pitch', 'yaw']].values

# =================================================
# 2. Load saved scalers (must exist from training)
# =================================================
model_dir = Path(__file__).resolve().parent  # <-- now src/model/

scaler_X = joblib.load(model_dir / "scaler_X.pkl")
scaler_y = joblib.load(model_dir / "scaler_y.pkl")

X = scaler_X.transform(X)
y = scaler_y.transform(y)

# =================================================
# 3. Create sliding windows
# =================================================
def create_windows(X, y, window_size=50):
    Xs, ys = [], []
    for i in range(len(X) - window_size):
        Xs.append(X[i:(i+window_size)])
        ys.append(y[i+window_size])
    return np.array(Xs), np.array(ys)

window_size = 50
X_seq, y_seq = create_windows(X, y, window_size)

# Convert to torch tensors
X_tensor = torch.tensor(X_seq, dtype=torch.float32)
y_tensor = torch.tensor(y_seq, dtype=torch.float32)

# =================================================
# 4. Define LSTM Model (must match training)
# =================================================
class DroneLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=2):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        _, (h_n, _) = self.lstm(x)  # last hidden state
        return self.fc(h_n[-1])

# Model params (must match training)
input_dim = X.shape[1]   # 9 features
hidden_dim = 128
output_dim = y.shape[1]  # 6 outputs
num_layers = 2

model = DroneLSTM(input_dim, hidden_dim, output_dim, num_layers)

# =================================================
# 5. Load trained model
# =================================================
model.load_state_dict(torch.load(model_dir / "drone_lstm_model.pth", map_location="cpu"))
model.eval()
print("✅ Model and scalers loaded successfully!")

# =================================================
# 6. Run predictions
# =================================================
dataset = TensorDataset(X_tensor, y_tensor)
loader = DataLoader(dataset, batch_size=256)

preds = []
with torch.no_grad():
    for batch_X, _ in loader:
        batch_pred = model(batch_X).cpu().numpy()
        preds.append(batch_pred)

y_pred = np.vstack(preds)

# Undo normalization
y_true = scaler_y.inverse_transform(y_seq)
y_pred = scaler_y.inverse_transform(y_pred)

# =================================================
# 7. Report performance
# =================================================
r2_scores = [r2_score(y_true[:, i], y_pred[:, i]) for i in range(y_true.shape[1])]
mae_scores = [mean_absolute_error(y_true[:, i], y_pred[:, i]) for i in range(y_true.shape[1])]
rmse_scores = [np.sqrt(mean_squared_error(y_true[:, i], y_pred[:, i])) for i in range(y_true.shape[1])]

print("\n📊 Model Evaluation on Dataset")
print("R² per output:", r2_scores)
print("MAE per output:", mae_scores)
print("RMSE per output:", rmse_scores)

# Overall score
mean_r2 = np.mean(r2_scores)
print(f"\n✅ Overall Score (Mean R²): {mean_r2:.4f}")

# =================================================
# 8. Example predictions
# =================================================
print("\n🔮 Example predictions vs actual:")
for i in range(5):  # show 5 samples
    print(f"True: {y_true[i]}, Pred: {y_pred[i]}")
