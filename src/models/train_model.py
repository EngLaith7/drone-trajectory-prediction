import sys
from pathlib import Path

# === Ensure script folder is in sys.path ===
script_dir = Path(__file__).resolve().parent
if str(script_dir) not in sys.path:
    sys.path.append(str(script_dir))

# =================================================
# Imports
# =================================================
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
import joblib
from src.data.clean_data import get_cleaned_data

# =================================================
# 1. Load cleaned dataset
# =================================================
df = get_cleaned_data()

# Features (9 inputs) & targets (6 outputs)
X = df[['accel_x', 'accel_y', 'accel_z',
        'gyro_x', 'gyro_y', 'gyro_z',
        'mag_x', 'mag_y', 'mag_z']].values

y = df[['pos_x', 'pos_y', 'pos_z',
        'roll', 'pitch', 'yaw']].values

# =================================================
# 2. Scaling
# =================================================
scaler_X = StandardScaler()
scaler_y = StandardScaler()

X = scaler_X.fit_transform(X)
y = scaler_y.fit_transform(y)

# Save scalers for later evaluation
model_dir = script_dir  # ✅ save in same folder as this script
joblib.dump(scaler_X, model_dir / "scaler_X.pkl")
joblib.dump(scaler_y, model_dir / "scaler_y.pkl")

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

dataset = TensorDataset(X_tensor, y_tensor)
dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

# =================================================
# 4. Define LSTM Model
# =================================================
class DroneLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=2):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        _, (h_n, _) = self.lstm(x)
        return self.fc(h_n[-1])

input_dim = X.shape[1]   # 9 features
hidden_dim = 128
output_dim = y.shape[1]  # 6 targets
num_layers = 2

model = DroneLSTM(input_dim, hidden_dim, output_dim, num_layers)

# =================================================
# 5. Train the model
# =================================================
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
epochs = 20

for epoch in range(epochs):
    model.train()
    epoch_loss = 0
    for batch_X, batch_y in dataloader:
        optimizer.zero_grad()
        y_pred = model(batch_X)
        loss = criterion(y_pred, batch_y)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    print(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss/len(dataloader):.6f}")

# =================================================
# 6. Save the model
# =================================================
torch.save(model.state_dict(), model_dir / "drone_lstm_model.pth")
print(f"✅ Model and scalers saved successfully in {model_dir.resolve()}")
