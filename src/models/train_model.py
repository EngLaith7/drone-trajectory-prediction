import sys
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

# === Ensure repo root is in sys.path ===
repo_root = Path(__file__).resolve().parents[2]
if str(repo_root) not in sys.path:
    sys.path.append(str(repo_root))

# Import the cleaned data (with grid search option if desired)
from src.data.clean_data import get_cleaned_data


# ======================
# 1. Load cleaned dataset
# ======================
df = get_cleaned_data(use_grid_search=True)  # <-- set to False if you want fixed σ

# Features & targets
X = df[['accel_x', 'accel_y', 'accel_z',
        'gyro_x', 'gyro_y', 'gyro_z',
        'mag_x', 'mag_y', 'mag_z']].values

y = df[['pos_x', 'pos_y', 'pos_z',
        'roll', 'pitch', 'yaw']].values


# ======================
# 2. Normalize features & targets
# ======================
scaler_X = StandardScaler()
scaler_y = StandardScaler()
X = scaler_X.fit_transform(X)
y = scaler_y.fit_transform(y)


# ======================
# 3. Create sliding windows (sequences)
# ======================
def create_windows(X, y, window_size=50):
    Xs, ys = [], []
    for i in range(len(X) - window_size):
        Xs.append(X[i:(i+window_size)])
        ys.append(y[i+window_size])
    return np.array(Xs), np.array(ys)

window_size = 50
X_seq, y_seq = create_windows(X, y, window_size)

# Train/test split
split = int(0.8 * len(X_seq))
X_train, X_test = X_seq[:split], X_seq[split:]
y_train, y_test = y_seq[:split], y_seq[split:]


# ======================
# 4. Convert to Torch tensors
# ======================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

X_train = torch.tensor(X_train, dtype=torch.float32).to(device)
y_train = torch.tensor(y_train, dtype=torch.float32).to(device)
X_test = torch.tensor(X_test, dtype=torch.float32).to(device)
y_test = torch.tensor(y_test, dtype=torch.float32).to(device)


# ======================
# 5. Define LSTM Model
# ======================
class DroneLSTM(nn.Module):
    def __init__(self, n_features, n_outputs, hidden_size=128, num_layers=2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True
        )
        self.fc = nn.Linear(hidden_size, n_outputs)

    def forward(self, x):
        _, (h_n, _) = self.lstm(x)   # h_n shape: (num_layers, batch, hidden)
        return self.fc(h_n[-1])      # use last layer’s hidden state


n_features = X.shape[1]  # 9 sensors
n_outputs = y.shape[1]   # 6 labels
model = DroneLSTM(n_features, n_outputs).to(device)


# ======================
# 6. Training Loop
# ======================
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
epochs = 50
batch_size = 64

for epoch in range(epochs):
    permutation = torch.randperm(X_train.size(0))
    for i in range(0, X_train.size(0), batch_size):
        idx = permutation[i:i+batch_size]
        batch_X, batch_y = X_train[idx], y_train[idx]

        optimizer.zero_grad()
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()

    print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}")


# ======================
# 7. Evaluation
# ======================
model.eval()
with torch.no_grad():
    y_pred = model(X_test).cpu().numpy()
    y_true = y_test.cpu().numpy()

# Undo normalization for evaluation
y_pred = scaler_y.inverse_transform(y_pred)
y_true = scaler_y.inverse_transform(y_true)

# Compute metrics
r2_scores = [r2_score(y_true[:, i], y_pred[:, i]) for i in range(y_true.shape[1])]
mae_scores = [mean_absolute_error(y_true[:, i], y_pred[:, i]) for i in range(y_true.shape[1])]
rmse_scores = [np.sqrt(mean_squared_error(y_true[:, i], y_pred[:, i])) for i in range(y_true.shape[1])]
mean_r2 = np.mean(r2_scores)

print("\n📊 Model Evaluation on Test Dataset")
print("R² per output:", r2_scores)
print("MAE per output:", mae_scores)
print("RMSE per output:", rmse_scores)
print(f"\n✅ Overall Score (Mean R²): {mean_r2:.4f}")


# ======================
# 8. Example predictions
# ======================
print("\n🔮 Example predictions vs actual:")
for i in range(5):
    print(f"True: {y_true[i]}, Pred: {y_pred[i]}")
# === Save the trained model ===
torch.save(model.state_dict(), "drone_lstm_model.pth")
print("✅ Model saved as drone_lstm_model.pth")
