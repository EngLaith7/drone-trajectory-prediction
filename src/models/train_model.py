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
from torch.utils.data import DataLoader, TensorDataset, random_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import joblib

from src.data.clean_data import get_cleaned_data

# =================================================
# Config
# =================================================
EPOCHS = 20
BATCH_SIZE = 64
HIDDEN_DIM = 128
NUM_LAYERS = 2
WINDOW_SIZE = 50

# Paths
script_dir = Path(__file__).resolve().parent
scaler_X_path = script_dir / "scaler_X.pkl"
scaler_y_path = script_dir / "scaler_y.pkl"
model_path = script_dir / "drone_lstm_model.pth"

# =================================================
# Helper: create sliding windows
# =================================================
def create_windows(X, y, window_size=50):
    Xs, ys = [], []
    for i in range(len(X) - window_size):
        Xs.append(X[i:(i+window_size)])
        ys.append(y[i+window_size])
    return np.array(Xs), np.array(ys)

# =================================================
# Model definition
# =================================================
class DroneLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=2):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        _, (h_n, _) = self.lstm(x)
        return self.fc(h_n[-1])

# =================================================
# Evaluation function
# =================================================
def evaluate_model(model, test_loader, scaler_y):
    model.eval()
    y_true_list, y_pred_list = [], []

    with torch.no_grad():
        for batch_X, batch_y in test_loader:
            preds = model(batch_X)
            y_true_list.append(batch_y.cpu().numpy())
            y_pred_list.append(preds.cpu().numpy())

    y_true = np.vstack(y_true_list)
    y_pred = np.vstack(y_pred_list)

    # inverse transform
    y_true = scaler_y.inverse_transform(y_true)
    y_pred = scaler_y.inverse_transform(y_pred)

    # overall metrics
    r2 = r2_score(y_true, y_pred, multioutput="variance_weighted")
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))

    print("\n✅ Test Set Evaluation:")
    print(f"R² Score (overall): {r2:.4f}")
    print(f"MAE (overall): {mae:.4f}")
    print(f"RMSE (overall): {rmse:.4f}")

    # per-output metrics
    outputs = ['pos_x', 'pos_y', 'pos_z', 'roll', 'pitch', 'yaw']
    print("\n📊 Per-output metrics:")
    for i, name in enumerate(outputs):
        r2_i = r2_score(y_true[:, i], y_pred[:, i])
        mae_i = mean_absolute_error(y_true[:, i], y_pred[:, i])
        rmse_i = np.sqrt(mean_squared_error(y_true[:, i], y_pred[:, i]))
        print(f"{name:>7} | R²: {r2_i:.4f} | MAE: {mae_i:.4f} | RMSE: {rmse_i:.4f}")

# =================================================
# Main workflow
# =================================================
if model_path.exists() and scaler_X_path.exists() and scaler_y_path.exists():
    print("⚡ Found existing model + scalers, skipping training...")

    # load scalers
    scaler_X = joblib.load(scaler_X_path)
    scaler_y = joblib.load(scaler_y_path)

    # load dataset (scaled)
    df = get_cleaned_data()
    X = scaler_X.transform(df[['accel_x','accel_y','accel_z',
                               'gyro_x','gyro_y','gyro_z',
                               'mag_x','mag_y','mag_z']].values)
    y = scaler_y.transform(df[['pos_x','pos_y','pos_z',
                               'roll','pitch','yaw']].values)

    X_seq, y_seq = create_windows(X, y, WINDOW_SIZE)
    X_tensor = torch.tensor(X_seq, dtype=torch.float32)
    y_tensor = torch.tensor(y_seq, dtype=torch.float32)
    dataset = TensorDataset(X_tensor, y_tensor)

    # 80/20 split
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    _, test_dataset = random_split(dataset, [train_size, test_size])
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # rebuild model + load weights
    model = DroneLSTM(input_dim=X.shape[1], hidden_dim=HIDDEN_DIM, 
                      output_dim=y.shape[1], num_layers=NUM_LAYERS)
    model.load_state_dict(torch.load(model_path, map_location="cpu"))

    evaluate_model(model, test_loader, scaler_y)

else:
    print("🚀 Training new model...")

    # load dataset
    df = get_cleaned_data()
    X = df[['accel_x','accel_y','accel_z',
            'gyro_x','gyro_y','gyro_z',
            'mag_x','mag_y','mag_z']].values
    y = df[['pos_x','pos_y','pos_z','roll','pitch','yaw']].values

    # scaling
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    X = scaler_X.fit_transform(X)
    y = scaler_y.fit_transform(y)

    joblib.dump(scaler_X, scaler_X_path)
    joblib.dump(scaler_y, scaler_y_path)

    # windows
    X_seq, y_seq = create_windows(X, y, WINDOW_SIZE)
    X_tensor = torch.tensor(X_seq, dtype=torch.float32)
    y_tensor = torch.tensor(y_seq, dtype=torch.float32)
    dataset = TensorDataset(X_tensor, y_tensor)

    # split
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # model
    model = DroneLSTM(input_dim=X.shape[1], hidden_dim=HIDDEN_DIM, 
                      output_dim=y.shape[1], num_layers=NUM_LAYERS)

    # training
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(EPOCHS):
        model.train()
        epoch_loss = 0
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            preds = model(batch_X)
            loss = criterion(preds, batch_y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        print(f"Epoch {epoch+1}/{EPOCHS}, Train Loss: {epoch_loss/len(train_loader):.6f}")

    # save
    torch.save(model.state_dict(), model_path)
    print(f"✅ Model and scalers saved at {script_dir.resolve()}")

    # final evaluation
    evaluate_model(model, test_loader, scaler_y)
