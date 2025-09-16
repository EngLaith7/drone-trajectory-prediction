import sys
from pathlib import Path

# === Ensure repo root is in sys.path ===
repo_root = Path(__file__).resolve().parents[2]
if str(repo_root) not in sys.path:
    sys.path.append(str(repo_root))

# =================================================
# Imports
# =================================================
import streamlit as st
import numpy as np
import torch
import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

# Import dataset
from src.data.clean_data import get_cleaned_data

# =================================================
# Path setup
# =================================================
model_dir = repo_root / "src" / "models"

# =================================================
# Model definitions
# =================================================
class DroneLSTM(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=2):
        super().__init__()
        self.lstm = torch.nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = torch.nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])  # last timestep


class DroneGRU(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=2):
        super().__init__()
        self.gru = torch.nn.GRU(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = torch.nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        out, _ = self.gru(x)
        return self.fc(out[:, -1, :])  # last timestep


# =================================================
# Load Model + Scalers (NO TRAINING)
# =================================================
def load_model(model_type="LSTM"):
    model_file = model_dir / f"drone_{model_type.lower()}_model.pth"
    scaler_X_file = model_dir / "scaler_X.pkl"
    scaler_y_file = model_dir / "scaler_y.pkl"

    if not model_file.exists() or not scaler_X_file.exists() or not scaler_y_file.exists():
        st.error(f"❌ {model_type} model or scalers not found in {model_dir}")
        st.stop()

    # Load scalers
    scaler_X = joblib.load(scaler_X_file)
    scaler_y = joblib.load(scaler_y_file)

    # Define model
    input_dim, hidden_dim, output_dim, num_layers = 9, 128, 6, 2
    model = DroneLSTM(input_dim, hidden_dim, output_dim, num_layers) if model_type == "LSTM" \
        else DroneGRU(input_dim, hidden_dim, output_dim, num_layers)

    # Load weights
    model.load_state_dict(torch.load(model_file, map_location="cpu"))
    model.eval()

    st.sidebar.success(f"✅ Loaded pre-trained {model_type} model")
    return model, scaler_X, scaler_y


# =================================================
# Helper: sliding window creation
# =================================================
def create_windows(X, y, window_size=50):
    Xs, ys = [], []
    for i in range(len(X) - window_size):
        Xs.append(X[i:(i+window_size)])
        ys.append(y[i+window_size])
    return np.array(Xs), np.array(ys)


# =================================================
# Streamlit UI
# =================================================
st.title("🚁 Drone Model Evaluation Dashboard")

# Sidebar
model_type = st.sidebar.radio("Choose Model Type:", ["LSTM", "GRU"])
menu = st.sidebar.radio("Choose Action:", ["Run Evaluation", "Manual Prediction"])

# Load pre-trained model
model, scaler_X, scaler_y = load_model(model_type)


# === Evaluation Mode ===
if menu == "Run Evaluation":
    st.header("📊 Run Evaluation on Test Dataset")
    df = get_cleaned_data()

    X = df[['accel_x','accel_y','accel_z',
            'gyro_x','gyro_y','gyro_z',
            'mag_x','mag_y','mag_z']].values
    y = df[['pos_x','pos_y','pos_z',
            'roll','pitch','yaw']].values

    X = scaler_X.transform(X)
    y = scaler_y.transform(y)

    X_seq, y_seq = create_windows(X, y, 50)
    X_tensor = torch.tensor(X_seq, dtype=torch.float32)

    # Batch inference to save memory
    batch_size = 256
    preds = []
    with torch.no_grad():
        for i in range(0, len(X_tensor), batch_size):
            batch = X_tensor[i:i+batch_size]
            preds.append(model(batch).numpy())
    y_pred = np.vstack(preds)

    y_true = scaler_y.inverse_transform(y_seq)
    y_pred = scaler_y.inverse_transform(y_pred)

    # Metrics
    r2_scores = [r2_score(y_true[:, i], y_pred[:, i]) for i in range(y_true.shape[1])]
    mae_scores = [mean_absolute_error(y_true[:, i], y_pred[:, i]) for i in range(y_true.shape[1])]
    rmse_scores = [np.sqrt(mean_squared_error(y_true[:, i], y_pred[:, i])) for i in range(y_true.shape[1])]
    mean_r2 = np.mean(r2_scores)

    st.subheader("Model Performance")
    st.write(f"**R² per output:** {r2_scores}")
    st.write(f"**MAE per output:** {mae_scores}")
    st.write(f"**RMSE per output:** {rmse_scores}")
    st.write(f"✅ **Overall Score (Mean R²): {mean_r2:.4f}**")

    # Graphs
    st.subheader("📈 Predicted vs Actual (First 200 samples)")
    fig, axes = plt.subplots(3, 2, figsize=(12, 8))
    outputs = ["pos_x","pos_y","pos_z","roll","pitch","yaw"]

    for i, ax in enumerate(axes.flatten()):
        ax.plot(y_true[:200, i], label="True", alpha=0.7)
        ax.plot(y_pred[:200, i], label="Pred", alpha=0.7)
        ax.set_title(outputs[i])
        ax.legend()

    st.pyplot(fig)

    st.subheader("🔮 Example Predictions")
    for i in range(5):
        st.write(f"True: {y_true[i]}, Pred: {y_pred[i]}")

# === Manual Input Mode ===
else:
    st.header("🎛️ Manual Prediction")
    st.write("Enter sensor values to predict drone position + orientation")

    accel_x = st.number_input("Accel X", value=0.0)
    accel_y = st.number_input("Accel Y", value=0.0)
    accel_z = st.number_input("Accel Z", value=0.0)
    gyro_x  = st.number_input("Gyro X", value=0.0)
    gyro_y  = st.number_input("Gyro Y", value=0.0)
    gyro_z  = st.number_input("Gyro Z", value=0.0)
    mag_x   = st.number_input("Mag X", value=0.0)
    mag_y   = st.number_input("Mag Y", value=0.0)
    mag_z   = st.number_input("Mag Z", value=0.0)

    if st.button("Predict"):
        X_input = np.array([[accel_x, accel_y, accel_z,
                             gyro_x, gyro_y, gyro_z,
                             mag_x, mag_y, mag_z]])
        X_input = scaler_X.transform(X_input)

        # Fake sequence history by repeating same input
        X_seq = np.tile(X_input, (50,1)).reshape(1,50,9)

        X_tensor = torch.tensor(X_seq, dtype=torch.float32)
        with torch.no_grad():
            y_pred = model(X_tensor).numpy()

        y_pred = scaler_y.inverse_transform(y_pred)
        st.success(f"Predicted Output (pos_x, pos_y, pos_z, roll, pitch, yaw): {y_pred[0]}")
