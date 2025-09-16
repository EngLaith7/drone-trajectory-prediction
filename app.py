#!/usr/bin/env python3
"""
Drone Path Prediction App
Streamlit application for visualizing drone 3D path from IMU CSV input using a pre-trained model.
"""

import streamlit as st
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib

# paths
repo_root = Path(__file__).parent
model_dir = repo_root / "src" / "models"
data_dir = repo_root / "data"
sample_file = data_dir / "imu_data.csv"  

# Load model and scalers
@st.cache_resource
def load_model_and_scalers():
    model = joblib.load(model_dir / "drone_model.pkl")
    scaler_X = joblib.load(model_dir / "scaler_X.pkl")
    scaler_y = joblib.load(model_dir / "scaler_y.pkl")
    return model, scaler_X, scaler_y

# Data division function into windows
def create_windows(X, window_size=50):
    Xs = []
    for i in range(len(X) - window_size):
        Xs.append(X[i:(i+window_size)].flatten())
    return np.array(Xs)

# Streamlit preparation row
st.set_page_config(
    page_title="Drone Path Predictor",
    page_icon="🚁",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""<style>
.big-title { font-size:2.5rem; font-weight:bold; text-align:center; color:#0066cc; margin-bottom:1rem; }
</style>""", unsafe_allow_html=True)
st.markdown('<div class="big-title">🚁 Drone Path Predictor</div>', unsafe_allow_html=True)
st.markdown("---")

st.write("""
Upload your IMU CSV file (must include columns: accel_x, accel_y, accel_z, gyro_x, gyro_y, gyro_z, mag_x, mag_y, mag_z),  
or try the app with a sample dataset.
""")

# 🟢 Choose a data source
option = st.radio(
    "Select data source:",
    ["Upload my own CSV", "Use sample data (fixed 1000 rows)"],
    index=0
)

uploaded = None
sample_rows = 1000  # A fixed number of rows for sample data

if option == "Upload my own CSV":
    uploaded = st.file_uploader("Upload IMU data CSV here", type=["csv"])

elif option == "Use sample data (fixed 1000 rows)":
    if sample_file.exists():
        uploaded = sample_file
        st.info(f"Using sample data (first {sample_rows} rows).")
    else:
        st.error(f"Sample file not found at {sample_file}")


if uploaded:
    # Load model
    model, scaler_X, scaler_y = load_model_and_scalers()

    # Read data
    try:
        if isinstance(uploaded, Path):  # sample file
            df = pd.read_csv(uploaded, nrows=sample_rows)  # ← Fixed number
        else:  # user uploaded file
            df = pd.read_csv(uploaded)  # ← All data raised
    except Exception as e:
        st.error(f"Error reading CSV: {e}")
        st.stop()

    sensor_cols = ['accel_x', 'accel_y', 'accel_z',
                   'gyro_x', 'gyro_y', 'gyro_z',
                   'mag_x', 'mag_y', 'mag_z']

    missing_cols = [c for c in sensor_cols if c not in df.columns]
    if missing_cols:
        st.error(f"Missing columns in CSV: {missing_cols}")
        st.stop()

    # Prediction
    X = df[sensor_cols].values
    X_scaled = scaler_X.transform(X)
    window_size = model.n_features_in_ // len(sensor_cols)
    if len(X_scaled) < window_size+1:
        st.error(f"Not enough rows. At least {window_size+1} rows required for window size {window_size}.")
        st.stop()

    X_windows = create_windows(X_scaled, window_size=window_size)
    y_pred_scaled = model.predict(X_windows)
    y_pred = scaler_y.inverse_transform(y_pred_scaled)

    pred_df = pd.DataFrame(
        y_pred,
        columns=['pos_x', 'pos_y', 'pos_z', 'roll', 'pitch', 'yaw']
    )
    pred_df.index = np.arange(window_size, window_size + len(pred_df))

    # Show the values of schedule (first 10)
    st.subheader("First 10 Predicted Drone States (Position & Orientation)")
    st.dataframe(pred_df.head(10).style.format(precision=3))

    # Drawing three-dimensional path
    st.subheader("Predicted Drone 3D Path")
    fig = plt.figure(figsize=(10,7))
    ax = fig.add_subplot(111, projection='3d')

    #Sample data: draw only 1000 Uploaded:draw all points
    if isinstance(uploaded, Path):
        pts = pred_df[['pos_x','pos_y','pos_z']].values[:sample_rows]
    else:
        pts = pred_df[['pos_x','pos_y','pos_z']].values

    colors = np.linspace(0, 1, len(pts))
    ax.scatter(pts[:,0], pts[:,1], pts[:,2], c=colors, cmap='viridis', s=10)
    ax.plot(pts[:,0], pts[:,1], pts[:,2], color='blue', alpha=0.5)
    ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')
    ax.set_title("Predicted Drone Path")
    st.pyplot(fig)

    st.success("✅ Prediction and visualization complete!")
else:
    st.info("Awaiting CSV upload or sample data to begin prediction...")
