#!/usr/bin/env python3
"""
Drone Path Prediction App
Streamlit application for visualizing drone 3D path from IMU CSV input using a pre-trained model.
"""

import streamlit as st
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib

# Set up paths for importing utilities/models
repo_root = Path(__file__).parent
model_dir = repo_root / "models"

# Helper to load model and scalers
@st.cache_resource
def load_model_and_scalers():
    model = joblib.load(model_dir / "drone_model.pkl")
    scaler_X = joblib.load(model_dir / "drone_scaler_X.pkl")
    scaler_y = joblib.load(model_dir / "drone_scaler_y.pkl")
    return model, scaler_X, scaler_y

def create_windows(X, window_size=50):
    Xs = []
    for i in range(len(X) - window_size):
        Xs.append(X[i:(i+window_size)].flatten())
    return np.array(Xs)

st.set_page_config(
    page_title="Drone Path Predictor",
    page_icon="🚁",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
.big-title { font-size:2.5rem; font-weight:bold; text-align:center; color:#0066cc; margin-bottom:1rem; }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="big-title">🚁 Drone Path Predictor</div>', unsafe_allow_html=True)
st.markdown("---")

st.write("""
Upload your IMU CSV file (must include columns: accel_x, accel_y, accel_z, gyro_x, gyro_y, gyro_z, mag_x, mag_y, mag_z). 
The app predicts the drone's 3D path and displays the first 10 predicted states.
""")

uploaded = st.file_uploader("Upload IMU data CSV here", type=["csv"])

if uploaded:
    # Load model and scalers
    model, scaler_X, scaler_y = load_model_and_scalers()

    # Read data
    try:
        df = pd.read_csv(uploaded)
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

    X = df[sensor_cols].values
    X_scaled = scaler_X.transform(X)
    window_size = model.n_features_in_ // len(sensor_cols)
    if len(X_scaled) < window_size+1:
        st.error(f"Not enough rows. At least {window_size+1} rows required for window size {window_size}.")
        st.stop()

    X_windows = create_windows(X_scaled, window_size=window_size)
    y_pred_scaled = model.predict(X_windows)
    y_pred = scaler_y.inverse_transform(y_pred_scaled)

    # Get DataFrame for predictions (align index)
    pred_df = pd.DataFrame(
        y_pred,
        columns=['pos_x', 'pos_y', 'pos_z', 'roll', 'pitch', 'yaw']
    )
    pred_df.index = np.arange(window_size, window_size + len(pred_df))

    st.subheader("First 10 Predicted Drone States (Position & Orientation)")
    st.dataframe(pred_df.head(10).style.format(precision=3))

    # 3D Path Plot
    st.subheader("Predicted Drone 3D Path (first 1000 points)")
    fig = plt.figure(figsize=(10,7))
    ax = fig.add_subplot(111, projection='3d')
    pts = pred_df[['pos_x','pos_y','pos_z']].values[:1000]
    colors = np.linspace(0, 1, len(pts))
    ax.scatter(pts[:,0], pts[:,1], pts[:,2], c=colors, cmap='viridis', s=10)
    ax.plot(pts[:,0], pts[:,1], pts[:,2], color='blue', alpha=0.5)
    ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')
    ax.set_title("Predicted Drone Path")
    st.pyplot(fig)

    st.success("✅ Prediction and visualization complete!")
else:
    st.info("Awaiting CSV upload to begin prediction...")