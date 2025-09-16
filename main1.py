#!/usr/bin/env python3
"""
Simple AI Project - Drone Sensor Data Regressor
Main application entry point (function calls only)
"""
import sys
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os

# Add src/data and src/models to path for imports
repo_root = Path(__file__).resolve().parents[0]
sys.path.append(str(repo_root / "src" / "data"))
sys.path.append(str(repo_root / "src" / "models"))
MODEL_PATH = str(repo_root / "src" / "models" / "drone_model.pkl")

from clean_data import get_cleaned_data
from Visualization import (plot_accelerometer_distribution, plot_gyroscope_distribution, plot_magnetometer_distribution,
                            plot_accelerometer_timeseries, plot_gyroscope_timeseries, plot_3d_trajectory,
                            plot_orientation, plot_correlation_heatmap, plot_trajectory_partial)
from training import train_model

def print_project_info():
    print("="*50)
    print("🚁 Drone Sensor ML Project - Trajectory & Orientation Prediction")
    print("="*50)
    print("Predict drone state (position & orientation) from raw IMU/gyro/magnetometer data.")
    print("Data: imu_data.csv (put in ./data/)")
    print("="*50)


def main():
    # 1. Print project info
    print_project_info()
    # 2. Load data
    print("📊 Loading and cleaning data...")
    df = get_cleaned_data()
    print(f"Loaded cleaned dataset with shape: {df.shape}")
    # 3. Visualizations
    print("="*50)
    print("📊 Visualizing feature distributions...")
    plot_accelerometer_distribution(df)
    plot_gyroscope_distribution(df)
    plot_magnetometer_distribution(df)
    plot_accelerometer_timeseries(df)
    plot_gyroscope_timeseries(df)
    plot_correlation_heatmap(df)
    plot_3d_trajectory(df)
    plot_orientation(df)

    print("="*50)
    print("🤖 Checking for trained model...")

    if os.path.exists(MODEL_PATH):
        print(f"✅ Found trained model at {MODEL_PATH}, loading instead of training.")
        model = joblib.load(MODEL_PATH)
    else:
        print("⚠️ No trained model found. Training a new one...")
        model = train_model()
        joblib.dump(model, MODEL_PATH)
        print(f"💾 Model saved at {MODEL_PATH}")

if __name__ == "__main__":
    main()