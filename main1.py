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

# Add src/data and src/models to path for imports
repo_root = Path(__file__).resolve().parents[0]
sys.path.append(str(repo_root / "src" / "data"))
sys.path.append(str(repo_root / "src" / "models"))

from clean_data import get_cleaned_data
from Visualization import (plot_accelerometer_distribution, plot_gyroscope_distribution, plot_magnetometer_distribution,
                            plot_accelerometer_timeseries, plot_gyroscope_timeseries, plot_3d_trajectory,
                            plot_orientation, plot_correlation_heatmap)
from training import train_model

def print_project_info():
    print("="*50)
    print("🚁 Drone Sensor ML Project - Trajectory & Orientation Prediction")
    print("="*50)
    print("Predict drone state (position & orientation) from raw IMU/gyro/magnetometer data.")
    print("Data: imu_data.csv (put in ./data/)")
    print("="*50)

def plot_feature_distributions(df): ...
def plot_correlation_heatmap(df): ...
def plot_3d_trajectory(df, title="Drone 3D Trajectory (first 1000 points)"): ...
def create_windows(X, y, window_size=50): ...
def save_model(model, scaler_X, scaler_y, path_prefix="models/drone"): ...

def main():
    # 1. Print project info
    print_project_info()
    # 2. Load data
    print("📊 Loading and cleaning data...")
    df = get_cleaned_data()
    print(f"Loaded cleaned dataset with shape: {df.shape}")
    # 3. Visualizations
    print("📊 Visualizing feature distributions...")
    plot_accelerometer_distribution(df)
    plot_gyroscope_distribution(df)
    plot_magnetometer_distribution(df)
    plot_accelerometer_timeseries(df)
    plot_gyroscope_timeseries(df)
    plot_3d_trajectory(df)
    plot_orientation(df)
    plot_correlation_heatmap(df)

    print("🤖 Training the model...")
    train_model()

if __name__ == "__main__":
    main()