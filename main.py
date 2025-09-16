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
sys.path.append(str(repo_root / "src" / "utils"))
MODEL_PATH = str(repo_root / "src" / "models" / "drone_model.pkl")

from src.data.clean_data import get_cleaned_data
from src.data.Visualization import (plot_accelerometer_distribution, plot_gyroscope_distribution, plot_magnetometer_distribution,
                            plot_accelerometer_timeseries, plot_gyroscope_timeseries, plot_3dplot_trajectory,
                            plot_orientation, plot_correlation_heatmap_plot, plot_trajectory_partial)
from src.models.training import train_model
from src.utils.helpers import print_project_info, load_data



def main():
    # 1. Print project info
    print_project_info()
    # 2. Load data
    print("📊 Loading and cleaning data...")
    df = load_data()
    print(f"Loaded cleaned dataset with shape: {df.shape}")
    # 3. Visualizations
    print("="*50)
    print("📊 Visualizing feature distributions...")
    plot_accelerometer_distribution(df)
    plot_gyroscope_distribution(df)
    plot_magnetometer_distribution(df)
    plot_accelerometer_timeseries(df)
    plot_gyroscope_timeseries(df)
    plot_correlation_heatmap_plot(df)
    plot_3dplot_trajectory(df)
    plot_trajectory_partial(df,10000)
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