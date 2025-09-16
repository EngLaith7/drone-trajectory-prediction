#!/usr/bin/env python3
"""
Simple AI Project - Drone Sensor Data Regressor
Main application entry point (function calls only)
"""

from src.data.clean_data import get_cleaned_data
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import numpy as np
import joblib
from pathlib import Path
from src.data.Visualization import (plot_accelerometer_distribution, plot_gyroscope_distribution, plot_magnetometer_distribution,
                            plot_accelerometer_timeseries, plot_gyroscope_timeseries, plot_3dplot_trajectory,
                            plot_orientation, plot_correlation_heatmap_plot, plot_trajectory_partial)


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
def create_windows(X, y, window_size):
    X_seq, y_seq = [], []
    for i in range(len(X) - window_size):
        X_seq.append(X[i:(i+window_size)].flatten())
        y_seq.append(y[i+window_size])
    return np.array(X_seq), np.array(y_seq)
def save_model(model, scaler_X, scaler_y, path_prefix="models/drone"): ...

def main():
    # 1. Print project info
    print_project_info()
    # 2. Load data
    df = get_cleaned_data()
    # 3. Visualizations
    plot_accelerometer_distribution(df)
    plot_gyroscope_distribution(df)
    plot_magnetometer_distribution(df)
    plot_accelerometer_timeseries(df)
    plot_gyroscope_timeseries(df)
    plot_3dplot_trajectory(df)
    plot_orientation(df)
    plot_correlation_heatmap_plot(df)
    plot_trajectory_partial(df,1000)
    # 4. Feature selection
    sensor_cols = ['accel_x', 'accel_y', 'accel_z', 'gyro_x', 'gyro_y', 'gyro_z', 'mag_x', 'mag_y', 'mag_z']
    target_cols = ['pos_x', 'pos_y', 'pos_z', 'roll', 'pitch', 'yaw']
    X, y = df[sensor_cols].values, df[target_cols].values
    # 5. Normalization
    scaler_X, scaler_y = StandardScaler(), StandardScaler()
    X, y = scaler_X.fit_transform(X), scaler_y.fit_transform(y)
    # 6. Sliding window
    window_size = 50
    X_seq, y_seq = create_windows(X, y, window_size)
    # 7. Train/test split
    X_train, X_test, y_train, y_test = train_test_split(X_seq, y_seq, test_size=0.2, random_state=42)
    # 8. Model training
    model = MLPRegressor(hidden_layer_sizes=(128, 64), activation='relu', solver='adam', max_iter=50, batch_size=64, random_state=42, verbose=True)
    model.fit(X_train, y_train)
    # 9. Evaluation
    y_pred = model.predict(X_test)
    y_test_inv = scaler_y.inverse_transform(y_test)
    y_pred_inv = scaler_y.inverse_transform(y_pred)
    print("R² per output:", [r2_score(y_test_inv[:, i], y_pred_inv[:, i]) for i in range(y_test_inv.shape[1])])
    print("MAE per output:", [mean_absolute_error(y_test_inv[:, i], y_pred_inv[:, i]) for i in range(y_test_inv.shape[1])])
    print("RMSE per output:", [np.sqrt(mean_squared_error(y_test_inv[:, i], y_pred_inv[:, i])) for i in range(y_test_inv.shape[1])])
    # 10. Save model
    save_model(model, scaler_X, scaler_y)
    # 11. Example predictions
    for i in range(5):
        print(f"True: {np.round(y_test_inv[i], 2)}, Pred: {np.round(y_pred_inv[i], 2)}")
    print("✅ Project completed successfully!")
    print("📁 Feature plots: drone_feature_distributions.png, drone_corr_heatmap.png, drone_trajectory.png")
    print("📦 Models saved in 'models/' directory.")

if __name__ == "__main__":
    main()