#!/usr/bin/env python3
"""
Simple AI Project - Drone Sensor Data Regressor
Main application entry point
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

def print_project_info():
    print("="*50)
    print("🚁 Drone Sensor ML Project - Trajectory & Orientation Prediction")
    print("="*50)
    print("Predict drone state (position & orientation) from raw IMU/gyro/magnetometer data.")
    print("Data: imu_data.csv (put in ./data/)")
    print("="*50)

def plot_feature_distributions(df):
    """Visualize sensor distributions and label distributions."""
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    for idx, sensor in enumerate(['accel', 'gyro', 'mag']):
        cols = [f"{sensor}_x", f"{sensor}_y", f"{sensor}_z"]
        df[cols].hist(bins=50, ax=axes[0, idx])
        axes[0, idx].set_title(f"{sensor.capitalize()} Distribution")
    for idx, target in enumerate([['pos_x','pos_y','pos_z'], ['roll','pitch','yaw']]):
        df[target].hist(bins=50, ax=axes[1, idx])
        axes[1, idx].set_title(f"{'Position' if idx==0 else 'Orientation'} Distribution")
    axes[1, 2].axis('off')
    plt.tight_layout()
    plt.savefig("drone_feature_distributions.png")
    plt.show()

def plot_correlation_heatmap(df):
    plt.figure(figsize=(10,7))
    sns.heatmap(df.corr(), annot=False, cmap="coolwarm")
    plt.title("Correlation Heatmap of Drone Features")
    plt.tight_layout()
    plt.savefig("drone_corr_heatmap.png")
    plt.show()

def plot_3d_trajectory(df, title="Drone 3D Trajectory (first 1000 points)"):
    fig = plt.figure(figsize=(10,7))
    ax = fig.add_subplot(111, projection='3d')
    pts = df[['pos_x','pos_y','pos_z']].values[:1000]
    colors = np.linspace(0, 1, len(pts))
    ax.scatter(pts[:,0], pts[:,1], pts[:,2], c=colors, cmap='viridis', s=10)
    ax.plot(pts[:,0], pts[:,1], pts[:,2], color='blue', alpha=0.5)
    ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')
    ax.set_title(title)
    plt.tight_layout()
    plt.savefig("drone_trajectory.png")
    plt.show()

def create_windows(X, y, window_size=50):
    Xs, ys = [], []
    for i in range(len(X) - window_size):
        Xs.append(X[i:(i+window_size)].flatten())
        ys.append(y[i+window_size])
    return np.array(Xs), np.array(ys)

def save_model(model, scaler_X, scaler_y, path_prefix="models/drone"):
    Path("models").mkdir(exist_ok=True)
    joblib.dump(model, f"{path_prefix}_model.pkl")
    joblib.dump(scaler_X, f"{path_prefix}_scaler_X.pkl")
    joblib.dump(scaler_y, f"{path_prefix}_scaler_y.pkl")
    print(f"✅ Saved model and scalers at prefix {path_prefix}")

def load_model(path_prefix="models/drone"):
    model = joblib.load(f"{path_prefix}_model.pkl")
    scaler_X = joblib.load(f"{path_prefix}_scaler_X.pkl")
    scaler_y = joblib.load(f"{path_prefix}_scaler_y.pkl")
    return model, scaler_X, scaler_y

def main():
    print_project_info()
    print("📊 Loading and cleaning data...")
    df = get_cleaned_data()
    print(f"Loaded cleaned dataset with shape: {df.shape}")

    # Visualizations
    print("📊 Visualizing feature distributions...")
    plot_feature_distributions(df)
    plot_correlation_heatmap(df)
    plot_3d_trajectory(df)

    # Features and targets
    sensor_cols = ['accel_x', 'accel_y', 'accel_z',
                   'gyro_x', 'gyro_y', 'gyro_z',
                   'mag_x', 'mag_y', 'mag_z']
    target_cols = ['pos_x', 'pos_y', 'pos_z', 'roll', 'pitch', 'yaw']

    X = df[sensor_cols].values
    y = df[target_cols].values

    # Normalize
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    X = scaler_X.fit_transform(X)
    y = scaler_y.fit_transform(y)

    # Sliding window for sequence learning
    window_size = 50
    X_seq, y_seq = create_windows(X, y, window_size)

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_seq, y_seq, test_size=0.2, random_state=42
    )

    # Model
    print("🤖 Training MLPRegressor...")
    model = MLPRegressor(hidden_layer_sizes=(128, 64),
                         activation='relu',
                         solver='adam',
                         max_iter=50,
                         batch_size=64,
                         random_state=42,
                         verbose=True)
    model.fit(X_train, y_train)

    # Evaluation
    print("📈 Evaluating model...")
    y_pred = model.predict(X_test)
    y_test_inv = scaler_y.inverse_transform(y_test)
    y_pred_inv = scaler_y.inverse_transform(y_pred)
    print("R² per output:", [r2_score(y_test_inv[:, i], y_pred_inv[:, i]) for i in range(y_test_inv.shape[1])])
    print("MAE per output:", [mean_absolute_error(y_test_inv[:, i], y_pred_inv[:, i]) for i in range(y_test_inv.shape[1])])
    print("RMSE per output:", [np.sqrt(mean_squared_error(y_test_inv[:, i], y_pred_inv[:, i])) for i in range(y_test_inv.shape[1])])

    # Save model and scalers
    print("💾 Saving model and scalers...")
    save_model(model, scaler_X, scaler_y)

    # Example predictions
    print("\n🔮 Example predictions (first 5):")
    for i in range(5):
        print(f"True: {np.round(y_test_inv[i], 2)}, Pred: {np.round(y_pred_inv[i], 2)}")
    print("\n✅ Project completed successfully!")
    print(f"📁 Feature plots: drone_feature_distributions.png, drone_corr_heatmap.png, drone_trajectory.png")
    print(f"📦 Models saved in 'models/' directory.")

if __name__ == "__main__":
    main()