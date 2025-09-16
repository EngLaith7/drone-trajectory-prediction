import pandas as pd
import numpy as np
from pathlib import Path

def print_project_info():
    print("="*50)
    print("🚁 Drone Sensor ML Project - Trajectory & Orientation Prediction")
    print("="*50)
    print("Predict drone state (position & orientation) from raw IMU/gyro/magnetometer data.")
    print("Data: imu_data.csv (put in ./data/)")
    print("="*50)

def load_data():
    repo_root = Path(__file__).resolve().parents[2]
    data_dir = repo_root / "data"

    csv_files = list(data_dir.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV file found in {data_dir}")

    file_path = csv_files[0]
    print(f"📂 Using dataset: {file_path.name}")

    df = pd.read_csv(file_path)
    return df