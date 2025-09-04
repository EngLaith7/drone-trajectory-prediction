import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import StandardScaler

def clean_drone_data(file_path: Path):
    # Load dataset
    df = pd.read_csv(file_path)

    # Replace junk values with NaN
    df = df.replace([' ', '', 'NaN', 'null', '----'], np.nan)

    # Convert all to numeric if possible
    df = df.apply(pd.to_numeric, errors="coerce")

    # Drop rows with missing values
    df = df.dropna()

    # Remove duplicates
    df = df.drop_duplicates()

    # If "time" column exists, sort + interpolate gaps
    if "time" in df.columns:
        df = df.sort_values("time")
        df = df.set_index("time").interpolate(method="linear").reset_index()

    # Filter unrealistic sensor ranges
    conditions = (
        df["accel_x"].between(-20, 20) &
        df["accel_y"].between(-20, 20) &
        df["accel_z"].between(-20, 20) &
        df["gyro_x"].between(-2000, 2000) &
        df["gyro_y"].between(-2000, 2000) &
        df["gyro_z"].between(-2000, 2000) &
        df["mag_x"].between(-100, 100) &
        df["mag_y"].between(-100, 100) &
        df["mag_z"].between(-100, 100)
    )
    df = df[conditions]

    return df

def get_cleaned_data():
    """
    Finds the first CSV inside /data, cleans it, and returns the DataFrame.
    """
    repo_root = Path(__file__).resolve().parents[2]
    data_dir = repo_root / "data"

    csv_files = list(data_dir.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV file found in {data_dir}")

    print(f"📂 Using dataset: {csv_files[0].name}")
    return clean_drone_data(csv_files[0])
