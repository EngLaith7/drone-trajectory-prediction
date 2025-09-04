import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from pathlib import Path

def clean_drone_data(file_path):
    # Make sure it's a Path object
    file_path = Path(file_path)
    
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
    
    # Select features to normalize
    features = ["accel_x","accel_y","accel_z",
                "gyro_x","gyro_y","gyro_z",
                "mag_x","mag_y","mag_z"]
    
    # Normalize features
    scaler = StandardScaler()
    df[features] = scaler.fit_transform(df[features])
    
    return df, scaler
