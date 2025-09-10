import pandas as pd
import numpy as np
from pathlib import Path

def clean_drone_data(file_path: Path, drop_constant_cols: bool = True) -> pd.DataFrame:
    """
    Clean drone dataset:
    1. Replace junk values
    2. Drop NaN + duplicates
    3. Remove columns with >50% NaN
    4. Strip whitespace & lowercase columns
    5. Sort by time & interpolate gaps
    6. Remove outliers (>3σ) for sensor columns
    7. Optionally drop constant columns
    """
    # Load dataset
    df = pd.read_csv(file_path)

    # Clean column names
    df.columns = df.columns.str.strip().str.lower()

    # Strip whitespace from string/object columns
    for col in df.select_dtypes(include=['object']).columns:
        df[col] = df[col].str.strip()

    # Replace junk placeholders with NaN
    df = df.replace([' ', '', 'NaN', 'nan', 'null', '----'], np.nan)

    # Remove columns with >50% missing values
    thresh = len(df) * 0.5
    df = df.dropna(axis=1, thresh=thresh)

    # Convert all to numeric where possible
    df = df.apply(pd.to_numeric, errors="coerce")

    # Drop rows with missing values
    df = df.dropna()

    # Remove duplicates
    df = df.drop_duplicates()

    # Sort + interpolate if time column exists
    if "time" in df.columns:
        df = df.sort_values("time")
        df = df.set_index("time").interpolate(method="linear").reset_index()

    # Define sensor columns to filter
    sensor_cols = [
        "accel_x", "accel_y", "accel_z",
        "gyro_x", "gyro_y", "gyro_z",
        "mag_x", "mag_y", "mag_z"
    ]
    # Apply 3σ rule for each sensor column
    for col in sensor_cols:
        if col in df.columns:  # avoid KeyError
            mean, std = df[col].mean(), df[col].std()
            df = df[df[col].between(mean - 3*std, mean + 3*std)]

    # Optionally drop constant columns
    if drop_constant_cols:
        nunique = df.nunique()
        const_cols = nunique[nunique == 1].index
        df = df.drop(columns=const_cols)

    df = df.reset_index(drop=True)
    return df


def get_cleaned_data():
    """
    Finds the first CSV inside /data, cleans it, and returns the DataFrame.
    """
    repo_root = Path().resolve().parent  # adjust if needed
    data_dir = repo_root / "data"

    csv_files = list(data_dir.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV file found in {data_dir}")

    print(f"📂 Using dataset: {csv_files[0].name}")
    return clean_drone_data(csv_files[0])



df = get_cleaned_data()
print("✅ Cleaned dataset shape:", df.shape)
print(df.head())
