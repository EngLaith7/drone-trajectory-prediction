import pandas as pd
import numpy as np
from pathlib import Path

def clean_drone_data(file_path):
    # Ensure it's a Path
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
    
    return df

def get_cleaned_data():
    # Path to raw file
    raw_path = Path("D:/fifth grade/Data project/.venv/drone-trajectory-prediction/data/drone_data.csv")
    return clean_drone_data(raw_path)

# Allow quick test if running directly
if __name__ == "__main__":
    df = get_cleaned_data()
    print(df.head())
