import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import ParameterGrid


def clean_drone_data(file_path: Path, sigma_threshold: float = 3.0, dropna: bool = True):
    """
    Cleans drone dataset with hyperparameterized filtering.

    Parameters
    ----------
    file_path : Path
        Path to the CSV file.
    sigma_threshold : float, default=3.0
        Number of standard deviations for outlier removal (z-score rule).
    dropna : bool, default=True
        Whether to drop rows with missing values.
    """
    # Load dataset
    df = pd.read_csv(file_path)

    # Replace junk values with NaN
    df = df.replace([' ', '', 'NaN', 'null', '----'], np.nan)

    # Convert all to numeric if possible
    df = df.apply(pd.to_numeric, errors="coerce")

    # Drop rows with missing values (configurable)
    if dropna:
        df = df.dropna()

    # Remove duplicates
    df = df.drop_duplicates()

    # If "time" column exists, sort + interpolate gaps
    if "time" in df.columns:
        df = df.sort_values("time")
        df = df.set_index("time").interpolate(method="linear").reset_index()

    sensor_cols = [
        "accel_x", "accel_y", "accel_z",
        "gyro_x", "gyro_y", "gyro_z",
        "mag_x", "mag_y", "mag_z"
    ]

    # Apply σ-rule (hyperparameterized)
    for col in sensor_cols:
        if col in df.columns:  # avoid KeyError
            mean, std = df[col].mean(), df[col].std()
            df = df[df[col].between(mean - sigma_threshold * std,
                                    mean + sigma_threshold * std)]

    return df


def _grid_search_cleaning(file_path: Path,
                          sigma_values=[2.0, 2.5, 3.0, 3.5, 4.0],
                          dropna_values=[True]):
    """
    Internal function: Performs a grid search over sigma_threshold and dropna.
    Uses a scoring function to pick the best hyperparameters.
    """
    results = []
    original_df = pd.read_csv(file_path)

    param_grid = ParameterGrid({
        "sigma_threshold": sigma_values,
        "dropna": dropna_values
    })

    for params in param_grid:
        df_cleaned = clean_drone_data(file_path,
                                      sigma_threshold=params["sigma_threshold"],
                                      dropna=params["dropna"])

        # Compute row ratio
        row_ratio = len(df_cleaned) / len(original_df)

        # Compute variance ratio (average across numeric columns)
        numeric_cols = df_cleaned.select_dtypes(include=[np.number]).columns
        var_original = original_df[numeric_cols].var().mean()
        var_cleaned = df_cleaned[numeric_cols].var().mean()
        var_ratio = var_cleaned / var_original if var_original != 0 else 0

        # Final score = balance between row_ratio & var_ratio
        score = 0.5 * row_ratio + 0.5 * var_ratio

        results.append({
            "sigma_threshold": params["sigma_threshold"],
            "dropna": params["dropna"],
            "rows_remaining": len(df_cleaned),
            "row_ratio": row_ratio,
            "var_ratio": var_ratio,
            "score": score
        })

    results_df = pd.DataFrame(results)
    best_params = results_df.loc[results_df["score"].idxmax()]

    print("✅ Best parameters found:")
    print(best_params)

    return best_params


def get_cleaned_data(sigma_threshold: float = 3.0,
                     dropna: bool = True,
                     use_grid_search: bool = False):
    """
    Finds the first CSV inside /data, cleans it, and returns the DataFrame.
    Allows tuning of hyperparameters for cleaning.

    Parameters
    ----------
    sigma_threshold : float, default=3.0
        Standard deviation rule threshold.
    dropna : bool, default=True
        Whether to drop NaN rows.
    use_grid_search : bool, default=False
        If True, automatically finds the best sigma/dropna using grid search.
    """
    repo_root = Path(__file__).resolve().parents[2]
    data_dir = repo_root / "data"

    csv_files = list(data_dir.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV file found in {data_dir}")

    file_path = csv_files[0]
    print(f"📂 Using dataset: {file_path.name}")

    # Optionally run grid search
    if use_grid_search:
        best_params = _grid_search_cleaning(file_path)
        sigma_threshold = best_params["sigma_threshold"]
        dropna = best_params["dropna"]

    return clean_drone_data(file_path,
                            sigma_threshold=sigma_threshold,
                            dropna=dropna)
