# Full ML pipeline: Predict drone trajectory from IMU sensor readings

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
import joblib
import sys
from pathlib import Path

repo_root = Path(__file__).resolve().parents[2]
if str(repo_root) not in sys.path:
    sys.path.append(str(repo_root))

from src.data.clean_data import get_cleaned_data


def create_sliding_window(X, y, window_size=5):
    Xs, ys = [], []
    for i in range(len(X) - window_size):
        Xs.append(X[i:i+window_size].flatten())
        ys.append(y[i+window_size])
    return np.array(Xs), np.array(ys)


def train_model(window_size=5, use_grid_search=True):
    """
    Train drone trajectory prediction model.

    Parameters
    ----------
    window_size : int, default=5
        Sliding window size for time-series features.
    use_grid_search : bool, default=True
        Whether to run grid search when cleaning data.

    Returns
    -------
    model : RandomForestRegressor
        Trained model.
    scaler_X : StandardScaler
        Scaler for input features.
    scaler_y : StandardScaler
        Scaler for target outputs.
    metrics : dict
        Dictionary with MSE and R2 scores.
    """
    # 1. Load and clean data
    df = get_cleaned_data(use_grid_search=use_grid_search)
    print("✅ Cleaned dataset shape:", df.shape)

    # 2. Select features and targets
    X = df[['accel_x', 'accel_y', 'accel_z',
            'gyro_x', 'gyro_y', 'gyro_z',
            'mag_x', 'mag_y', 'mag_z']].values
    y = df[['pos_x', 'pos_y', 'pos_z', 'roll', 'pitch', 'yaw']].values

    # 3. Sliding window
    X_windowed, y_windowed = create_sliding_window(X, y, window_size=window_size)

    # 4. Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X_windowed, y_windowed, test_size=0.2, random_state=42
    )

    # 5. Normalize features and targets
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    X_train_scaled = scaler_X.fit_transform(X_train)
    X_test_scaled = scaler_X.transform(X_test)
    y_train_scaled = scaler_y.fit_transform(y_train)
    y_test_scaled = scaler_y.transform(y_test)

    # 6. Train model
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train_scaled, y_train_scaled)

    # 7. Predict
    y_pred_scaled = model.predict(X_test_scaled)
    y_pred = scaler_y.inverse_transform(y_pred_scaled)

    # 8. Evaluate
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    metrics = {"MSE": mse, "R2": r2}
    print(f"Test MSE: {mse:.4f}")
    print(f"Test R2: {r2:.4f}")

    # 9. Save model + scalers
    model_dir = repo_root / "src" / "models"
    model_dir.mkdir(parents=True, exist_ok=True)

    joblib.dump(scaler_X, model_dir / "scaler_X.pkl")
    joblib.dump(scaler_y, model_dir / "scaler_y.pkl")
    joblib.dump(model, model_dir / "drone_model.pkl")

    print("✅ Model and scalers saved successfully in src/models/")

    return model, scaler_X, scaler_y, metrics


if __name__ == "__main__":
    train_model()

