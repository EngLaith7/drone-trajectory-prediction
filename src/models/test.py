# train_using_clean.py
import sys
from pathlib import Path
import time
import joblib
import numpy as np
import logging

# sklearn imports
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

# ----- make repo root importable (so we can import src.data.clean_data) -----
repo_root = Path(__file__).resolve().parents[0]  # adjust if you run from different folder
# if your repo layout is project_root/scripts/train_using_clean.py then use .parents[1] etc.
# safe add parents until we find src/
if str(repo_root) not in sys.path:
    sys.path.append(str(repo_root))

# try to import get_cleaned_data from your colleague's module
try:
    from src.data.clean_data import get_cleaned_data
except Exception as e:
    # fallback: try parent level
    repo_root = Path(__file__).resolve().parents[2]
    if str(repo_root) not in sys.path:
        sys.path.append(str(repo_root))
    try:
        from src.data.clean_data import get_cleaned_data
    except Exception as ee:
        raise ImportError("Could not import get_cleaned_data from src.data.clean_data. "
                          "Ensure file exists and package path is correct.") from ee

# logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("train_using_clean")


WINDOW_SIZE = 5       
TEST_SIZE = 0.2        # نسبة الاختبار
RANDOM_STATE = 42
N_ESTIMATORS = 100
MAX_DEPTH = None       # أو 15 مثلاً
MODEL_DIR = Path("models")   # سيتم الحفظ هنا
MODEL_DIR.mkdir(parents=True, exist_ok=True)
# ----------------------------------------

def create_sliding_window(X, y, window_size=5):
    Xs, ys = [], []
    n = len(X)
    if n <= window_size:
        raise ValueError("Data length must be greater than window_size.")
    for i in range(n - window_size):
        Xs.append(X[i:i+window_size].flatten())
        ys.append(y[i+window_size])
    return np.array(Xs), np.array(ys)


def main(use_grid_search=False, sigma_threshold=3.0, dropna=True):
    """
    use_grid_search: إذا أردت أن يدير زميلك الـ grid search قبل التنظيف
    sigma_threshold, dropna: تُمرّر إلى get_cleaned_data إذا أردت
    """
    logger.info("Calling get_cleaned_data() from teammate's code...")
    try:
        df = get_cleaned_data(sigma_threshold=sigma_threshold,
                              dropna=dropna,
                              use_grid_search=use_grid_search)
    except FileNotFoundError as e:
        logger.error(f"No CSV found or cleaning failed: {e}")
        return
    except Exception as e:
        logger.error(f"Error while cleaning data via teammate function: {e}")
        raise

    logger.info(f"✅ Cleaned dataset shape: {df.shape}")

    # ensure expected columns exist
    X_cols = ['accel_x', 'accel_y', 'accel_z',
              'gyro_x', 'gyro_y', 'gyro_z',
              'mag_x', 'mag_y', 'mag_z']
    y_cols = ['pos_x', 'pos_y', 'pos_z', 'roll', 'pitch', 'yaw']

    missing = [c for c in X_cols + y_cols if c not in df.columns]
    if missing:
        logger.error(f"Missing expected columns in cleaned df: {missing}")
        raise ValueError(f"Missing columns: {missing}")

    # select arrays
    X = df[X_cols].values.astype(np.float32)
    y = df[y_cols].values.astype(np.float32)

    # create windowed features
    X_windowed, y_windowed = create_sliding_window(X, y, window_size=WINDOW_SIZE)
    logger.info(f"Windowed shapes: X={X_windowed.shape}, y={y_windowed.shape}")

    # split (time-aware default: shuffle=False)
    X_train, X_test, y_train, y_test = train_test_split(
        X_windowed, y_windowed, test_size=TEST_SIZE, random_state=RANDOM_STATE, shuffle=False)

    # scale (fit on train)
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    X_train_scaled = scaler_X.fit_transform(X_train)
    X_test_scaled = scaler_X.transform(X_test)
    y_train_scaled = scaler_y.fit_transform(y_train)
    # note: keep y_test in original units for final evaluation, but scale for training
    y_test_orig = y_test.copy()
    y_test_scaled = scaler_y.transform(y_test)

    # train RandomForest (multi-output supported natively)
    model = RandomForestRegressor(n_estimators=N_ESTIMATORS, max_depth=MAX_DEPTH,
                                  random_state=RANDOM_STATE, n_jobs=-1)
    logger.info("Starting RandomForest training...")
    t0 = time.time()
    model.fit(X_train_scaled, y_train_scaled)
    train_time = time.time() - t0
    logger.info(f"Training finished in {train_time:.2f} seconds")

    # predict and inverse-scale
    y_pred_scaled = model.predict(X_test_scaled)
    y_pred = scaler_y.inverse_transform(y_pred_scaled)

    # evaluate per-target
    logger.info("Evaluation per target:")
    for i, col in enumerate(y_cols):
        mae = mean_absolute_error(y_test_orig[:, i], y_pred[:, i])
        rmse = np.sqrt(mean_squared_error(y_test_orig[:, i], y_pred[:, i]))
        r2 = r2_score(y_test_orig[:, i], y_pred[:, i])
        logger.info(f" {col:6s} | MAE={mae:.4f} | RMSE={rmse:.4f} | R2={r2:.4f}")

    overall_mae = np.mean([mean_absolute_error(y_test_orig[:, i], y_pred[:, i]) for i in range(len(y_cols))])
    overall_rmse = np.mean([np.sqrt(mean_squared_error(y_test_orig[:, i], y_pred[:, i])) for i in range(len(y_cols))])
    overall_r2 = np.mean([r2_score(y_test_orig[:, i], y_pred[:, i]) for i in range(len(y_cols))])

    print("\n=== Summary ===")
    print(f"Train time (s): {train_time:.2f}")
    print(f"Overall MAE: {overall_mae:.6f}")
    print(f"Overall RMSE: {overall_rmse:.6f}")
    print(f"Overall R2: {overall_r2:.6f}")

    # save model & scalers & metadata
    joblib.dump(model, MODEL_DIR / "drone_rf_model.joblib")
    joblib.dump(scaler_X, MODEL_DIR / "scaler_X.joblib")
    joblib.dump(scaler_y, MODEL_DIR / "scaler_y.joblib")
    joblib.dump({'X_cols': X_cols, 'y_cols': y_cols, 'window_size': WINDOW_SIZE}, MODEL_DIR / "metadata.joblib")
    logger.info(f"Saved model & scalers to {MODEL_DIR}")

if __name__ == "__main__":
    # examples:
    # main(use_grid_search=False)
    # or to enable teammate grid-search: main(use_grid_search=True)
    main(use_grid_search=False)
