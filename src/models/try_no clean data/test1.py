import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error
import joblib
import matplotlib.pyplot as plt

# Load model and scalers
model = joblib.load('saved_models_sklearn/best_model.joblib')
scaler_X = joblib.load('saved_models_sklearn/scaler_X.joblib')
scaler_y = joblib.load('saved_models_sklearn/scaler_y.joblib')

# Load test data
df_test = pd.read_csv(r"E:\drone-trajectory-prediction\drone-trajectory-prediction\data\imu_data.csv")
X_test = df_test.iloc[:, 1:10].values  # sensor measurements
y_test = df_test.iloc[:, 10:16].values  # position and orientation

# Scale test data
X_test_scaled = scaler_X.transform(X_test)
y_test_scaled = scaler_y.transform(y_test)

# Create time sequences
def create_sequences(X, window_size):
    X_seq = []
    for i in range(len(X) - window_size):
        X_seq.append(X[i:i+window_size].flatten())
    return np.array(X_seq)

WINDOW_SIZE = 30
X_test_seq = create_sequences(X_test_scaled, WINDOW_SIZE)
y_test_seq = y_test_scaled[WINDOW_SIZE:]

# Make predictions
y_pred_scaled = model.predict(X_test_seq)
y_pred = scaler_y.inverse_transform(y_pred_scaled)
y_true = scaler_y.inverse_transform(y_test_seq)

# Calculate metrics
output_columns = ['pos_x', 'pos_y', 'pos_z', 'roll', 'pitch', 'yaw']
metrics_df = pd.DataFrame(columns=['MAE', 'RMSE'], index=output_columns)

for i, col in enumerate(output_columns):
    mae = mean_absolute_error(y_true[:, i], y_pred[:, i])
    rmse = np.sqrt(mean_squared_error(y_true[:, i], y_pred[:, i]))
    metrics_df.loc[col] = [mae, rmse]

print("Test performance metrics:")
print(metrics_df)

# Save results
results_df = pd.DataFrame()
for i, col in enumerate(output_columns):
    results_df[f'true_{col}'] = y_true[:, i]
    results_df[f'pred_{col}'] = y_pred[:, i]

results_df.to_csv('saved_models_sklearn/test_predictions.csv', index=False)
print("Test predictions saved to saved_models_sklearn/test_predictions.csv")