import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_absolute_error
import joblib
import time
from pathlib import Path

class LocalDronePredictor:
    def __init__(self, window_size=30):
        self.window_size = window_size
        self.model = None
        self.scaler_X = StandardScaler()
        self.scaler_y = StandardScaler()
    
    def load_data(self, data_path):
        """Load data from a local path"""
        data_path = Path(data_path)
        print(f"Loading data from: {data_path}")
        
        if not data_path.exists():
            raise FileNotFoundError(f"File not found: {data_path}")
        
        df = pd.read_csv(data_path)
        print(f"Successfully loaded {len(df)} records")
        
        # Assuming the CSV structure is similar to the Kaggle dataset
        # Features: first 9 columns after index (if any)
        # Targets: next 6 columns
        return df.iloc[:, 1:10].values, df.iloc[:, 10:16].values
    
    def create_sequences(self, X, y):
        """Create time sequences"""
        X_seq, y_seq = [], []
        for i in range(len(X) - self.window_size):
            X_seq.append(X[i:i+self.window_size].flatten())
            y_seq.append(y[i+self.window_size])
        return np.array(X_seq), np.array(y_seq)
    
    def train(self, data_path, test_size=0.2):
        """Train the model with local data"""
        # Load data
        X, y = self.load_data(data_path)
        
        # Scale data
        X_scaled = self.scaler_X.fit_transform(X)
        y_scaled = self.scaler_y.fit_transform(y)
        
        # Create sequences
        X_seq, y_seq = self.create_sequences(X_scaled, y_scaled)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_seq, y_seq, test_size=test_size, random_state=42, shuffle=False
        )
        
        print(f"Training data dimensions: {X_train.shape}")
        
        # Build and train model
        print("Starting training...")
        start_time = time.time()
        
        self.model = MultiOutputRegressor(RandomForestRegressor(
            n_estimators=50,
            max_depth=10,
            random_state=42,
            n_jobs=-1
        ))
        
        self.model.fit(X_train, y_train)
        training_time = time.time() - start_time
        
        print(f"Training completed in {training_time/60:.2f} minutes")
        
        # Evaluate model
        score = self.evaluate_model(X_test, y_test)
        return training_time, score
    
    def evaluate_model(self, X_test, y_test):
        """Evaluate model performance"""
        y_pred = self.model.predict(X_test)
        
        # Reverse scaling
        y_test_orig = self.scaler_y.inverse_transform(y_test)
        y_pred_orig = self.scaler_y.inverse_transform(y_pred)
        
        # Calculate performance metrics
        output_columns = ['pos_x', 'pos_y', 'pos_z', 'roll', 'pitch', 'yaw']
        mae_scores = []
        
        print("\nModel performance:")
        for i, col in enumerate(output_columns):
            mae = mean_absolute_error(y_test_orig[:, i], y_pred_orig[:, i])
            mae_scores.append(mae)
            print(f"{col}: MAE = {mae:.6f}")
        
        return np.mean(mae_scores)
    
    def save_model(self, save_dir):
        """Save model to local directory"""
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        
        joblib.dump(self.model, save_path / "model.joblib")
        joblib.dump(self.scaler_X, save_path / "scaler_X.joblib")
        joblib.dump(self.scaler_y, save_path / "scaler_y.joblib")
        print(f"Model saved at: {save_path}")

if __name__ == "__main__":
    # Initialize predictor
    predictor = LocalDronePredictor(window_size=20)
    
    # Train model with local data (replace with your CSV file path)
    data_path = "path/to/your/imu_data.csv"  # Update this path
    training_time, avg_mae = predictor.train(data_path)
    
    print(f"\nTraining Summary:")
    print(f"Training time: {training_time/60:.2f} minutes")
    print(f"Average MAE: {avg_mae:.6f}")
    
    # Save model
    predictor.save_model("drone_model")