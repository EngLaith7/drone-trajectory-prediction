# visualizations.py
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D  # مطلوب لرسم 3D



# ========================
# 1. Histogram Distribution
# ========================
def plot_accelerometer_distribution(df: pd.DataFrame):
    df[['accel_x','accel_y','accel_z']].hist(bins=50, figsize=(15,5))
    plt.suptitle("Accelerometer Distribution")
    plt.show()

def plot_gyroscope_distribution(df: pd.DataFrame):
    df[['gyro_x','gyro_y','gyro_z']].hist(bins=50, figsize=(15,5))
    plt.suptitle("Gyroscope Distribution")
    plt.show()

def plot_magnetometer_distribution(df: pd.DataFrame):
    df[['mag_x','mag_y','mag_z']].hist(bins=50, figsize=(15,5))
    plt.suptitle("Magnetometer Distribution")
    plt.show()

# ========================
# 2. Time Series Plots
# ========================
def plot_accelerometer_timeseries(df: pd.DataFrame):
    plt.figure(figsize=(15,5))
    plt.plot(df['time'], df['accel_x'], label='accel_x')
    plt.plot(df['time'], df['accel_y'], label='accel_y')
    plt.plot(df['time'], df['accel_z'], label='accel_z')
    plt.title("Accelerometer over Time")
    plt.xlabel("Time")
    plt.ylabel("Acceleration")
    plt.legend()
    plt.show()

def plot_gyroscope_timeseries(df: pd.DataFrame):
    plt.figure(figsize=(15,5))
    plt.plot(df['time'], df['gyro_x'], label='gyro_x')
    plt.plot(df['time'], df['gyro_y'], label='gyro_y')
    plt.plot(df['time'], df['gyro_z'], label='gyro_z')
    plt.title("Gyroscope over Time")
    plt.xlabel("Time")
    plt.ylabel("Angular Velocity")
    plt.legend()
    plt.show()

# ========================
# 3. 3D Trajectory (Position)
# ========================
def plot_3dplot_trajectory(df: pd.DataFrame):
    fig = plt.figure(figsize=(10,7))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(df['pos_x'], df['pos_y'], df['pos_z'], color='blue')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title("Drone Trajectory")
    plt.show()

# ========================
# 4. Orientation (Roll, Pitch, Yaw)
# ========================
def plot_orientation(df: pd.DataFrame):
    plt.figure(figsize=(15,5))
    plt.plot(df['time'], df['roll'], label='Roll')
    plt.plot(df['time'], df['pitch'], label='Pitch')
    plt.plot(df['time'], df['yaw'], label='Yaw')
    plt.title("Orientation over Time")
    plt.xlabel("Time")
    plt.ylabel("Angle (deg)")
    plt.legend()
    plt.show()

# ========================
# 5. Correlation Heatmap
# ========================
def plot_correlation_heatmap_plot(df: pd.DataFrame):
    plt.figure(figsize=(12,8))
    sns.heatmap(df.corr(), annot=False, cmap="coolwarm")
    plt.title("Correlation Heatmap of Features")
    plt.show()

# ========================
# 6. trajectory_partial
# ========================
def plot_trajectory_partial(df: pd.DataFrame, k: int):
    """
    drone (Trajectory) only point of the k draw
    """
    if k > len(df):
        k = len(df)  # If the user use a larger number of data size

    df_part = df.iloc[:k]

    fig = plt.figure(figsize=(10,7))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(df_part['pos_x'], df_part['pos_y'], df_part['pos_z'], color='blue')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(f"Drone Trajectory - First {k} Points")
    plt.show()