import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import os

# ---------- Project root ----------
base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

# ---------- Dataset ----------
dataset_path = os.path.join(base_dir, "data", "imu_data.csv")
df = pd.read_csv(dataset_path)
df_sample = df.iloc[:10000]

# ---------- 3D Plot ----------
fig = plt.figure(figsize=(10,7))
ax = fig.add_subplot(111, projection='3d')

points = df_sample[['pos_x','pos_y','pos_z']].values
colors = np.linspace(0, 1, len(points))  # gradient along time

ax.scatter(points[:,0], points[:,1], points[:,2], c=colors, cmap='viridis', s=50)
ax.plot(points[:,0], points[:,1], points[:,2], color='blue', alpha=0.5)

ax.set_xlabel('X Position')
ax.set_ylabel('Y Position')
ax.set_zlabel('Z Position')
ax.set_title('Drone Trajectory - First 100 Points')
plt.show()


