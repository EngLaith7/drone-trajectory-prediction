from pathlib import Path
import Visualization as vz

# Load data
#make sure that you copied the imu_data.csv file to the data folder
data_path = Path("data") / "imu_data.csv"
df = vz.load_data(data_path)

# Call visualization functions according to the need
vz.plot_accelerometer_distribution(df)
vz.plot_gyroscope_distribution(df)
vz.plot_magnetometer_distribution(df)
vz.plot_accelerometer_timeseries(df)
vz.plot_gyroscope_timeseries(df)
vz.plot_3d_trajectory(df)
vz.plot_orientation(df)
vz.plot_correlation_heatmap(df)
vz.plot_trajectory_partial(df, 1000)  