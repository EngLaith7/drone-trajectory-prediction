from pathlib import Path
import Visualization_clean_data as vzc
from clean_data import get_cleaned_data
# Load data
#make sure that you copied the imu_data.csv file to the data folder

df = get_cleaned_data(use_grid_search=True)

# Call visualization functions according to the need
vzc.plot_accelerometer_distribution(df)
vzc.plot_gyroscope_distribution(df)
vzc.plot_magnetometer_distribution(df)
vzc.plot_accelerometer_timeseries(df)
vzc.plot_gyroscope_timeseries(df)
vzc.plot_3d_trajectory(df)
vzc.plot_orientation(df)
vzc.plot_correlation_heatmap(df)
vzc.plot_trajectory_partial(df, 1000)  