from pathlib import Path
import Visualization as vz
from clean_data import get_cleaned_data
df = get_cleaned_data(use_grid_search=True)

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