"""
Methods for calculating some of the settings.
"""
from pathlib import Path
from matplotlib import pyplot as plt
from tool.measurement.idle_stats import create_combined_df, calc_stats_for_split_data, calc_stdev_mean_ratios
from tool.measurement.measurement_parse import parse_cpu_temperature
from tool.measurement.measurement_config import CHECK_LAST_N_POINTS


IDLE_DATA_DIR = Path("../data/other/settings/idle_data/")

# code used to calculate the standard deviation to mean ratios 
# gathered data by running the server application, nothing else
def stdev_mean_ratios(plot_data=False):
    combined_df = create_combined_df(directory=IDLE_DATA_DIR)
    mean_stats = calc_stats_for_split_data(CHECK_LAST_N_POINTS, combined_df)
    cpu_std_mean, ram_std_mean, gpu_std_mean = calc_stdev_mean_ratios(mean_stats)

    with open("out/idle_data.txt", 'w') as f:
        f.write(str(mean_stats))
        f.write("\n")
        f.writelines([
            "\ncpu_std_mean: " + cpu_std_mean,
            "\nram_std_mean: " + ram_std_mean,
            "\ngpu_std_mean: " + gpu_std_mean
        ])
    if plot_data:
        combined_df.plot()
        plt.show()

# code used to calculate the maximum cpu temperature
# gathered data by running the server application and cpu_temperature.py, nothing else
def cpu_temperature():
    df_cpu_temp = parse_cpu_temperature(IDLE_DATA_DIR/"cpu_temperature.txt")
    with open("out/idle_cpu_temperature_stats.txt", 'w') as f:
        f.write(f"mean CPU temperature: {df_cpu_temp.iloc[:,1].mean()}\n")
        f.write(f"min CPU temperature: {df_cpu_temp.iloc[:,1].min()}\n")
        f.write(f"max CPU temperature: {df_cpu_temp.iloc[:,1].max()}\n")

if __name__ == "__main__":
    ### commented code has already been run, uncomment to replicate

    # stdev_mean_ratios()
    # cpu_temperature()
    pass
