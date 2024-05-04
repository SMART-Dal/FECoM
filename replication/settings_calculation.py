"""
Methods for calculating some of the settings.
"""
from pathlib import Path
from matplotlib import pyplot as plt
from fecom.measurement.idle_stats import create_combined_df, calc_stats_for_split_data, calc_stdev_mean_ratios
from fecom.measurement.measurement_parse import parse_cpu_temperature
from fecom.measurement.measurement_config import CHECK_LAST_N_POINTS, MEASUREMENT_INTERVAL_MS


# IDLE_DATA_DIR = Path("../data/other/settings/idle_data/")
IDLE_DATA_DIR = Path(f'settings/setting-{MEASUREMENT_INTERVAL_MS}ms/out/')
OUTPUT_DIR = Path(f'settings/setting-{MEASUREMENT_INTERVAL_MS}ms')

# code used to calculate the standard deviation to mean ratios 
# gathered data by running start_measurement.py, nothing else
def stdev_mean_ratios(plot_data=True):
    combined_df = create_combined_df(directory=IDLE_DATA_DIR)
    mean_stats = calc_stats_for_split_data(CHECK_LAST_N_POINTS, combined_df)
    cpu_std_mean, ram_std_mean, gpu_std_mean = calc_stdev_mean_ratios(mean_stats)

    with open(OUTPUT_DIR / "idle_data.txt", 'w') as f:
        f.write(str(mean_stats))
        f.write("\n")
        f.writelines([
            "\ncpu_std_mean: " + cpu_std_mean,
            "\nram_std_mean: " + ram_std_mean,
            "\ngpu_std_mean: " + gpu_std_mean
        ])
    if plot_data:
        combined_df.plot()
        plt.savefig(OUTPUT_DIR / "idle_data.png")
        plt.show()


def plot_coefficients_of_var():
    intervals = [1, 10, 50, 100, 200, 300, 400, 500, 1000]
    cpu_data = []
    ram_data = []
    gpu_data = []

    for interval in intervals:
        IDLE_DATA_DIR = Path(f'settings/setting-{interval}ms/out/')
        OUTPUT_DIR = Path(f'settings/setting-{interval}ms')
        combined_df = create_combined_df(directory=IDLE_DATA_DIR)
        mean_stats = calc_stats_for_split_data(CHECK_LAST_N_POINTS, combined_df)
        cpu_std_mean, ram_std_mean, gpu_std_mean = calc_stdev_mean_ratios(mean_stats)
        # change datatype of each to numerical
        cpu_std_mean = float(cpu_std_mean)
        ram_std_mean = float(ram_std_mean)
        gpu_std_mean = float(gpu_std_mean)
        cpu_data.append(cpu_std_mean)
        ram_data.append(ram_std_mean)
        gpu_data.append(gpu_std_mean)

    print("cpu:",cpu_data)
    print("ram:",ram_data)
    print("gpu:",gpu_data)
    fig, ax = plt.subplots(figsize=(8, 6))

    # Plot the data points in the original order
    ax.plot(intervals, cpu_data, marker='o', label="CPU")
    ax.plot(intervals, ram_data, marker='o', label="RAM")
    ax.plot(intervals, gpu_data, marker='o', label="GPU")
    ax.legend(fontsize=12)

    ax.set_xlabel('Sampling Interval (ms)', fontsize=16)
    ax.set_ylabel('Coefficient of Variation', fontsize=16)
    # ax.set_title('Coefficients of Variation Across Sampling Intervals', fontsize=16)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    ax.legend()

    # Set the y-axis to use a logarithmic scale
    ax.set_yscale('log')

    # Set the y-axis range to cover the full range of data
    # ax.set_ylim(min(min(cpu_data), min(ram_data), min(gpu_data)),
    #             max(max(cpu_data), max(ram_data), max(gpu_data)))

    plt.savefig(Path("settings") / "coefficients_of_variation.png", bbox_inches='tight')

    plt.show()

# code used to calculate the maximum cpu temperature
# gathered data by running start_measurement.py and cpu_temperature.py, nothing else
def cpu_temperature():
    df_cpu_temp = parse_cpu_temperature(IDLE_DATA_DIR/"cpu_temperature.txt")
    with open(OUTPUT_DIR / "idle_cpu_temperature_stats.txt", 'w') as f:
        f.write(f"mean CPU temperature: {df_cpu_temp.iloc[:,1].mean()}\n")
        f.write(f"min CPU temperature: {df_cpu_temp.iloc[:,1].min()}\n")
        f.write(f"max CPU temperature: {df_cpu_temp.iloc[:,1].max()}\n")

if __name__ == "__main__":
    ### commented code has already been run, uncomment to replicate

    # stdev_mean_ratios()
    # cpu_temperature()
    plot_coefficients_of_var()
    pass
