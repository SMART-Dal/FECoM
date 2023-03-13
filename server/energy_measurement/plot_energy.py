import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from pathlib import Path

import sys
sys.path.insert(0,'..')
from measurement_parse import parse_nvidia_smi, parse_perf

from config import CPU_TEMPERATURE_FILE

def normalised(lst, normalise=True):
    """
    normalise a list of data using max-min normalisation
    """
    if not normalise:
        return lst

    min_x = min(lst)
    max_x = max(lst)

    return [(x - min_x) / (max_x - min_x) for x in lst]

def plot_cpu_and_ram(filename, n=100, normalise=True):
    """
    plot the last n data points without using parse_perf/parse_nvidia_smi
    """

    data_lines = []
    with open(filename, 'r') as f:
        data_lines = f.read().splitlines(True)

    cpu_energies = normalised([float(line.strip(' ').split(';')[1]) for line in data_lines[2::2][-n:]], normalise=normalise)
    ram_energies = normalised([float(line.strip(' ').split(';')[1]) for line in data_lines[3::2][-n:]], normalise=normalise)
    ram_plus_cpu = normalised([sum(x) for x in zip(cpu_energies, ram_energies)], normalise=normalise)

    time = [i/2 for i in range(len(cpu_energies))]

    fig, [[ax1, ax2],[ax3, ax4]] = plt.subplots(nrows=2, ncols=2)

    ax1.set_title("CPU Energy over time")
    ax1.plot(time, cpu_energies, 'o')

    ax2.set_title("RAM Energy over time")
    ax2.plot(time, ram_energies, 'o')

    ax3.set_title("RAM Energy vs CPU Energy")
    ax3.plot(cpu_energies, ram_energies, 'o')

    ax4.set_title("RAM+CPU Energy over time")
    ax4.plot(time, ram_plus_cpu, 'o')

def combined_plot(cpu_energy=None, ram_energy=None, gpu_power=None, directory=None):
    """
    Requires either
    - a path to a directory containing nvidia_smi.txt and perf.txt or
    - the 3 dataframes as returned by parse_perf and parse_nvidia_smi
    And concatenates these three dataframes into one containing only energy consumption of each hardware component
    as well as the sum of these three values over time. It does not attempt to merge the perf and nvidia-smi data
    in a way that synchronises the measurements in same rows to be at the same time.
    """
    if directory is not None:
        gpu_power = parse_nvidia_smi(f"{directory}nvidia_smi.txt")
        cpu_energy, ram_energy = parse_perf(f"{directory}perf.txt")
    min_len = min([len(gpu_power), len(cpu_energy), len(ram_energy)]) - 1
    print(min_len)
    df = pd.concat([gpu_power.iloc[:min_len]['power_draw (W)'], cpu_energy.iloc[:min_len]['energy (J)'], ram_energy.iloc[:min_len]['energy (J)']], axis=1)
    df.columns = ['gpu_power', 'cpu_energy', 'ram_energy']
  
    # # apply normalization techniques
    # column = 'Column 1'
    # for column in df:
    #     df[column] = MinMaxScaler().fit_transform(np.array(df[column]).reshape(-1,1))

    df['sum'] = df.sum(axis=1)

    print("Combined plot:")
    print(df)
    print("Statistics (stdv, mean):")
    print(df.std())
    print(df.mean())
    return df

def plot_energy_from_dfs(cpu_df, ram_df, gpu_df, perf_times: list, nvidia_times: list):
    """
    Given the dataframes returned by parse_perf and parse_nvidia_smi, and the start and end times for each file,
    create a plot with 3 graphs showing the energy consumption over time for CPU, RAM and GPU with start/end
    times indicated by lines.

    perf_times and nvidia_times are lists of tuples in the format (time, label, color), where
    - time (float, seconds) is the time relative to the start of perf/nvidia (exact times can be found in START_TIMES_FILE)
    - label (str) is a description of this time
    - color (str) is a matplotlib color, e.g. 'r','b','g'
    """
    fig, [ax1, ax2, ax3] = plt.subplots(nrows=1, ncols=3)

    ax1.set_title("CPU Energy over time")
    ax1.plot(cpu_df["time_elapsed"], cpu_df["energy (J)"])
    for time, label, color in perf_times:
        ax1.axvline(x=time, color=color,linewidth=1)
        ax1.text(time, 0, label, rotation=90)

    ax2.set_title("RAM Energy over time")
    ax2.plot(ram_df["time_elapsed"], ram_df["energy (J)"])
    for time, label, color in perf_times:
        ax2.axvline(x=time, color=color,linewidth=1)
        ax2.text(time, 0, label, rotation=90)

    ax3.set_title("GPU Power over time")
    ax3.plot(gpu_df["time_elapsed"], gpu_df["power_draw (W)"])
    for time, label, color in nvidia_times:
        ax3.axvline(x=time, color=color,linewidth=1)
        ax3.text(time, 0, label, rotation=90)

def split_df_into_n(df: pd.DataFrame, n) -> list:
    """
    Split the given dataframe into a list of new dataframes, each with n rows.
    E.g. a dataframe df with 150 rows will be split into
    [df[0:50], df[50:100], df[100:150]]
    """
    dfs = []
    prev_i = 0
    for i in range(n, len(df.index), n):
        dfs.append(df.iloc[prev_i:i])
    return dfs

def calc_stats_for_split_data(combined_df: pd.DataFrame, n=20):
    dfs = split_df_into_n(combined_df, n)

    columns=["cpu_energy_stdv", "cpu_energy_mean", "ram_energy_stdv", "ram_energy_mean", "gpu_power_stdv", "gpu_power_mean"]
    stats = []
    for i, df in enumerate(dfs):
        # TODO: continue here by changing labels of the statistics series and appending them to a list to create a df out of
        df_mean = df.mean()
        df_stdv = df.std()

        current_stats = pd.DataFrame(
            [[
                df_stdv["cpu_energy"],
                df_mean["cpu_energy"],
                df_stdv["ram_energy"],
                df_mean["ram_energy"],
                df_stdv["gpu_power"],
                df_mean["gpu_power"]
            ]],
            index=[i],
            columns=columns)

        stats.append(current_stats)
    
    total = pd.concat(stats)
    
    return total.mean()

def load_and_plot_temperature(temperature_file=Path("../")/CPU_TEMPERATURE_FILE):
    temperature_df = pd.read_csv(temperature_file, sep=';', names=["time_elapsed","temperature","timestamp"], dtype={
        "time_elapsed": float,
        "temperature": int,
        "timestamp": float
    })

    print(temperature_df["temperature"].mean())
    print(temperature_df["temperature"].std())
    temperature_df["temperature"].plot()
    plt.show()



# def plot_energy(time, energy, start_time, end_time, title=None):
#     fig, ax = plt.subplots()
#     if title is not None:
#         ax.set_title(title)
#     ax.plot(time, energy)
#     ax.axvline(x=start_time, color='r',linewidth=1)
#     ax.axvline(x=end_time, color='r',linewidth=1)



if __name__ == "__main__":
    directory = "./out/2022-12-10/"
    # plot_cpu_and_ram(f"{directory}perf.txt", n=10000, normalise=True)
    # parse_nvidia_smi(f"{directory}nvidia_smi.txt").plot(y='power_draw (W)')
    # print(parse_nvidia_smi(f"{directory}nvidia_smi.txt"))
    # print(parse_perf(f"{directory}perf.txt"))
    # combined_plot(directory=directory)#.plot()
    # print(calc_stats_for_split_data(combined_plot(directory=directory)))
    # plt.show()
    load_and_plot_temperature()