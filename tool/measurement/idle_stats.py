from pathlib import Path

import pandas as pd

from tool.measurement.measurement_parse import parse_nvidia_smi, parse_perf
from tool.measurement.measurement_config import CHECK_LAST_N_POINTS

def split_df_into_n(df: pd.DataFrame, n: int) -> list:
    """
    Split the given dataframe into a list of new dataframes, each with n rows.
    E.g. a dataframe df with 150 rows and n=50 will be split into
    [df[0:50], df[50:100], df[100:150]]
    """
    dfs = []
    prev_i = 0
    for i in range(n, len(df.index), n):
        dfs.append(df.iloc[prev_i:i])
    return dfs


def create_combined_df(cpu_energy: pd.DataFrame = None, ram_energy: pd.DataFrame = None, gpu_power: pd.DataFrame = None, directory: Path = None):
    """
    Requires EITHER
        - a path to a directory containing nvidia_smi.txt and perf.txt OR
        - the 3 dataframes as returned by parse_perf and parse_nvidia_smi

    And concatenates these three dataframes into one containing only energy consumption of each hardware component
    as well as the sum of these three values over time. It does not attempt to merge the perf and nvidia-smi data
    in a way that synchronises the measurements in same rows to be at the same time.
    """
    if directory is not None:
        gpu_power = parse_nvidia_smi(directory/"nvidia_smi.txt")
        cpu_energy, ram_energy = parse_perf(directory/"perf.txt")
    min_len = min([len(gpu_power), len(cpu_energy), len(ram_energy)]) - 1
    df = pd.concat([gpu_power.iloc[:min_len]['power_draw (W)'], cpu_energy.iloc[:min_len]['energy (J)'], ram_energy.iloc[:min_len]['energy (J)']], axis=1)
    df.columns = ['gpu_power', 'cpu_energy', 'ram_energy']
  
    # df['sum'] = df.sum(axis=1)

    return df


def calc_stats_for_split_data(n: int, combined_df: pd.DataFrame):
    """ 
    The given combined energy data is split into dataframes containing n consecutive values.
    For each dataframe, the mean & standard deviation (stdev) is calculated.
    This is supposed to emulate the behaviour of the stable checking mechanism,
    which considers the last n values only, which can increase standard deviation due to
    the smaller sample size.
    
    Returns a Series containing the mean values of these calculated statistics in this order:
        - CPU energy stdev
        - CPU energy mean
        - RAM energy stdev
        - RAM energy mean
        - GPU power stdev
        - GPU power mean
    """
    
    dfs = split_df_into_n(combined_df, n)

    columns=["cpu_energy_stdv", "cpu_energy_mean", "ram_energy_stdv", "ram_energy_mean", "gpu_power_stdv", "gpu_power_mean"]
    stats = []
    for i, df in enumerate(dfs):
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


def calc_stdev_mean_ratios(mean_stats: pd.Series):
    """
    It returns the standard deviation (stdev) to mean ratios of the given mean idle energy stats:
        - CPU Energy
        - RAM Energy
        - GPU Power
    """
    cpu_std_mean = str(round(mean_stats[0] / mean_stats[1], 3))
    ram_std_mean = str(round(mean_stats[2] / mean_stats[3], 3))
    gpu_std_mean = str(round(mean_stats[4] / mean_stats[5], 3))
    return cpu_std_mean, ram_std_mean, gpu_std_mean


def calc_ratios_from_data(n: int, cpu_energy: pd.DataFrame = None, ram_energy: pd.DataFrame = None, gpu_power: pd.DataFrame = None, directory: Path = None):
    """
    The module's "main" method.
    Calculate the stdev-to-mean ratios given the energy data as DataFrames or
    as a directory containing the perf & nvidia-smi files.
    """
    combined_df = create_combined_df(cpu_energy, ram_energy, gpu_power, directory)
    idle_mean_stats = calc_stats_for_split_data(n, combined_df)
    cpu_std_mean, ram_std_mean, gpu_std_mean = calc_stdev_mean_ratios(idle_mean_stats)
    return cpu_std_mean, ram_std_mean, gpu_std_mean
