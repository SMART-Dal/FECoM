from datetime import datetime

import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

def parse_nvidia_smi(filename) -> pd.DataFrame:
    """
    Given a filename returns a dataframe with columns 
        - timestamp (datetime)
        - power_draw (W) (float)
        - time_elapsed (seconds) (float)
        - temperature (Core GPU temperature in degrees Celsius) (int)   
    """
    data_list = []
    with open(filename, 'r') as f:
        time_zero = None
        for i, line in enumerate(f):
            # file format is timestamp, power_draw, temperature: 2023/02/06 11:23:08.654, 20.28 W, 32
            raw_data = line.strip('\n').split(',')
            current_time = datetime.strptime(raw_data[0], '%Y/%m/%d %H:%M:%S.%f').timestamp()
            # get the time of the first measurement for the time_elapsed column
            if i==0:
                time_zero = current_time
            data = [
                current_time, # timestamp
                float(raw_data[1].split()[0]), # power_draw
                current_time - time_zero, # add time elapsed column to data for graphing
                int(raw_data[2]) # temperature
                ]

            data_list.append(data)
    
    df = pd.DataFrame(data_list,
                      columns=['timestamp', 'power_draw (W)', 'time_elapsed', 'temperature'])

    return df


def parse_perf(filename) -> tuple((pd.DataFrame, pd.DataFrame)):
    """
    Given a filename returns a 4-tuple with
    - 2 dataframes (cpu_energy, ram_energy) with columns 
        - time_elapsed (float)
        - energy (J) (float)
    - start_time
    - end_time
    The times are determined by parsing the special execution markers START_EXECUTION, END_EXECUTION inserted into the perf file by the server.
    If no markers are found, start_time and end_time are None.
    """
    data_list = []
    with open(filename, 'r') as f:
        for i, line in enumerate(f):
            # skip over the first two lines
            if i < 2:
                continue

            # the last two values in each line are always empty because the line ends with ;;
            data_list.append(line.strip(' \n').split(';')[:-2])


    # create dataframe, and ignore the last two lines because they are always unrealistically low 
    df = pd.DataFrame(data_list[:-2],
                      columns=['time_elapsed', 'energy (J)', 'unit', 'event_name',
                               'counter_runtime', 'percent_measure_time'])

    # drop 'counter_runtime' and 'percent_measure_time'
    df.drop(['counter_runtime', 'percent_measure_time', 'unit'], axis=1, inplace=True)
    df[["time_elapsed", "energy (J)"]] = df[['time_elapsed', 'energy (J)']].apply(pd.to_numeric)

    # split df by event_name
    df_pkg = df[df['event_name'] == 'power/energy-pkg/'].reset_index(drop=True).drop(columns='event_name')
    df_ram = df[df['event_name'] == 'power/energy-ram/'].reset_index(drop=True).drop(columns='event_name')
    return df_pkg, df_ram


if __name__ == "__main__":
    directory = "energy_measurement/out/"
    gpu_energy = parse_nvidia_smi(f"{directory}nvidia_smi.txt")
    ax = gpu_energy.plot(x="timestamp", y="power_draw (W)")
    ax.axvline(x=start_time, color='r',linewidth=1)
    ax.axvline(x=end_time, color='r',linewidth=1)
    plt.savefig('gpu_plot.png')
    print(gpu_energy)
    print(gpu_energy.dtypes)

    cpu_energy, ram_energy = parse_perf(f"{directory}perf.txt")
    ax = cpu_energy.plot(x="time_elapsed", y="energy (J)")
    ax.axvline(x=start_time, color='r',linewidth=1)
    ax.axvline(x=end_time, color='r',linewidth=1)
    plt.savefig('cpu_plot.png')
    print(cpu_energy)
    print(cpu_energy.dtypes)

    ax = ram_energy.plot(x="time_elapsed", y="energy (J)")
    ax.axvline(x=start_time, color='r',linewidth=1)
    ax.axvline(x=end_time, color='r',linewidth=1)
    plt.savefig('ram_plot.png')
    print(ram_energy)
    print(ram_energy.dtypes)
   # print(parse_perf(f"{directory}perf.txt"))
