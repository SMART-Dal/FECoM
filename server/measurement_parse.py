from datetime import datetime

import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from config import START_EXECUTION, END_EXECUTION


def parse_nvidia_smi(filename):
    """
    Given a filename returns a dataframe with columns 
    - timestamp (datetime)
    - power_draw (W) (float)
    """
    start_time = None
    end_time = None
    data_list = []
    with open(filename, 'r') as f:
        in_execution = False

        for line in f:
            line = line.strip('\n')
            # look for special execution markers
            if line == START_EXECUTION:
                in_execution = True
                # TODO keep in mind: start & end time will always be slightly sooner than the actual start time because we take the time of the last measurement before execution starts
                start_time = data_list[-1][0]
                continue
            elif line == END_EXECUTION:
                if in_execution == False:
                    raise ValueError("END_EXECUTION must be after START_EXECUTION")
                in_execution = False
                end_time = data_list[-1][0]
                continue

            # 2023/02/06 11:23:08.654, 20.28 W
            raw_data = line.split(',')
            data = [
                datetime.strptime(raw_data[0], '%Y/%m/%d %H:%M:%S.%f').timestamp(),
                float(raw_data[1].split()[0]),
                in_execution # add boolean in_execution column to data to indicate when the method is executing
                ]

            data_list.append(data)
    
    df = pd.DataFrame(data_list,
                      columns=['timestamp', 'power_draw (W)', 'in_execution'])

    return df, start_time, end_time


def parse_perf(filename):
    """
    Given a filename returns the tuple of dataframes (cpu_energy, ram_energy) with columns 
    - time_elapsed (float)
    - energy (J) (float)
    """
    start_time = None
    end_time = None
    data_list = []
    with open(filename, 'r') as f:
        in_execution = False

        for i, line in enumerate(f):
            # skip over the first two lines
            if i < 2:
                continue

            line = line.strip(' \n')
            # look for special execution markers
            if line == START_EXECUTION:
                in_execution = True
                # TODO keep in mind: start & end time will always be slightly sooner than the actual start time because we take the time of the last measurement before execution starts
                start_time = float(data_list[-1][0])
                continue
            elif line == END_EXECUTION:
                if in_execution == False:
                    raise ValueError("END_EXECUTION must be after START_EXECUTION")
                in_execution = False
                end_time = float(data_list[-1][0])
                continue
            # the last two values in each line are always empty because the line ends with ;;
            data = line.split(';')[:-2]
            # add boolean in_execution column to data to indicate when the method is executing
            data.append(in_execution)
            data_list.append(data)


    # create dataframe, and ignore the last two lines because they are always unrealistically low 
    df = pd.DataFrame(data_list[:-2],
                      columns=['time_elapsed', 'energy (J)', 'unit', 'event_name',
                               'counter_runtime', 'percent_measure_time', 'in_execution'])

    # drop 'counter_runtime' and 'percent_measure_time'
    df.drop(['counter_runtime', 'percent_measure_time', 'unit'], axis=1, inplace=True)
    df[["time_elapsed", "energy (J)"]] = df[['time_elapsed', 'energy (J)']].apply(pd.to_numeric)

    # split df by event_name
    df_pkg = df[df['event_name'] == 'power/energy-pkg/'].reset_index(drop=True).drop(columns='event_name')
    df_ram = df[df['event_name'] == 'power/energy-ram/'].reset_index(drop=True).drop(columns='event_name')
    return df_pkg, df_ram, start_time, end_time


if __name__ == "__main__":
    directory = "energy_measurement/out/"
    gpu_energy, start_time, end_time = parse_nvidia_smi(f"{directory}nvidia_smi.txt")
    ax = gpu_energy.plot(x="timestamp", y="power_draw (W)")
    ax.axvline(x=start_time, color='r',linewidth=1)
    ax.axvline(x=end_time, color='r',linewidth=1)
    plt.savefig('gpu_plot.png')
    print(gpu_energy)
    print(gpu_energy.dtypes)

    cpu_energy, ram_energy, start_time, end_time = parse_perf(f"{directory}perf.txt")
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
