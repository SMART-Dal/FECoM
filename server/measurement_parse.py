from datetime import datetime

import pandas as pd


def parse_nvidia_smi(filename):
    """
    Given a filename returns a dataframe with columns 
    - timestamp (datetime)
    - power_draw (W) (float)
    """
    headers = ['timestamp', 'power_draw (W)']
    dtypes = {'timestamp': 'str'}
    parse_dates = ['timestamp']
    convert_power_draw = lambda x: float(x.split()[0])
    converters = {'power_draw (W)': convert_power_draw}
    df = pd.read_csv(filename, header=None, names=headers, dtype=dtypes, parse_dates=parse_dates, converters=converters)
    return df


def parse_perf(filename):
    """
    Given a filename returns the tuple of dataframes (cpu_energy, ram_energy) with columns 
    - time_elapsed (float)
    - energy (J) (float)
    """
    # remove first two lines (with start datetime)
    with open(filename, 'r') as fin:
        data_with_datetime = fin.read().splitlines(True)
    start_datetime_str = data_with_datetime[0][len('# started on '):-1]
    start_datetime = datetime.strptime(start_datetime_str, '%c')
    data_without_datetime = data_with_datetime[2:]

    # remove first 5 whitespaces from each line
    data_without_whitespace = [line.strip(' ') for line in data_without_datetime]

    # create dataframe
    data_list_of_lists = [element.split(';') for element in data_without_whitespace]
    data_list_of_lists_clean = [row_list[:-2] for row_list in data_list_of_lists]
    df = pd.DataFrame(data_list_of_lists_clean,
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
    directory = "./out/2022-12-09/"
    gpu_energy = parse_nvidia_smi(f"{directory}nvidia_smi.txt")
    print(gpu_energy)
    print(gpu_energy.dtypes)
    cpu_energy, ram_energy = parse_perf(f"{directory}perf.txt")
    print(cpu_energy)
    print(cpu_energy.dtypes)
    print(ram_energy)
    print(ram_energy.dtypes)
   # print(parse_perf(f"{directory}perf.txt"))
