import pandas as pd


def parse_nvidia_smi(filename):
    df = pd.read_csv(filename, header=['timestamp', 'power_draw'])
    return df


def parse_perf(filename):
    # remove first two lines (with start datetime)
    with open(filename, 'r') as fin:
        data_with_datetime = fin.read().splitlines(True)
    data_without_datetime = data_with_datetime[2:]

    # remove first 5 whitespaces from each line
    data_without_whitespace = [line.strip(' ') for line in data_without_datetime]

    # create dataframe
    data_list_of_lists = [element.split(';') for element in data_without_whitespace]
    data_list_of_lists_clean = [row_list[:-2] for row_list in data_list_of_lists]
    df = pd.DataFrame(data_list_of_lists_clean,
                      columns=['time_elapsed', 'counter_value', 'unit', 'event_name',
                               'counter_runtime', 'percent_measure_time'])

    # drop 'counter_runtime' and 'percent_measure_time'
    df.drop(['counter_runtime', 'percent_measure_time'], axis=1, inplace=True)
    df[["time_elapsed", "counter_value"]] = df[["time_elapsed", "counter_value"]].apply(pd.to_numeric)
    print(df.dtypes)

    # split df by event_name
    df_pkg = df[df['event_name'] == 'power/energy-pkg/']
    df_ram = df[df['event_name'] == 'power/energy-ram/']
    return df_pkg, df_ram


if __name__ == "__main__":
    parse_nvidia_smi("nvidia_smi.txt")
    parse_perf("perf.txt")
