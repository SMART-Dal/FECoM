import matplotlib.pyplot as plt
from measurement_parse import parse_nvidia_smi, parse_perf
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import numpy as np

def normalised(lst, normalise=True):
    if not normalise:
        return lst

    min_x = min(lst)
    max_x = max(lst)

    return [(x - min_x) / (max_x - min_x) for x in lst]

def plot_cpu_and_ram(filename, n=100, normalise=True):
    """
    plot the last n data points
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
    if directory is not None:
        gpu_power = parse_nvidia_smi(f"{directory}nvidia_smi.txt")
        cpu_energy, ram_energy = parse_perf(f"{directory}perf.txt")
    min_len = min([len(gpu_power), len(cpu_energy), len(ram_energy)]) - 1
    print(min_len)
    df = pd.concat([gpu_power.loc[:min_len]['power_draw (W)'], cpu_energy.loc[:min_len]['energy (J)'], ram_energy.loc[:min_len]['energy (J)']], axis=1)
    df.columns = ['gpu_power', 'cpu_energy', 'ram_energy']
  
    # # apply normalization techniques
    # column = 'Column 1'
    # for column in df:
    #     df[column] = MinMaxScaler().fit_transform(np.array(df[column]).reshape(-1,1))

    df['sum'] = df.sum(axis=1)

    print(df)
    print(df.std())
    print(df.mean())
    return df
    

if __name__ == "__main__":
    directory = "./out/2022-12-10/"
    # plot_cpu_and_ram(f"{directory}perf.txt", n=10000, normalise=True)
    parse_nvidia_smi(f"{directory}nvidia_smi.txt").plot(y='power_draw (W)')
    # print(parse_nvidia_smi(f"{directory}nvidia_smi.txt"))
    # print(parse_perf(f"{directory}perf.txt"))
    # combined_plot(directory=directory).plot()
    plt.show()