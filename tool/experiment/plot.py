import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import pandas as pd
from typing import List

from tool.experiment.data import EnergyData, ProjectEnergyData
from tool.experiment.analysis import prepare_total_energy_from_project
from tool.server.server_config import COUNT_INTERVAL_S, STABLE_CPU_ENERGY_MEAN, STABLE_RAM_ENERGY_MEAN, STABLE_GPU_POWER_MEAN
from tool.client.client_config import WAIT_AFTER_RUN_S


def get_perf_times(energy_data: EnergyData) -> list:
    """
    Helper method for format_ax_cpu and format_ax_ram.
    perf_times and gpu_times (see format_ax_gpu) are lists of tuples in the format (time, label, color, style), where
    - time (float, seconds) is the time relative to the start of perf/nvidia (exact times can be found in START_TIMES_FILE)
    - label (str) is a description of this time
    - color (str) is a matplotlib color, e.g. 'r','b','g'
    - style (str|tuple) is a matplotlib line style, e.g. 'dashed' or (0, (1, 10))
    """
    perf_times = [
        (energy_data.start_time_perf, "method_start", 'r', 'dashed'),
        (energy_data.end_time_perf, "method_end", 'r', 'solid'),
        (energy_data.pickle_load_time_perf, "pickle_load", 'b', 'solid'),
        (energy_data.import_time_perf, "import", 'g', 'dotted'),
        (energy_data.begin_stable_check_time_perf, "stable_check", 'y', 'dashed'),
        (energy_data.begin_temperature_check_time_perf, "temperature_check", 'c', (0, (1, 10)))
    ]
    return perf_times


def add_stable_mean_to_ax(ax: plt.Axes, ax_legend_handles: list, hardware_device: str):
    stable_mean = eval(f"STABLE_{hardware_device.upper()}_ENERGY_MEAN") / COUNT_INTERVAL_S # convert energy (J) to power (W)
    label, color, linestyle = "stable_mean_power", "grey", "dashed"
    ax.axhline(y=stable_mean, color=color, linewidth=1, linestyle=linestyle, alpha=0.7)
    ax_legend_handles.append(mlines.Line2D([], [], color=color, label=label, linestyle=linestyle))
    return ax, ax_legend_handles


def format_ax_cpu(energy_data: EnergyData, ax: plt.Axes, graph_stable_mean=False):
    """
    Populate the given Axes object with CPU energy data needed for plotting energy consumption over time,
    including markers for the key times.
    """
    cpu_times = get_perf_times(energy_data)
    cpu_times.append(
        (energy_data.lag_end_time_cpu, "lag_end", 'm', 'dotted')
    )

    ax.set_title("CPU Power over time")
    ax.plot(energy_data.cpu_energy["time_elapsed"], energy_data.cpu_energy["energy (J)"].div(COUNT_INTERVAL_S))
    ax_legend_handles = []
    for time, label, color, linestyle in cpu_times:
        ax.axvline(x=time, color=color, linewidth=1,linestyle=linestyle, alpha=0.7)
        ax_legend_handles.append(mlines.Line2D([], [], color=color, label=label, linestyle=linestyle))
    
    if graph_stable_mean:
        ax, ax_legend_handles = add_stable_mean_to_ax(ax, ax_legend_handles, "CPU")
    
    ax.legend(handles=ax_legend_handles, loc="upper left")
    
    ax.set_ylabel("Power (W)")
    ax.set_xlabel("Time elapsed (s)")

    return ax


def format_ax_ram(energy_data: EnergyData, ax: plt.Axes, graph_stable_mean=False):
    """
    Populate the given Axes object with RAM energy data needed for plotting energy consumption over time,
    including markers for the key times.
    """
    ram_times = get_perf_times(energy_data)
    ram_times.append(
        (energy_data.lag_end_time_ram, "lag_end", 'm', 'dotted')
    )

    ax.set_title("RAM Power over time")
    ax.plot(energy_data.ram_energy["time_elapsed"], energy_data.ram_energy["energy (J)"].div(COUNT_INTERVAL_S))
    ax_legend_handles = []
    for time, label, color, linestyle in ram_times:
        ax.axvline(x=time, color=color, linewidth=1, linestyle=linestyle, alpha=0.7)
        ax_legend_handles.append(mlines.Line2D([], [], color=color, label=label, linestyle=linestyle))
    
    if graph_stable_mean:
        ax, ax_legend_handles = add_stable_mean_to_ax(ax, ax_legend_handles, "RAM")

    ax.legend(handles=ax_legend_handles, loc="upper left")
    
    ax.set_ylabel("Power (W)")
    ax.set_xlabel("Time elapsed (s)")

    return ax


def format_ax_gpu(energy_data: EnergyData, ax: plt.Axes, graph_stable_mean=False):
    gpu_times = [
        (energy_data.start_time_nvidia, "method_start", 'r', 'dashed'),
        (energy_data.end_time_nvidia, "method_end", 'r', 'solid'),
        (energy_data.pickle_load_time_nvidia, "pickle_load", 'b', 'solid'),
        (energy_data.import_time_nvidia, "import", 'g', 'dotted'),
        (energy_data.begin_stable_check_time_nvidia, "stable_check", 'y', 'dashed'),
        (energy_data.lag_end_time_gpu, "lag_end", 'm', 'dotted'),
        (energy_data.begin_temperature_check_time_nvidia, "temperature_check", 'c', (0, (1, 10)))
    ]

    ax.set_title("GPU Power over time")
    ax.plot(energy_data.gpu_energy["time_elapsed"], energy_data.gpu_energy["power_draw (W)"])
    ax_legend_handles = []
    for time, label, color, linestyle in gpu_times:
        ax.axvline(x=time, color=color, linewidth=1, linestyle=linestyle, alpha=0.7)
        ax_legend_handles.append(mlines.Line2D([], [], color=color, label=label, linestyle=linestyle))
    
    if graph_stable_mean:
        ax, ax_legend_handles = add_stable_mean_to_ax(ax, ax_legend_handles, "GPU")

    ax.legend(handles=ax_legend_handles, loc="upper left")
    
    ax.set_ylabel("Power (W)")
    ax.set_xlabel("Time elapsed (s)")

    return ax


def plot_single_energy_with_times(energy_data: EnergyData, hardware_component: str = "gpu", start_at_stable_state = False, title = True, graph_stable_mean = False):
    """
    Given an EnergyData object, create a single plot showing the energy consumption over time
    with key start/end times indicated by lines.
    The hardware_component parameter must be one of "cpu", "ram", "gpu".
    If start_at_stable_state is True, plot energy data starting at the beginning of stable state checking
    """
    fig, ax = plt.subplots()
    if title:
        fig.suptitle(f"Data for {energy_data.function_name} from {energy_data.project_name}", fontsize=16)

    ax = eval(f"format_ax_{hardware_component}(energy_data, ax, graph_stable_mean)")

    if start_at_stable_state:
        if hardware_component in ["cpu", "ram"]:
            start_stable_state_time = energy_data.begin_stable_check_time_perf
            last_time = energy_data.end_time_perf + WAIT_AFTER_RUN_S
        elif hardware_component == "gpu":
            start_stable_state_time = energy_data.begin_stable_check_time_nvidia
            last_time = energy_data.end_time_nvidia + WAIT_AFTER_RUN_S
        ax.set_xlim(left = start_stable_state_time - 30, right = last_time)
        # ax.margins(x=0, tight=True)

    figure = plt.gcf() # get current figure
    figure.set_size_inches(12, 6)
    plt.savefig('energy_plot.png', dpi=200)

    plt.show()


def plot_energy_with_times(energy_data: EnergyData):
    """
    Given an EnergyData object, create a plot with 3 graphs showing the energy consumption over time
    for CPU, RAM and GPU with start/end times indicated by lines.
    Set one or more of the parameters cpu, ram, gpu to False to exclude it from the graph.
    """

    fig, [ax1, ax2, ax3] = plt.subplots(nrows=1, ncols=3)

    fig.suptitle(f"Data for {energy_data.function_name} from {energy_data.project_name}", fontsize=16)
    
    ax1 = format_ax_cpu(energy_data, ax1)
    ax2 = format_ax_ram(energy_data, ax2)
    ax3 = format_ax_gpu(energy_data, ax3)
    
    # fig.tight_layout()
    
    figure = plt.gcf() # get current figure
    figure.set_size_inches(20, 6)
    plt.savefig('energy_plot.png', dpi=200)

    plt.show()

def plot_combined(energy_data: EnergyData):
    """
    Concatenates the three energy dataframes from the EnergyData object into one containing only energy consumption of each hardware component
    as well as the sum of these three values over time. It does not attempt to merge the perf and nvidia-smi data
    in a way that synchronises the measurements in same rows to be at the same time.
    TODO implement that.
    """
    min_len = min([len(energy_data.gpu_energy), len(energy_data.cpu_energy), len(energy_data.ram_energy)]) - 1
    print(min_len)
    combined_df = pd.concat(
        [
        energy_data.gpu_energy.iloc[:min_len]['power_draw (W)'],
        energy_data.cpu_energy.iloc[:min_len]['energy (J)'].div(COUNT_INTERVAL_S),
        energy_data.ram_energy.iloc[:min_len]['energy (J)'].div(COUNT_INTERVAL_S)
        ],
        axis=1)
    combined_df.columns = ['gpu_power', 'cpu_power', 'ram_power']
  
    combined_df['sum'] = combined_df.sum(axis=1)

    print("Combined plot:")
    print(combined_df)
    print("Statistics (stdv, mean):")
    print(combined_df.std())
    print(combined_df.mean())
    combined_df.plot()
    plt.show()


def plot_total_energy_vs_execution_time(method_level_energies: List[ProjectEnergyData], title=True):
    """
    Takes a list of ProjectEnergyData objects, and plots the total normalised energy consumption
    versus mean execution time for all functions in the ProjectEnergyData objects.
    """
    data_list = []
    for method_level_energy in method_level_energies:
        project_data_list, column_names = prepare_total_energy_from_project(method_level_energy)
        data_list.extend(project_data_list)
    
    total_df = pd.DataFrame(data_list, columns=column_names)

    for hardware in ["CPU", "RAM", "GPU"]:
        plt.figure(f"{hardware}_total")
        # allow the option to not set a title for graphs included in a report
        if title:
            plt.title(f"Total normalised energy consumption vs time ({hardware})", fontsize=16)
        plt.xlabel("Mean execution time (s)")
        plt.ylabel("Normalised energy consumption (Joules)")
        scatter_1 = f"{hardware} (mean)"
        scatter_2 = f"{hardware} (median)"
        plt.scatter(total_df.loc[:,"run time"], total_df.loc[:, scatter_1])
        plt.scatter(total_df.loc[:,"run time"], total_df.loc[:, scatter_2])
        plt.legend([scatter_1, scatter_2])

    plt.show()


def plot_total_energy_vs_data_size_scatter(project_energy: ProjectEnergyData, title=True):
    """
    Takes a ProjectEnergyData object from any kind of experiment (typically data-size),
    and plots the total normalised energy consumption versus total args size for all
    data points (every experiment for every function) in the ProjectEnergyData object.
    Creates 3 plots, one for each hardware device.
    """
    for hardware in ["cpu", "ram", "gpu"]:
        hardware_label = hardware.upper()
        function_energies = getattr(project_energy, hardware)
        args_sizes = []
        total_energies = []
        for function_energy in function_energies:
            args_sizes.extend(function_energy.total_args_size)
            total_energies.extend(function_energy.total_normalised)

        plt.figure(f"{hardware_label}_total_vs_data_size")
        # allow the option to not set a title for graphs included in a report
        if title:
            plt.title(f"Total normalised energy consumption vs args size ({hardware_label})", fontsize=16)
        plt.xlabel("Total args size (MB)")
        plt.ylabel("Normalised energy consumption (Joules)")
        plt.scatter(args_sizes, total_energies)

    plt.show()


def plot_total_energy_vs_data_size_boxplot(project_energy: ProjectEnergyData, title=True):
    """
    Takes a ProjectEnergyData object from any kind of experiment (typically data-size),
    and plots the total normalised energy consumption versus total args size as a boxplot.
    It draws a box of the different data points for every datasize.
    Creates 3 plots, one for each hardware device.
    """
    for hardware in ["cpu", "ram", "gpu"]:
        hardware_label = hardware.upper()
        # function_energies = project_energy.cpu
        function_energies = getattr(project_energy, hardware)
        total_energies = []
        args_sizes = []
        for function_energy in function_energies:
            assert len(set(function_energy.total_args_size)) == 1, "The argument size of the same function should be the same across experiments."
            args_sizes.append(int(function_energy.total_args_size[0]))
            total_energies.append(function_energy.total_normalised)

        plt.figure(f"{hardware_label}_total_vs_data_size")
        # allow the option to not set a title for graphs included in a report
        if title:
            plt.title(f"Total normalised energy consumption vs args size ({hardware_label})", fontsize=16)
        plt.xlabel("Total args size (MB)")
        plt.ylabel("Normalised energy consumption (Joules)")
        plt.boxplot(total_energies, labels=args_sizes)

    plt.show()