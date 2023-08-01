import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import pandas as pd
from typing import List
import numpy as np
import matplotlib.ticker as ticker

from fecom.experiment.data import EnergyData, ProjectEnergyData
from fecom.experiment.analysis import prepare_total_energy_from_project, prepare_total_energy_and_size_from_project


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
        (energy_data.begin_stable_check_time_perf, "stable_check", 'y', 'dashed'),
        (energy_data.begin_temperature_check_time_perf, "temperature_check", 'c', 'solid')
    ]
    return perf_times


def add_stable_mean_to_ax(ax: plt.Axes, ax_legend_handles: list, hardware_device: str, energy_data: EnergyData):
    hardware_device = hardware_device.lower()
    stable_mean = getattr(energy_data, f"stable_{hardware_device}_power_mean")
    
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
    ax.plot(energy_data.cpu_energy["time_elapsed"], energy_data.cpu_energy["energy (J)"].div(energy_data.measurement_interval_s)) # convert energy (J) to power (W)
    ax_legend_handles = []
    for time, label, color, linestyle in cpu_times:
        ax.axvline(x=time, color=color, linewidth=1,linestyle=linestyle, alpha=0.7)
        ax_legend_handles.append(mlines.Line2D([], [], color=color, label=label, linestyle=linestyle))
    
    if graph_stable_mean:
        ax, ax_legend_handles = add_stable_mean_to_ax(ax, ax_legend_handles, "CPU", energy_data)
    
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
    ax.plot(energy_data.ram_energy["time_elapsed"], energy_data.ram_energy["energy (J)"].div(energy_data.measurement_interval_s)) # convert energy (J) to power (W)
    ax_legend_handles = []
    for time, label, color, linestyle in ram_times:
        ax.axvline(x=time, color=color, linewidth=1, linestyle=linestyle, alpha=0.7)
        ax_legend_handles.append(mlines.Line2D([], [], color=color, label=label, linestyle=linestyle))
    
    if graph_stable_mean:
        ax, ax_legend_handles = add_stable_mean_to_ax(ax, ax_legend_handles, "RAM", energy_data)

    ax.legend(handles=ax_legend_handles, loc="upper left", bbox_to_anchor=(1, 1))
    
    ax.set_ylabel("Power (W)")
    ax.set_xlabel("Time elapsed (s)")

    return ax


def format_ax_gpu(energy_data: EnergyData, ax: plt.Axes, graph_stable_mean=False):
    gpu_times = [
        (energy_data.start_time_nvidia, "method_start", 'r', 'dashed'),
        (energy_data.end_time_nvidia, "method_end", 'r', 'solid'),
        (energy_data.begin_stable_check_time_nvidia, "stable_check", 'y', 'dashed'),
        (energy_data.lag_end_time_gpu, "lag_end", 'm', 'dotted'),
        (energy_data.begin_temperature_check_time_nvidia, "temperature_check", 'c', 'solid')
    ]

    ax.set_title("GPU Power over time")
    ax.plot(energy_data.gpu_energy["time_elapsed"], energy_data.gpu_energy["power_draw (W)"])
    ax_legend_handles = []
    for time, label, color, linestyle in gpu_times:
        ax.axvline(x=time, color=color, linewidth=1, linestyle=linestyle, alpha=0.7)
        ax_legend_handles.append(mlines.Line2D([], [], color=color, label=label, linestyle=linestyle))
    
    if graph_stable_mean:
        ax, ax_legend_handles = add_stable_mean_to_ax(ax, ax_legend_handles, "GPU", energy_data)

    ax.legend(handles=ax_legend_handles, loc='upper left', bbox_to_anchor=(1, 1))
    
    ax.set_ylabel("Power (W)")
    ax.set_xlabel("Time elapsed (s)")

    return ax

def plot_args_size_vs_gpu_mean(total_energy_dfs):
    function_values = set()

    for df in total_energy_dfs:
        function_values.update(df['function'].unique())

    for function in function_values:
        fig, ax = plt.subplots(figsize=(8, 6))

        for df in total_energy_dfs:
            data = df[df['function'] == function]
            args_size_mean = data['Args Size (mean)']
            gpu_mean = data['GPU (mean)']
            ax.plot(args_size_mean, gpu_mean, marker='o', linestyle='-', label=f'Project: {df.iloc[0]["Project Name"]}')

        ax.set_xlabel('Args Size (mean)')
        ax.set_ylabel('GPU (mean)')
        ax.set_title(f'Function: {function}')
        ax.legend()

        plt.tight_layout()
        plt.savefig(f'./rq2_analysis/plot_args_size_vs_gpu_mean_{function}.png')
        plt.close()


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
    
    # run format_ax_{hardware_component} to populate the ax object with the correct data for the given hardware component
    ax = eval(f"format_ax_{hardware_component}(energy_data, ax, graph_stable_mean)")

    if start_at_stable_state:
        if hardware_component in ["cpu", "ram"]:
            start_stable_state_time = energy_data.begin_stable_check_time_perf
            last_time = energy_data.end_time_perf + energy_data.wait_after_run_s
        elif hardware_component == "gpu":
            start_stable_state_time = energy_data.begin_stable_check_time_nvidia
            last_time = energy_data.end_time_nvidia + energy_data.wait_after_run_s
        ax.set_xlim(left = start_stable_state_time - 30, right = last_time)
        # ax.margins(x=0, tight=True)

    figure = plt.gcf() # get current figure
    figure.set_size_inches(12, 6)
    plt.tight_layout()
    plt.savefig(f'./rq2_analysis/tail_state_plot_{hardware_component}.png', dpi=200, bbox_inches='tight')

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
    """
    min_len = min([len(energy_data.gpu_energy), len(energy_data.cpu_energy), len(energy_data.ram_energy)]) - 1
    print(min_len)
    combined_df = pd.concat(
        [
        energy_data.gpu_energy.iloc[:min_len]['power_draw (W)'],
        energy_data.cpu_energy.iloc[:min_len]['energy (J)'].div(energy_data.measurement_interval_s), # convert energy (J) to power (W)
        energy_data.ram_energy.iloc[:min_len]['energy (J)'].div(energy_data.measurement_interval_s) # convert energy (J) to power (W)
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


# def plot_total_energy_vs_execution_time(method_level_energies: List[ProjectEnergyData], title=True):
#     """
#     Takes a list of ProjectEnergyData objects, and plots the total normalised energy consumption
#     versus mean execution time for all functions in the ProjectEnergyData objects.
#     """
#     data_list = []
#     for method_level_energy in method_level_energies:
#         project_data_list, column_names = prepare_total_energy_from_project(method_level_energy)
#         data_list.extend(project_data_list)
    
#     total_df = pd.DataFrame(data_list, columns=column_names)

#     for hardware in ["CPU", "RAM", "GPU"]:
#         plt.figure(f"{hardware}_total")
#         # allow the option to not set a title for graphs included in a report
#         if title:
#             plt.title(f"Total normalised energy consumption vs time ({hardware})", fontsize=16)
#         plt.xlabel("Mean execution time (s)")
#         plt.ylabel("Normalised energy consumption (Joules)")
#         scatter_1 = f"{hardware} (mean)"
#         scatter_2 = f"{hardware} (median)"
#         plt.scatter(total_df.loc[:,"run time"], total_df.loc[:, scatter_1])
#         plt.scatter(total_df.loc[:,"run time"], total_df.loc[:, scatter_2])
#         plt.legend([scatter_1, scatter_2])
#         plt.savefig(f'{hardware}_energy_vs_time_plot.png')
#         plt.show()

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

    plt.figure("Total Energy vs Time")
    # allow the option to not set a title for graphs included in a report
    if title:
        plt.title("Total normalised energy consumption vs time", fontsize=16)
    plt.xlabel("Mean execution time (s)",fontsize=12)
    plt.ylabel("Net Energy consumption (Joules)",fontsize=12)

    for hardware in ["CPU", "RAM", "GPU"]:
        scatter_col = f"{hardware} (mean)"
        plt.scatter(total_df.loc[:, "run time"], total_df.loc[:, scatter_col], label=hardware, alpha=0.4 )
        # scatter_col = f"{hardware} (median)"
        # plt.scatter(total_df.loc[:, "run time"], total_df.loc[:, scatter_col], label=hardware, marker="*")
        
    plt.legend()
    # Function to format y-axis tick labels to show values in thousands (K)
    def format_y_axis(value, _):
        if value >= 1000:
            return f"{value/1000:.0f}K"
        return value

    # Apply the custom formatter to the y-axis
    plt.gca().yaxis.set_major_formatter(ticker.FuncFormatter(format_y_axis))
    plt.tight_layout()
    plt.savefig('energy_vs_time_plot.png')
    plt.show()

def plot_combined_total_energy_vs_execution_time(method_level_energies: List[ProjectEnergyData], title=True):
    """
    Takes a list of ProjectEnergyData objects, and plots the total normalised energy consumption
    versus mean execution time for all functions in the ProjectEnergyData objects.
    """
    data_list = []
    for method_level_energy in method_level_energies:
        # project_data_list, column_names = prepare_total_energy_from_project(method_level_energy)
        project_data_list, column_names = prepare_total_energy_and_size_from_project(method_level_energy)
        data_list.extend(project_data_list)

    total_df = pd.DataFrame(data_list, columns=column_names)
    sorted_total_df = total_df.sort_values(by='run time')


    fig, axes = plt.subplots(1, 3, figsize=(36, 12))
    if title:
        fig.suptitle("Total normalized energy consumption vs time", fontsize=16)

    hardware_labels = ["CPU", "RAM", "GPU"]
    scatter_labels = ["(mean)", "(median)","(min)","(max)"]

    for i, ax in enumerate(axes):
        hardware = hardware_labels[i]
        scatter_1 = f"{hardware} {scatter_labels[0]}"
        scatter_2 = f"{hardware} {scatter_labels[1]}"
        scatter_min = f"{hardware} {scatter_labels[2]}"
        scatter_max = f"{hardware} {scatter_labels[3]}"
        # ax.plot(total_df.loc[:, "run time"], total_df.loc[:, scatter_1], label=scatter_1)
        # ax.plot(total_df.loc[:, "run time"], total_df.loc[:, scatter_2], label=scatter_2)
        ax.plot(sorted_total_df["run time"], sorted_total_df[scatter_1], label=scatter_1)
        # ax.plot(sorted_total_df["run time"], sorted_total_df[scatter_2], label=scatter_2)
        ax.fill_between(sorted_total_df["run time"], sorted_total_df[scatter_min], sorted_total_df[scatter_max], alpha=0.2)
        
        ax.set_xlabel("Mean execution time (s)", fontsize=24)
        ax.set_ylabel("Normalized energy consumption (Joules)", fontsize=24)
        ax.set_title(hardware, fontsize=28)

        ax.legend()

    
    plt.yticks(fontsize=18)
    plt.xticks(fontsize=18)
    plt.legend(fontsize=20)
    plt.tight_layout()
    plt.savefig("energy_vs_time_combined_plot.pdf")
    plt.show()



def plot_project_level_energy_vs_method_level_energy(total_energy_projects):
    x = []
    y_cpu_method = []
    y_cpu_project = []
    y_gpu_method = []
    y_gpu_project = []
    y_ram_method = []
    y_ram_project = []

    for project_name, total_energy_df in total_energy_projects.items():
        project_data = total_energy_df[total_energy_df['function'].isin(['project-level', 'method-level (sum)'])]

        if 'method-level (sum)' not in project_data['function'].values:
            project_data = project_data.append({'function': 'method-level (sum)'}, ignore_index=True)
            project_data.fillna(0, inplace=True)

        x.append(project_name)
        y_cpu_method.append(project_data.loc[project_data['function'] == 'method-level (sum)', 'CPU (mean)'].tolist()[0])
        y_cpu_project.append(project_data.loc[project_data['function'] == 'project-level', 'CPU (mean)'].tolist()[0])
        y_gpu_method.append(project_data.loc[project_data['function'] == 'method-level (sum)', 'GPU (mean)'].tolist()[0])
        y_gpu_project.append(project_data.loc[project_data['function'] == 'project-level', 'GPU (mean)'].tolist()[0])
        y_ram_method.append(project_data.loc[project_data['function'] == 'method-level (sum)', 'RAM (mean)'].tolist()[0])
        y_ram_project.append(project_data.loc[project_data['function'] == 'project-level', 'RAM (mean)'].tolist()[0])

    attributes = ['CPU Method-Level', 'CPU Project-Level', 'GPU Method-Level', 'GPU Project-Level', 'RAM Method-Level', 'RAM Project-Level']
    measurements = [y_cpu_method, y_cpu_project, y_gpu_method, y_gpu_project, y_ram_method, y_ram_project]
    hardware = ['CPU', 'GPU', 'RAM']

    x_pos = np.arange(len(x))  # the label locations
    bar_width = 0.4  # the width of the bars
    spacing = 0.04  # spacing between grouped bars

    fig, axes = plt.subplots(3, 1, figsize=(8, 10), sharex=True, gridspec_kw={'hspace': 0.04})

    for i, ax in enumerate(axes):
        for j in range(2):
            measurement = measurements[i * 2 + j]
            ax.bar(x_pos + j * (bar_width + spacing), measurement, bar_width, label=attributes[i * 2 + j])
        
        if i == 1:
            ax.set_ylabel('Energy consumption (Joules)', fontsize=12)
        
        # ax.set_title(attributes[i * 2] + ' vs ' + attributes[i * 2 + 1])
        ax.margins(x=0.01)
        if i == 0:
            ax.legend(labels=['Method-Level', 'Project-Level'], bbox_to_anchor=(0.02, 0.98), loc='upper left')

        ax.text(0.98, 0.94, hardware[i], transform=ax.transAxes, fontsize=12, ha='right', va='top')

    axes[-1].set_xlabel('Project', fontsize=12)
    plt.xticks(x_pos, ["P" + str(x + 1) for x in x_pos], rotation=45, ha='right', fontsize='small')
    plt.tight_layout()
    plt.savefig('project_vs_method_energy_plot.png', bbox_inches='tight')
    plt.show()



# def plot_total_energy_vs_data_size_scatter(project_energy: ProjectEnergyData, title=True):
#     """
#     Takes a ProjectEnergyData object from any kind of experiment (typically data-size),
#     and plots the total normalised energy consumption versus total args size for all
#     data points (every experiment for every function) in the ProjectEnergyData object.
#     Creates 3 plots, one for each hardware device.
#     """
#     raise DeprecationWarning("Reconsider using this function. The plot_total_energy_vs_data_size_boxplot function is preferrable.")
#     for hardware in ["cpu", "ram", "gpu"]:
#         hardware_label = hardware.upper()
#         function_energies = getattr(project_energy, hardware)
#         args_sizes = []
#         total_energies = []
#         for function_energy in function_energies:
#             args_sizes.extend(function_energy.total_args_size)
#             total_energies.extend(function_energy.total_normalised)

#         plt.figure(f"{hardware_label}_total_vs_data_size")
#         # allow the option to not set a title for graphs included in a report
#         if title:
#             plt.title(f"Total normalised energy consumption vs args size ({hardware_label})", fontsize=16)
#         plt.xlabel("Total args size (MB)")
#         plt.ylabel("Normalised energy consumption (Joules)")
#         plt.scatter(args_sizes, total_energies)

#     plt.show()


def plot_total_energy_vs_data_size_boxplot(project_energy: ProjectEnergyData, title=True):
    """
    Takes a ProjectEnergyData object from any kind of experiment (typically data-size),
    and plots the total normalised energy consumption versus total args size as a boxplot.
    It draws a box of the different data points for every datasize.
    Creates 3 plots, one for each hardware device.
    """
    for hardware in ["cpu", "ram", "gpu"]:
        hardware_label = hardware.upper()
        # below is same as function_energies = project_energy.cpu_data (and same for ram and gpu)
        function_energies = getattr(project_energy, f"{hardware}_data")
        total_energies = []
        args_sizes = []
        for function_energy in function_energies:
            assert len(set(function_energy.total_args_size)) == 1, f"The argument size of the same function call ({function_energy.name}) should be the same across experiments."
            args_sizes.append(int(function_energy.total_args_size[0]))
            total_energies.append(function_energy.total_normalised)

        plt.figure(f"{hardware_label}_total_vs_data_size")
        # allow the option to not set a title for graphs included in a report
        if title:
            plt.title(f"Total normalised energy consumption vs args size ({hardware_label})", fontsize=16)
        plt.xlabel("Total args size (MB)")
        plt.ylabel("Total normalised energy consumption (Joules)")
        plt.boxplot(total_energies, labels=args_sizes)
        plt.savefig(f'./rq2_analysis/plot_total_energy_vs_data_size_boxplot_{hardware_label}_{project_energy.name.replace("/","_",1)}.png')
        plt.show()

    
def plot_total_energy_vs_data_size_scatter(project_energy: ProjectEnergyData, title=True):
    """
    Takes a ProjectEnergyData object from any kind of experiment (typically data-size),
    and plots the total normalised energy consumption versus total args size as a boxplot.
    It draws a box of the different data points for every datasize.
    Creates 3 plots, one for each hardware device.
    """
    plt.figure(f"total_vs_data_size")
        # allow the option to not set a title for graphs included in a report
    if title:
        plt.title(f"Energy consumption vs args size", fontsize=16)
    plt.xlabel("Total args size (MB)")
    plt.ylabel("Energy consumption (Joules)")
    for hardware in ["cpu", "ram", "gpu"]:
        hardware_label = hardware.upper()
        # function_energies = project_energy.cpu
        function_energies = getattr(project_energy, hardware)
        total_energies = []
        args_sizes = []
        for function_energy in function_energies:
            assert len(set(function_energy.total_args_size)) == 1, "The argument size of the same function should be the same across experiments."
            args_sizes.append(int(function_energy.total_args_size[0]))
            total_energies.append(function_energy.mean_total_normalised)

        plt.scatter(args_sizes, total_energies,label=hardware_label)
    plt.legend()
    plt.savefig(f'./rq2_analysis/plot_total_energy_vs_data_size_scatterplot.png')
    plt.show()

# def plot_total_energy_vs_data_size_scatter_combined(project_energy_list, title=True):
#     """
#     Takes a ProjectEnergyData object list from any kind of experiment (typically data-size),
#     and plots the total normalised energy consumption versus total args size as a boxplot.
#     It draws a box of the different data points for every datasize.
#     Creates 3 plots, one for each hardware device.
#     """
#     plt.figure(f"total_vs_data_size")
#         # allow the option to not set a title for graphs included in a report
#     if title:
#         plt.title(f"Energy consumption vs args size", fontsize=16)
#     plt.xlabel("Total args size (MB)",fontsize=22)
#     plt.ylabel("Net Energy consumption (Joules)",fontsize=22)
#     for hardware in ["cpu", "ram", "gpu"]:
#         hardware_label = hardware.upper()
#         # function_energies = project_energy.cpu

#         for i, project_energy in enumerate(project_energy_list):
#             function_energies = getattr(project_energy, hardware)
#             total_energies = []
#             args_sizes = []
#             for function_name, function_energy in function_energies.items():
#                 # assert len(set(function_energy.total_args_size)) == 1, "The argument size of the same function should be the same across experiments."
#                 args_sizes.append(int(function_energy.total_args_size[0]))
#                 total_energies.append(function_energy.mean_total_normalised)

#             plt.scatter(args_sizes, total_energies,label=hardware_label+"_P"+str(i))
#     plt.legend(fontsize=16)
#     plt.xticks(fontsize=16)
#     plt.yticks(fontsize=16)
#     plt.savefig(f'./rq2_analysis/plot_total_energy_vs_data_size_scatterplot_combined.png')
#     plt.show()

def plot_total_energy_vs_data_size_scatter_combined(project_energy_list, title=True):
    """
    Takes a ProjectEnergyData object list from any kind of experiment (typically data-size),
    and plots the total normalised energy consumption versus total args size as a boxplot.
    It draws a box of the different data points for every datasize.
    Creates 3 plots, one for each hardware device.
    """
    plt.figure(f"total_vs_data_size")
        # allow the option to not set a title for graphs included in a report
    if title:
        plt.title(f"Energy consumption vs args size", fontsize=16)
    plt.xlabel("Total args size (MB)",fontsize=16)
    plt.ylabel("Net Energy consumption (Joules)",fontsize=16)
    for hardware in ["cpu", "ram", "gpu"]:
        hardware_label = hardware.upper()
        # function_energies = project_energy.cpu
        total_energies = []
        args_sizes = []
        for project_energy in project_energy_list:
            function_energies = getattr(project_energy, hardware)
            for function_energy in function_energies.values():
                assert len(set(function_energy.total_args_size)) == 1, "The argument size of the same function should be the same across experiments."
                args_sizes.append(int(function_energy.total_args_size[0]))
                total_energies.append(function_energy.mean_total_normalised)
       
        # Convert energy consumption values to a common scale (Min-Max normalization)
        # total_energies = np.array(total_energies)
        # normalized_total_energies = (total_energies - np.min(total_energies)) / (np.max(total_energies) - np.min(total_energies))
        # normalized_total_energies = (total_energies - np.mean(total_energies)) / np.std(total_energies)

        plt.scatter(args_sizes, total_energies,label=hardware_label, alpha=0.4)
    # plt.legend()
    plt.legend(fontsize=14)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    def format_y_axis(value, _):
        if value >= 1000:
            return f"{value/1000:.0f}K"
        return value

    # Apply the custom formatter to the y-axis
    plt.gca().yaxis.set_major_formatter(ticker.FuncFormatter(format_y_axis))
    plt.tight_layout()
    plt.savefig(f'./rq2_analysis/plot_total_energy_vs_data_size_scatterplot_combined.png')
    plt.show()

def plot_total_unnormalised_energy_vs_data_size_boxplot(project_energy: ProjectEnergyData, title=True):
    """
    Takes a ProjectEnergyData object from any kind of experiment (typically data-size),
    and plots the total (unnormalised) energy consumption versus total args size as a boxplot.
    It draws a box of the different data points for every datasize.
    Creates 3 plots, one for each hardware device.
    """
    for hardware in ["cpu", "ram", "gpu"]:
        hardware_label = hardware.upper()
        # below is same as function_energies = project_energy.cpu_data (and same for ram and gpu)
        function_energies = getattr(project_energy, f"{hardware}_data")
        total_energies = []
        args_sizes = []
        for function_energy in function_energies:
            assert len(set(function_energy.total_args_size)) == 1, "The argument size of the same function should be the same across experiments."
            args_sizes.append(int(function_energy.total_args_size[0]))
            total_energies.append(function_energy.total)

        plt.figure(f"{hardware_label}_total_vs_data_size")
        # allow the option to not set a title for graphs included in a report
        if title:
            plt.title(f"Total energy consumption vs args size ({hardware_label})", fontsize=16)
        plt.xlabel("Total args size (MB)")
        plt.ylabel("Total energy consumption (Joules)")
        plt.boxplot(total_energies, labels=args_sizes)
        plt.savefig(f'./rq2_analysis/plot_total_unnormalised_energy_vs_data_size_boxplot_{hardware_label}_{project_energy.name.replace("/","_",1)}.png')
        plt.show()