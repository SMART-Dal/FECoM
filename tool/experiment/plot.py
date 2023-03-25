import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import pandas as pd

from tool.experiment.analysis import EnergyData

def plot_energy_with_times(energy_data: EnergyData):
    """
    Given an EnergyData object, create a plot with 3 graphs showing the energy consumption over time
    for CPU, RAM and GPU with start/end times indicated by lines.
    """

    """
    perf_times and nvidia_times are lists of tuples in the format (time, label, color), where
    - time (float, seconds) is the time relative to the start of perf/nvidia (exact times can be found in START_TIMES_FILE)
    - label (str) is a description of this time
    - color (str) is a matplotlib color, e.g. 'r','b','g'
    """
    perf_times = [
        (energy_data.start_time_perf, "method_start", 'r', 'dashed'),
        (energy_data.end_time_perf, "method_end", 'r', 'solid'),
        (energy_data.pickle_load_time_perf, "pickle_load", 'b', 'solid'),
        (energy_data.import_time_perf, "import", 'g', 'dotted'),
        (energy_data.begin_stable_check_time_perf, "stable_check", 'y', 'dashed'),
        (energy_data.begin_temperature_check_time_perf, "temperature_check", 'c', (0, (1, 10)))
    ]

    cpu_times = perf_times.copy()
    cpu_times.append(
        (energy_data.lag_end_time_cpu, "lag_end", 'm', 'dotted')
    )

    ram_times = perf_times.copy()
    ram_times.append(
        (energy_data.lag_end_time_ram, "lag_end", 'm', 'dotted')
    )

    gpu_times = [
        (energy_data.start_time_nvidia, "method_start", 'r', 'solid'),
        (energy_data.end_time_nvidia, "method_end", 'r', 'solid'),
        (energy_data.pickle_load_time_nvidia, "pickle_load", 'b', 'solid'),
        (energy_data.import_time_nvidia, "import", 'g', 'dotted'),
        (energy_data.begin_stable_check_time_nvidia, "stable_check", 'y', 'dashed'),
        (energy_data.lag_end_time_gpu, "lag_end", 'm', 'dotted'),
        (energy_data.begin_temperature_check_time_nvidia, "temperature_check", 'c', (0, (1, 10)))
    ]

    fig, [ax1, ax2, ax3] = plt.subplots(nrows=1, ncols=3)

    fig.suptitle(f"Data for {energy_data.function_name} from {energy_data.project_name}", fontsize=16)
    
    ax1.set_title("CPU Energy over time")
    ax1.plot(energy_data.cpu_energy["time_elapsed"], energy_data.cpu_energy["energy (J)"])
    ax1_legend_handles = []
    for time, label, color, linestyle in cpu_times:
        ax1.axvline(x=time, color=color, linewidth=1,linestyle=linestyle, alpha=0.7)
        ax1_legend_handles.append(mlines.Line2D([], [], color=color, label=label, linestyle=linestyle))
        
    ax1.legend(handles=ax1_legend_handles)

    ax2.set_title("RAM Energy over time")
    ax2.plot(energy_data.ram_energy["time_elapsed"], energy_data.ram_energy["energy (J)"])
    ax2_legend_handles = []
    for time, label, color, linestyle in ram_times:
        ax2.axvline(x=time, color=color, linewidth=1, linestyle=linestyle, alpha=0.7)
        ax2_legend_handles.append(mlines.Line2D([], [], color=color, label=label, linestyle=linestyle))
    ax2.legend(handles=ax2_legend_handles)

    ax3.set_title("GPU Power over time")
    ax3.plot(energy_data.gpu_energy["time_elapsed"], energy_data.gpu_energy["power_draw (W)"])
    ax3_legend_handles = []
    for time, label, color, linestyle in gpu_times:
        ax3.axvline(x=time, color=color, linewidth=1, linestyle=linestyle, alpha=0.7)
        ax3_legend_handles.append(mlines.Line2D([], [], color=color, label=label, linestyle=linestyle))
    ax3.legend(handles=ax3_legend_handles)
    
    # fig.tight_layout()
    
    figure = plt.gcf() # get current figure
    figure.set_size_inches(20, 6)
    plt.savefig('energy_plot.png', dpi=200)

    plt.show()

def plot_combined(energy_data):
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
        energy_data.cpu_energy.iloc[:min_len]['energy (J)'],
        energy_data.ram_energy.iloc[:min_len]['energy (J)']
        ],
        axis=1)
    combined_df.columns = ['gpu_power', 'cpu_energy', 'ram_energy']
  
    combined_df['sum'] = combined_df.sum(axis=1)

    print("Combined plot:")
    print(combined_df)
    print("Statistics (stdv, mean):")
    print(combined_df.std())
    print(combined_df.mean())
    combined_df.plot()
    plt.show()