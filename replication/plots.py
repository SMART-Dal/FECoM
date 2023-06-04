"""
Replication code for plots used in the paper. Currently only used for Tim's Final Year Project Report.
"""

from tool.experiment.data import DataLoader
from tool.experiment.experiments import ExperimentKinds
from tool.experiment.analysis import init_project_energy_data
from tool.patching.patching_config import EXPERIMENT_DIR
from tool.experiment.plot import plot_single_energy_with_times, plot_total_energy_vs_execution_time, plot_total_energy_vs_data_size_boxplot, plot_total_unnormalised_energy_vs_data_size_boxplot

def implementation_plot_GPU_energy_with_times():
    """
    create the GPU plot used in Design & Implementation, Processed Data Representation
    """
    dl = DataLoader("keras/classification", EXPERIMENT_DIR, ExperimentKinds.METHOD_LEVEL)
    energy_data_list = dl.load_single_file("experiment-6.json")
    for energy_data in energy_data_list:
        if energy_data.function_name == "tf.keras.Sequential.fit(*args, **kwargs)":
            plot_single_energy_with_times(energy_data, hardware_component="gpu")

### RQ 1 PLOTS
def rq1_plot_total_energy_vs_time():
    project_name = "keras/classification"
    data_1 = init_project_energy_data(project_name, ExperimentKinds.METHOD_LEVEL, first_experiment=1, last_experiment=10)
    project_name = "images/cnn"
    data_2 = init_project_energy_data(project_name, ExperimentKinds.METHOD_LEVEL, first_experiment=1, last_experiment=10)

    plot_total_energy_vs_execution_time([data_1, data_2], title=False)


def rq1_plot_tail_power_states_cpu():
    dl = DataLoader("images/cnn", EXPERIMENT_DIR, ExperimentKinds.METHOD_LEVEL)
    energy_data_list = dl.load_single_file("experiment-1.json")
    for energy_data in energy_data_list:
        if energy_data.function_name == "models.Sequential.fit(*args, **kwargs)":
            plot_single_energy_with_times(energy_data, hardware_component="cpu", graph_stable_mean=True, start_at_stable_state=True, title=False)

def rq1_plot_tail_power_states_ram():
    dl = DataLoader("images/cnn", EXPERIMENT_DIR, ExperimentKinds.METHOD_LEVEL)
    energy_data_list = dl.load_single_file("experiment-1.json")
    for energy_data in energy_data_list:
        if energy_data.function_name == "models.Sequential.fit(*args, **kwargs)":
            plot_single_energy_with_times(energy_data, hardware_component="ram", graph_stable_mean=True, start_at_stable_state=True, title=False)
            

def rq1_plot_tail_power_states_gpu():
    dl = DataLoader("images/cnn", EXPERIMENT_DIR, ExperimentKinds.METHOD_LEVEL)
    energy_data_list = dl.load_single_file("experiment-1.json")
    for energy_data in energy_data_list:
        if energy_data.function_name == "models.Sequential.fit(*args, **kwargs)":
            plot_single_energy_with_times(energy_data, hardware_component="gpu", graph_stable_mean=True, start_at_stable_state=True, title=False)

### RQ 2 PLOTS
def rq2_plot_data_size_vs_energy():
    project_name = "keras/classification"
    project_data = init_project_energy_data(project_name, ExperimentKinds.DATA_SIZE, first_experiment=1, last_experiment=7)
    plot_total_energy_vs_data_size_boxplot(project_data, title=False)

def rq2_plot_data_size_vs_unnormalised_energy():
    project_name = "keras/classification"
    project_data = init_project_energy_data(project_name, ExperimentKinds.DATA_SIZE, first_experiment=1, last_experiment=7)
    plot_total_unnormalised_energy_vs_data_size_boxplot(project_data, title=False)

def rq2_plot_smallest_data_size_ram():
    dl = DataLoader("keras/classification", EXPERIMENT_DIR, ExperimentKinds.DATA_SIZE)
    energy_data_list = dl.load_single_file("experiment-1.json")
    # first sample of a data-size experiment has the smallest data size
    print(f"Total args size: {energy_data_list[0].total_args_size}")
    plot_single_energy_with_times(energy_data_list[0], hardware_component="ram", start_at_stable_state=True, title=False, graph_stable_mean=True)

def rq2_plot_largest_data_size_ram():
    dl = DataLoader("keras/classification", EXPERIMENT_DIR, ExperimentKinds.DATA_SIZE)
    energy_data_list = dl.load_single_file("experiment-1.json")
    # last sample of a data-size experiment has the largest data size
    print(f"Total args size: {energy_data_list[-1].total_args_size}")
    plot_single_energy_with_times(energy_data_list[-1], hardware_component="ram", start_at_stable_state=True, title=False, graph_stable_mean=True)


if __name__ == "__main__":
    ### commented code has already been run, uncomment to replicate

    # implementation_plot_GPU_energy_with_times()
    # rq1_plot_total_energy_vs_time()
    # rq1_plot_tail_power_states_gpu()
    # rq1_plot_tail_power_states_cpu()
    # rq1_plot_tail_power_states_ram()
    # rq2_plot_data_size_vs_energy()
    # rq2_plot_smallest_data_size_ram()
    # rq2_plot_largest_data_size_ram()
    # rq2_plot_data_size_vs_unnormalised_energy()
    pass