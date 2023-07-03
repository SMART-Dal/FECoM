"""
Replication code for plots used in the paper. Currently only used for Tim's Final Year Project Report.
"""

from tool.experiment.data import DataLoader
from tool.experiment.experiment_kinds import ExperimentKinds
from tool.experiment.analysis import init_project_energy_data, build_total_energy_df, build_total_energy_and_size_df
from tool.patching.patching_config import EXPERIMENT_DIR
from tool.experiment.plot import plot_single_energy_with_times, plot_total_energy_vs_execution_time, plot_total_energy_vs_data_size_boxplot, plot_total_unnormalised_energy_vs_data_size_boxplot, plot_project_level_energy_vs_method_level_energy, plot_args_size_vs_gpu_mean

from replication.executed_experiments import EXECUTED_EXPERIMENTS

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
    experiments_data = []
    for project_name in EXECUTED_EXPERIMENTS:
        try:
            data = init_project_energy_data(project_name, ExperimentKinds.METHOD_LEVEL, first_experiment=1, last_experiment=10)
            experiments_data.append(data)
        except Exception as e:
            print("Exception in project: ", project_name)
            raise e

    plot_total_energy_vs_execution_time(experiments_data, title=True)

def rq1_plot_project_level_energy_vs_method_level_energy():
    total_energy_projects = {}
    for project_name in EXECUTED_EXPERIMENTS:
        method_level_data = init_project_energy_data(project_name, ExperimentKinds.METHOD_LEVEL, first_experiment=1)
        project_level_data = init_project_energy_data(project_name, ExperimentKinds.PROJECT_LEVEL, first_experiment=1)
        total_energy_df = build_total_energy_df(method_level_data, project_level_data)
        total_energy_projects[project_name] = total_energy_df
    
    plot_project_level_energy_vs_method_level_energy(total_energy_projects)


def rq1_plot_tail_power_states_cpu():
    dl = DataLoader("images/cnn", EXPERIMENT_DIR, ExperimentKinds.METHOD_LEVEL)
    energy_data_list = dl.load_single_file("experiment-1.json")
    for energy_data in energy_data_list:
        if energy_data.function_name == "tensorflow.keras.models.Sequential.fit()":
            plot_single_energy_with_times(energy_data, hardware_component="cpu", graph_stable_mean=True, start_at_stable_state=True, title=True)
            return
    print("could not find function")

def rq1_plot_tail_power_states_ram():
    dl = DataLoader("images/cnn", EXPERIMENT_DIR, ExperimentKinds.METHOD_LEVEL)
    energy_data_list = dl.load_single_file("experiment-1.json")
    for energy_data in energy_data_list:
        if energy_data.function_name == "tensorflow.keras.models.Sequential.fit()":
            plot_single_energy_with_times(energy_data, hardware_component="ram", graph_stable_mean=True, start_at_stable_state=True, title=True)
            return
    print("could not find function")

def rq1_plot_tail_power_states_gpu():
    dl = DataLoader("images/cnn", EXPERIMENT_DIR, ExperimentKinds.METHOD_LEVEL)
    energy_data_list = dl.load_single_file("experiment-1.json")
    for energy_data in energy_data_list:
        if energy_data.function_name == "tensorflow.keras.models.Sequential.fit()":
            plot_single_energy_with_times(energy_data, hardware_component="gpu", graph_stable_mean=True, start_at_stable_state=True, title=True)
            return
    print("could not find function")
    
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

def rq2_plot_args_size_vs_gpu_mean():
    total_energy_dfs = []

    for project_name in EXECUTED_EXPERIMENTS:
        print(f"Project: {project_name}")
        method_level_data = init_project_energy_data(project_name, ExperimentKinds.METHOD_LEVEL, first_experiment=1)
        total_energy_df = build_total_energy_and_size_df(method_level_data)

        # Add the project name as the first column in the DataFrame
        total_energy_df.insert(0, 'Project Name', project_name)

        if not total_energy_df.empty:
            total_energy_dfs.append(total_energy_df)

    plot_args_size_vs_gpu_mean(total_energy_dfs)
    

if __name__ == "__main__":
    ### commented code has already been run, uncomment to replicate

    # implementation_plot_GPU_energy_with_times()
    # rq1_plot_total_energy_vs_time()
    # rq1_plot_project_level_energy_vs_method_level_energy()
    # rq1_plot_tail_power_states_gpu()
    # rq1_plot_tail_power_states_cpu()
    # rq1_plot_tail_power_states_ram()
    # rq2_plot_data_size_vs_energy()
    # rq2_plot_smallest_data_size_ram()
    # rq2_plot_largest_data_size_ram()
    # rq2_plot_data_size_vs_unnormalised_energy()
    rq2_plot_args_size_vs_gpu_mean()
    pass