"""
Replication code for plots used in the paper.
"""

from fecom.experiment.data import DataLoader
from fecom.experiment.experiment_kinds import ExperimentKinds
from fecom.experiment.analysis import init_project_energy_data, build_total_energy_df, build_total_energy_and_size_df
from fecom.patching.patching_config import EXPERIMENT_DIR
from fecom.experiment.plot import plot_single_energy_with_times, plot_total_energy_vs_data_size_scatter_combined, plot_total_energy_vs_data_size_scatter,plot_combined_total_energy_vs_execution_time, plot_total_energy_vs_execution_time, plot_total_energy_vs_data_size_boxplot, plot_total_unnormalised_energy_vs_data_size_boxplot, plot_project_level_energy_vs_method_level_energy, plot_args_size_vs_gpu_mean
import pandas as pd
from executed_experiments import EXECUTED_RQ1_EXPERIMENTS, EXECUTED_RQ2_EXPERIMENTS
import matplotlib.pyplot as plt
import scipy.stats as stats

def implementation_plot_GPU_energy_with_times():
    """
    create the GPU plot used in Design & Implementation, Processed Data Representation
    """
    function_name = "tensorflow.keras.Sequential.fit()"
    dl = DataLoader("keras/classification", EXPERIMENT_DIR, ExperimentKinds.METHOD_LEVEL)
    energy_data_dict = dl.load_single_file("experiment-6.json")
    energy_data = energy_data_dict[function_name]
    plot_single_energy_with_times(energy_data, hardware_component="gpu")

### RQ 1 PLOTS
def rq1_plot_total_energy_vs_time():
    experiments_data = []
    for project_name in EXECUTED_RQ1_EXPERIMENTS:
        try:
            data = init_project_energy_data(project_name, ExperimentKinds.METHOD_LEVEL, first_experiment=1, last_experiment=10)
            experiments_data.append(data)
        except Exception as e:
            print("Exception in project: ", project_name)
            raise e

    plot_total_energy_vs_execution_time(experiments_data, title=False)

def rq1_plot_total_energy_vs_time_combined():
    experiments_data = []
    for project_name in EXECUTED_RQ1_EXPERIMENTS:
        try:
            data = init_project_energy_data(project_name, ExperimentKinds.METHOD_LEVEL, first_experiment=1, last_experiment=10)
            experiments_data.append(data)
        except Exception as e:
            print("Exception in project: ", project_name)
            raise e

    plot_combined_total_energy_vs_execution_time(experiments_data, title=False)

def rq1_plot_project_level_energy_vs_method_level_energy():
    total_energy_projects = {}
    for project_name in EXECUTED_RQ1_EXPERIMENTS:
        method_level_data = init_project_energy_data(project_name, ExperimentKinds.METHOD_LEVEL, first_experiment=1)
        project_level_data = init_project_energy_data(project_name, ExperimentKinds.PROJECT_LEVEL, first_experiment=1)
        total_energy_df = build_total_energy_df(method_level_data, project_level_data)
        total_energy_projects[project_name] = total_energy_df
    
    plot_project_level_energy_vs_method_level_energy(total_energy_projects)


def rq1_plot_tail_power_states_cpu():
    function_name = "tensorflow.keras.models.Sequential.fit()"
    dl = DataLoader("images/cnn", EXPERIMENT_DIR, ExperimentKinds.METHOD_LEVEL)
    energy_data_dict = dl.load_single_file("experiment-1.json")
    energy_data = energy_data_dict[function_name]
    plot_single_energy_with_times(energy_data, hardware_component="cpu", graph_stable_mean=True, start_at_stable_state=True, title=True)

def rq1_plot_tail_power_states_ram():
    # function_name = "tensorflow.keras.models.Sequential.fit()"
    # dl = DataLoader("images/cnn", EXPERIMENT_DIR, ExperimentKinds.METHOD_LEVEL)
    function_name = "tensorflow.data.Dataset.from_tensor_slices()_0"
    # dl = DataLoader("load_data/numpy", EXPERIMENT_DIR, ExperimentKinds.METHOD_LEVEL)
    dl = DataLoader("load_data/numpy", EXPERIMENT_DIR, ExperimentKinds.DATA_SIZE)
    energy_data_dict = dl.load_single_file("experiment-1.json")
    energy_data = energy_data_dict[function_name]
    plot_single_energy_with_times(energy_data, hardware_component="ram", graph_stable_mean=True, start_at_stable_state=True, title=False)

def rq1_plot_tail_power_states_gpu():
    function_name = "tensorflow.keras.models.Sequential.fit()"
    dl = DataLoader("images/cnn", EXPERIMENT_DIR, ExperimentKinds.METHOD_LEVEL)
    energy_data_dict = dl.load_single_file("experiment-1.json")
    energy_data = energy_data_dict[function_name]
    plot_single_energy_with_times(energy_data, hardware_component="gpu", graph_stable_mean=True, start_at_stable_state=True, title=True)
    
### RQ 2 PLOTS
def rq2_plot_data_size_vs_energy():
    project_name = "load_data/numpy"
    # project_name = "images/cnn_fit"
    project_data, dl_list = init_project_energy_data(project_name, ExperimentKinds.DATA_SIZE, first_experiment=1, last_experiment=10)
    # print project_data object
    print("dl_list: ",dl_list)
    print(project_data.cpu_data)
    plot_total_energy_vs_data_size_boxplot(project_data, title=True)

def rq2_plot_data_size_vs_energy_scatter():
    project_name = "images/cnn_fit"
    project_data = init_project_energy_data(project_name, ExperimentKinds.DATA_SIZE, first_experiment=1, last_experiment=10)

    plot_total_energy_vs_data_size_scatter(project_data, title=True)

def rq2_plot_data_size_vs_energy_scatter_combined():
    project_data = []
    for project_name in EXECUTED_RQ2_EXPERIMENTS:
        project_data.append(init_project_energy_data(project_name, ExperimentKinds.DATA_SIZE, first_experiment=1, last_experiment=10))
    plot_total_energy_vs_data_size_scatter_combined(project_data, title=False)

def rq2_plot_data_size_vs_unnormalised_energy():
    project_name = "images/cnn_fit"
    project_data = init_project_energy_data(project_name, ExperimentKinds.DATA_SIZE, first_experiment=1, last_experiment=7)
    plot_total_unnormalised_energy_vs_data_size_boxplot(project_data, title=False)

def rq2_plot_smallest_data_size_ram():
    # if there are 10 experiments, _0 is the smallest data size
    function_name = 'tensorflow.keras.models.Sequential.fit()_0'
    dl = DataLoader("images/cnn_fit", EXPERIMENT_DIR, ExperimentKinds.DATA_SIZE)
    energy_data_dict = dl.load_single_file("experiment-1.json")
    # first sample of a data-size experiment has the smallest data size
    print(f"Total args size: {energy_data_dict[function_name].total_args_size}")
    plot_single_energy_with_times(energy_data_dict[function_name], hardware_component="ram", start_at_stable_state=True, title=False, graph_stable_mean=True)

def rq2_plot_largest_data_size_ram():
    # if there are 10 experiments, _9 is the largest data size
    function_name = 'tensorflow.keras.models.Sequential.fit()_9'
    dl = DataLoader("images/cnn_fit", EXPERIMENT_DIR, ExperimentKinds.DATA_SIZE)
    energy_data_dict = dl.load_single_file("experiment-1.json")
    # last sample of a data-size experiment has the largest data size
    print(f"Total args size: {energy_data_dict[function_name].total_args_size}")
    plot_single_energy_with_times(energy_data_dict[function_name], hardware_component="ram", start_at_stable_state=True, title=False, graph_stable_mean=True)

def rq2_plot_args_size_vs_gpu_mean():
    # this method is used for analysis of argsize vs energy for rq2 method-level dataset using rq1 data
    total_energy_dfs = []

    for project_name in EXECUTED_RQ1_EXPERIMENTS:
        print(f"Project: {project_name}")
        method_level_data = init_project_energy_data(project_name, ExperimentKinds.METHOD_LEVEL, first_experiment=1)
        total_energy_df = build_total_energy_and_size_df(method_level_data)

        # Add the project name as the first column in the DataFrame
        total_energy_df.insert(0, 'Project Name', project_name)

        if not total_energy_df.empty:
            total_energy_dfs.append(total_energy_df)

    plot_args_size_vs_gpu_mean(total_energy_dfs)

def rq2_analyze_api_groups_compute(csv_file):
    # Read the CSV file
    df = pd.read_csv(csv_file)

    # Split the APIs into two groups based on runtime
    runtime_threshold = 1  # 1 second (1000 ms)
    fast_apis = df[df['run time'] <= runtime_threshold]
    slow_apis = df[df['run time'] > runtime_threshold]

    print(f"Number of fast APIs (runtime <= {runtime_threshold} seconds): {len(fast_apis)}")
    print(f"Number of slow APIs (runtime > {runtime_threshold} seconds): {len(slow_apis)}")

    # Perform further analysis on the two groups
    analyze_group_compute(fast_apis, "Fast APIs")
    analyze_group_compute(slow_apis, "Slow APIs")

    # Plot the input data size vs. energy consumption
    plot_input_size_vs_energy(fast_apis, slow_apis)

def analyze_group_compute(group, group_name):
    print(f"\nAnalysis for {group_name}:")

    # Calculate the mean and median values for each feature
    print("Mean values:")
    print(group.mean())

    print("\nMedian values:")
    print(group.median())

    # Compute the correlation between input data size and energy consumption
    input_size_feature = "Args Size (mean)"
    energy_features = ["CPU (mean)", "RAM (mean)", "GPU (mean)"]
    significance_level = 0.05  # Choose the desired significance level

    for energy_feature in energy_features:
        corr, p_value = stats.pearsonr(group[input_size_feature], group[energy_feature])
        print(f"\nPearson correlation between {input_size_feature} and {energy_feature}: {corr:.2f}")
        print(f"p-value: {p_value:.2e}")

        # Determine the effect size based on the correlation coefficient
        if abs(corr) < 0.3:
            effect_size = "small"
        elif abs(corr) < 0.5:
            effect_size = "medium"
        else:
            effect_size = "large"
        print(f"Effect size: {effect_size}")

        # Test for statistical significance
        if p_value < significance_level:
            print("The correlation is statistically significant.")
        else:
            print("The correlation is not statistically significant.")

def rq2_analyze_api_groups_memory(csv_file):
    # Read the CSV file
    df = pd.read_csv(csv_file)

    # Split the APIs into two groups based on "Args Size (mean)"
    size_threshold = 1  # 1 MB
    small_apis = df[df['Args Size (mean)'] <= size_threshold]
    large_apis = df[df['Args Size (mean)'] > size_threshold]

    print(f"Number of small APIs (Args Size <= {size_threshold} MB): {len(small_apis)}")
    print(f"Number of large APIs (Args Size > {size_threshold} MB): {len(large_apis)}")

    # Perform further analysis on the two groups
    analyze_group_memory(small_apis, "Small APIs")
    analyze_group_memory(large_apis, "Large APIs")

    # Plot the input data size vs. energy consumption
    # plot_input_size_vs_energy(small_apis, large_apis)

def analyze_group_memory(group, group_name):
    print(f"\nAnalysis for {group_name}:")

    # Calculate the mean and median values for each feature
    print("Mean values:")
    print(group.mean())

    print("\nMedian values:")
    print(group.median())

    # Compute the correlation between input data size and energy consumption
    input_size_feature = "Args Size (mean)"
    energy_features = ["CPU (mean)", "GPU (mean)", "RAM (mean)"]
    significance_level = 0.05  # Choose the desired significance level

    for energy_feature in energy_features:
        corr, p_value = stats.pearsonr(group[input_size_feature], group[energy_feature])
        print(f"\nPearson correlation between {input_size_feature} and {energy_feature}: {corr:.2f}")
        print(f"p-value: {p_value:.2e}")

        # Determine the effect size based on the correlation coefficient
        if abs(corr) < 0.3:
            effect_size = "small"
        elif abs(corr) < 0.5:
            effect_size = "medium"
        else:
            effect_size = "large"
        print(f"Effect size: {effect_size}")

        # Test for statistical significance
        if p_value < significance_level:
            print("The correlation is statistically significant.")
        else:
            print("The correlation is not statistically significant.")

def plot_input_size_vs_energy(fast_apis, slow_apis):
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))

    for ax, energy_feature in zip(axs, ["CPU (mean)", "RAM (mean)", "GPU (mean)"]):
        ax.scatter(fast_apis["Args Size (mean)"], fast_apis[energy_feature], label="Fast APIs", alpha=0.5)
        ax.scatter(slow_apis["Args Size (mean)"], slow_apis[energy_feature], label="Slow APIs", alpha=0.5)
        ax.set_xlabel("Input Data Size (MB)")
        ax.set_ylabel(f"{energy_feature} (Joules)")
        ax.set_title(f"Input Data Size vs. {energy_feature}")
        ax.legend()

    plt.tight_layout()
    plt.savefig("input_size_vs_energy_rq2.png")
    plt.show()

if __name__ == "__main__":
    ### commented code has already been run, uncomment to replicate

    # implementation_plot_GPU_energy_with_times()
    # rq1_plot_total_energy_vs_time()
    # rq1_plot_total_energy_vs_time_combined()
    # rq1_plot_project_level_energy_vs_method_level_energy()
    # rq1_plot_tail_power_states_gpu()
    rq1_plot_tail_power_states_cpu()
    # rq1_plot_tail_power_states_ram()
    # rq2_plot_data_size_vs_energy()
    # rq2_plot_data_size_vs_energy_scatter()
    # rq2_plot_smallest_data_size_ram()
    # rq2_plot_largest_data_size_ram()
    # rq2_plot_data_size_vs_unnormalised_energy()
    # rq2_plot_data_size_vs_energy_scatter_combined()
    # rq2_plot_args_size_vs_gpu_mean()
    # rq2_plot_data_size_vs_energy()
    # rq2_analyze_api_groups_memory("/home/saurabh/code-energy-consumption/replication/rq2_analysis/rq2_analysis.csv")
    # rq2_analyze_api_groups_compute("/home/saurabh/code-energy-consumption/replication/rq2_analysis/rq2_analysis.csv")
    pass