"""
Analyse the experimental results using the data structures from data.py.
"""

import pandas as pd
from typing import List, Dict, Tuple
from pathlib import Path

from tool.experiment.data import DataLoader, FunctionEnergyData, ProjectEnergyData
from tool.experiment.experiments import ExperimentKinds
from tool.patching.patching_config import EXPERIMENT_DIR


SUMMARY_DF_COLUMNS = ['function', 'exec time (s)', 'total', 'total (normalised)', 'lag time (s)', 'lag', 'lag (normalised)', 'total + lag (normalised)', 'stdev power', 'avg power']


def init_project_energy_data(project: str, experiment_kind: ExperimentKinds, first_experiment: int = 1, last_experiment: int = 10) -> ProjectEnergyData: 
    """
    Initialise a ProjectEnergyData object containing 3 lists of FunctionEnergyData objects (CPU, RAM, GPU),
    each ob which holds lists of data for one function collected over multiple experiments.
    """
    # input sanity check
    assert last_experiment >= first_experiment

    dl = DataLoader(project, EXPERIMENT_DIR, experiment_kind)
    # get all experiment data files for this project
    experiment_files = dl.get_all_data_files()

    # initialse the project energy data holder object with the number of functions in the first experiment
    function_count = len(dl.load_single_file(experiment_files[0]))
    project_energy_data = ProjectEnergyData(function_count, project, experiment_kind, (last_experiment-first_experiment+1))

    # experiment 1 has index 0, so subtract 1 from the first_experiment variable
    # TODO the current indexing into experiment_files is buggy if first_experiment is not 1!
    for exp_file in experiment_files[first_experiment-1:last_experiment]:
        # exp_data is a list of EnergyData objects for this experiment. Each list entry corresponds to a unique function executed in this experiment.
        exp_data = dl.load_single_file(exp_file)
        # the number of functions executed should be the same for every experiment, otherwise something went wrong
        assert len(exp_data) == function_count, f"{experiment_kind.value}/{project}/{exp_file} contains data for {len(exp_data)} functions, but it should contain {function_count}!"
        for function_number, function_data in enumerate(exp_data):
            # skip a function if it has no energy data, but keep track of it
            if not function_data.has_energy_data:
                project_energy_data.no_energy_functions.add(function_data.function_name)
                continue

            ### add CPU data
            # general data
            project_energy_data.cpu[function_number].name = function_data.function_name
            project_energy_data.cpu[function_number].execution_time.append(function_data.execution_time_s)
            project_energy_data.cpu[function_number].total_args_size.append(function_data.total_args_size)
            project_energy_data.cpu[function_number].total_input_size.append(function_data.total_input_size)
            # device-specific data
            project_energy_data.cpu[function_number].total.append(function_data.total_cpu)
            project_energy_data.cpu[function_number].total_normalised.append(function_data.total_cpu_normalised)
            project_energy_data.cpu[function_number].lag_time.append(function_data.cpu_lag_time)
            project_energy_data.cpu[function_number].lag.append(function_data.cpu_lag)
            project_energy_data.cpu[function_number].lag_normalised.append(function_data.cpu_lag_normalised)
            project_energy_data.cpu[function_number].total_lag_normalised.append(function_data.total_cpu_lag_normalised)
            project_energy_data.cpu[function_number].stdev_power.append(function_data.stdev_power_cpu)
            project_energy_data.cpu[function_number].mean_power.append(function_data.mean_power_cpu)
            project_energy_data.cpu[function_number].median_power.append(function_data.median_power_cpu)

            ### add RAM data
            # general data
            project_energy_data.ram[function_number].name = function_data.function_name
            project_energy_data.ram[function_number].execution_time.append(function_data.execution_time_s)
            project_energy_data.ram[function_number].total_args_size.append(function_data.total_args_size)
            project_energy_data.ram[function_number].total_input_size.append(function_data.total_input_size)
            # device-specific data
            project_energy_data.ram[function_number].total.append(function_data.total_ram)
            project_energy_data.ram[function_number].total_normalised.append(function_data.total_ram_normalised)
            project_energy_data.ram[function_number].lag_time.append(function_data.ram_lag_time)
            project_energy_data.ram[function_number].lag.append(function_data.ram_lag)
            project_energy_data.ram[function_number].lag_normalised.append(function_data.ram_lag_normalised)
            project_energy_data.ram[function_number].total_lag_normalised.append(function_data.total_ram_lag_normalised)
            project_energy_data.ram[function_number].stdev_power.append(function_data.stdev_power_ram)
            project_energy_data.ram[function_number].mean_power.append(function_data.mean_power_ram)
            project_energy_data.ram[function_number].median_power.append(function_data.median_power_ram)

            ### add GPU data
            # general data
            project_energy_data.gpu[function_number].name = function_data.function_name
            project_energy_data.gpu[function_number].execution_time.append(function_data.execution_time_s)
            project_energy_data.gpu[function_number].total_args_size.append(function_data.total_args_size)
            project_energy_data.gpu[function_number].total_input_size.append(function_data.total_input_size)
            # device-specific data
            project_energy_data.gpu[function_number].total.append(function_data.total_gpu)
            project_energy_data.gpu[function_number].total_normalised.append(function_data.total_gpu_normalised)
            project_energy_data.gpu[function_number].lag_time.append(function_data.gpu_lag_time)
            project_energy_data.gpu[function_number].lag.append(function_data.gpu_lag)
            project_energy_data.gpu[function_number].lag_normalised.append(function_data.gpu_lag_normalised)
            project_energy_data.gpu[function_number].total_lag_normalised.append(function_data.total_gpu_lag_normalised)
            project_energy_data.gpu[function_number].stdev_power.append(function_data.stdev_power_gpu)
            project_energy_data.gpu[function_number].mean_power.append(function_data.mean_power_gpu)
            project_energy_data.gpu[function_number].median_power.append(function_data.median_power_gpu)
    
    return project_energy_data


def format_df(data_list: list, column_names: List[str]) -> pd.DataFrame:
    summary_df = pd.DataFrame(data_list, columns=column_names)
    
    # add a sum row for method-level experiments where each row value is equal to the sum of all values in its column
    if len(summary_df) > 1:
        summary_df.loc[len(summary_df)] = summary_df.sum(numeric_only=True)
        summary_df.iloc[-1,0] = "method-level (sum)"
    
    # round to 2 decimal places
    return summary_df.round(2)


def build_summary_df_median(energy_data_list: List[FunctionEnergyData]) -> pd.DataFrame:
    data_list = []
    for function_data in energy_data_list:
        data_list.append([
            function_data.name,
            function_data.median_execution_time,
            function_data.median_total,
            function_data.median_total_normalised,
            function_data.median_lag_time,
            function_data.median_lag,
            function_data.median_lag_normalised,
            function_data.median_total_lag_normalised,
            function_data.median_stdev_power,
            function_data.median_median_power
        ])
    
    return format_df(data_list, SUMMARY_DF_COLUMNS)


def build_summary_df_mean(energy_data_list: List[FunctionEnergyData]) -> pd.DataFrame:
    data_list = []
    for function_data in energy_data_list:
        data_list.append([
            function_data.name,
            function_data.mean_execution_time,
            function_data.mean_total,
            function_data.mean_total_normalised,
            function_data.mean_lag_time,
            function_data.mean_lag,
            function_data.mean_lag_normalised,
            function_data.mean_total_lag_normalised,
            function_data.mean_stdev_power,
            function_data.mean_mean_power
        ])
    return format_df(data_list, SUMMARY_DF_COLUMNS)


def prepare_total_energy_from_project(project_energy_data: ProjectEnergyData) -> Tuple[List[list], List[str]]:
    """
    Given a ProjectEnergyData object, construct a list of lists for constructing a DataFrame
    that contains total normalised energy data. Also return the column names for this DataFrame.
    """
    data_list = []
    for cpu, ram, gpu in zip(project_energy_data.cpu_data, project_energy_data.ram_data, project_energy_data.gpu_data):
        assert cpu.name == ram.name and cpu.name == gpu.name, "The hardware components should list the functions in the same order."
        data_list.append([
            cpu.name,
            cpu.mean_execution_time,
            cpu.mean_total_normalised,
            cpu.median_total_normalised,
            ram.mean_total_normalised,
            ram.median_total_normalised,
            gpu.mean_total_normalised,
            gpu.median_total_normalised
        ])
    
    column_names = ["function", "run time", "CPU (mean)", "CPU (median)", "RAM (mean)", "RAM (median)", "GPU (mean)", "GPU (median)"]

    return data_list, column_names


def build_total_energy_df(method_level_energy: ProjectEnergyData, project_level_energy: ProjectEnergyData) -> pd.DataFrame:
    """
    Construct a DataFrame containing total normalised method-level and project-level energy.
    Used to evaluate RQ1.
    """
    data_list, column_names = prepare_total_energy_from_project(method_level_energy)
    total_energy_df = format_df(data_list, column_names)
    
    data_list_project_level, _ = prepare_total_energy_from_project(project_level_energy)
    total_energy_df.loc[len(total_energy_df)] = data_list_project_level[0]
    
    return total_energy_df


def export_summary_to_latex(output_dir: Path, summary_dfs: Dict[str, pd.DataFrame]):
    """
    Write a given set of summary dfs returned by create_summary() to latex files.
    """
    for name, df in summary_dfs.items():
        df.style.format(precision=2).to_latex(buf = output_dir/f"{name}.tex")


def create_summary(project_energy_data: ProjectEnergyData) -> Dict[str, pd.DataFrame]:
    cpu_summary_mean = build_summary_df_mean(project_energy_data.cpu_data)
    ram_summary_mean = build_summary_df_mean(project_energy_data.ram_data)
    gpu_summary_mean = build_summary_df_mean(project_energy_data.gpu_data)

    cpu_summary_median = build_summary_df_median(project_energy_data.cpu_data)
    ram_summary_median = build_summary_df_median(project_energy_data.ram_data)
    gpu_summary_median = build_summary_df_median(project_energy_data.gpu_data)

    print(f"\n### SUMMARY FOR {project_energy_data.name} | {project_energy_data.experiment_kind.value} ###\n")
    print("Mean CPU results (Joules)")
    print(cpu_summary_mean)
    print("\nMedian CPU results (Joules)")
    print(cpu_summary_median)
    print("\n###=======###\n")
    print("Mean RAM results (Joules)")
    print(ram_summary_mean)
    print("\nMedian RAM results (Joules)")
    print(ram_summary_median)
    print("\n###=======###\n")
    print("Mean GPU results (Joules)")
    print(gpu_summary_mean)
    print("\nMedian GPU results (Joules)")
    print(gpu_summary_median)
    print("\n###=======###\n")

    return {
        "cpu_summary_mean": cpu_summary_mean,
        "ram_summary_mean": ram_summary_mean,
        "gpu_summary_mean": gpu_summary_mean,
        "cpu_summary_median": cpu_summary_median,
        "ram_summary_median": ram_summary_median,
        "gpu_summary_median": gpu_summary_median,
    }


if __name__ == "__main__":
    project_name = "images/cnn"
    method_level_data = init_project_energy_data(project_name, ExperimentKinds.METHOD_LEVEL, first_experiment=6)
    print(method_level_data.no_energy_functions)
    print(f"Number of functions: {len(method_level_data)}")
    # create_summary(method_level_data)

    project_level_data = init_project_energy_data(project_name, ExperimentKinds.PROJECT_LEVEL, first_experiment=6)
    print(build_total_energy_df(method_level_data, project_level_data))

    method_level_energies = [method_level_data]

    project_name = "keras/classification"
    method_level_data = init_project_energy_data(project_name, ExperimentKinds.METHOD_LEVEL, first_experiment=6)
    method_level_energies.append(method_level_data)

    from tool.experiment.plot import plot_total_energy_vs_execution_time
    plot_total_energy_vs_execution_time(method_level_energies)