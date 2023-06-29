from pathlib import Path
from statistics import median, mean

from tool.experiment.analysis import init_project_energy_data, create_summary, export_summary_to_latex, build_total_energy_df
from tool.experiment.experiments import ExperimentKinds

from executed_experiments import EXECUTED_EXPERIMENTS

LATEX_OUTPUT_PATH = Path("out/latex")

def create_summary_and_save_to_latex(project_name: str, experiment_kind: ExperimentKinds, first_experiment: int, last_experiment: int):
    data = init_project_energy_data(project_name, experiment_kind, first_experiment, last_experiment)
    summary_dfs = create_summary(data)
    export_summary_to_latex(LATEX_OUTPUT_PATH/experiment_kind.value/project_name.replace('/','-'), summary_dfs=summary_dfs)


def appendix_rq1_summary_dfs_all_experiments():
    first_exp = 1
    last_exp = 10
    project_name = "keras/classification"
    create_summary_and_save_to_latex(project_name, ExperimentKinds.METHOD_LEVEL, first_exp, last_exp)
    create_summary_and_save_to_latex(project_name, ExperimentKinds.PROJECT_LEVEL, first_exp, last_exp)

    project_name = "images/cnn"
    create_summary_and_save_to_latex(project_name, ExperimentKinds.METHOD_LEVEL, first_exp, last_exp)
    create_summary_and_save_to_latex(project_name, ExperimentKinds.PROJECT_LEVEL, first_exp, last_exp)


def results_rq1_total_energy_consumption():
    file_name = "total_energy_df.tex"
    sub_dir = "combined"
    first_exp = 1
    last_exp = 10

    project_name = "keras/classification"
    method_level  = init_project_energy_data(project_name, ExperimentKinds.METHOD_LEVEL, first_experiment=first_exp, last_experiment=last_exp)
    project_level = init_project_energy_data(project_name, ExperimentKinds.PROJECT_LEVEL, first_experiment=first_exp, last_experiment=last_exp)
    df = build_total_energy_df(method_level, project_level)
    print(df)
    df.style.format(precision=2).to_latex(buf = LATEX_OUTPUT_PATH/sub_dir/project_name.replace('/','-')/file_name)

    project_name = "images/cnn"
    method_level  = init_project_energy_data(project_name, ExperimentKinds.METHOD_LEVEL, first_experiment=first_exp, last_experiment=last_exp)
    project_level = init_project_energy_data(project_name, ExperimentKinds.PROJECT_LEVEL, first_experiment=first_exp, last_experiment=last_exp)
    df = build_total_energy_df(method_level, project_level)
    print(df)
    df.style.format(precision=2).to_latex(buf = LATEX_OUTPUT_PATH/sub_dir/project_name.replace('/','-')/file_name)


def calculate_function_counts():
    """
    Calculate the number of functions with and without energy data.
    """
    energy_function_counts = []
    no_energy_function_counts = []

    for project_name in EXECUTED_EXPERIMENTS:
        try:
            project_energy_data = init_project_energy_data(project_name, ExperimentKinds.METHOD_LEVEL, first_experiment=1, last_experiment=10)
            energy_function_counts.append(project_energy_data.energy_function_count)
            no_energy_function_counts.append(project_energy_data.no_energy_function_count)
        except Exception as e:
            print("Exception in project: ", project_name)
            raise e
    
    print("Total number of projects: ", len(energy_function_counts))
    print("Total number of functions with energy data: ", sum(energy_function_counts))
    print("Total number of functions without energy data: ", sum(no_energy_function_counts))
    print("Median number of functions with energy data: ", median(energy_function_counts))
    print("Median number of functions without energy data: ", median(no_energy_function_counts))

def calculate_no_energy_functions_execution_time_stats():
    stdev_times = []
    median_times = []
    range_times = []
    min_times = []
    max_times = []

    for project_name in EXECUTED_EXPERIMENTS:
        try:
            project_energy_data = init_project_energy_data(project_name, ExperimentKinds.METHOD_LEVEL, first_experiment=1, last_experiment=10)
            execution_time_stats = project_energy_data.no_energy_functions_execution_time_stats
            stdev_times.extend([stats_tuple[0] for stats_tuple in execution_time_stats])
            median_times.extend([stats_tuple[1] for stats_tuple in execution_time_stats])
            range_times.extend([stats_tuple[2] for stats_tuple in execution_time_stats])
            min_times.extend([stats_tuple[3] for stats_tuple in execution_time_stats])
            max_times.extend([stats_tuple[4] for stats_tuple in execution_time_stats])
        except Exception as e:
            print("Exception in project: ", project_name)
            raise e
    
    print(stdev_times)
    print("Median stdev of execution times: ", round(mean(stdev_times), 3))
    print("Median execution time: ", round(median(median_times), 3))
    print("Median range of execution times: ", round(median(range_times), 3))
    print("Min execution time: ", round(min(min_times), 3))
    print("Max execution time: ", round(max(max_times), 3))
    

if __name__ == "__main__":
    # import warnings
    # warnings.simplefilter(action='ignore', category=FutureWarning)
    ### commented code has already been run, uncomment to replicate
    
    # appendix_rq1_summary_dfs_all_experiments()
    # results_rq1_total_energy_consumption()

    # calculate_function_counts()
    calculate_no_energy_functions_execution_time_stats()
    pass