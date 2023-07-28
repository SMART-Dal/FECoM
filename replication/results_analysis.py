"""
Analyse the experimental data.
"""

from pathlib import Path
from statistics import median, mean
import os

from tool.experiment.analysis import init_project_energy_data, create_summary, export_summary_to_latex, build_total_energy_df, build_total_energy_and_size_df
from tool.experiment.experiment_kinds import ExperimentKinds

from executed_experiments import EXECUTED_RQ1_EXPERIMENTS

LATEX_OUTPUT_PATH = Path("rq1_analysis/latex")

# helper method to create a summary table and save it to latex for one project
def create_summary_and_save_to_latex(project_name: str, experiment_kind: ExperimentKinds, first_experiment: int, last_experiment: int):
    data = init_project_energy_data(project_name, experiment_kind, first_experiment, last_experiment)
    summary_dfs = create_summary(data)
    export_summary_to_latex(LATEX_OUTPUT_PATH/experiment_kind.value/project_name.replace('/','-'), summary_dfs=summary_dfs)


def results_rq1_total_energy_consumption():
    """
    Compare project and method-level total normalised energy consumption for all projects
    and save the results to latex.
    """

    file_name = "total_energy_df.tex"
    sub_dir = "combined"
    first_exp = 1
    last_exp = 10

    for project_name in EXECUTED_RQ1_EXPERIMENTS:
        try:
            print("Project: ", project_name)
            method_level  = init_project_energy_data(project_name, ExperimentKinds.METHOD_LEVEL, first_experiment=first_exp, last_experiment=last_exp)
            project_level = init_project_energy_data(project_name, ExperimentKinds.PROJECT_LEVEL, first_experiment=first_exp, last_experiment=last_exp)
            df = build_total_energy_df(method_level, project_level)
            print(df)

            # Specify the file path
            file_path = os.path.join(LATEX_OUTPUT_PATH, sub_dir, project_name.replace('/', '-'), file_name)

            # Create the directory structure if it doesn't exist
            os.makedirs(os.path.dirname(file_path), exist_ok=True)

            # Save the DataFrame as LaTeX to the file
            df.style.format(precision=2).to_latex(buf=file_path)
            # df.style.format(precision=2).to_latex(buf = LATEX_OUTPUT_PATH/sub_dir/project_name.replace('/','-')/file_name)
        except Exception as e:
            print("Exception in project: ", project_name)
            raise e


def calculate_function_counts():
    """
    Calculate the number of functions with and without energy data.
    """
    energy_function_counts = []
    no_energy_function_counts = []

    for project_name in EXECUTED_RQ1_EXPERIMENTS:
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
    print("Median number of functions with energy data in one project: ", median(energy_function_counts))
    print("Median number of functions without energy data in one project: ", median(no_energy_function_counts))


def calculate_no_energy_functions_execution_time_stats():
    """
    Calculate statistics about the execution times of functions without energy data.
    """
    stdev_times = []
    median_times = []
    range_times = []
    min_times = []
    max_times = []

    for project_name in EXECUTED_RQ1_EXPERIMENTS:
        try:
            project_energy_data = init_project_energy_data(project_name, ExperimentKinds.METHOD_LEVEL, first_experiment=1, last_experiment=10)
            # this data is not available for projects that use skip_calls
            if project_energy_data.skip_calls:
                continue
            execution_time_stats = project_energy_data.no_energy_functions_execution_time_stats
            stdev_times.extend([stats_tuple[0] for stats_tuple in execution_time_stats])
            median_times.extend([stats_tuple[1] for stats_tuple in execution_time_stats])
            range_times.extend([stats_tuple[2] for stats_tuple in execution_time_stats])
            min_times.extend([stats_tuple[3] for stats_tuple in execution_time_stats])
            max_times.extend([stats_tuple[4] for stats_tuple in execution_time_stats])
        except Exception as e:
            print("Exception in project: ", project_name)
            raise e
    
    print("Median stdev of execution times: ", round(mean(stdev_times), 3))
    print("Median execution time: ", round(median(median_times), 3))
    print("Median range of execution times: ", round(median(range_times), 3))
    print("Min execution time: ", round(min(min_times), 3))
    print("Max execution time: ", round(max(max_times), 3))


def collect_significant_energy_consumption_functions():
    """
    Create a CSV file of all functions that consume a significant amount of energy.
    This is useful for finding functions that are worth investigating further with
    data size experiments for RQ2.
    """
    csv_file = './rq2_analysis/rq2_analysis.csv'
    if os.path.exists(csv_file):
        os.remove(csv_file)

    for project_name in EXECUTED_RQ1_EXPERIMENTS:
        print(f"Project: {project_name}")
        method_level_data = init_project_energy_data(project_name, ExperimentKinds.METHOD_LEVEL, first_experiment=1)
        # create_summary(method_level_data)

        total_energy_df = build_total_energy_and_size_df(method_level_data)
        print(total_energy_df)

        # Add the project name as the first column in the DataFrame
        total_energy_df.insert(0, 'Project Name', project_name)

        # Append the data to rq2_analysis.csv if the file already exists, otherwise create a new file
        if os.path.exists(csv_file):
            total_energy_df.to_csv(csv_file, mode='a', index=False, header=False)
        else:
            total_energy_df.to_csv(csv_file, index=False)
    

if __name__ == "__main__":
    ### below code can be used to suppress certain warnings occuring in pandas
    # import warnings
    # warnings.simplefilter(action='ignore', category=FutureWarning)
    
    ### commented code has already been run, uncomment to replicate
    
    # appendix_rq1_summary_dfs_all_experiments()
    # results_rq1_total_energy_consumption()
    # calculate_function_counts()
    # calculate_no_energy_functions_execution_time_stats()
    collect_significant_energy_consumption_functions()
    pass