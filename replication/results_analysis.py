from pathlib import Path

from tool.experiment.analysis import init_project_energy_data, create_summary, export_summary_to_latex, build_total_energy_df
from tool.experiment.experiments import ExperimentKinds

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

# TESTING LOCAL EXECUTION

def local_vs_server_execution():
    first_exp = 1
    last_exp = 10
    project_name = "keras/classification"
    
    server_data = init_project_energy_data(project_name, ExperimentKinds.METHOD_LEVEL, first_exp, last_exp)
    summary_dfs_server = create_summary(server_data)

    local_data = init_project_energy_data(project_name, ExperimentKinds.LOCAL_EXECUTION, first_exp, last_exp)
    summary_dfs_local = create_summary(local_data)

def local_execution_keras_estimator():
    first_exp = 1
    last_exp = 10
    project_name = "estimator/keras_model_to_estimator"

    local_data = init_project_energy_data(project_name, ExperimentKinds.LOCAL_EXECUTION, first_exp, last_exp)
    summary_dfs_local = create_summary(local_data)

if __name__ == "__main__":
    # import warnings
    # warnings.simplefilter(action='ignore', category=FutureWarning)
    ### commented code has already been run, uncomment to replicate
    
    # appendix_rq1_summary_dfs_all_experiments()
    # results_rq1_total_energy_consumption()

    # local_vs_server_execution()
    local_execution_keras_estimator()
    pass