from pathlib import Path

from tool.experiment.analysis import init_project_energy_data, create_summary, export_summary_to_latex, build_total_energy_df
from tool.experiment.experiments import ExperimentKinds

LATEX_OUTPUT_PATH = Path("out/latex")

def appendix_rq1_summary_dfs():
    project_name = "keras/classification"
    data = init_project_energy_data(project_name, ExperimentKinds.METHOD_LEVEL, first_experiment=6, last_experiment=10)
    summary_dfs = create_summary(data)
    export_summary_to_latex(LATEX_OUTPUT_PATH/ExperimentKinds.METHOD_LEVEL.value/"keras-classification", summary_dfs=summary_dfs)

    project_name = "keras/classification"
    data = init_project_energy_data(project_name, ExperimentKinds.PROJECT_LEVEL, first_experiment=6, last_experiment=10)
    summary_dfs = create_summary(data)
    export_summary_to_latex(LATEX_OUTPUT_PATH/ExperimentKinds.PROJECT_LEVEL.value/"keras-classification", summary_dfs=summary_dfs)

    # TODO: currently experiments 1-5 didn't use the GPU (you can confirm this by comparing the
    # summary statistics with those of experiments 6-10). We have to re-run these experiments
    # project_name = "images/cnn"
    # data = init_project_energy_data(project_name, ExperimentKinds.METHOD_LEVEL, first_experiment=1, last_experiment=5)
    # create_summary(data)

    project_name = "images/cnn"
    data = init_project_energy_data(project_name, ExperimentKinds.METHOD_LEVEL, first_experiment=6, last_experiment=10)
    summary_dfs = create_summary(data)
    export_summary_to_latex(LATEX_OUTPUT_PATH/ExperimentKinds.METHOD_LEVEL.value/"images-cnn", summary_dfs=summary_dfs)

    # TODO: see the todo above
    # project_name = "images/cnn"
    # data = init_project_energy_data(project_name, ExperimentKinds.PROJECT_LEVEL, first_experiment=1, last_experiment=5)
    # create_summary(data)

    project_name = "images/cnn"
    data = init_project_energy_data(project_name, ExperimentKinds.PROJECT_LEVEL, first_experiment=6, last_experiment=10)
    summary_dfs = create_summary(data)
    export_summary_to_latex(LATEX_OUTPUT_PATH/data.experiment_kind.value/"images-cnn", summary_dfs=summary_dfs)


def results_rq1_total_energy_consumption():
    file_name = "total_energy_df.tex"
    sub_dir = "combined"

    project_name = "keras/classification"
    method_level  = init_project_energy_data(project_name, ExperimentKinds.METHOD_LEVEL, first_experiment=6, last_experiment=10)
    project_level = init_project_energy_data(project_name, ExperimentKinds.PROJECT_LEVEL, first_experiment=6, last_experiment=10)
    df = build_total_energy_df(method_level, project_level)
    print(df)
    df.style.format(precision=2).to_latex(buf = LATEX_OUTPUT_PATH/sub_dir/project_name.replace('/','-')/file_name)

    project_name = "images/cnn"
    method_level  = init_project_energy_data(project_name, ExperimentKinds.METHOD_LEVEL, first_experiment=6, last_experiment=10)
    project_level = init_project_energy_data(project_name, ExperimentKinds.PROJECT_LEVEL, first_experiment=6, last_experiment=10)
    df = build_total_energy_df(method_level, project_level)
    print(df)
    df.style.format(precision=2).to_latex(buf = LATEX_OUTPUT_PATH/sub_dir/project_name.replace('/','-')/file_name)


if __name__ == "__main__":
    results_rq1_total_energy_consumption()
    