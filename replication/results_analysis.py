from pathlib import Path

from tool.experiment.analysis import init_project_energy_data, create_summary, export_summary_to_latex
from tool.experiment.experiments import ExperimentKinds

LATEX_OUTPUT_PATH = Path("out/latex")

if __name__ == "__main__":
    project_name = "keras/classification"
    data = init_project_energy_data(project_name, ExperimentKinds.METHOD_LEVEL, first_experiment=6, last_experiment=10)
    create_summary(data)
    export_summary_to_latex(data, LATEX_OUTPUT_PATH/"keras-classification-method")

    project_name = "keras/classification"
    data = init_project_energy_data(project_name, ExperimentKinds.PROJECT_LEVEL, first_experiment=6, last_experiment=10)
    create_summary(data)
    # TODO export to latex, too.

    # TODO: currently experiments 1-5 didn't use the GPU (you can confirm this by comparing the
    # summary statistics with those of experiments 6-10). We have to re-run these experiments
    # project_name = "images/cnn"
    # data = init_project_energy_data(project_name, ExperimentKinds.METHOD_LEVEL, first_experiment=1, last_experiment=5)
    # create_summary(data)

    project_name = "images/cnn"
    data = init_project_energy_data(project_name, ExperimentKinds.METHOD_LEVEL, first_experiment=6, last_experiment=10)
    create_summary(data)

    # TODO: see the todo above
    # project_name = "images/cnn"
    # data = init_project_energy_data(project_name, ExperimentKinds.PROJECT_LEVEL, first_experiment=1, last_experiment=5)
    # create_summary(data)

    project_name = "images/cnn"
    data = init_project_energy_data(project_name, ExperimentKinds.PROJECT_LEVEL, first_experiment=6, last_experiment=10)
    create_summary(data)