from tool.experiment.analysis import init_project_energy_data, create_summary
from tool.experiment.experiments import ExperimentKinds

if __name__ == "__main__":
    project_name = "keras/classification"
    data = init_project_energy_data(project_name, ExperimentKinds.METHOD_LEVEL, first_experiment=6, last_experiment=10)
    create_summary(data)

    project_name = "keras/classification"
    data = init_project_energy_data(project_name, ExperimentKinds.PROJECT_LEVEL, first_experiment=6, last_experiment=10)
    create_summary(data)

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