from tool.experiment.analysis import init_project_energy_data, build_total_energy_df
from tool.experiment.experiments import ExperimentKinds
from executed_experiments import EXECUTED_RQ1_EXPERIMENTS

if __name__ == '__main__':
    # project_name = "images/cnn" -- DONE
    # project_name = "keras/classification" # -- DONE
    # project_name = "keras/overfit_and_underfit" # -- DONE (but weird)
    # project_name = "keras/regression" # -- ASSERTION ERROR
    # project_name = "keras/save_and_load" # -- DONE (but weird)
    # project_name = "load_data/numpy" # -- DONE
    # project_name = "quickstart/advanced" # -- ASSERTION ERROR (time elapsed)
    # project_name = "quickstart/beginner"  # -- DONE
     # project_name = "audio/transfer_learning_audio"  # -- 

    for project_name in EXECUTED_RQ1_EXPERIMENTS:
        print(f"Project: {project_name}")
        method_level_data = init_project_energy_data(project_name, ExperimentKinds.METHOD_LEVEL, first_experiment=1)
        print(method_level_data.no_energy_functions)
        print(f"Number of functions: {len(method_level_data)}")
        # create_summary(method_level_data)

        project_level_data = init_project_energy_data(project_name, ExperimentKinds.PROJECT_LEVEL, first_experiment=1)
        total_energy_df = build_total_energy_df(method_level_data, project_level_data)
        print(total_energy_df)

