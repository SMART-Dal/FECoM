from tool.experiment.experiments import ProjectLevelExperiment
from tool.experiment.run import run_experiments
from tool.patching.patching_config import EXPERIMENT_DIR, CODE_DIR, MAX_WAIT_S, WAIT_AFTER_RUN_S

def run_keras_classification_project_level():
    experiment = ProjectLevelExperiment("keras/classification", EXPERIMENT_DIR, CODE_DIR, MAX_WAIT_S, WAIT_AFTER_RUN_S)
    run_experiments(experiment, count=10, start=1)

def run_images_cnn_project_level():
    experiment = ProjectLevelExperiment("images/cnn", EXPERIMENT_DIR, CODE_DIR, MAX_WAIT_S, WAIT_AFTER_RUN_S)
    run_experiments(experiment, count=10, start=1)

if __name__ == "__main__":
    ### commented code has already been run, uncomment to replicate

    # run_keras_classification_project_level()
    # run_images_cnn_project_level()
    pass