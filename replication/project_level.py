from tool.experiment.experiments import ProjectLevelExperiment
from tool.experiment.run import run_experiments
from tool.client.client_config import EXPERIMENT_DIR, CODE_DIR, MAX_WAIT_S, WAIT_AFTER_RUN_S

if __name__ == "__main__":
    start_number = 1
    # commented code has already been run, uncomment to replicate

    # experiment = ProjectLevelExperiment("keras/classification", EXPERIMENT_DIR, CODE_DIR, MAX_WAIT_S, WAIT_AFTER_RUN_S)
    # run_experiments(experiment, count=10, start=start_number)

    # experiment = ProjectLevelExperiment("images/cnn", EXPERIMENT_DIR, CODE_DIR, MAX_WAIT_S, WAIT_AFTER_RUN_S)
    # run_experiments(experiment, count=10, start=start_number)