from tool.experiment.experiments import MethodLevelExperiment
from tool.experiment.run import run_experiments
from tool.client.client_config import EXPERIMENT_DIR, CODE_DIR

on_hold = [
    "images/cnn",
    "generative/cvae",
    "generative/autoencoder",
    "generative/data_compression"
]

if __name__ == "__main__":
    start_number = 1
    # commented code has already been run, uncomment to replicate

    # experiment = MethodLevelExperiment("keras/classification", EXPERIMENT_DIR, CODE_DIR)
    # run_experiments(experiment, count=10, start=start_number)

    # experiment = MethodLevelExperiment("images/cnn", EXPERIMENT_DIR, CODE_DIR)
    # run_experiments(experiment, count=10, start=start_number)