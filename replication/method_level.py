from tool.experiment.experiments import MethodLevelExperiment
from tool.experiment.run import run_experiments
from tool.client.client_config import EXPERIMENT_DIR, CODE_DIR

if __name__ == "__main__":
    keras_classification = MethodLevelExperiment("keras/classification", EXPERIMENT_DIR, CODE_DIR)
    run_experiments(keras_classification, 5)