from tool.experiment.experiments import MethodLevelExperiment
from tool.experiment.run import run_experiments
from tool.client.client_config import EXPERIMENT_DIR, CODE_DIR

on_hold = [
    "generative/cvae",
    "generative/autoencoder",
    "generative/data_compression"
]


def run_keras_classification_method_level():
    experiment = MethodLevelExperiment("keras/classification", EXPERIMENT_DIR, CODE_DIR)
    run_experiments(experiment, count=10, start=1)

def run_images_cnn_method_level():
    experiment = MethodLevelExperiment("images/cnn", EXPERIMENT_DIR, CODE_DIR)
    run_experiments(experiment, count=10, start=1)
    
def run_estimator_keras_model_to_estimator_method_level():
    experiment = MethodLevelExperiment("estimator/keras_model_to_estimator", EXPERIMENT_DIR, CODE_DIR)
    run_experiments(experiment, count=10, start=1)

def run_images_data_augmentation_method_level():
    experiment = MethodLevelExperiment("images/data_augmentation", EXPERIMENT_DIR, CODE_DIR)
    run_experiments(experiment, count=10, start=1)


if __name__ == "__main__":
    ### commented code has already been run, uncomment to replicate

    # run_keras_classification_method_level()
    # run_images_cnn_method_level()
    # run_images_data_augmentation_method_level()

    # below need some fixing
    run_estimator_keras_model_to_estimator_method_level()
    pass