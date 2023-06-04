from tool.experiment.experiments import MethodLevelExperiment, MethodLevelLocalExperiment
from tool.experiment.run import run_experiments
from tool.patching.patching_config import EXPERIMENT_DIR, CODE_DIR

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

def run_images_transfer_learning_method_level():
    experiment = MethodLevelExperiment("images/transfer_learning", EXPERIMENT_DIR, CODE_DIR)
    run_experiments(experiment, count=10, start=1)

def run_text_word2vec_method_level():
    experiment = MethodLevelExperiment("text/word2vec", EXPERIMENT_DIR, CODE_DIR)
    run_experiments(experiment, count=10, start=1)

def run_interpretability_integrated_gradients_method_level():
    experiment = MethodLevelExperiment("interpretability/integrated_gradients", EXPERIMENT_DIR, CODE_DIR)
    run_experiments(experiment, count=10, start=1)

def run_customization_basics_method_level():
    experiment = MethodLevelExperiment("customization/basics", EXPERIMENT_DIR, CODE_DIR)
    run_experiments(experiment, count=10, start=1)

# LOCAL EXECUTION

def run_keras_classification_method_level_local():
    experiment = MethodLevelLocalExperiment("keras/classification", EXPERIMENT_DIR, CODE_DIR)
    run_experiments(experiment, count=10, start=1)

def run_estimator_keras_model_to_estimator_method_level_local():
    experiment = MethodLevelLocalExperiment("estimator/keras_model_to_estimator", EXPERIMENT_DIR, CODE_DIR)
    run_experiments(experiment, count=10, start=1)

if __name__ == "__main__":
    ### commented code has already been run, uncomment to replicate

    # run_keras_classification_method_level()
    # run_images_cnn_method_level()
    # run_estimator_keras_model_to_estimator_method_level()
    
    # Try later
    # run_images_data_augmentation_method_level()
    # run_images_transfer_learning_method_level()
    # run_text_word2vec_method_level()
    # run_interpretability_integrated_gradients_method_level()

    # currently running
    # run_customization_basics_method_level()

    # local execution
    # run_keras_classification_method_level_local()
    run_estimator_keras_model_to_estimator_method_level_local()
    pass