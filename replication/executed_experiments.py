"""
Keep track of projects for which experiment have been executed such that
their data can be analysed and plotted.
"""

# Executed project-level and method-level experiments for RQ1,
# found in energy-dataset/project-level and energy-dataset/method-level.
EXECUTED_RQ1_EXPERIMENTS = [
    "audio/simple_audio", # using skip_calls
    "audio/transfer_learning_audio",
    "distribute/custom_training", # using skip_calls
    "estimator/keras_model_to_estimator",
    "estimator/linear",
    "estimator/premade",
    "generative/adversarial_fgsm",
    "generative/autoencoder",
    "generative/cvae", # using skip_calls
    "images/cnn",
    "keras/classification",
    "keras/overfit_and_underfit",
    "keras/regression",
    "keras/save_and_load",
    "load_data/numpy",
    "load_data/tfrecord", # using skip_calls
    "quickstart/advanced",
    "quickstart/beginner",
    "structured_data/feature_columns",
    "structured_data/imbalanced_data"
]

# Executed data-size experiments for RQ2,
# found in energy-dataset/data-size.
EXECUTED_RQ2_EXPERIMENTS = [
    "estimator/keras_model_to_estimator_train",
    "generative/autoencoder",
    "images/cnn_evaluate",
    "images/cnn_fit",
    "keras/classification_evaluate",
    "keras/classification_fit",
    "keras/regression_adapt",
    "keras/regression_predict"
    "load_data/numpy",
    "quickstart/beginner_fit"
]