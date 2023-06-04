import tensorflow as tf
from tensorflow.keras import datasets, layers, models

from tool.measurement.function_details import build_function_details
from tool.patching.patching_config import EXPERIMENT_DIR
from tool.experiment.experiments import DataSizeExperiment
from tool.experiment.run import run_experiments

raise DeprecationWarning("This module needs to be updated to work with the new local execution environment")

def run_keras_classification_model_fit_datasize_experiment():
    # (1) setup like in the original project

    #### begin copied code (from keras/classification)
    (train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()

    # Normalize pixel values to be between 0 and 1
    train_images, test_images = train_images / 255.0, test_images / 255.0

    model = models.Sequential()
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.Flatten())
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(10))

    model.compile(optimizer='adam',
                loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                metrics=['accuracy'])

    

    ## commented out relevant method call
    # history = model.fit(train_images, train_labels, epochs=10, 
    #                     validation_data=(test_images, test_labels))
    ## end comment

    #### end copied code

    # (2) check data as described in tensorflow documentation: https://www.tensorflow.org/api_docs/python/tf/keras/datasets/cifar10/load_data
    assert train_images.shape == (50000, 32, 32, 3)
    assert test_images.shape == (10000, 32, 32, 3)
    assert train_labels.shape == (50000, 1)
    assert test_labels.shape == (10000, 1)

    # (3) build function details for function
    function_details = build_function_details(
        imports = "import tensorflow as tf\nfrom tensorflow.keras import datasets, layers, models",
        function_to_run = "obj.fit(*args,**kwargs)",
        kwargs = {
            "epochs": 10,
            "validation_data": (test_images, test_labels)
        },
        method_object = model,
        max_wait_secs = MAX_WAIT_S,
        wait_after_run_secs = WAIT_AFTER_RUN_S,
        object_signature = "tf.keras.Sequential"
    )

    vary_args = [train_images, train_labels]

    # (4) Initialise and run the experiment
    experiment = DataSizeExperiment(
        project = "keras/classification",
        experiment_dir = EXPERIMENT_DIR,
        n_runs = 10,
        function_details = function_details,
        vary_args = vary_args,
        start_at = 6
    )

    # TODO continue with running experiments 4-10, already reflected in settings below
    run_experiments(experiment, count=1, start=9)

    experiment = DataSizeExperiment(
        project = "keras/classification",
        experiment_dir = EXPERIMENT_DIR,
        n_runs = 10,
        function_details = function_details,
        vary_args = vary_args
    )
    run_experiments(experiment, count=1, start=10)


if __name__ == "__main__":
    run_keras_classification_model_fit_datasize_experiment()