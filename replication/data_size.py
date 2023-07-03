from tool.patching.patching_config import EXPERIMENT_DIR
from tool.experiment.experiments import DataSizeExperiment
from tool.experiment.run import run_experiments

# raise DeprecationWarning("This module needs to be updated to work with the new local execution environment")

def run_images_cnn_model_fit_datasize_experiment():
    # (1) setup like in the original project

    #### begin copied code (from images/cnn)
    import tensorflow as tf
    from tensorflow.keras import datasets, layers, models

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
    function_to_run = "obj.fit(*args, **kwargs)"
    function_args = [train_images, train_labels]
    function_kwargs = {
        "epochs": 10,
        "validation_data": (test_images, test_labels)
    }
    method_object = model
    function_signature = "tf.keras.Sequential.fit()"

    def vary_arg_sizes(fraction: float, function_args: list, function_kwargs: dict) -> tuple:
        """
        E.g. if an arg in vary_args has shape (100,10,10) and fraction=0.5, return an array of shape (50,10,10).
        So this method only scales the first dimension of the array by the given fraction.
        """
        varied_args = [arg[:int(arg.shape[0]*fraction)] for arg in function_args]
        
        return varied_args, function_kwargs


    # (4) Initialise and run the experiment
    experiment = DataSizeExperiment(
        project = "images/cnn",
        experiment_dir = EXPERIMENT_DIR,
        n_runs = 10,
        vary_arg_sizes = vary_arg_sizes,
        function_to_run = function_to_run,
        function_signature = function_signature,
        function_args = function_args,
        function_kwargs = function_kwargs,
        method_object = method_object
    )

    run_experiments(experiment, count=10, start=1)


if __name__ == "__main__":
    run_images_cnn_model_fit_datasize_experiment()