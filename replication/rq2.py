"""
Run data size experiments for RQ2.
"""

from tool.patching.patching_config import EXPERIMENT_DIR
from tool.experiment.experiments import DataSizeExperiment
from tool.experiment.run import run_experiments


# This is for energy consumption of tensorflow.keras.models.Sequential.fit() api in images/cnn
def run_images_cnn_model_fit_datasize_experiment():
    # (1) create prepare_experiment function
    def prepare_experiment(fraction: float):
        # (1a) setup like in the original project

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

        # (1b) check data as described in tensorflow documentation: https://www.tensorflow.org/api_docs/python/tf/keras/datasets/cifar10/load_data
        assert train_images.shape == (50000, 32, 32, 3)
        assert test_images.shape == (10000, 32, 32, 3)
        assert train_labels.shape == (50000, 1)
        assert test_labels.shape == (10000, 1)

        # (1c) build function details for function
        original_args = [train_images, train_labels]
        function_kwargs = {
            "epochs": 10,
            "validation_data": (test_images, test_labels)
        }
        method_object = model

        # (1d) vary the data size
        # E.g. if an arg in vary_args has shape (100,10,10) and fraction=0.5, return an array of shape (50,10,10).
        # So this method only scales the first dimension of the array by the given fraction.
        function_args = [arg[:int(arg.shape[0]*fraction)] for arg in original_args]

        return function_args, function_kwargs, method_object
    
    # (2) create function details
    function_to_run = "obj.fit(*args, **kwargs)"
    function_signature = "tensorflow.keras.models.Sequential.fit()"

    # (3) Initialise and run the experiment
    experiment = DataSizeExperiment(
        project = "images/cnn",
        experiment_dir = EXPERIMENT_DIR,
        n_runs = 10,
        prepare_experiment = prepare_experiment,
        function_to_run = function_to_run,
        function_signature = function_signature
    )

    run_experiments(experiment, count=10, start=1)

# This is for energy consumption of tensorflow.keras.models.Model.fit() api in generative/autoencoder
def run_generative_autoencoder_fit_datasize_experiment():
    
    # (1) create prepare_experiment function
    def prepare_experiment(fraction: float):
        # (1a) setup like in the original project

        #### begin copied code (from generative/autoencoder)
        import tensorflow as tf
        from tensorflow.keras import layers, losses
        from tensorflow.keras.datasets import fashion_mnist
        from tensorflow.keras.models import Model
        (x_train, _), (x_test, _) = fashion_mnist.load_data()

        x_train = x_train.astype('float32') / 255.
        x_test = x_test.astype('float32') / 255.

        print (x_train.shape)
        print (x_test.shape)
        latent_dim = 64 

        class Autoencoder(Model):
            def __init__(self, latent_dim):
                super(Autoencoder, self).__init__()
                self.latent_dim = latent_dim   
                self.encoder = tf.keras.Sequential([
                layers.Flatten(),
                layers.Dense(latent_dim, activation='relu'),
                ])
                self.decoder = tf.keras.Sequential([
                layers.Dense(784, activation='sigmoid'),
                layers.Reshape((28, 28))
                ])

            def call(self, x):
                encoded = self.encoder(x)
                decoded = self.decoder(encoded)
                return decoded
        
        autoencoder = Autoencoder(latent_dim) 
        autoencoder.compile(optimizer='adam', loss=losses.MeanSquaredError())
        ## commented out relevant method call
        # autoencoder.fit(x_train, x_train,
        #                 epochs=10,
        #                 shuffle=True,
        #                 validation_data=(x_test, x_test))
        ## end comment
        #### end copied code

        # (1b) check data as described in tensorflow documentation: https://www.tensorflow.org/api_docs/python/tf/keras/datasets/fashion_mnist/load_data
        assert x_train.shape == (60000, 28, 28)
        assert x_test.shape == (10000, 28, 28)

        # (1c) build function details for function
        original_args = [x_train, x_train]
        function_kwargs = {
            "epochs": 10,
            "shuffle": True,
            "validation_data": (x_test, x_test)
        }
        method_object = autoencoder

        # (1d) vary the data size
        # E.g. if an arg in vary_args has shape (100,10,10) and fraction=0.5, return an array of shape (50,10,10).
        # So this method only scales the first dimension of the array by the given fraction.
        function_args = [arg[:int(arg.shape[0]*fraction)] for arg in original_args]

        return function_args, function_kwargs, method_object
    
    # (2) create function details
    function_to_run = "obj.fit(*args, **kwargs)"
    function_signature = "tensorflow.keras.models.Model.fit()"

    # (3) Initialise and run the experiment
    experiment = DataSizeExperiment(
        project = "generative/autoencoder",
        experiment_dir = EXPERIMENT_DIR,
        n_runs = 10,
        prepare_experiment = prepare_experiment,
        function_to_run = function_to_run,
        function_signature = function_signature
    )

    run_experiments(experiment, count=10, start=1)

# This is for energy consumption of tensorflow.keras.models.Sequential.evaluate() api in images/cnn
def run_images_cnn_model_evaluate_datasize_experiment():
    # (1) create prepare_experiment function
    def prepare_experiment(fraction: float):
        # (1a) setup like in the original project

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

        history = model.fit(train_images, train_labels, epochs=10, 
                            validation_data=(test_images, test_labels))
        
        ## commented out relevant method call
        # test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)
        ## end comment

        #### end copied code

        # (1b) check data as described in tensorflow documentation: https://www.tensorflow.org/api_docs/python/tf/keras/datasets/cifar10/load_data
        assert train_images.shape == (50000, 32, 32, 3)
        assert test_images.shape == (10000, 32, 32, 3)
        assert train_labels.shape == (50000, 1)
        assert test_labels.shape == (10000, 1)

        # (1c) build function details for function
        original_args = [test_images, test_labels]
        function_kwargs = {
            "verbose": 2
        }
        method_object = model

        # (1d) vary the data size
        # E.g. if an arg in vary_args has shape (100,10,10) and fraction=0.5, return an array of shape (50,10,10).
        # So this method only scales the first dimension of the array by the given fraction.
        function_args = [arg[:int(arg.shape[0]*fraction)] for arg in original_args]

        return function_args, function_kwargs, method_object
    
    # (2) create function details
    function_to_run = "obj.evaluate(*args, **kwargs)"
    function_signature = "tensorflow.keras.models.Sequential.evaluate()"

    # (3) Initialise and run the experiment
    experiment = DataSizeExperiment(
        project = "images/cnn_evaluate",
        experiment_dir = EXPERIMENT_DIR,
        n_runs = 10,
        prepare_experiment = prepare_experiment,
        function_to_run = function_to_run,
        function_signature = function_signature
    )

    run_experiments(experiment, count=10, start=1)

# This is for energy consumption of tensorflow.keras.models.Sequential.fit() api in quickstart/beginner
def run_quickstart_beginner_model_fit_datasize_experiment():
    # (1) create prepare_experiment function
    def prepare_experiment(fraction: float):
        # (1a) setup like in the original project

        #### begin copied code (from quickstart/beginner)
        import tensorflow as tf
        print("TensorFlow version:", tf.__version__)
        mnist = tf.keras.datasets.mnist

        (x_train, y_train), (x_test, y_test) = mnist.load_data()
        x_train, x_test = x_train / 255.0, x_test / 255.0
        model = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(input_shape=(28, 28)),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(10)
        ])
        predictions = model(x_train[:1]).numpy()
        predictions
        tf.nn.softmax(predictions).numpy()
        loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        loss_fn(y_train[:1], predictions).numpy()
        model.compile(optimizer='adam',
                    loss=loss_fn,
                    metrics=['accuracy'])
        
        ## commented out relevant method call
        # model.fit(x_train, y_train, epochs=5)
        ## end comment

        #### end copied code

        # (1b) check data as described in tensorflow documentation: https://www.tensorflow.org/api_docs/python/tf/keras/datasets/mnist/load_data
        assert x_train.shape == (60000, 28, 28)
        assert x_test.shape == (10000, 28, 28)
        assert y_train.shape == (60000,)
        assert y_test.shape == (10000,)

        # (1c) build function details for function
        original_args = [x_train, y_train]
        function_kwargs = {
            "epochs": 5
        }
        method_object = model

        # (1d) vary the data size
        # E.g. if an arg in vary_args has shape (100,10,10) and fraction=0.5, return an array of shape (50,10,10).
        # So this method only scales the first dimension of the array by the given fraction.
        function_args = [arg[:int(arg.shape[0]*fraction)] for arg in original_args]

        return function_args, function_kwargs, method_object
    
    # (2) create function details
    function_to_run = "obj.fit(*args, **kwargs)"
    function_signature = "tensorflow.keras.models.Sequential.fit()"

    # (3) Initialise and run the experiment
    experiment = DataSizeExperiment(
        project = "quickstart/beginner_fit",
        experiment_dir = EXPERIMENT_DIR,
        n_runs = 10,
        prepare_experiment = prepare_experiment,
        function_to_run = function_to_run,
        function_signature = function_signature
    )

    run_experiments(experiment, count=10, start=1)

# This is for energy consumption of tensorflow.data.Dataset.from_tensor_slices api in load_data/numpy
def run_loaddata_numpy_Dataset_from_tensor_slices_datasize_experiment():
    
    # (1) create prepare_experiment function
    def prepare_experiment(fraction: float):
        # (1a) setup like in the original project

        #### begin copied code (from quickstart/beginner)
        import numpy as np
        import tensorflow as tf
        DATA_URL = 'https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz'

        path = tf.keras.utils.get_file('mnist.npz', DATA_URL)
        with np.load(path) as data:
            train_examples = data['x_train']
            train_labels = data['y_train']
            test_examples = data['x_test']
            test_labels = data['y_test']

        ## commented out relevant method call    
        # train_dataset = tf.data.Dataset.from_tensor_slices((train_examples, train_labels))
        ## end comment

        #### end copied code

        # (1b) check data as described in tensorflow documentation: https://www.tensorflow.org/api_docs/python/tf/keras/datasets/mnist/load_data
        assert train_examples.shape == (60000, 28, 28)
        assert test_examples.shape == (10000, 28, 28)
        assert train_labels.shape == (60000,)
        assert test_labels.shape == (10000,)

        # (1c) build function details for function
        original_args = [(train_examples, train_labels)]
        function_kwargs = None
        method_object = None

        # (1d) vary the data size
        # E.g. if an arg in vary_args has shape (100,10,10) and fraction=0.5, return an array of shape (50,10,10).
        # So this method only scales the first dimension of the array by the given fraction.
        function_args = [(arg[0][:int(arg[0].shape[0]*fraction)], arg[1][:int(arg[1].shape[0]*fraction)]) for arg in original_args]

        return function_args, function_kwargs, method_object
    
    # (2) create function details
    function_to_run = "tf.data.Dataset.from_tensor_slices(*args)"
    function_signature = "tensorflow.data.Dataset.from_tensor_slices()"

    # (3) Initialise and run the experiment
    experiment = DataSizeExperiment(
        project = "load_data/numpy",
        experiment_dir = EXPERIMENT_DIR,
        n_runs = 10,
        prepare_experiment = prepare_experiment,
        function_to_run = function_to_run,
        function_signature = function_signature,
        imports = "import tensorflow as tf"
    )

    run_experiments(experiment, count=10, start=1)

# This is for energy consumption of tensorflow.keras.Sequential.fit() api in keras/classification
def run_keras_classification_sequential_fit_datasize_experiment():
    # (1) create prepare_experiment function
    def prepare_experiment(fraction: float):
        # (1a) setup like in the original project

        #### begin copied code (from keras/classification)
        import tensorflow as tf

        import numpy as np
        import matplotlib.pyplot as plt

        print(tf.__version__)
        fashion_mnist = tf.keras.datasets.fashion_mnist

        (train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
        ## commented out irrelevant data visualisation code
        # class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
        #             'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
        # train_images.shape
        # len(train_labels)
        # train_labels
        # test_images.shape
        # len(test_labels)
        # plt.figure()
        # plt.imshow(train_images[0])
        # plt.colorbar()
        # plt.grid(False)
        # plt.show()
        ## end comment
        train_images = train_images / 255.0

        test_images = test_images / 255.0

        ## commented out irrelevant data visualisation code
        # plt.figure(figsize=(10,10))
        # for i in range(25):
        #     plt.subplot(5,5,i+1)
        #     plt.xticks([])
        #     plt.yticks([])
        #     plt.grid(False)
        #     plt.imshow(train_images[i], cmap=plt.cm.binary)
        #     plt.xlabel(class_names[train_labels[i]])
        # plt.show()
        ## end comment

        model = tf.keras.Sequential([
            tf.keras.layers.Flatten(input_shape=(28, 28)),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dense(10)
        ])
        model.compile(optimizer='adam',
                    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                    metrics=['accuracy'])
        
        ## commented out relevant method call
        # model.fit(train_images, train_labels, epochs=10)
        ## end comment

        #### end copied code

        # (1b) check data as described in tensorflow documentation: https://www.tensorflow.org/api_docs/python/tf/keras/datasets/fashion_mnist/load_data
        assert train_images.shape == (60000, 28, 28)
        assert test_images.shape == (10000, 28, 28)
        assert train_labels.shape == (60000,)
        assert test_labels.shape == (10000,)

        # (1c) build function details for function
        original_args = [train_images, train_labels]
        function_kwargs = {
            "epochs": 10
        }
        method_object = model

        # (1d) vary the data size
        # E.g. if an arg in vary_args has shape (100,10,10) and fraction=0.5, return an array of shape (50,10,10).
        # So this method only scales the first dimension of the array by the given fraction.
        function_args = [arg[:int(arg.shape[0]*fraction)] for arg in original_args]

        return function_args, function_kwargs, method_object
    
    # (2) create function details
    function_to_run = "obj.fit(*args, **kwargs)"
    function_signature = "tensorflow.keras.Sequential.fit()"

    # (3) Initialise and run the experiment
    experiment = DataSizeExperiment(
        project = "keras/classification_fit",
        experiment_dir = EXPERIMENT_DIR,
        n_runs = 10,
        prepare_experiment = prepare_experiment,
        function_to_run = function_to_run,
        function_signature = function_signature
    )

    run_experiments(experiment, count=10, start=1)


# This is for energy consumption of tensorflow.keras.Sequential.evaluate() api in keras/classification
def run_keras_classification_sequential_evaluate_datasize_experiment():
    # (1) create prepare_experiment function
    def prepare_experiment(fraction: float):
        # (1a) setup like in the original project

        #### begin copied code (from keras/classification)
        import tensorflow as tf

        import numpy as np
        import matplotlib.pyplot as plt

        print(tf.__version__)
        fashion_mnist = tf.keras.datasets.fashion_mnist

        (train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
        ## commented out irrelevant data visualisation code
        # class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
        #             'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
        # train_images.shape
        # len(train_labels)
        # train_labels
        # test_images.shape
        # len(test_labels)
        # plt.figure()
        # plt.imshow(train_images[0])
        # plt.colorbar()
        # plt.grid(False)
        # plt.show()
        ## end comment
        train_images = train_images / 255.0

        test_images = test_images / 255.0

        ## commented out irrelevant data visualisation code
        # plt.figure(figsize=(10,10))
        # for i in range(25):
        #     plt.subplot(5,5,i+1)
        #     plt.xticks([])
        #     plt.yticks([])
        #     plt.grid(False)
        #     plt.imshow(train_images[i], cmap=plt.cm.binary)
        #     plt.xlabel(class_names[train_labels[i]])
        # plt.show()
        ## end comment

        model = tf.keras.Sequential([
            tf.keras.layers.Flatten(input_shape=(28, 28)),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dense(10)
        ])
        model.compile(optimizer='adam',
                    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                    metrics=['accuracy'])
        
        model.fit(train_images, train_labels, epochs=10)
        
        ## commented out relevant method call
        # test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)
        ## end comment

        #### end copied code

        # (1b) check data as described in tensorflow documentation: https://www.tensorflow.org/api_docs/python/tf/keras/datasets/fashion_mnist/load_data
        assert train_images.shape == (60000, 28, 28)
        assert test_images.shape == (10000, 28, 28)
        assert train_labels.shape == (60000,)
        assert test_labels.shape == (10000,)

        # (1c) build function details for function
        original_args = [test_images, test_labels]
        function_kwargs = {
            "verbose": 2
        }
        method_object = model

        # (1d) vary the data size
        # E.g. if an arg in vary_args has shape (100,10,10) and fraction=0.5, return an array of shape (50,10,10).
        # So this method only scales the first dimension of the array by the given fraction.
        function_args = [arg[:int(arg.shape[0]*fraction)] for arg in original_args]

        return function_args, function_kwargs, method_object
    
    # (2) create function details
    function_to_run = "obj.evaluate(*args, **kwargs)"
    function_signature = "tensorflow.keras.Sequential.evaluate()"

    # (3) Initialise and run the experiment
    experiment = DataSizeExperiment(
        project = "keras/classification_evaluate",
        experiment_dir = EXPERIMENT_DIR,
        n_runs = 10,
        prepare_experiment = prepare_experiment,
        function_to_run = function_to_run,
        function_signature = function_signature
    )

    run_experiments(experiment, count=10, start=1)


# This is for energy consumption of tensorflow.keras.estimator.model_to_estimator.train() api in estimator/keras_model_to_estimator
def run_estimator_keras_model_to_estimator_train_datasize_experiment():
    # (1) create prepare_experiment function
    def prepare_experiment(fraction: float):
        # (1a) setup like in the original project

        #### begin copied code (from keras/classification)
        import tensorflow as tf

        import numpy as np
        import tensorflow_datasets as tfds
        model = tf.keras.models.Sequential([
            tf.keras.layers.Dense(16, activation='relu', input_shape=(4,), name="dense"), ## added name to layer to prevent auto-generated names with incremented numbers
            tf.keras.layers.Dropout(0.2, name="dropout"), ## added name to layer to prevent auto-generated names with incremented numbers
            tf.keras.layers.Dense(3, name="output") ## added name to layer to prevent auto-generated names with incremented numbers
        ],
        name="sequential_model") ## added name to layer to prevent auto-generated names with incremented numbers
        model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                    optimizer='adam')
        model.summary()
        def input_fn():
            import tensorflow_datasets as tfds

            ## MODIFIED CODE (1d) vary the data size (due to the way the data is loaded with an input_fn, we need to vary the data size here)
            # split = tfds.Split.TRAIN
            split = 'train[:{}%]'.format(int(fraction * 100))
            ## END MODIFIED CODE
        
            dataset = tfds.load('iris', split=split, as_supervised=True)
            dataset = dataset.map(lambda features, labels: ({'dense_input':features}, labels))
            dataset = dataset.batch(32).repeat()
            return dataset
        
        ## commented out irrelevant data visualisation code
        # for features_batch, labels_batch in input_fn().take(1):
        #     print(features_batch)
        #     print(labels_batch)
        ## end comment

        import tempfile
        model_dir = tempfile.mkdtemp()
        keras_estimator = tf.keras.estimator.model_to_estimator(
            keras_model=model, model_dir=model_dir)
        
        ## commented out relevant method call
        # keras_estimator.train(input_fn=input_fn, steps=500)
        ## end comment

        #### end copied code

        # (1c) build function details for function
        function_kwargs = {
            "input_fn": input_fn,
            "steps": 500
        }
        method_object = keras_estimator

        function_args = None

        return function_args, function_kwargs, method_object
    
    # (2) create function details
    function_to_run = "obj.train(**kwargs)"
    function_signature = "tensorflow.keras.estimator.model_to_estimator.train()"

    # (3) Initialise and run the experiment
    experiment = DataSizeExperiment(
        project = "estimator/keras_model_to_estimator_train",
        experiment_dir = EXPERIMENT_DIR,
        n_runs = 10,
        prepare_experiment = prepare_experiment,
        function_to_run = function_to_run,
        function_signature = function_signature
    )

    run_experiments(experiment, count=10, start=1)


if __name__ == "__main__":
    ### commented code has already been run, uncomment to replicate
    
    # run_images_cnn_model_fit_datasize_experiment()
    # run_generative_autoencoder_fit_datasize_experiment()
    # run_images_cnn_model_evaluate_datasize_experiment()
    # run_quickstart_beginner_model_fit_datasize_experiment()
    # run_loaddata_numpy_Dataset_from_tensor_slices_datasize_experiment()
    # run_keras_classification_sequential_fit_datasize_experiment()
    # run_keras_classification_sequential_evaluate_datasize_experiment()
    run_estimator_keras_model_to_estimator_train_datasize_experiment()
    pass