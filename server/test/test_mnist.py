import tensorflow as tf
import numpy as np
from send_request import send_request

def test_mnist_load():
    imports = "import tensorflow as tf"
    function_to_run = "tf.keras.datasets.mnist.load_data()"
    function_args = None
    function_kwargs = None

    result = send_request(imports, function_to_run, function_args, function_kwargs)

    # compare result from server with the real results
    mnist = tf.keras.datasets.mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    assert(result[0][0].all() == x_train.all())
    assert(result[0][1].all() == y_train.all())
    assert(result[1][0].all() == x_test.all())
    assert(result[1][1].all() == y_test.all())

def test_mnist_model_compile():
    model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(10)
    ])

    # loss function for training
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

    # configure & compile model
    imports = "import tensorflow as tf"
    function_to_run = "obj.compile(**kwargs)"
    function_args = None
    function_kwargs = {"optimizer": "adam",
                       "loss": loss_fn,
                       "metrics": ['accuracy']
    }
    method_object = model
    
    result = send_request(imports, function_to_run, function_args, function_kwargs, method_object=method_object)

    assert(result is None)

def test_mnist_model_train():
    mnist = tf.keras.datasets.mnist

    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0

    model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(10)
    ])

    # loss function for training
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

    # configure & compile model
    model.compile(optimizer='adam',
                loss=loss_fn,
                metrics=['accuracy'])

    # training
    imports = "import tensorflow as tf"
    function_to_run = "obj.fit(*args,**kwargs)"
    function_args = x_train, y_train
    function_kwargs = {"epochs": 5}
    method_object = model
    
    test_history = send_request(imports, function_to_run, function_args, function_kwargs, method_object=method_object)
    real_history = model.fit(x_train, y_train, epochs=5)

    assert(type(test_history) == type(real_history))
    assert(test_history.params == real_history.params)