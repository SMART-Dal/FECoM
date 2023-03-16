"""
Test 
"""

import tensorflow as tf
import pytest
from tool.server.send_request import send_request

@pytest.fixture
def imports():
    return "import tensorflow as tf"

@pytest.fixture
def model():
    model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(10)
    ])
    return model

@pytest.fixture
def loss_fn():
    # loss function for training
    return tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

@pytest.fixture
def mnist():
    mnist = tf.keras.datasets.mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0
    return x_train, y_train, x_test, y_test

@pytest.fixture
def compiled_model(model, loss_fn):
    # configure & compile model
    model.compile(optimizer='adam',
                loss=loss_fn,
                metrics=['accuracy'])
    return model

def test_mnist_load(imports):
    function_to_run = "tf.keras.datasets.mnist.load_data()"

    resp = send_request(imports, function_to_run, return_result=True)[function_to_run]

    result = resp["return"]

    # compare result from server with the real results
    mnist = tf.keras.datasets.mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    assert(result[0][0].all() == x_train.all())
    assert(result[0][1].all() == y_train.all())
    assert(result[1][0].all() == x_test.all())
    assert(result[1][1].all() == y_test.all())

def test_mnist_model_compile(model, loss_fn):

    # configure & compile model
    imports = "import tensorflow as tf"
    function_to_run = "obj.compile(**kwargs)"
    function_kwargs = {"optimizer": "adam",
                       "loss": loss_fn,
                       "metrics": ['accuracy']
    }
    method_object = model
    
    return_dict = send_request(imports, function_to_run, function_kwargs=function_kwargs, method_object=method_object, return_result=True)[function_to_run]

    assert return_dict["return"] is None
    assert type(return_dict["method_object"]) == type(model)

def test_mnist_model_train(compiled_model, mnist):
    epochs = 2
    x_train, y_train, _, _ = mnist

    # training
    imports = "import tensorflow as tf"
    function_to_run = "obj.fit(*args,**kwargs)"
    function_args = x_train, y_train
    function_kwargs = {"epochs": epochs}
    method_object = compiled_model
    
    return_dict = send_request(imports, function_to_run, function_args, function_kwargs, method_object=method_object, return_result=True)[function_to_run]
    test_history = return_dict["return"]
    real_history = compiled_model.fit(x_train, y_train, epochs=epochs)

    assert(type(test_history) == type(real_history))
    assert(test_history.params == real_history.params)

    test_trained_model = return_dict["method_object"]

    assert test_trained_model.layers[1].get_weights()[0].all() == compiled_model.layers[1].get_weights()[0].all()
    assert test_trained_model.layers[1].get_weights()[1].all() == compiled_model.layers[1].get_weights()[1].all()
    assert test_trained_model.layers[3].get_weights()[0].all() == compiled_model.layers[3].get_weights()[0].all()
    assert test_trained_model.layers[3].get_weights()[1].all() == compiled_model.layers[3].get_weights()[1].all()

def test_mnist_model_testing(compiled_model, mnist):
    x_train, y_train, x_test, y_test = mnist

    # training
    compiled_model.fit(x_train, y_train, epochs=5)

    # testing
    imports = "import tensorflow as tf"
    function_to_run = "obj.evaluate(*args,**kwargs)"
    function_args = x_test, y_test
    function_kwargs = {"verbose": 2}
    method_object = compiled_model
    
    return_dict = send_request(imports, function_to_run, function_args, function_kwargs, method_object=method_object, return_result=True)[function_to_run]
    test_evaluation = return_dict["return"]
    real_evaluation = compiled_model.evaluate(x_test, y_test, verbose=2)

    assert(test_evaluation == real_evaluation)