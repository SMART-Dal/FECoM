import tensorflow as tf
import os
from pathlib import Path
import dill as pickle
import sys
from tool.client.client_config import EXPERIMENT_DIR, MAX_WAIT_S, WAIT_AFTER_RUN_S
from tool.server.send_request import send_request
from tool.server.function_details import FunctionDetails
current_path = os.path.abspath(__file__)
experiment_number = sys.argv[1]
experiment_project = sys.argv[2]
EXPERIMENT_FILE_PATH = EXPERIMENT_DIR / 'method-level' / experiment_project / f'experiment-{experiment_number}.json'

def custom_method(func, imports: str, function_to_run: str, method_object=None, object_signature=None, function_args: list=None, function_kwargs: dict=None, custom_class=None):
    result = send_request(imports=imports, function_to_run=function_to_run, function_args=function_args, function_kwargs=function_kwargs, max_wait_secs=MAX_WAIT_S, wait_after_run_secs=WAIT_AFTER_RUN_S, method_object=method_object, object_signature=object_signature, custom_class=custom_class, experiment_file_path=EXPERIMENT_FILE_PATH)
print('TensorFlow version:', tf.__version__)
mnist = tf.keras.datasets.mnist
((x_train, y_train), (x_test, y_test)) = mnist.load_data()
(x_train, x_test) = (x_train / 255.0, x_test / 255.0)
custom_method(imports='import tensorflow as tf', function_to_run='tf.keras.models.Sequential(*args)', method_object=None, object_signature=None, function_args=[eval("[\n  tf.keras.layers.Flatten(input_shape=(28, 28)),\n  tf.keras.layers.Dense(128, activation='relu'),\n  tf.keras.layers.Dropout(0.2),\n  tf.keras.layers.Dense(10)\n]")], function_kwargs={})
model = tf.keras.models.Sequential([tf.keras.layers.Flatten(input_shape=(28, 28)), tf.keras.layers.Dense(128, activation='relu'), tf.keras.layers.Dropout(0.2), tf.keras.layers.Dense(10)])
custom_method(imports='import tensorflow as tf', function_to_run='obj(x_train[:1]).numpy()', method_object=eval('model'), object_signature=None, function_args=[], function_kwargs={}, custom_class=None)
predictions = model(x_train[:1]).numpy()
predictions
custom_method(imports='import tensorflow as tf', function_to_run='tf.nn.softmax(predictions).numpy()', method_object=None, object_signature=None, function_args=[], function_kwargs={})
tf.nn.softmax(predictions).numpy()
custom_method(imports='import tensorflow as tf', function_to_run='tf.keras.losses.SparseCategoricalCrossentropy(**kwargs)', method_object=None, object_signature=None, function_args=[], function_kwargs={'from_logits': eval('True')})
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
custom_method(imports='import tensorflow as tf', function_to_run='obj(y_train[:1], predictions).numpy()', method_object=eval('loss_fn'), object_signature=None, function_args=[], function_kwargs={}, custom_class=None)
loss_fn(y_train[:1], predictions).numpy()
custom_method(imports='import tensorflow as tf', function_to_run='obj.compile(**kwargs)', method_object=eval('model'), object_signature=None, function_args=[], function_kwargs={'optimizer': eval("'adam'"), 'loss': eval('loss_fn'), 'metrics': eval("['accuracy']")}, custom_class=None)
model.compile(optimizer='adam', loss=loss_fn, metrics=['accuracy'])
custom_method(imports='import tensorflow as tf', function_to_run='obj.fit(*args, **kwargs)', method_object=eval('model'), object_signature=None, function_args=[eval('x_train'), eval('y_train')], function_kwargs={'epochs': eval('5')}, custom_class=None)
model.fit(x_train, y_train, epochs=5)
custom_method(imports='import tensorflow as tf', function_to_run='obj.evaluate(*args, **kwargs)', method_object=eval('model'), object_signature=None, function_args=[eval('x_test'), eval('y_test')], function_kwargs={'verbose': eval('2')}, custom_class=None)
model.evaluate(x_test, y_test, verbose=2)
custom_method(imports='import tensorflow as tf', function_to_run='tf.keras.Sequential(*args)', method_object=None, object_signature=None, function_args=[eval('[\n  model,\n  tf.keras.layers.Softmax()\n]')], function_kwargs={})
probability_model = tf.keras.Sequential([model, tf.keras.layers.Softmax()])
custom_method(imports='import tensorflow as tf', function_to_run='obj(*args)', method_object=eval('probability_model'), object_signature=None, function_args=[eval('x_test[:5]')], function_kwargs={}, custom_class=None)
probability_model(x_test[:5])
