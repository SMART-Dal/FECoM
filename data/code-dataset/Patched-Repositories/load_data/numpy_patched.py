import numpy as np
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

def custom_method(func, imports: str, function_to_run: str, method_object=None, function_args: list=None, function_kwargs: dict=None, max_wait_secs=MAX_WAIT_S, custom_class=None, wait_after_run_secs=WAIT_AFTER_RUN_S):
    result = send_request(imports=imports, function_to_run=function_to_run, function_args=function_args, function_kwargs=function_kwargs, max_wait_secs=max_wait_secs, wait_after_run_secs=wait_after_run_secs, method_object=method_object, custom_class=custom_class, experiment_file_path=EXPERIMENT_FILE_PATH)
    return func
if __name__ == '__main__':
    print(EXPERIMENT_FILE_PATH)
DATA_URL = 'https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz'
path = custom_method(
tf.keras.utils.get_file('mnist.npz', DATA_URL), imports='import numpy as np;import tensorflow as tf', function_to_run='tf.keras.utils.get_file(*args)', method_object=None, function_args=[eval("'mnist.npz'"), eval('DATA_URL')], function_kwargs={}, max_wait_secs=0)
with np.load(path) as data:
    train_examples = data['x_train']
    train_labels = data['y_train']
    test_examples = data['x_test']
    test_labels = data['y_test']
train_dataset = custom_method(
tf.data.Dataset.from_tensor_slices((train_examples, train_labels)), imports='import numpy as np;import tensorflow as tf', function_to_run='tf.data.Dataset.from_tensor_slices(*args)', method_object=None, function_args=[eval('(train_examples, train_labels)')], function_kwargs={}, max_wait_secs=0)
test_dataset = custom_method(
tf.data.Dataset.from_tensor_slices((test_examples, test_labels)), imports='import numpy as np;import tensorflow as tf', function_to_run='tf.data.Dataset.from_tensor_slices(*args)', method_object=None, function_args=[eval('(test_examples, test_labels)')], function_kwargs={}, max_wait_secs=0)
BATCH_SIZE = 64
SHUFFLE_BUFFER_SIZE = 100
train_dataset = train_dataset.shuffle(SHUFFLE_BUFFER_SIZE).batch(BATCH_SIZE)
test_dataset = custom_method(
test_dataset.batch(BATCH_SIZE), imports='import numpy as np;import tensorflow as tf', function_to_run='obj.batch(*args)', method_object=eval('test_dataset'), function_args=[eval('BATCH_SIZE')], function_kwargs={}, max_wait_secs=0, custom_class=None)
model = custom_method(
tf.keras.Sequential([tf.keras.layers.Flatten(input_shape=(28, 28)), tf.keras.layers.Dense(128, activation='relu'), tf.keras.layers.Dense(10)]), imports='import numpy as np;import tensorflow as tf', function_to_run='tf.keras.Sequential(*args)', method_object=None, function_args=[eval("[\n    tf.keras.layers.Flatten(input_shape=(28, 28)),\n    tf.keras.layers.Dense(128, activation='relu'),\n    tf.keras.layers.Dense(10)\n]")], function_kwargs={}, max_wait_secs=0)
custom_method(
model.compile(optimizer=tf.keras.optimizers.RMSprop(), loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=['sparse_categorical_accuracy']), imports='import numpy as np;import tensorflow as tf', function_to_run='obj.compile(**kwargs)', method_object=eval('model'), function_args=[], function_kwargs={'optimizer': eval('tf.keras.optimizers.RMSprop()'), 'loss': eval('tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)'), 'metrics': eval("['sparse_categorical_accuracy']")}, max_wait_secs=0, custom_class=None)
custom_method(
model.fit(train_dataset, epochs=10), imports='import numpy as np;import tensorflow as tf', function_to_run='obj.fit(*args, **kwargs)', method_object=eval('model'), function_args=[eval('train_dataset')], function_kwargs={'epochs': eval('10')}, max_wait_secs=0, custom_class=None)
custom_method(
model.evaluate(test_dataset), imports='import numpy as np;import tensorflow as tf', function_to_run='obj.evaluate(*args)', method_object=eval('model'), function_args=[eval('test_dataset')], function_kwargs={}, max_wait_secs=0, custom_class=None)
