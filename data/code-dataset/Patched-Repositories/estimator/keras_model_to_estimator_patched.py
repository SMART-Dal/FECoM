import tensorflow as tf
import numpy as np
import tensorflow_datasets as tfds
import os
from pathlib import Path
import dill as pickle
import sys
from tool.client.client_config import EXPERIMENT_DIR, MAX_WAIT_S, WAIT_AFTER_RUN_S
from tool.server.send_request import send_request
from tool.server.function_details import FunctionDetails
current_path = os.path.abspath(__file__)
(immediate_folder, file_name) = os.path.split(current_path)
immediate_folder = os.path.basename(immediate_folder)
experiment_number = int(sys.argv[0])
experiment_file_name = os.path.splitext(file_name)[0]
EXPERIMENT_FILE_PATH = EXPERIMENT_DIR / 'method-level' / immediate_folder / experiment_file_name / f'experiment-{experiment_number}.json'

def custom_method(func, imports: str, function_to_run: str, method_object=None, function_args: list=None, function_kwargs: dict=None, max_wait_secs=MAX_WAIT_S, custom_class=None, wait_after_run_secs=WAIT_AFTER_RUN_S):
    result = send_request(imports=imports, function_to_run=function_to_run, function_args=function_args, function_kwargs=function_kwargs, max_wait_secs=max_wait_secs, method_object=method_object, custom_class=custom_class, experiment_file_path=EXPERIMENT_FILE_PATH)
    return func
if __name__ == '__main__':
    print(EXPERIMENT_FILE_PATH)
model = custom_method(
tf.keras.models.Sequential([tf.keras.layers.Dense(16, activation='relu', input_shape=(4,)), tf.keras.layers.Dropout(0.2), tf.keras.layers.Dense(3)]), imports='import tensorflow as tf;import tensorflow_datasets as tfds;import tempfile;import numpy as np', function_to_run='tf.keras.models.Sequential(*args)', method_object=None, function_args=[eval("[\n    tf.keras.layers.Dense(16, activation='relu', input_shape=(4,)),\n    tf.keras.layers.Dropout(0.2),\n    tf.keras.layers.Dense(3)\n]")], function_kwargs={}, max_wait_secs=0)
custom_method(
model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), optimizer='adam'), imports='import tensorflow as tf;import tensorflow_datasets as tfds;import tempfile;import numpy as np', function_to_run='obj.compile(**kwargs)', method_object=eval('model'), function_args=[], function_kwargs={'loss': eval('tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)'), 'optimizer': eval("'adam'")}, max_wait_secs=0, custom_class=None)
custom_method(
model.summary(), imports='import tensorflow as tf;import tensorflow_datasets as tfds;import tempfile;import numpy as np', function_to_run='obj.summary()', method_object=eval('model'), function_args=[], function_kwargs={}, max_wait_secs=0, custom_class=None)

def input_fn():
    split = tfds.Split.TRAIN
    dataset = custom_method(
    tfds.load('iris', split=split, as_supervised=True), imports='import tensorflow as tf;import tensorflow_datasets as tfds;import tempfile;import numpy as np', function_to_run='tfds.load(*args, **kwargs)', method_object=None, function_args=[eval("'iris'")], function_kwargs={'split': eval('split'), 'as_supervised': eval('True')}, max_wait_secs=0)
    dataset = custom_method(
    dataset.map(lambda features, labels: ({'dense_input': features}, labels)), imports='import tensorflow as tf;import tensorflow_datasets as tfds;import tempfile;import numpy as np', function_to_run='obj.map(*args)', method_object=eval('dataset'), function_args=[eval("lambda features, labels: ({'dense_input':features}, labels)")], function_kwargs={}, max_wait_secs=0, custom_class=None)
    dataset = custom_method(
    dataset.batch(32).repeat(), imports='import tensorflow as tf;import tensorflow_datasets as tfds;import tempfile;import numpy as np', function_to_run='obj.batch(32).repeat()', method_object=eval('dataset'), function_args=[], function_kwargs={}, max_wait_secs=0, custom_class=None)
    return dataset
for (features_batch, labels_batch) in input_fn().take(1):
    print(features_batch)
    print(labels_batch)
import tempfile
model_dir = tempfile.mkdtemp()
keras_estimator = custom_method(
tf.keras.estimator.model_to_estimator(keras_model=model, model_dir=model_dir), imports='import tensorflow as tf;import tensorflow_datasets as tfds;import tempfile;import numpy as np', function_to_run='tf.keras.estimator.model_to_estimator(**kwargs)', method_object=None, function_args=[], function_kwargs={'keras_model': eval('model'), 'model_dir': eval('model_dir')}, max_wait_secs=0)
custom_method(
keras_estimator.train(input_fn=input_fn, steps=500), imports='import tensorflow as tf;import tensorflow_datasets as tfds;import tempfile;import numpy as np', function_to_run='obj.train(**kwargs)', method_object=eval('keras_estimator'), function_args=[], function_kwargs={'input_fn': eval('input_fn'), 'steps': eval('500')}, max_wait_secs=0, custom_class=None)
eval_result = custom_method(
keras_estimator.evaluate(input_fn=input_fn, steps=10), imports='import tensorflow as tf;import tensorflow_datasets as tfds;import tempfile;import numpy as np', function_to_run='obj.evaluate(**kwargs)', method_object=eval('keras_estimator'), function_args=[], function_kwargs={'input_fn': eval('input_fn'), 'steps': eval('10')}, max_wait_secs=0, custom_class=None)
print('Eval result: {}'.format(eval_result))
