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
experiment_number = sys.argv[1]
experiment_project = sys.argv[2]
EXPERIMENT_FILE_PATH = EXPERIMENT_DIR / 'method-level' / experiment_project / f'experiment-{experiment_number}.json'

def custom_method(imports: str, function_to_run: str, method_object=None, object_signature=None, function_args: list=None, function_kwargs: dict=None, custom_class=None):
    result = send_request(imports=imports, function_to_run=function_to_run, function_args=function_args, function_kwargs=function_kwargs, max_wait_secs=MAX_WAIT_S, wait_after_run_secs=WAIT_AFTER_RUN_S, method_object=method_object, object_signature=object_signature, custom_class=custom_class, experiment_file_path=EXPERIMENT_FILE_PATH)
custom_method(imports='import tensorflow as tf;import tensorflow_datasets as tfds;import tempfile;import numpy as np', function_to_run='tf.keras.models.Sequential(*args)', method_object=None, object_signature=None, function_args=[eval("[\n    tf.keras.layers.Dense(16, activation='relu', input_shape=(4,)),\n    tf.keras.layers.Dropout(0.2),\n    tf.keras.layers.Dense(3)\n]")], function_kwargs={})
model = tf.keras.models.Sequential([tf.keras.layers.Dense(16, activation='relu', input_shape=(4,)), tf.keras.layers.Dropout(0.2), tf.keras.layers.Dense(3)])
custom_method(imports='import tensorflow as tf;import tensorflow_datasets as tfds;import tempfile;import numpy as np', function_to_run='obj.compile(**kwargs)', method_object=eval('model'), object_signature=None, function_args=[], function_kwargs={'loss': eval('tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)'), 'optimizer': eval("'adam'")}, custom_class=None)
model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), optimizer='adam')
custom_method(imports='import tensorflow as tf;import tensorflow_datasets as tfds;import tempfile;import numpy as np', function_to_run='obj.summary()', method_object=eval('model'), object_signature=None, function_args=[], function_kwargs={}, custom_class=None)
model.summary()

def input_fn():
    import tensorflow_datasets as tfds
    split = tfds.Split.TRAIN
    dataset = tfds.load('iris', split=split, as_supervised=True)
    dataset = dataset.map(lambda features, labels: ({'dense_2_input': features}, labels))
    dataset = dataset.batch(32).repeat()
    return dataset
for (features_batch, labels_batch) in input_fn().take(1):
    print(features_batch)
    print(labels_batch)
import tempfile
model_dir = tempfile.mkdtemp()
custom_method(imports='import tensorflow as tf;import tensorflow_datasets as tfds;import tempfile;import numpy as np', function_to_run='tf.keras.estimator.model_to_estimator(**kwargs)', method_object=None, object_signature=None, function_args=[], function_kwargs={'keras_model': eval('model'), 'model_dir': eval('model_dir')})
keras_estimator = tf.keras.estimator.model_to_estimator(keras_model=model, model_dir=model_dir)
custom_method(imports='import tensorflow as tf;import tensorflow_datasets as tfds;import tempfile;import numpy as np', function_to_run='obj.train(**kwargs)', method_object=eval('keras_estimator'), object_signature=None, function_args=[], function_kwargs={'input_fn': eval('input_fn'), 'steps': eval('500')}, custom_class=None)
keras_estimator.train(input_fn=input_fn, steps=500)
custom_method(imports='import tensorflow as tf;import tensorflow_datasets as tfds;import tempfile;import numpy as np', function_to_run='obj.evaluate(**kwargs)', method_object=eval('keras_estimator'), object_signature=None, function_args=[], function_kwargs={'input_fn': eval('input_fn'), 'steps': eval('10')}, custom_class=None)
eval_result = keras_estimator.evaluate(input_fn=input_fn, steps=10)
print('Eval result: {}'.format(eval_result))
