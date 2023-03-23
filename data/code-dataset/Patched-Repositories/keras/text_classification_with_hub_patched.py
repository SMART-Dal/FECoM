import os
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
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

def custom_method(func, imports: str, function_to_run: str, method_object=None, object_signature=None, function_args: list=None, function_kwargs: dict=None, custom_class=None):
    result = send_request(imports=imports, function_to_run=function_to_run, function_args=function_args, function_kwargs=function_kwargs, max_wait_secs=MAX_WAIT_S, wait_after_run_secs=WAIT_AFTER_RUN_S, method_object=method_object, object_signature=object_signature, custom_class=custom_class, experiment_file_path=EXPERIMENT_FILE_PATH)
    return func
print('Version: ', tf.__version__)
print('Eager mode: ', tf.executing_eagerly())
print('Hub version: ', hub.__version__)
print('GPU is', 'available' if tf.config.list_physical_devices('GPU') else 'NOT AVAILABLE')
(train_data, validation_data, test_data) = custom_method(
tfds.load(name='imdb_reviews', split=('train[:60%]', 'train[60%:]', 'test'), as_supervised=True), imports='import numpy as np;import os;import tensorflow_hub as hub;import tensorflow_datasets as tfds;import tensorflow as tf', function_to_run='tfds.load(**kwargs)', method_object=None, object_signature=None, function_args=[], function_kwargs={'name': eval('"imdb_reviews"'), 'split': eval("('train[:60%]', 'train[60%:]', 'test')"), 'as_supervised': eval('True')})
(train_examples_batch, train_labels_batch) = next(iter(train_data.batch(10)))
train_examples_batch
train_labels_batch
embedding = 'https://tfhub.dev/google/nnlm-en-dim50/2'
hub_layer = hub.KerasLayer(embedding, input_shape=[], dtype=tf.string, trainable=True)
hub_layer(train_examples_batch[:3])
model = custom_method(
tf.keras.Sequential(), imports='import numpy as np;import os;import tensorflow_hub as hub;import tensorflow_datasets as tfds;import tensorflow as tf', function_to_run='tf.keras.Sequential()', method_object=None, object_signature=None, function_args=[], function_kwargs={})
custom_method(
model.add(hub_layer), imports='import numpy as np;import os;import tensorflow_hub as hub;import tensorflow_datasets as tfds;import tensorflow as tf', function_to_run='obj.add(*args)', method_object=eval('model'), object_signature='tf.keras.Sequential', function_args=[eval('hub_layer')], function_kwargs={}, custom_class=None)
custom_method(
model.add(tf.keras.layers.Dense(16, activation='relu')), imports='import numpy as np;import os;import tensorflow_hub as hub;import tensorflow_datasets as tfds;import tensorflow as tf', function_to_run='obj.add(*args)', method_object=eval('model'), object_signature='tf.keras.Sequential', function_args=[eval("tf.keras.layers.Dense(16, activation='relu')")], function_kwargs={}, custom_class=None)
custom_method(
model.add(tf.keras.layers.Dense(1)), imports='import numpy as np;import os;import tensorflow_hub as hub;import tensorflow_datasets as tfds;import tensorflow as tf', function_to_run='obj.add(*args)', method_object=eval('model'), object_signature='tf.keras.Sequential', function_args=[eval('tf.keras.layers.Dense(1)')], function_kwargs={}, custom_class=None)
custom_method(
model.summary(), imports='import numpy as np;import os;import tensorflow_hub as hub;import tensorflow_datasets as tfds;import tensorflow as tf', function_to_run='obj.summary()', method_object=eval('model'), object_signature='tf.keras.Sequential', function_args=[], function_kwargs={}, custom_class=None)
custom_method(
model.compile(optimizer='adam', loss=tf.keras.losses.BinaryCrossentropy(from_logits=True), metrics=['accuracy']), imports='import numpy as np;import os;import tensorflow_hub as hub;import tensorflow_datasets as tfds;import tensorflow as tf', function_to_run='obj.compile(**kwargs)', method_object=eval('model'), object_signature='tf.keras.Sequential', function_args=[], function_kwargs={'optimizer': eval("'adam'"), 'loss': eval('tf.keras.losses.BinaryCrossentropy(from_logits=True)'), 'metrics': eval("['accuracy']")}, custom_class=None)
history = custom_method(
model.fit(train_data.shuffle(10000).batch(512), epochs=10, validation_data=validation_data.batch(512), verbose=1), imports='import numpy as np;import os;import tensorflow_hub as hub;import tensorflow_datasets as tfds;import tensorflow as tf', function_to_run='obj.fit(*args, **kwargs)', method_object=eval('model'), object_signature='tf.keras.Sequential', function_args=[eval('train_data.shuffle(10000).batch(512)')], function_kwargs={'epochs': eval('10'), 'validation_data': eval('validation_data.batch(512)'), 'verbose': eval('1')}, custom_class=None)
results = custom_method(
model.evaluate(test_data.batch(512), verbose=2), imports='import numpy as np;import os;import tensorflow_hub as hub;import tensorflow_datasets as tfds;import tensorflow as tf', function_to_run='obj.evaluate(*args, **kwargs)', method_object=eval('model'), object_signature='tf.keras.Sequential', function_args=[eval('test_data.batch(512)')], function_kwargs={'verbose': eval('2')}, custom_class=None)
for (name, value) in zip(model.metrics_names, results):
    print('%s: %.3f' % (name, value))
