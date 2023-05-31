import os
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_datasets as tfds
import os
from pathlib import Path
import dill as pickle
import sys
import numpy as np
from tool.client.client_config import EXPERIMENT_DIR, MAX_WAIT_S, WAIT_AFTER_RUN_S
from tool.server.send_request import send_request
from tool.server.function_details import FunctionDetails
import json
current_path = os.path.abspath(__file__)
experiment_number = sys.argv[1]
experiment_project = sys.argv[2]
EXPERIMENT_FILE_PATH = EXPERIMENT_DIR / 'method-level' / experiment_project / f'experiment-{experiment_number}.json'
skip_calls_file_path = EXPERIMENT_FILE_PATH.parent / 'skip_calls.json'
if skip_calls_file_path.exists():
    with open(skip_calls_file_path, 'r') as f:
        skip_calls = json.load(f)
else:
    skip_calls = []
    with open(skip_calls_file_path, 'w') as f:
        json.dump(skip_calls, f)

def custom_method(imports: str, function_to_run: str, method_object=None, object_signature=None, function_args: list=None, function_kwargs: dict=None, custom_class=None):
    if skip_calls is not None and any((call['function_to_run'] == function_to_run and np.array_equal(call['function_args'], function_args) and (call['function_kwargs'] == function_kwargs) for call in skip_calls)):
        print('skipping call: ', function_to_run)
        return
    result = send_request(imports=imports, function_to_run=function_to_run, function_args=function_args, function_kwargs=function_kwargs, max_wait_secs=MAX_WAIT_S, wait_after_run_secs=WAIT_AFTER_RUN_S, method_object=method_object, object_signature=object_signature, custom_class=custom_class, experiment_file_path=EXPERIMENT_FILE_PATH)
    if result is not None and isinstance(result, dict) and (len(result) == 1):
        energy_data = next(iter(result.values()))
        if skip_calls is not None and 'start_time_perf' in energy_data['times'] and ('end_time_perf' in energy_data['times']) and ('start_time_nvidia' in energy_data['times']) and ('end_time_nvidia' in energy_data['times']) and (energy_data['times']['start_time_perf'] == energy_data['times']['end_time_perf']) and (energy_data['times']['start_time_nvidia'] == energy_data['times']['end_time_nvidia']):
            call_to_skip = {'function_to_run': function_to_run, 'function_args': function_args, 'function_kwargs': function_kwargs}
            try:
                json.dumps(call_to_skip)
                if call_to_skip not in skip_calls:
                    skip_calls.append(call_to_skip)
                    with open(skip_calls_file_path, 'w') as f:
                        json.dump(skip_calls, f)
                    print('skipping call added, current list is: ', skip_calls)
                else:
                    print('Skipping call already exists.')
            except TypeError:
                print('Ignore: Skipping call is not JSON serializable, skipping append and dump.')
    else:
        print('Invalid dictionary object or does not have one key-value pair.')
print('Version: ', tf.__version__)
print('Eager mode: ', tf.executing_eagerly())
print('Hub version: ', hub.__version__)
print('GPU is', 'available' if tf.config.list_physical_devices('GPU') else 'NOT AVAILABLE')
(train_data, validation_data, test_data) = tfds.load(name='imdb_reviews', split=('train[:60%]', 'train[60%:]', 'test'), as_supervised=True)
(train_examples_batch, train_labels_batch) = next(iter(train_data.batch(10)))
train_examples_batch
train_labels_batch
embedding = 'https://tfhub.dev/google/nnlm-en-dim50/2'
hub_layer = hub.KerasLayer(embedding, input_shape=[], dtype=tf.string, trainable=True)
hub_layer(train_examples_batch[:3])
custom_method(imports='import tensorflow as tf;import tensorflow_hub as hub;import numpy as np;import tensorflow_datasets as tfds;import os', function_to_run='tf.keras.Sequential()', method_object=None, object_signature=None, function_args=[], function_kwargs={})
model = tf.keras.Sequential()
custom_method(imports='import tensorflow as tf;import tensorflow_hub as hub;import numpy as np;import tensorflow_datasets as tfds;import os', function_to_run='obj.add(*args)', method_object=eval('model'), object_signature=None, function_args=[eval('hub_layer')], function_kwargs={}, custom_class=None)
model.add(hub_layer)
custom_method(imports='import tensorflow as tf;import tensorflow_hub as hub;import numpy as np;import tensorflow_datasets as tfds;import os', function_to_run='obj.add(*args)', method_object=eval('model'), object_signature=None, function_args=[eval("tf.keras.layers.Dense(16, activation='relu')")], function_kwargs={}, custom_class=None)
model.add(tf.keras.layers.Dense(16, activation='relu'))
custom_method(imports='import tensorflow as tf;import tensorflow_hub as hub;import numpy as np;import tensorflow_datasets as tfds;import os', function_to_run='obj.add(*args)', method_object=eval('model'), object_signature=None, function_args=[eval('tf.keras.layers.Dense(1)')], function_kwargs={}, custom_class=None)
model.add(tf.keras.layers.Dense(1))
custom_method(imports='import tensorflow as tf;import tensorflow_hub as hub;import numpy as np;import tensorflow_datasets as tfds;import os', function_to_run='obj.summary()', method_object=eval('model'), object_signature=None, function_args=[], function_kwargs={}, custom_class=None)
model.summary()
custom_method(imports='import tensorflow as tf;import tensorflow_hub as hub;import numpy as np;import tensorflow_datasets as tfds;import os', function_to_run='obj.compile(**kwargs)', method_object=eval('model'), object_signature=None, function_args=[], function_kwargs={'optimizer': eval("'adam'"), 'loss': eval('tf.keras.losses.BinaryCrossentropy(from_logits=True)'), 'metrics': eval("['accuracy']")}, custom_class=None)
model.compile(optimizer='adam', loss=tf.keras.losses.BinaryCrossentropy(from_logits=True), metrics=['accuracy'])
custom_method(imports='import tensorflow as tf;import tensorflow_hub as hub;import numpy as np;import tensorflow_datasets as tfds;import os', function_to_run='obj.fit(*args, **kwargs)', method_object=eval('model'), object_signature=None, function_args=[eval('train_data.shuffle(10000).batch(512)')], function_kwargs={'epochs': eval('10'), 'validation_data': eval('validation_data.batch(512)'), 'verbose': eval('1')}, custom_class=None)
history = model.fit(train_data.shuffle(10000).batch(512), epochs=10, validation_data=validation_data.batch(512), verbose=1)
custom_method(imports='import tensorflow as tf;import tensorflow_hub as hub;import numpy as np;import tensorflow_datasets as tfds;import os', function_to_run='obj.evaluate(*args, **kwargs)', method_object=eval('model'), object_signature=None, function_args=[eval('test_data.batch(512)')], function_kwargs={'verbose': eval('2')}, custom_class=None)
results = model.evaluate(test_data.batch(512), verbose=2)
for (name, value) in zip(model.metrics_names, results):
    print('%s: %.3f' % (name, value))
