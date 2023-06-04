import tensorflow as tf
import numpy as np
import tensorflow_datasets as tfds
# import os
# from pathlib import Path
# import dill as pickle
import sys
# import numpy as np
from tool.patching.patching_config import EXPERIMENT_DIR #, MAX_WAIT_S, WAIT_AFTER_RUN_S
# from tool.server.send_request import send_request
# from tool.server.function_details import FunctionDetails
# import json
# current_path = os.path.abspath(__file__)
experiment_number = sys.argv[1]
experiment_project = sys.argv[2]
EXPERIMENT_FILE_PATH = EXPERIMENT_DIR / 'local-execution' / experiment_project / f'experiment-{experiment_number}.json'

# NEW
# changed names to make function name clash less likely
from tool.server.local_execution import before_execution as before_execution_INSERTED_INTO_SCRIPT
from tool.server.local_execution import after_execution as after_execution_INSERTED_INTO_SCRIPT
# END NEW

# skip_calls_file_path = EXPERIMENT_FILE_PATH.parent / 'skip_calls.json'
# if skip_calls_file_path.exists():
#     with open(skip_calls_file_path, 'r') as f:
#         skip_calls = json.load(f)
# else:
#     skip_calls = []
#     with open(skip_calls_file_path, 'w') as f:
#         json.dump(skip_calls, f)

# def custom_method(imports: str, function_to_run: str, method_object=None, object_signature=None, function_args: list=None, function_kwargs: dict=None, custom_class=None):
#     if skip_calls is not None and any((call['function_to_run'] == function_to_run and np.array_equal(call['function_args'], function_args) and (call['function_kwargs'] == function_kwargs) for call in skip_calls)):
#         print('skipping call: ', function_to_run)
#         return
#     result = send_request(imports=imports, function_to_run=function_to_run, function_args=function_args, function_kwargs=function_kwargs, max_wait_secs=MAX_WAIT_S, wait_after_run_secs=WAIT_AFTER_RUN_S, method_object=method_object, object_signature=object_signature, custom_class=custom_class, experiment_file_path=EXPERIMENT_FILE_PATH)
#     if result is not None and isinstance(result, dict) and (len(result) == 1):
#         energy_data = next(iter(result.values()))
#         if skip_calls is not None and 'start_time_perf' in energy_data['times'] and ('end_time_perf' in energy_data['times']) and ('start_time_nvidia' in energy_data['times']) and ('end_time_nvidia' in energy_data['times']) and (energy_data['times']['start_time_perf'] == energy_data['times']['end_time_perf']) and (energy_data['times']['start_time_nvidia'] == energy_data['times']['end_time_nvidia']):
#             call_to_skip = {'function_to_run': function_to_run, 'function_args': function_args, 'function_kwargs': function_kwargs}
#             try:
#                 json.dumps(call_to_skip)
#                 if call_to_skip not in skip_calls:
#                     skip_calls.append(call_to_skip)
#                     with open(skip_calls_file_path, 'w') as f:
#                         json.dump(skip_calls, f)
#                     print('skipping call added, current list is: ', skip_calls)
#                 else:
#                     print('Skipping call already exists.')
#             except TypeError:
#                 print('Ignore: Skipping call is not JSON serializable, skipping append and dump.')
#     else:
#         print('Invalid dictionary object or does not have one key-value pair.')

# NEW
# custom_method(imports='import tensorflow as tf;import tempfile;import numpy as np;import tensorflow_datasets as tfds', function_to_run='tf.keras.models.Sequential(*args)', method_object=None, object_signature=None, function_args=[eval("[\n    tf.keras.layers.Dense(16, activation='relu', input_shape=(4,)),\n    tf.keras.layers.Dropout(0.2),\n    tf.keras.layers.Dense(3)\n]")], function_kwargs={})
start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT()
model = tf.keras.models.Sequential([tf.keras.layers.Dense(16, activation='relu', input_shape=(4,)), tf.keras.layers.Dropout(0.2), tf.keras.layers.Dense(3)])
after_execution_INSERTED_INTO_SCRIPT(
    start_times=start_times_INSERTED_INTO_SCRIPT,
    experiment_file_path=EXPERIMENT_FILE_PATH,
    function_to_run='tf.keras.models.Sequential(*args)',
    function_args=[[tf.keras.layers.Dense(16, activation='relu', input_shape=(4,)), tf.keras.layers.Dropout(0.2), tf.keras.layers.Dense(3)]]
)
# END NEW

# NEW
# custom_method(imports='import tensorflow as tf;import tempfile;import numpy as np;import tensorflow_datasets as tfds', function_to_run='obj.compile(**kwargs)', method_object=eval('model'), object_signature=None, function_args=[], function_kwargs={'loss': eval('tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)'), 'optimizer': eval("'adam'")}, custom_class=None)
start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT()
model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), optimizer='adam')
after_execution_INSERTED_INTO_SCRIPT(
    start_times=start_times_INSERTED_INTO_SCRIPT,
    experiment_file_path=EXPERIMENT_FILE_PATH,
    function_to_run='obj.compile(**kwargs)',
    method_object=model,
    object_signature='tf.keras.models.Sequential',
    function_kwargs={'optimizer': 'adam', 'loss': tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)}
)
# END NEW

# NEW
# custom_method(imports='import tensorflow as tf;import tempfile;import numpy as np;import tensorflow_datasets as tfds', function_to_run='obj.summary()', method_object=eval('model'), object_signature=None, function_args=[], function_kwargs={}, custom_class=None)
start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT()
model.summary()
after_execution_INSERTED_INTO_SCRIPT(
    start_times=start_times_INSERTED_INTO_SCRIPT,
    experiment_file_path=EXPERIMENT_FILE_PATH,
    function_to_run='obj.summary()',
    method_object=model,
    object_signature='tf.keras.models.Sequential'
)
# END NEW

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

# NEW
# custom_method(imports='import tensorflow as tf;import tempfile;import numpy as np;import tensorflow_datasets as tfds', function_to_run='tf.keras.estimator.model_to_estimator(**kwargs)', method_object=None, object_signature=None, function_args=[], function_kwargs={'keras_model': eval('model'), 'model_dir': eval('model_dir')})
start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT()
keras_estimator = tf.keras.estimator.model_to_estimator(keras_model=model, model_dir=model_dir)
after_execution_INSERTED_INTO_SCRIPT(
    start_times=start_times_INSERTED_INTO_SCRIPT,
    experiment_file_path=EXPERIMENT_FILE_PATH,
    function_to_run='tf.keras.estimator.model_to_estimator(**kwargs)',
    function_kwargs={'keras_model': model, 'model_dir': model_dir}
)
# END NEW

# NEW
# custom_method(imports='import tensorflow as tf;import tempfile;import numpy as np;import tensorflow_datasets as tfds', function_to_run='obj.train(**kwargs)', method_object=eval('keras_estimator'), object_signature=None, function_args=[], function_kwargs={'input_fn': eval('input_fn'), 'steps': eval('500')}, custom_class=None)
start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT()
keras_estimator.train(input_fn=input_fn, steps=500)
after_execution_INSERTED_INTO_SCRIPT(
    start_times=start_times_INSERTED_INTO_SCRIPT,
    experiment_file_path=EXPERIMENT_FILE_PATH,
    function_to_run='obj.train(**kwargs)',
    method_object=keras_estimator,
    object_signature='tf.estimator.Estimator',
    function_kwargs={'input_fn': input_fn, 'steps': 500}
)
# END NEW

# NEW
# custom_method(imports='import tensorflow as tf;import tempfile;import numpy as np;import tensorflow_datasets as tfds', function_to_run='obj.evaluate(**kwargs)', method_object=eval('keras_estimator'), object_signature=None, function_args=[], function_kwargs={'input_fn': eval('input_fn'), 'steps': eval('10')}, custom_class=None)
start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT()
eval_result = keras_estimator.evaluate(input_fn=input_fn, steps=10)
after_execution_INSERTED_INTO_SCRIPT(
    start_times=start_times_INSERTED_INTO_SCRIPT,
    experiment_file_path=EXPERIMENT_FILE_PATH,
    function_to_run='obj.evaluate(**kwargs)',
    method_object=keras_estimator,
    object_signature='tf.estimator.Estimator',
    function_kwargs={'input_fn': input_fn, 'steps': 10}
)
# END NEW
print('Eval result: {}'.format(eval_result))
