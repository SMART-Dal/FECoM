import tensorflow as tf
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
print(tf.math.add(1, 2))
print(tf.math.add([1, 2], [3, 4]))
print(tf.math.square(5))
print(tf.math.reduce_sum([1, 2, 3]))
print(tf.math.square(2) + tf.math.square(3))
custom_method(imports='import time;import tempfile;import tensorflow as tf;import numpy as np', function_to_run='tf.linalg.matmul(*args)', method_object=None, object_signature=None, function_args=[eval('[[1]]'), eval('[[2, 3]]')], function_kwargs={})
x = tf.linalg.matmul([[1]], [[2, 3]])
print(x)
print(x.shape)
print(x.dtype)
import numpy as np
ndarray = np.ones([3, 3])
print('TensorFlow operations convert numpy arrays to Tensors automatically')
custom_method(imports='import time;import tempfile;import tensorflow as tf;import numpy as np', function_to_run='tf.math.multiply(*args)', method_object=None, object_signature=None, function_args=[eval('ndarray'), eval('42')], function_kwargs={})
tensor = tf.math.multiply(ndarray, 42)
print(tensor)
print('And NumPy operations convert Tensors to NumPy arrays automatically')
print(np.add(tensor, 1))
print('The .numpy() method explicitly converts a Tensor to a numpy array')
print(tensor.numpy())
custom_method(imports='import time;import tempfile;import tensorflow as tf;import numpy as np', function_to_run='tf.random.uniform(*args)', method_object=None, object_signature=None, function_args=[eval('[3, 3]')], function_kwargs={})
x = tf.random.uniform([3, 3])
(print('Is there a GPU available: '),)
print(tf.config.list_physical_devices('GPU'))
(print('Is the Tensor on GPU #0:  '),)
print(x.device.endswith('GPU:0'))
import time

def time_matmul(x):
    start = time.time()
    for loop in range(10):
        custom_method(imports='import time;import tempfile;import tensorflow as tf;import numpy as np', function_to_run='tf.linalg.matmul(*args)', method_object=None, object_signature=None, function_args=[eval('x'), eval('x')], function_kwargs={})
        tf.linalg.matmul(x, x)
    result = time.time() - start
    print('10 loops: {:0.2f}ms'.format(1000 * result))
print('On CPU:')
with tf.device('CPU:0'):
    custom_method(imports='import time;import tempfile;import tensorflow as tf;import numpy as np', function_to_run='tf.random.uniform(*args)', method_object=None, object_signature=None, function_args=[eval('[1000, 1000]')], function_kwargs={})
    x = tf.random.uniform([1000, 1000])
    assert x.device.endswith('CPU:0')
    time_matmul(x)
if tf.config.list_physical_devices('GPU'):
    print('On GPU:')
    with tf.device('GPU:0'):
        custom_method(imports='import time;import tempfile;import tensorflow as tf;import numpy as np', function_to_run='tf.random.uniform(*args)', method_object=None, object_signature=None, function_args=[eval('[1000, 1000]')], function_kwargs={})
        x = tf.random.uniform([1000, 1000])
        assert x.device.endswith('GPU:0')
        time_matmul(x)
custom_method(imports='import time;import tempfile;import tensorflow as tf;import numpy as np', function_to_run='tf.data.Dataset.from_tensor_slices(*args)', method_object=None, object_signature=None, function_args=[eval('[1, 2, 3, 4, 5, 6]')], function_kwargs={})
ds_tensors = tf.data.Dataset.from_tensor_slices([1, 2, 3, 4, 5, 6])
import tempfile
(_, filename) = tempfile.mkstemp()
with open(filename, 'w') as f:
    f.write('Line 1\nLine 2\nLine 3\n  ')
custom_method(imports='import time;import tempfile;import tensorflow as tf;import numpy as np', function_to_run='tf.data.TextLineDataset(*args)', method_object=None, object_signature=None, function_args=[eval('filename')], function_kwargs={})
ds_file = tf.data.TextLineDataset(filename)
custom_method(imports='import time;import tempfile;import tensorflow as tf;import numpy as np', function_to_run='obj.map(tf.math.square).shuffle(2).batch(*args)', method_object=eval('ds_tensors'), object_signature=None, function_args=[eval('2')], function_kwargs={}, custom_class=None)
ds_tensors = ds_tensors.map(tf.math.square).shuffle(2).batch(2)
custom_method(imports='import time;import tempfile;import tensorflow as tf;import numpy as np', function_to_run='obj.batch(*args)', method_object=eval('ds_file'), object_signature=None, function_args=[eval('2')], function_kwargs={}, custom_class=None)
ds_file = ds_file.batch(2)
print('Elements of ds_tensors:')
for x in ds_tensors:
    print(x)
print('\nElements in ds_file:')
for x in ds_file:
    print(x)
