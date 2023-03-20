import tensorflow as tf
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
print(tf.math.add(1, 2))
print(tf.math.add([1, 2], [3, 4]))
print(tf.math.square(5))
print(tf.math.reduce_sum([1, 2, 3]))
print(tf.math.square(2) + tf.math.square(3))
x = custom_method(
tf.linalg.matmul([[1]], [[2, 3]]), imports='import time;import tensorflow as tf;import numpy as np;import tempfile', function_to_run='tf.linalg.matmul(*args)', method_object=None, function_args=[eval('[[1]]'), eval('[[2, 3]]')], function_kwargs={}, max_wait_secs=0)
print(x)
print(x.shape)
print(x.dtype)
import numpy as np
ndarray = np.ones([3, 3])
print('TensorFlow operations convert numpy arrays to Tensors automatically')
tensor = custom_method(
tf.math.multiply(ndarray, 42), imports='import time;import tensorflow as tf;import numpy as np;import tempfile', function_to_run='tf.math.multiply(*args)', method_object=None, function_args=[eval('ndarray'), eval('42')], function_kwargs={}, max_wait_secs=0)
print(tensor)
print('And NumPy operations convert Tensors to NumPy arrays automatically')
print(np.add(tensor, 1))
print('The .numpy() method explicitly converts a Tensor to a numpy array')
print(tensor.numpy())
x = custom_method(
tf.random.uniform([3, 3]), imports='import time;import tensorflow as tf;import numpy as np;import tempfile', function_to_run='tf.random.uniform(*args)', method_object=None, function_args=[eval('[3, 3]')], function_kwargs={}, max_wait_secs=0)
(print('Is there a GPU available: '),)
print(tf.config.list_physical_devices('GPU'))
(print('Is the Tensor on GPU #0:  '),)
print(x.device.endswith('GPU:0'))
import time

def time_matmul(x):
    start = time.time()
    for loop in range(10):
        custom_method(
        tf.linalg.matmul(x, x), imports='import time;import tensorflow as tf;import numpy as np;import tempfile', function_to_run='tf.linalg.matmul(*args)', method_object=None, function_args=[eval('x'), eval('x')], function_kwargs={}, max_wait_secs=0)
    result = time.time() - start
    print('10 loops: {:0.2f}ms'.format(1000 * result))
print('On CPU:')
with custom_method(
tf.device('CPU:0'), imports='import time;import tensorflow as tf;import numpy as np;import tempfile', function_to_run='tf.device(*args)', method_object=None, function_args=[eval('"CPU:0"')], function_kwargs={}, max_wait_secs=0):
    x = custom_method(
    tf.random.uniform([1000, 1000]), imports='import time;import tensorflow as tf;import numpy as np;import tempfile', function_to_run='tf.random.uniform(*args)', method_object=None, function_args=[eval('[1000, 1000]')], function_kwargs={}, max_wait_secs=0)
    assert custom_method(
    x.device.endswith('CPU:0'), imports='import time;import tensorflow as tf;import numpy as np;import tempfile', function_to_run='obj.device.endswith(*args)', method_object=eval('x'), function_args=[eval('"CPU:0"')], function_kwargs={}, max_wait_secs=0, custom_class=None)
    time_matmul(x)
if custom_method(
tf.config.list_physical_devices('GPU'), imports='import time;import tensorflow as tf;import numpy as np;import tempfile', function_to_run='tf.config.list_physical_devices(*args)', method_object=None, function_args=[eval('"GPU"')], function_kwargs={}, max_wait_secs=0):
    print('On GPU:')
    with custom_method(
    tf.device('GPU:0'), imports='import time;import tensorflow as tf;import numpy as np;import tempfile', function_to_run='tf.device(*args)', method_object=None, function_args=[eval('"GPU:0"')], function_kwargs={}, max_wait_secs=0):
        x = custom_method(
        tf.random.uniform([1000, 1000]), imports='import time;import tensorflow as tf;import numpy as np;import tempfile', function_to_run='tf.random.uniform(*args)', method_object=None, function_args=[eval('[1000, 1000]')], function_kwargs={}, max_wait_secs=0)
        assert custom_method(
        x.device.endswith('GPU:0'), imports='import time;import tensorflow as tf;import numpy as np;import tempfile', function_to_run='obj.device.endswith(*args)', method_object=eval('x'), function_args=[eval('"GPU:0"')], function_kwargs={}, max_wait_secs=0, custom_class=None)
        time_matmul(x)
ds_tensors = custom_method(
tf.data.Dataset.from_tensor_slices([1, 2, 3, 4, 5, 6]), imports='import time;import tensorflow as tf;import numpy as np;import tempfile', function_to_run='tf.data.Dataset.from_tensor_slices(*args)', method_object=None, function_args=[eval('[1, 2, 3, 4, 5, 6]')], function_kwargs={}, max_wait_secs=0)
import tempfile
(_, filename) = tempfile.mkstemp()
with open(filename, 'w') as f:
    f.write('Line 1\nLine 2\nLine 3\n  ')
ds_file = custom_method(
tf.data.TextLineDataset(filename), imports='import time;import tensorflow as tf;import numpy as np;import tempfile', function_to_run='tf.data.TextLineDataset(*args)', method_object=None, function_args=[eval('filename')], function_kwargs={}, max_wait_secs=0)
ds_tensors = custom_method(
ds_tensors.map(tf.math.square).shuffle(2).batch(2), imports='import time;import tensorflow as tf;import numpy as np;import tempfile', function_to_run='ds_tensors.map(tf.math.square).shuffle(2).batch(*args)', method_object=None, function_args=[eval('2')], function_kwargs={}, max_wait_secs=0)
ds_file = custom_method(
ds_file.batch(2), imports='import time;import tensorflow as tf;import numpy as np;import tempfile', function_to_run='obj.batch(*args)', method_object=eval('ds_file'), function_args=[eval('2')], function_kwargs={}, max_wait_secs=0, custom_class=None)
print('Elements of ds_tensors:')
for x in ds_tensors:
    print(x)
print('\nElements in ds_file:')
for x in ds_file:
    print(x)
