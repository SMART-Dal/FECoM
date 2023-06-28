import tensorflow as tf
import sys
from tool.patching.patching_config import EXPERIMENT_DIR
from tool.measurement.execution import before_execution as before_execution_INSERTED_INTO_SCRIPT
from tool.measurement.execution import after_execution as after_execution_INSERTED_INTO_SCRIPT
from tool.experiment.experiments import ExperimentKinds
experiment_number = sys.argv[1]
experiment_project = sys.argv[2]
EXPERIMENT_FILE_PATH = EXPERIMENT_DIR / ExperimentKinds.METHOD_LEVEL.value / experiment_project / f'experiment-{experiment_number}.json'
print(tf.math.add(1, 2))
print(tf.math.add([1, 2], [3, 4]))
print(tf.math.square(5))
print(tf.math.reduce_sum([1, 2, 3]))
print(tf.math.square(2) + tf.math.square(3))
start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT(experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tensorflow.linalg.matmul()')
x = tf.linalg.matmul([[1]], [[2, 3]])
after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tensorflow.linalg.matmul()', method_object=None, function_args=[[[1]], [[2, 3]]], function_kwargs=None)
print(x)
print(x.shape)
print(x.dtype)
import numpy as np
ndarray = np.ones([3, 3])
print('TensorFlow operations convert numpy arrays to Tensors automatically')
start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT(experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tensorflow.math.multiply()')
tensor = tf.math.multiply(ndarray, 42)
after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tensorflow.math.multiply()', method_object=None, function_args=[ndarray, 42], function_kwargs=None)
print(tensor)
print('And NumPy operations convert Tensors to NumPy arrays automatically')
print(np.add(tensor, 1))
print('The .numpy() method explicitly converts a Tensor to a numpy array')
print(tensor.numpy())
start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT(experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tensorflow.random.uniform()')
x = tf.random.uniform([3, 3])
after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tensorflow.random.uniform()', method_object=None, function_args=[[3, 3]], function_kwargs=None)
(print('Is there a GPU available: '),)
print(tf.config.list_physical_devices('GPU'))
(print('Is the Tensor on GPU #0:  '),)
print(x.device.endswith('GPU:0'))
import time

def time_matmul(x):
    start = time.time()
    for loop in range(10):
        start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT(experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tensorflow.linalg.matmul()')
        tf.linalg.matmul(x, x)
        after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tensorflow.linalg.matmul()', method_object=None, function_args=[x, x], function_kwargs=None)
    result = time.time() - start
    print('10 loops: {:0.2f}ms'.format(1000 * result))
print('On CPU:')
with tf.device('CPU:0'):
    start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT(experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tensorflow.random.uniform()')
    x = tf.random.uniform([1000, 1000])
    after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tensorflow.random.uniform()', method_object=None, function_args=[[1000, 1000]], function_kwargs=None)
    assert x.device.endswith('CPU:0')
    time_matmul(x)
if tf.config.list_physical_devices('GPU'):
    print('On GPU:')
    with tf.device('GPU:0'):
        start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT(experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tensorflow.random.uniform()')
        x = tf.random.uniform([1000, 1000])
        after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tensorflow.random.uniform()', method_object=None, function_args=[[1000, 1000]], function_kwargs=None)
        assert x.device.endswith('GPU:0')
        time_matmul(x)
start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT(experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tensorflow.data.Dataset.from_tensor_slices()')
ds_tensors = tf.data.Dataset.from_tensor_slices([1, 2, 3, 4, 5, 6])
after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tensorflow.data.Dataset.from_tensor_slices()', method_object=None, function_args=[[1, 2, 3, 4, 5, 6]], function_kwargs=None)
import tempfile
(_, filename) = tempfile.mkstemp()
with open(filename, 'w') as f:
    f.write('Line 1\nLine 2\nLine 3\n  ')
start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT(experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tensorflow.data.TextLineDataset()')
ds_file = tf.data.TextLineDataset(filename)
after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tensorflow.data.TextLineDataset()', method_object=None, function_args=[filename], function_kwargs=None)
start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT(experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tensorflow.data.Dataset.from_tensor_slices.map(tf.math.square).shuffle(2).batch()')
ds_tensors = ds_tensors.map(tf.math.square).shuffle(2).batch(2)
after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tensorflow.data.Dataset.from_tensor_slices.map(tf.math.square).shuffle(2).batch()', method_object=ds_tensors, function_args=[2], function_kwargs=None)
start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT(experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tensorflow.data.TextLineDataset.batch()')
ds_file = ds_file.batch(2)
after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tensorflow.data.TextLineDataset.batch()', method_object=ds_file, function_args=[2], function_kwargs=None)
print('Elements of ds_tensors:')
for x in ds_tensors:
    print(x)
print('\nElements in ds_file:')
for x in ds_file:
    print(x)
