import tensorflow as tf
import numpy as np
import os
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
print(tf.__version__)
N_VIRTUAL_DEVICES = 2
custom_method(imports='import os;import tensorflow as tf;import numpy as np', function_to_run='tf.config.list_physical_devices(*args)', method_object=None, object_signature=None, function_args=[eval('"CPU"')], function_kwargs={})
physical_devices = tf.config.list_physical_devices('CPU')
custom_method(imports='import os;import tensorflow as tf;import numpy as np', function_to_run='tf.config.set_logical_device_configuration(*args)', method_object=None, object_signature=None, function_args=[eval('physical_devices[0]'), eval('[tf.config.LogicalDeviceConfiguration() for _ in range(N_VIRTUAL_DEVICES)]')], function_kwargs={})
tf.config.set_logical_device_configuration(physical_devices[0], [tf.config.LogicalDeviceConfiguration() for _ in range(N_VIRTUAL_DEVICES)])
print('Available devices:')
for (i, device) in enumerate(tf.config.list_logical_devices()):
    print('%d) %s' % (i, device))
global_batch_size = 16
custom_method(imports='import os;import tensorflow as tf;import numpy as np', function_to_run='tf.data.Dataset.from_tensors(([1.0], [1.0])).repeat(100).batch(*args)', method_object=None, object_signature=None, function_args=[eval('global_batch_size')], function_kwargs={})
dataset = tf.data.Dataset.from_tensors(([1.0], [1.0])).repeat(100).batch(global_batch_size)

@tf.function
def train_step(inputs):
    (features, labels) = inputs
    return labels - 0.3 * features
for inputs in dataset:
    print(train_step(inputs))
global_batch_size = 16
custom_method(imports='import os;import tensorflow as tf;import numpy as np', function_to_run='tf.distribute.MirroredStrategy()', method_object=None, object_signature=None, function_args=[], function_kwargs={})
mirrored_strategy = tf.distribute.MirroredStrategy()
custom_method(imports='import os;import tensorflow as tf;import numpy as np', function_to_run='tf.data.Dataset.from_tensors(([1.0], [1.0])).repeat(100).batch(*args)', method_object=None, object_signature=None, function_args=[eval('global_batch_size')], function_kwargs={})
dataset = tf.data.Dataset.from_tensors(([1.0], [1.0])).repeat(100).batch(global_batch_size)
custom_method(imports='import os;import tensorflow as tf;import numpy as np', function_to_run='obj.experimental_distribute_dataset(*args)', method_object=eval('mirrored_strategy'), object_signature=None, function_args=[eval('dataset')], function_kwargs={}, custom_class=None)
dist_dataset = mirrored_strategy.experimental_distribute_dataset(dataset)
print(next(iter(dist_dataset)))
custom_method(imports='import os;import tensorflow as tf;import numpy as np', function_to_run='tf.data.Dataset.from_tensors(([1.0], [1.0])).repeat(64).batch(*args)', method_object=None, object_signature=None, function_args=[eval('16')], function_kwargs={})
dataset = tf.data.Dataset.from_tensors(([1.0], [1.0])).repeat(64).batch(16)
custom_method(imports='import os;import tensorflow as tf;import numpy as np', function_to_run='tf.data.Options()', method_object=None, object_signature=None, function_args=[], function_kwargs={})
options = tf.data.Options()
options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.DATA
custom_method(imports='import os;import tensorflow as tf;import numpy as np', function_to_run='obj.with_options(*args)', method_object=eval('dataset'), object_signature=None, function_args=[eval('options')], function_kwargs={}, custom_class=None)
dataset = dataset.with_options(options)
custom_method(imports='import os;import tensorflow as tf;import numpy as np', function_to_run='tf.distribute.MirroredStrategy()', method_object=None, object_signature=None, function_args=[], function_kwargs={})
mirrored_strategy = tf.distribute.MirroredStrategy()

def dataset_fn(input_context):
    batch_size = input_context.get_per_replica_batch_size(global_batch_size)
    custom_method(imports='import os;import tensorflow as tf;import numpy as np', function_to_run='tf.data.Dataset.from_tensors(([1.0], [1.0])).repeat(64).batch(*args)', method_object=None, object_signature=None, function_args=[eval('16')], function_kwargs={})
    dataset = tf.data.Dataset.from_tensors(([1.0], [1.0])).repeat(64).batch(16)
    custom_method(imports='import os;import tensorflow as tf;import numpy as np', function_to_run='obj.shard(*args)', method_object=eval('dataset'), object_signature=None, function_args=[eval('input_context.num_input_pipelines'), eval('input_context.input_pipeline_id')], function_kwargs={}, custom_class=None)
    dataset = dataset.shard(input_context.num_input_pipelines, input_context.input_pipeline_id)
    custom_method(imports='import os;import tensorflow as tf;import numpy as np', function_to_run='obj.batch(*args)', method_object=eval('dataset'), object_signature=None, function_args=[eval('batch_size')], function_kwargs={}, custom_class=None)
    dataset = dataset.batch(batch_size)
    custom_method(imports='import os;import tensorflow as tf;import numpy as np', function_to_run='obj.prefetch(*args)', method_object=eval('dataset'), object_signature=None, function_args=[eval('2')], function_kwargs={}, custom_class=None)
    dataset = dataset.prefetch(2)
    return dataset
custom_method(imports='import os;import tensorflow as tf;import numpy as np', function_to_run='obj.distribute_datasets_from_function(*args)', method_object=eval('mirrored_strategy'), object_signature=None, function_args=[eval('dataset_fn')], function_kwargs={}, custom_class=None)
dist_dataset = mirrored_strategy.distribute_datasets_from_function(dataset_fn)
global_batch_size = 16
custom_method(imports='import os;import tensorflow as tf;import numpy as np', function_to_run='tf.distribute.MirroredStrategy()', method_object=None, object_signature=None, function_args=[], function_kwargs={})
mirrored_strategy = tf.distribute.MirroredStrategy()
custom_method(imports='import os;import tensorflow as tf;import numpy as np', function_to_run='tf.data.Dataset.from_tensors(([1.0], [1.0])).repeat(100).batch(*args)', method_object=None, object_signature=None, function_args=[eval('global_batch_size')], function_kwargs={})
dataset = tf.data.Dataset.from_tensors(([1.0], [1.0])).repeat(100).batch(global_batch_size)
custom_method(imports='import os;import tensorflow as tf;import numpy as np', function_to_run='obj.experimental_distribute_dataset(*args)', method_object=eval('mirrored_strategy'), object_signature=None, function_args=[eval('dataset')], function_kwargs={}, custom_class=None)
dist_dataset = mirrored_strategy.experimental_distribute_dataset(dataset)

@tf.function
def train_step(inputs):
    (features, labels) = inputs
    return labels - 0.3 * features
for x in dist_dataset:
    custom_method(imports='import os;import tensorflow as tf;import numpy as np', function_to_run='obj.run(*args, **kwargs)', method_object=eval('mirrored_strategy'), object_signature=None, function_args=[eval('train_step')], function_kwargs={'args': eval('(x,)')}, custom_class=None)
    loss = mirrored_strategy.run(train_step, args=(x,))
    print('Loss is ', loss)
num_epochs = 10
steps_per_epoch = 5
for epoch in range(num_epochs):
    dist_iterator = iter(dist_dataset)
    for step in range(steps_per_epoch):
        custom_method(imports='import os;import tensorflow as tf;import numpy as np', function_to_run='obj.run(*args, **kwargs)', method_object=eval('mirrored_strategy'), object_signature=None, function_args=[eval('train_step')], function_kwargs={'args': eval('(next(dist_iterator),)')}, custom_class=None)
        loss = mirrored_strategy.run(train_step, args=(next(dist_iterator),))
        print('Loss is ', loss)
global_batch_size = 4
steps_per_loop = 5
custom_method(imports='import os;import tensorflow as tf;import numpy as np', function_to_run='tf.distribute.MirroredStrategy()', method_object=None, object_signature=None, function_args=[], function_kwargs={})
strategy = tf.distribute.MirroredStrategy()
custom_method(imports='import os;import tensorflow as tf;import numpy as np', function_to_run='tf.data.Dataset.range(9).batch(*args)', method_object=None, object_signature=None, function_args=[eval('global_batch_size')], function_kwargs={})
dataset = tf.data.Dataset.range(9).batch(global_batch_size)
distributed_iterator = iter(strategy.experimental_distribute_dataset(dataset))

@tf.function
def train_fn(distributed_iterator):
    for _ in tf.range(steps_per_loop):
        optional_data = distributed_iterator.get_next_as_optional()
        if not optional_data.has_value():
            break
        custom_method(imports='import os;import tensorflow as tf;import numpy as np', function_to_run='obj.run(*args, **kwargs)', method_object=eval('strategy'), object_signature=None, function_args=[eval('lambda x: x')], function_kwargs={'args': eval('(optional_data.get_value(),)')}, custom_class=None)
        per_replica_results = strategy.run(lambda x: x, args=(optional_data.get_value(),))
        custom_method(imports='import os;import tensorflow as tf;import numpy as np', function_to_run='tf.print(*args)', method_object=None, object_signature=None, function_args=[eval('strategy.experimental_local_results(per_replica_results)')], function_kwargs={})
        tf.print(strategy.experimental_local_results(per_replica_results))
train_fn(distributed_iterator)
global_batch_size = 16
epochs = 5
steps_per_epoch = 5
custom_method(imports='import os;import tensorflow as tf;import numpy as np', function_to_run='tf.distribute.MirroredStrategy()', method_object=None, object_signature=None, function_args=[], function_kwargs={})
mirrored_strategy = tf.distribute.MirroredStrategy()
custom_method(imports='import os;import tensorflow as tf;import numpy as np', function_to_run='tf.data.Dataset.from_tensors(([1.0], [1.0])).repeat(100).batch(*args)', method_object=None, object_signature=None, function_args=[eval('global_batch_size')], function_kwargs={})
dataset = tf.data.Dataset.from_tensors(([1.0], [1.0])).repeat(100).batch(global_batch_size)
custom_method(imports='import os;import tensorflow as tf;import numpy as np', function_to_run='obj.experimental_distribute_dataset(*args)', method_object=eval('mirrored_strategy'), object_signature=None, function_args=[eval('dataset')], function_kwargs={}, custom_class=None)
dist_dataset = mirrored_strategy.experimental_distribute_dataset(dataset)

@tf.function(input_signature=[dist_dataset.element_spec])
def train_step(per_replica_inputs):

    def step_fn(inputs):
        return 2 * inputs
    return mirrored_strategy.run(step_fn, args=(per_replica_inputs,))
for _ in range(epochs):
    iterator = iter(dist_dataset)
    for _ in range(steps_per_epoch):
        output = train_step(next(iterator))
        custom_method(imports='import os;import tensorflow as tf;import numpy as np', function_to_run='tf.print(*args)', method_object=None, object_signature=None, function_args=[eval('output')], function_kwargs={})
        tf.print(output)
custom_method(imports='import os;import tensorflow as tf;import numpy as np', function_to_run='tf.distribute.MirroredStrategy()', method_object=None, object_signature=None, function_args=[], function_kwargs={})
strategy = tf.distribute.MirroredStrategy()
vocab = ['a', 'b', 'c', 'd', 'f']
with strategy.scope():
    custom_method(imports='import os;import tensorflow as tf;import numpy as np', function_to_run='tf.keras.layers.StringLookup(**kwargs)', method_object=None, object_signature=None, function_args=[], function_kwargs={'vocabulary': eval('vocab')})
    layer = tf.keras.layers.StringLookup(vocabulary=vocab)

def dataset_fn(input_context):
    custom_method(imports='import os;import tensorflow as tf;import numpy as np', function_to_run="tf.data.Dataset.from_tensor_slices(['a', 'c', 'e']).repeat()", method_object=None, object_signature=None, function_args=[], function_kwargs={})
    dataset = tf.data.Dataset.from_tensor_slices(['a', 'c', 'e']).repeat()
    global_batch_size = 4
    batch_size = input_context.get_per_replica_batch_size(global_batch_size)
    custom_method(imports='import os;import tensorflow as tf;import numpy as np', function_to_run='obj.batch(*args)', method_object=eval('dataset'), object_signature=None, function_args=[eval('batch_size')], function_kwargs={}, custom_class=None)
    dataset = dataset.batch(batch_size)
    custom_method(imports='import os;import tensorflow as tf;import numpy as np', function_to_run='obj.shard(*args)', method_object=eval('dataset'), object_signature=None, function_args=[eval('input_context.num_input_pipelines'), eval('input_context.input_pipeline_id')], function_kwargs={}, custom_class=None)
    dataset = dataset.shard(input_context.num_input_pipelines, input_context.input_pipeline_id)

    def preprocess_with_kpl(input):
        return layer(input)
    custom_method(imports='import os;import tensorflow as tf;import numpy as np', function_to_run='obj.map(*args)', method_object=eval('dataset'), object_signature=None, function_args=[eval('preprocess_with_kpl')], function_kwargs={}, custom_class=None)
    processed_ds = dataset.map(preprocess_with_kpl)
    return processed_ds
custom_method(imports='import os;import tensorflow as tf;import numpy as np', function_to_run='obj.distribute_datasets_from_function(*args)', method_object=eval('strategy'), object_signature=None, function_args=[eval('dataset_fn')], function_kwargs={}, custom_class=None)
distributed_dataset = strategy.distribute_datasets_from_function(dataset_fn)
distributed_dataset_iterator = iter(distributed_dataset)
for _ in range(3):
    print(next(distributed_dataset_iterator))
custom_method(imports='import os;import tensorflow as tf;import numpy as np', function_to_run='tf.distribute.MirroredStrategy()', method_object=None, object_signature=None, function_args=[], function_kwargs={})
mirrored_strategy = tf.distribute.MirroredStrategy()
dataset_size = 24
batch_size = 6
custom_method(imports='import os;import tensorflow as tf;import numpy as np', function_to_run='tf.data.Dataset.range(dataset_size).enumerate().batch(*args)', method_object=None, object_signature=None, function_args=[eval('batch_size')], function_kwargs={})
dataset = tf.data.Dataset.range(dataset_size).enumerate().batch(batch_size)
custom_method(imports='import os;import tensorflow as tf;import numpy as np', function_to_run='obj.experimental_distribute_dataset(*args)', method_object=eval('mirrored_strategy'), object_signature=None, function_args=[eval('dataset')], function_kwargs={}, custom_class=None)
dist_dataset = mirrored_strategy.experimental_distribute_dataset(dataset)

def predict(index, inputs):
    outputs = 2 * inputs
    return (index, outputs)
result = {}
for (index, inputs) in dist_dataset:
    custom_method(imports='import os;import tensorflow as tf;import numpy as np', function_to_run='obj.run(*args, **kwargs)', method_object=eval('mirrored_strategy'), object_signature=None, function_args=[eval('predict')], function_kwargs={'args': eval('(index, inputs)')}, custom_class=None)
    (output_index, outputs) = mirrored_strategy.run(predict, args=(index, inputs))
    indices = list(mirrored_strategy.experimental_local_results(output_index))
    rindices = []
    for a in indices:
        rindices.extend(a.numpy())
    outputs = list(mirrored_strategy.experimental_local_results(outputs))
    routputs = []
    for a in outputs:
        routputs.extend(a.numpy())
    for (i, value) in zip(rindices, routputs):
        result[i] = value
print(result)
custom_method(imports='import os;import tensorflow as tf;import numpy as np', function_to_run='tf.distribute.MirroredStrategy()', method_object=None, object_signature=None, function_args=[], function_kwargs={})
mirrored_strategy = tf.distribute.MirroredStrategy()

def value_fn(ctx):
    return tf.constant(ctx.replica_id_in_sync_group)
custom_method(imports='import os;import tensorflow as tf;import numpy as np', function_to_run='obj.experimental_distribute_values_from_function(*args)', method_object=eval('mirrored_strategy'), object_signature=None, function_args=[eval('value_fn')], function_kwargs={}, custom_class=None)
distributed_values = mirrored_strategy.experimental_distribute_values_from_function(value_fn)
for _ in range(4):
    custom_method(imports='import os;import tensorflow as tf;import numpy as np', function_to_run='obj.run(*args, **kwargs)', method_object=eval('mirrored_strategy'), object_signature=None, function_args=[eval('lambda x: x')], function_kwargs={'args': eval('(distributed_values,)')}, custom_class=None)
    result = mirrored_strategy.run(lambda x: x, args=(distributed_values,))
    print(result)
custom_method(imports='import os;import tensorflow as tf;import numpy as np', function_to_run='tf.distribute.MirroredStrategy()', method_object=None, object_signature=None, function_args=[], function_kwargs={})
mirrored_strategy = tf.distribute.MirroredStrategy()

def input_gen():
    while True:
        yield np.random.rand(4)
custom_method(imports='import os;import tensorflow as tf;import numpy as np', function_to_run='tf.data.Dataset.from_generator(*args, **kwargs)', method_object=None, object_signature=None, function_args=[eval('input_gen')], function_kwargs={'output_types': eval('tf.float32'), 'output_shapes': eval('tf.TensorShape([4])')})
dataset = tf.data.Dataset.from_generator(input_gen, output_types=tf.float32, output_shapes=tf.TensorShape([4]))
custom_method(imports='import os;import tensorflow as tf;import numpy as np', function_to_run='obj.experimental_distribute_dataset(*args)', method_object=eval('mirrored_strategy'), object_signature=None, function_args=[eval('dataset')], function_kwargs={}, custom_class=None)
dist_dataset = mirrored_strategy.experimental_distribute_dataset(dataset)
iterator = iter(dist_dataset)
for _ in range(4):
    custom_method(imports='import os;import tensorflow as tf;import numpy as np', function_to_run='obj.run(*args, **kwargs)', method_object=eval('mirrored_strategy'), object_signature=None, function_args=[eval('lambda x: x')], function_kwargs={'args': eval('(next(iterator),)')}, custom_class=None)
    result = mirrored_strategy.run(lambda x: x, args=(next(iterator),))
    print(result)
