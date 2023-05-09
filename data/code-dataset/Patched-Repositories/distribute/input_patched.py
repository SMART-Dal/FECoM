import tensorflow as tf
import numpy as np
import os
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
print(tf.__version__)
N_VIRTUAL_DEVICES = 2
physical_devices = custom_method(
tf.config.list_physical_devices('CPU'), imports='import tensorflow as tf;import numpy as np;import os', function_to_run='tf.config.list_physical_devices(*args)', method_object=None, object_signature=None, function_args=[eval('"CPU"')], function_kwargs={})
custom_method(
tf.config.set_logical_device_configuration(physical_devices[0], [tf.config.LogicalDeviceConfiguration() for _ in range(N_VIRTUAL_DEVICES)]), imports='import tensorflow as tf;import numpy as np;import os', function_to_run='tf.config.set_logical_device_configuration(*args)', method_object=None, object_signature=None, function_args=[eval('physical_devices[0]'), eval('[tf.config.LogicalDeviceConfiguration() for _ in range(N_VIRTUAL_DEVICES)]')], function_kwargs={})
print('Available devices:')
for (i, device) in enumerate(tf.config.list_logical_devices()):
    print('%d) %s' % (i, device))
global_batch_size = 16
dataset = custom_method(
tf.data.Dataset.from_tensors(([1.0], [1.0])).repeat(100).batch(global_batch_size), imports='import tensorflow as tf;import numpy as np;import os', function_to_run='tf.data.Dataset.from_tensors(([1.0], [1.0])).repeat(100).batch(*args)', method_object=None, object_signature=None, function_args=[eval('global_batch_size')], function_kwargs={})

@tf.function
def train_step(inputs):
    (features, labels) = inputs
    return labels - 0.3 * features
for inputs in dataset:
    print(train_step(inputs))
global_batch_size = 16
mirrored_strategy = custom_method(
tf.distribute.MirroredStrategy(), imports='import tensorflow as tf;import numpy as np;import os', function_to_run='tf.distribute.MirroredStrategy()', method_object=None, object_signature=None, function_args=[], function_kwargs={})
dataset = custom_method(
tf.data.Dataset.from_tensors(([1.0], [1.0])).repeat(100).batch(global_batch_size), imports='import tensorflow as tf;import numpy as np;import os', function_to_run='tf.data.Dataset.from_tensors(([1.0], [1.0])).repeat(100).batch(*args)', method_object=None, object_signature=None, function_args=[eval('global_batch_size')], function_kwargs={})
dist_dataset = mirrored_strategy.experimental_distribute_dataset(dataset)
print(next(iter(dist_dataset)))
dataset = custom_method(
tf.data.Dataset.from_tensors(([1.0], [1.0])).repeat(64).batch(16), imports='import tensorflow as tf;import numpy as np;import os', function_to_run='tf.data.Dataset.from_tensors(([1.0], [1.0])).repeat(64).batch(*args)', method_object=None, object_signature=None, function_args=[eval('16')], function_kwargs={})
options = custom_method(
tf.data.Options(), imports='import tensorflow as tf;import numpy as np;import os', function_to_run='tf.data.Options()', method_object=None, object_signature=None, function_args=[], function_kwargs={})
options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.DATA
dataset = dataset.with_options(options)
mirrored_strategy = custom_method(
tf.distribute.MirroredStrategy(), imports='import tensorflow as tf;import numpy as np;import os', function_to_run='tf.distribute.MirroredStrategy()', method_object=None, object_signature=None, function_args=[], function_kwargs={})

def dataset_fn(input_context):
    batch_size = input_context.get_per_replica_batch_size(global_batch_size)
    dataset = custom_method(
    tf.data.Dataset.from_tensors(([1.0], [1.0])).repeat(64).batch(16), imports='import tensorflow as tf;import numpy as np;import os', function_to_run='tf.data.Dataset.from_tensors(([1.0], [1.0])).repeat(64).batch(*args)', method_object=None, object_signature=None, function_args=[eval('16')], function_kwargs={})
    dataset = dataset.shard(input_context.num_input_pipelines, input_context.input_pipeline_id)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(2)
    return dataset
dist_dataset = mirrored_strategy.distribute_datasets_from_function(dataset_fn)
global_batch_size = 16
mirrored_strategy = custom_method(
tf.distribute.MirroredStrategy(), imports='import tensorflow as tf;import numpy as np;import os', function_to_run='tf.distribute.MirroredStrategy()', method_object=None, object_signature=None, function_args=[], function_kwargs={})
dataset = custom_method(
tf.data.Dataset.from_tensors(([1.0], [1.0])).repeat(100).batch(global_batch_size), imports='import tensorflow as tf;import numpy as np;import os', function_to_run='tf.data.Dataset.from_tensors(([1.0], [1.0])).repeat(100).batch(*args)', method_object=None, object_signature=None, function_args=[eval('global_batch_size')], function_kwargs={})
dist_dataset = mirrored_strategy.experimental_distribute_dataset(dataset)

@tf.function
def train_step(inputs):
    (features, labels) = inputs
    return labels - 0.3 * features
for x in dist_dataset:
    loss = mirrored_strategy.run(train_step, args=(x,))
    print('Loss is ', loss)
num_epochs = 10
steps_per_epoch = 5
for epoch in range(num_epochs):
    dist_iterator = iter(dist_dataset)
    for step in range(steps_per_epoch):
        loss = mirrored_strategy.run(train_step, args=(next(dist_iterator),))
        print('Loss is ', loss)
global_batch_size = 4
steps_per_loop = 5
strategy = custom_method(
tf.distribute.MirroredStrategy(), imports='import tensorflow as tf;import numpy as np;import os', function_to_run='tf.distribute.MirroredStrategy()', method_object=None, object_signature=None, function_args=[], function_kwargs={})
dataset = custom_method(
tf.data.Dataset.range(9).batch(global_batch_size), imports='import tensorflow as tf;import numpy as np;import os', function_to_run='tf.data.Dataset.range(9).batch(*args)', method_object=None, object_signature=None, function_args=[eval('global_batch_size')], function_kwargs={})
distributed_iterator = iter(strategy.experimental_distribute_dataset(dataset))

@tf.function
def train_fn(distributed_iterator):
    for _ in custom_method(
    tf.range(steps_per_loop), imports='import tensorflow as tf;import numpy as np;import os', function_to_run='tf.range(*args)', method_object=None, object_signature=None, function_args=[eval('steps_per_loop')], function_kwargs={}):
        optional_data = distributed_iterator.get_next_as_optional()
        if not optional_data.has_value():
            break
        per_replica_results = strategy.run(lambda x: x, args=(optional_data.get_value(),))
        custom_method(
        tf.print(strategy.experimental_local_results(per_replica_results)), imports='import tensorflow as tf;import numpy as np;import os', function_to_run='tf.print(*args)', method_object=None, object_signature=None, function_args=[eval('strategy.experimental_local_results(per_replica_results)')], function_kwargs={})
train_fn(distributed_iterator)
global_batch_size = 16
epochs = 5
steps_per_epoch = 5
mirrored_strategy = custom_method(
tf.distribute.MirroredStrategy(), imports='import tensorflow as tf;import numpy as np;import os', function_to_run='tf.distribute.MirroredStrategy()', method_object=None, object_signature=None, function_args=[], function_kwargs={})
dataset = custom_method(
tf.data.Dataset.from_tensors(([1.0], [1.0])).repeat(100).batch(global_batch_size), imports='import tensorflow as tf;import numpy as np;import os', function_to_run='tf.data.Dataset.from_tensors(([1.0], [1.0])).repeat(100).batch(*args)', method_object=None, object_signature=None, function_args=[eval('global_batch_size')], function_kwargs={})
dist_dataset = mirrored_strategy.experimental_distribute_dataset(dataset)

@custom_method(
tf.function(input_signature=[dist_dataset.element_spec]), imports='import tensorflow as tf;import numpy as np;import os', function_to_run='tf.function(**kwargs)', method_object=None, object_signature=None, function_args=[], function_kwargs={'input_signature': eval('[dist_dataset.element_spec]')})
def train_step(per_replica_inputs):

    def step_fn(inputs):
        return 2 * inputs
    return mirrored_strategy.run(step_fn, args=(per_replica_inputs,))
for _ in range(epochs):
    iterator = iter(dist_dataset)
    for _ in range(steps_per_epoch):
        output = train_step(next(iterator))
        custom_method(
        tf.print(output), imports='import tensorflow as tf;import numpy as np;import os', function_to_run='tf.print(*args)', method_object=None, object_signature=None, function_args=[eval('output')], function_kwargs={})
strategy = custom_method(
tf.distribute.MirroredStrategy(), imports='import tensorflow as tf;import numpy as np;import os', function_to_run='tf.distribute.MirroredStrategy()', method_object=None, object_signature=None, function_args=[], function_kwargs={})
vocab = ['a', 'b', 'c', 'd', 'f']
with strategy.scope():
    layer = custom_method(
    tf.keras.layers.StringLookup(vocabulary=vocab), imports='import tensorflow as tf;import numpy as np;import os', function_to_run='tf.keras.layers.StringLookup(**kwargs)', method_object=None, object_signature=None, function_args=[], function_kwargs={'vocabulary': eval('vocab')})

def dataset_fn(input_context):
    dataset = custom_method(
    tf.data.Dataset.from_tensor_slices(['a', 'c', 'e']).repeat(), imports='import tensorflow as tf;import numpy as np;import os', function_to_run="tf.data.Dataset.from_tensor_slices(['a', 'c', 'e']).repeat()", method_object=None, object_signature=None, function_args=[], function_kwargs={})
    global_batch_size = 4
    batch_size = input_context.get_per_replica_batch_size(global_batch_size)
    dataset = dataset.batch(batch_size)
    dataset = dataset.shard(input_context.num_input_pipelines, input_context.input_pipeline_id)

    def preprocess_with_kpl(input):
        return layer(input)
    processed_ds = dataset.map(preprocess_with_kpl)
    return processed_ds
distributed_dataset = strategy.distribute_datasets_from_function(dataset_fn)
distributed_dataset_iterator = iter(distributed_dataset)
for _ in range(3):
    print(next(distributed_dataset_iterator))
mirrored_strategy = custom_method(
tf.distribute.MirroredStrategy(), imports='import tensorflow as tf;import numpy as np;import os', function_to_run='tf.distribute.MirroredStrategy()', method_object=None, object_signature=None, function_args=[], function_kwargs={})
dataset_size = 24
batch_size = 6
dataset = custom_method(
tf.data.Dataset.range(dataset_size).enumerate().batch(batch_size), imports='import tensorflow as tf;import numpy as np;import os', function_to_run='tf.data.Dataset.range(dataset_size).enumerate().batch(*args)', method_object=None, object_signature=None, function_args=[eval('batch_size')], function_kwargs={})
dist_dataset = mirrored_strategy.experimental_distribute_dataset(dataset)

def predict(index, inputs):
    outputs = 2 * inputs
    return (index, outputs)
result = {}
for (index, inputs) in dist_dataset:
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
mirrored_strategy = custom_method(
tf.distribute.MirroredStrategy(), imports='import tensorflow as tf;import numpy as np;import os', function_to_run='tf.distribute.MirroredStrategy()', method_object=None, object_signature=None, function_args=[], function_kwargs={})

def value_fn(ctx):
    return custom_method(
    tf.constant(ctx.replica_id_in_sync_group), imports='import tensorflow as tf;import numpy as np;import os', function_to_run='tf.constant(*args)', method_object=None, object_signature=None, function_args=[eval('ctx.replica_id_in_sync_group')], function_kwargs={})
distributed_values = mirrored_strategy.experimental_distribute_values_from_function(value_fn)
for _ in range(4):
    result = mirrored_strategy.run(lambda x: x, args=(distributed_values,))
    print(result)
mirrored_strategy = custom_method(
tf.distribute.MirroredStrategy(), imports='import tensorflow as tf;import numpy as np;import os', function_to_run='tf.distribute.MirroredStrategy()', method_object=None, object_signature=None, function_args=[], function_kwargs={})

def input_gen():
    while True:
        yield np.random.rand(4)
dataset = custom_method(
tf.data.Dataset.from_generator(input_gen, output_types=tf.float32, output_shapes=tf.TensorShape([4])), imports='import tensorflow as tf;import numpy as np;import os', function_to_run='tf.data.Dataset.from_generator(*args, **kwargs)', method_object=None, object_signature=None, function_args=[eval('input_gen')], function_kwargs={'output_types': eval('tf.float32'), 'output_shapes': eval('tf.TensorShape([4])')})
dist_dataset = mirrored_strategy.experimental_distribute_dataset(dataset)
iterator = iter(dist_dataset)
for _ in range(4):
    result = mirrored_strategy.run(lambda x: x, args=(next(iterator),))
    print(result)
