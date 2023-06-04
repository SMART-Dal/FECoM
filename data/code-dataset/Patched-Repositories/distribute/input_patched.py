import tensorflow as tf
import numpy as np
import os
import sys
from tool.patching.patching_config import EXPERIMENT_DIR
from tool.measurement.execution import before_execution as before_execution_INSERTED_INTO_SCRIPT
from tool.measurement.execution import after_execution as after_execution_INSERTED_INTO_SCRIPT
experiment_number = sys.argv[1]
experiment_project = sys.argv[2]
EXPERIMENT_FILE_PATH = EXPERIMENT_DIR / 'local-execution' / experiment_project / f'experiment-{experiment_number}.json'
print(tf.__version__)
N_VIRTUAL_DEVICES = 2
start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT()
physical_devices = tf.config.list_physical_devices('CPU')
after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tf.config.list_physical_devices()', method_object=None, function_args=['CPU'], function_kwargs=None)
start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT()
tf.config.set_logical_device_configuration(physical_devices[0], [tf.config.LogicalDeviceConfiguration() for _ in range(N_VIRTUAL_DEVICES)])
after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tf.config.set_logical_device_configuration()', method_object=None, function_args=[physical_devices[0], [tf.config.LogicalDeviceConfiguration() for _ in range(N_VIRTUAL_DEVICES)]], function_kwargs=None)
print('Available devices:')
for (i, device) in enumerate(tf.config.list_logical_devices()):
    print('%d) %s' % (i, device))
global_batch_size = 16
start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT()
dataset = tf.data.Dataset.from_tensors(([1.0], [1.0])).repeat(100).batch(global_batch_size)
after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tf.data.Dataset.from_tensors(([1.0], [1.0])).repeat(100).batch()', method_object=None, function_args=[global_batch_size], function_kwargs=None)

@tf.function
def train_step(inputs):
    (features, labels) = inputs
    return labels - 0.3 * features
for inputs in dataset:
    print(train_step(inputs))
global_batch_size = 16
start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT()
mirrored_strategy = tf.distribute.MirroredStrategy()
after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tf.distribute.MirroredStrategy()', method_object=None, function_args=None, function_kwargs=None)
start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT()
dataset = tf.data.Dataset.from_tensors(([1.0], [1.0])).repeat(100).batch(global_batch_size)
after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tf.data.Dataset.from_tensors(([1.0], [1.0])).repeat(100).batch()', method_object=None, function_args=[global_batch_size], function_kwargs=None)
start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT()
dist_dataset = mirrored_strategy.experimental_distribute_dataset(dataset)
after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tf.distribute.MirroredStrategy.experimental_distribute_dataset()', method_object=mirrored_strategy, function_args=[dataset], function_kwargs=None)
print(next(iter(dist_dataset)))
start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT()
dataset = tf.data.Dataset.from_tensors(([1.0], [1.0])).repeat(64).batch(16)
after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tf.data.Dataset.from_tensors(([1.0], [1.0])).repeat(64).batch()', method_object=None, function_args=[16], function_kwargs=None)
start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT()
options = tf.data.Options()
after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tf.data.Options()', method_object=None, function_args=None, function_kwargs=None)
options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.DATA
start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT()
dataset = dataset.with_options(options)
after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='dataset.with_options.with_options()', method_object=dataset, function_args=[options], function_kwargs=None)
start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT()
mirrored_strategy = tf.distribute.MirroredStrategy()
after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tf.distribute.MirroredStrategy()', method_object=None, function_args=None, function_kwargs=None)

def dataset_fn(input_context):
    batch_size = input_context.get_per_replica_batch_size(global_batch_size)
    start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT()
    dataset = tf.data.Dataset.from_tensors(([1.0], [1.0])).repeat(64).batch(16)
    after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tf.data.Dataset.from_tensors(([1.0], [1.0])).repeat(64).batch()', method_object=None, function_args=[16], function_kwargs=None)
    start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT()
    dataset = dataset.shard(input_context.num_input_pipelines, input_context.input_pipeline_id)
    after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='dataset.shard.shard()', method_object=dataset, function_args=[input_context.num_input_pipelines, input_context.input_pipeline_id], function_kwargs=None)
    start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT()
    dataset = dataset.batch(batch_size)
    after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='dataset.batch.batch()', method_object=dataset, function_args=[batch_size], function_kwargs=None)
    start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT()
    dataset = dataset.prefetch(2)
    after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='dataset.prefetch.prefetch()', method_object=dataset, function_args=[2], function_kwargs=None)
    return dataset
start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT()
dist_dataset = mirrored_strategy.distribute_datasets_from_function(dataset_fn)
after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tf.distribute.MirroredStrategy.distribute_datasets_from_function()', method_object=mirrored_strategy, function_args=[dataset_fn], function_kwargs=None)
global_batch_size = 16
start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT()
mirrored_strategy = tf.distribute.MirroredStrategy()
after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tf.distribute.MirroredStrategy()', method_object=None, function_args=None, function_kwargs=None)
start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT()
dataset = tf.data.Dataset.from_tensors(([1.0], [1.0])).repeat(100).batch(global_batch_size)
after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tf.data.Dataset.from_tensors(([1.0], [1.0])).repeat(100).batch()', method_object=None, function_args=[global_batch_size], function_kwargs=None)
start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT()
dist_dataset = mirrored_strategy.experimental_distribute_dataset(dataset)
after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tf.distribute.MirroredStrategy.experimental_distribute_dataset()', method_object=mirrored_strategy, function_args=[dataset], function_kwargs=None)

@tf.function
def train_step(inputs):
    (features, labels) = inputs
    return labels - 0.3 * features
for x in dist_dataset:
    start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT()
    loss = mirrored_strategy.run(train_step, args=(x,))
    after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tf.distribute.MirroredStrategy.run()', method_object=mirrored_strategy, function_args=[train_step], function_kwargs={'args': (x,)})
    print('Loss is ', loss)
num_epochs = 10
steps_per_epoch = 5
for epoch in range(num_epochs):
    dist_iterator = iter(dist_dataset)
    for step in range(steps_per_epoch):
        start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT()
        loss = mirrored_strategy.run(train_step, args=(next(dist_iterator),))
        after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tf.distribute.MirroredStrategy.run()', method_object=mirrored_strategy, function_args=[train_step], function_kwargs={'args': (next(dist_iterator),)})
        print('Loss is ', loss)
global_batch_size = 4
steps_per_loop = 5
start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT()
strategy = tf.distribute.MirroredStrategy()
after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tf.distribute.MirroredStrategy()', method_object=None, function_args=None, function_kwargs=None)
start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT()
dataset = tf.data.Dataset.range(9).batch(global_batch_size)
after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tf.data.Dataset.range(9).batch()', method_object=None, function_args=[global_batch_size], function_kwargs=None)
distributed_iterator = iter(strategy.experimental_distribute_dataset(dataset))

@tf.function
def train_fn(distributed_iterator):
    for _ in tf.range(steps_per_loop):
        optional_data = distributed_iterator.get_next_as_optional()
        if not optional_data.has_value():
            break
        start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT()
        per_replica_results = strategy.run(lambda x: x, args=(optional_data.get_value(),))
        after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tf.distribute.MirroredStrategy.run()', method_object=strategy, function_args=[lambda x: x], function_kwargs={'args': (optional_data.get_value(),)})
        start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT()
        tf.print(strategy.experimental_local_results(per_replica_results))
        after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tf.print()', method_object=None, function_args=[strategy.experimental_local_results(per_replica_results)], function_kwargs=None)
train_fn(distributed_iterator)
global_batch_size = 16
epochs = 5
steps_per_epoch = 5
start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT()
mirrored_strategy = tf.distribute.MirroredStrategy()
after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tf.distribute.MirroredStrategy()', method_object=None, function_args=None, function_kwargs=None)
start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT()
dataset = tf.data.Dataset.from_tensors(([1.0], [1.0])).repeat(100).batch(global_batch_size)
after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tf.data.Dataset.from_tensors(([1.0], [1.0])).repeat(100).batch()', method_object=None, function_args=[global_batch_size], function_kwargs=None)
start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT()
dist_dataset = mirrored_strategy.experimental_distribute_dataset(dataset)
after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tf.distribute.MirroredStrategy.experimental_distribute_dataset()', method_object=mirrored_strategy, function_args=[dataset], function_kwargs=None)

@tf.function(input_signature=[dist_dataset.element_spec])
def train_step(per_replica_inputs):

    def step_fn(inputs):
        return 2 * inputs
    return mirrored_strategy.run(step_fn, args=(per_replica_inputs,))
for _ in range(epochs):
    iterator = iter(dist_dataset)
    for _ in range(steps_per_epoch):
        output = train_step(next(iterator))
        start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT()
        tf.print(output)
        after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tf.print()', method_object=None, function_args=[output], function_kwargs=None)
start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT()
strategy = tf.distribute.MirroredStrategy()
after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tf.distribute.MirroredStrategy()', method_object=None, function_args=None, function_kwargs=None)
vocab = ['a', 'b', 'c', 'd', 'f']
with strategy.scope():
    start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT()
    layer = tf.keras.layers.StringLookup(vocabulary=vocab)
    after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tf.keras.layers.StringLookup()', method_object=None, function_args=None, function_kwargs={'vocabulary': vocab})

def dataset_fn(input_context):
    start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT()
    dataset = tf.data.Dataset.from_tensor_slices(['a', 'c', 'e']).repeat()
    after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run="tf.data.Dataset.from_tensor_slices(['a', 'c', 'e']).repeat()", method_object=None, function_args=None, function_kwargs=None)
    global_batch_size = 4
    batch_size = input_context.get_per_replica_batch_size(global_batch_size)
    start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT()
    dataset = dataset.batch(batch_size)
    after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='dataset.batch.batch()', method_object=dataset, function_args=[batch_size], function_kwargs=None)
    start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT()
    dataset = dataset.shard(input_context.num_input_pipelines, input_context.input_pipeline_id)
    after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='dataset.shard.shard()', method_object=dataset, function_args=[input_context.num_input_pipelines, input_context.input_pipeline_id], function_kwargs=None)

    def preprocess_with_kpl(input):
        return layer(input)
    start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT()
    processed_ds = dataset.map(preprocess_with_kpl)
    after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='dataset.shard.map()', method_object=dataset, function_args=[preprocess_with_kpl], function_kwargs=None)
    return processed_ds
start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT()
distributed_dataset = strategy.distribute_datasets_from_function(dataset_fn)
after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tf.distribute.MirroredStrategy.distribute_datasets_from_function()', method_object=strategy, function_args=[dataset_fn], function_kwargs=None)
distributed_dataset_iterator = iter(distributed_dataset)
for _ in range(3):
    print(next(distributed_dataset_iterator))
start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT()
mirrored_strategy = tf.distribute.MirroredStrategy()
after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tf.distribute.MirroredStrategy()', method_object=None, function_args=None, function_kwargs=None)
dataset_size = 24
batch_size = 6
start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT()
dataset = tf.data.Dataset.range(dataset_size).enumerate().batch(batch_size)
after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tf.data.Dataset.range(dataset_size).enumerate().batch()', method_object=None, function_args=[batch_size], function_kwargs=None)
start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT()
dist_dataset = mirrored_strategy.experimental_distribute_dataset(dataset)
after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tf.distribute.MirroredStrategy.experimental_distribute_dataset()', method_object=mirrored_strategy, function_args=[dataset], function_kwargs=None)

def predict(index, inputs):
    outputs = 2 * inputs
    return (index, outputs)
result = {}
for (index, inputs) in dist_dataset:
    start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT()
    (output_index, outputs) = mirrored_strategy.run(predict, args=(index, inputs))
    after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tf.distribute.MirroredStrategy.run()', method_object=mirrored_strategy, function_args=[predict], function_kwargs={'args': (index, inputs)})
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
start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT()
mirrored_strategy = tf.distribute.MirroredStrategy()
after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tf.distribute.MirroredStrategy()', method_object=None, function_args=None, function_kwargs=None)

def value_fn(ctx):
    return tf.constant(ctx.replica_id_in_sync_group)
start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT()
distributed_values = mirrored_strategy.experimental_distribute_values_from_function(value_fn)
after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tf.distribute.MirroredStrategy.experimental_distribute_values_from_function()', method_object=mirrored_strategy, function_args=[value_fn], function_kwargs=None)
for _ in range(4):
    start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT()
    result = mirrored_strategy.run(lambda x: x, args=(distributed_values,))
    after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tf.distribute.MirroredStrategy.run()', method_object=mirrored_strategy, function_args=[lambda x: x], function_kwargs={'args': (distributed_values,)})
    print(result)
start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT()
mirrored_strategy = tf.distribute.MirroredStrategy()
after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tf.distribute.MirroredStrategy()', method_object=None, function_args=None, function_kwargs=None)

def input_gen():
    while True:
        yield np.random.rand(4)
start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT()
dataset = tf.data.Dataset.from_generator(input_gen, output_types=tf.float32, output_shapes=tf.TensorShape([4]))
after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tf.data.Dataset.from_generator()', method_object=None, function_args=[input_gen], function_kwargs={'output_types': tf.float32, 'output_shapes': tf.TensorShape([4])})
start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT()
dist_dataset = mirrored_strategy.experimental_distribute_dataset(dataset)
after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tf.distribute.MirroredStrategy.experimental_distribute_dataset()', method_object=mirrored_strategy, function_args=[dataset], function_kwargs=None)
iterator = iter(dist_dataset)
for _ in range(4):
    start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT()
    result = mirrored_strategy.run(lambda x: x, args=(next(iterator),))
    after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tf.distribute.MirroredStrategy.run()', method_object=mirrored_strategy, function_args=[lambda x: x], function_kwargs={'args': (next(iterator),)})
    print(result)
