import sys
from fecom.patching.patching_config import EXPERIMENT_DIR
from fecom.measurement.execution import before_execution as before_execution_INSERTED_INTO_SCRIPT
from fecom.measurement.execution import after_execution as after_execution_INSERTED_INTO_SCRIPT
from fecom.experiment.experiment_kinds import ExperimentKinds
experiment_number = sys.argv[1]
experiment_project = sys.argv[2]
EXPERIMENT_FILE_PATH = EXPERIMENT_DIR / ExperimentKinds.PROJECT_LEVEL.value / experiment_project / f'experiment-{experiment_number}.json'
start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT(experiment_file_path=EXPERIMENT_FILE_PATH, enable_skip_calls=False)
import tensorflow as tf
import numpy as np
import os
print(tf.__version__)
N_VIRTUAL_DEVICES = 2
physical_devices = tf.config.list_physical_devices('CPU')
tf.config.set_logical_device_configuration(physical_devices[0], [tf.config.LogicalDeviceConfiguration() for _ in range(N_VIRTUAL_DEVICES)])
print('Available devices:')
for (i, device) in enumerate(tf.config.list_logical_devices()):
    print('%d) %s' % (i, device))
global_batch_size = 16
dataset = tf.data.Dataset.from_tensors(([1.0], [1.0])).repeat(100).batch(global_batch_size)

@tf.function
def train_step(inputs):
    (features, labels) = inputs
    return labels - 0.3 * features
for inputs in dataset:
    print(train_step(inputs))
global_batch_size = 16
mirrored_strategy = tf.distribute.MirroredStrategy()
dataset = tf.data.Dataset.from_tensors(([1.0], [1.0])).repeat(100).batch(global_batch_size)
dist_dataset = mirrored_strategy.experimental_distribute_dataset(dataset)
print(next(iter(dist_dataset)))
dataset = tf.data.Dataset.from_tensors(([1.0], [1.0])).repeat(64).batch(16)
options = tf.data.Options()
options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.DATA
dataset = dataset.with_options(options)
mirrored_strategy = tf.distribute.MirroredStrategy()

def dataset_fn(input_context):
    batch_size = input_context.get_per_replica_batch_size(global_batch_size)
    dataset = tf.data.Dataset.from_tensors(([1.0], [1.0])).repeat(64).batch(16)
    dataset = dataset.shard(input_context.num_input_pipelines, input_context.input_pipeline_id)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(2)
    return dataset
dist_dataset = mirrored_strategy.distribute_datasets_from_function(dataset_fn)
global_batch_size = 16
mirrored_strategy = tf.distribute.MirroredStrategy()
dataset = tf.data.Dataset.from_tensors(([1.0], [1.0])).repeat(100).batch(global_batch_size)
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
strategy = tf.distribute.MirroredStrategy()
dataset = tf.data.Dataset.range(9).batch(global_batch_size)
distributed_iterator = iter(strategy.experimental_distribute_dataset(dataset))

@tf.function
def train_fn(distributed_iterator):
    for _ in tf.range(steps_per_loop):
        optional_data = distributed_iterator.get_next_as_optional()
        if not optional_data.has_value():
            break
        per_replica_results = strategy.run(lambda x: x, args=(optional_data.get_value(),))
        tf.print(strategy.experimental_local_results(per_replica_results))
train_fn(distributed_iterator)
global_batch_size = 16
epochs = 5
steps_per_epoch = 5
mirrored_strategy = tf.distribute.MirroredStrategy()
dataset = tf.data.Dataset.from_tensors(([1.0], [1.0])).repeat(100).batch(global_batch_size)
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
        tf.print(output)
strategy = tf.distribute.MirroredStrategy()
vocab = ['a', 'b', 'c', 'd', 'f']
with strategy.scope():
    layer = tf.keras.layers.StringLookup(vocabulary=vocab)

def dataset_fn(input_context):
    dataset = tf.data.Dataset.from_tensor_slices(['a', 'c', 'e']).repeat()
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
mirrored_strategy = tf.distribute.MirroredStrategy()
dataset_size = 24
batch_size = 6
dataset = tf.data.Dataset.range(dataset_size).enumerate().batch(batch_size)
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
mirrored_strategy = tf.distribute.MirroredStrategy()

def value_fn(ctx):
    return tf.constant(ctx.replica_id_in_sync_group)
distributed_values = mirrored_strategy.experimental_distribute_values_from_function(value_fn)
for _ in range(4):
    result = mirrored_strategy.run(lambda x: x, args=(distributed_values,))
    print(result)
mirrored_strategy = tf.distribute.MirroredStrategy()

def input_gen():
    while True:
        yield np.random.rand(4)
dataset = tf.data.Dataset.from_generator(input_gen, output_types=tf.float32, output_shapes=tf.TensorShape([4]))
dist_dataset = mirrored_strategy.experimental_distribute_dataset(dataset)
iterator = iter(dist_dataset)
for _ in range(4):
    result = mirrored_strategy.run(lambda x: x, args=(next(iterator),))
    print(result)
after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, enable_skip_calls=False)
