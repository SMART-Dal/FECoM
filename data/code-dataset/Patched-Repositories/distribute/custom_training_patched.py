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
fashion_mnist = tf.keras.datasets.fashion_mnist
((train_images, train_labels), (test_images, test_labels)) = fashion_mnist.load_data()
train_images = train_images[..., None]
test_images = test_images[..., None]
train_images = train_images / np.float32(255)
test_images = test_images / np.float32(255)
custom_method(imports='import numpy as np;import tensorflow as tf;import os', function_to_run='tf.distribute.MirroredStrategy()', method_object=None, object_signature=None, function_args=[], function_kwargs={})
strategy = tf.distribute.MirroredStrategy()
print('Number of devices: {}'.format(strategy.num_replicas_in_sync))
BUFFER_SIZE = len(train_images)
BATCH_SIZE_PER_REPLICA = 64
GLOBAL_BATCH_SIZE = BATCH_SIZE_PER_REPLICA * strategy.num_replicas_in_sync
EPOCHS = 10
custom_method(imports='import numpy as np;import tensorflow as tf;import os', function_to_run='tf.data.Dataset.from_tensor_slices((train_images, train_labels)).shuffle(BUFFER_SIZE).batch(*args)', method_object=None, object_signature=None, function_args=[eval('GLOBAL_BATCH_SIZE')], function_kwargs={})
train_dataset = tf.data.Dataset.from_tensor_slices((train_images, train_labels)).shuffle(BUFFER_SIZE).batch(GLOBAL_BATCH_SIZE)
custom_method(imports='import numpy as np;import tensorflow as tf;import os', function_to_run='tf.data.Dataset.from_tensor_slices((test_images, test_labels)).batch(*args)', method_object=None, object_signature=None, function_args=[eval('GLOBAL_BATCH_SIZE')], function_kwargs={})
test_dataset = tf.data.Dataset.from_tensor_slices((test_images, test_labels)).batch(GLOBAL_BATCH_SIZE)
custom_method(imports='import numpy as np;import tensorflow as tf;import os', function_to_run='obj.experimental_distribute_dataset(*args)', method_object=eval('strategy'), object_signature=None, function_args=[eval('train_dataset')], function_kwargs={}, custom_class=None)
train_dist_dataset = strategy.experimental_distribute_dataset(train_dataset)
custom_method(imports='import numpy as np;import tensorflow as tf;import os', function_to_run='obj.experimental_distribute_dataset(*args)', method_object=eval('strategy'), object_signature=None, function_args=[eval('test_dataset')], function_kwargs={}, custom_class=None)
test_dist_dataset = strategy.experimental_distribute_dataset(test_dataset)

def create_model():
    custom_method(imports='import numpy as np;import tensorflow as tf;import os', function_to_run='tf.keras.Sequential(*args)', method_object=None, object_signature=None, function_args=[eval("[\n      tf.keras.layers.Conv2D(32, 3, activation='relu'),\n      tf.keras.layers.MaxPooling2D(),\n      tf.keras.layers.Conv2D(64, 3, activation='relu'),\n      tf.keras.layers.MaxPooling2D(),\n      tf.keras.layers.Flatten(),\n      tf.keras.layers.Dense(64, activation='relu'),\n      tf.keras.layers.Dense(10)\n    ]")], function_kwargs={})
    model = tf.keras.Sequential([tf.keras.layers.Conv2D(32, 3, activation='relu'), tf.keras.layers.MaxPooling2D(), tf.keras.layers.Conv2D(64, 3, activation='relu'), tf.keras.layers.MaxPooling2D(), tf.keras.layers.Flatten(), tf.keras.layers.Dense(64, activation='relu'), tf.keras.layers.Dense(10)])
    return model
checkpoint_dir = './training_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, 'ckpt')
with strategy.scope():
    custom_method(imports='import numpy as np;import tensorflow as tf;import os', function_to_run='tf.keras.losses.SparseCategoricalCrossentropy(**kwargs)', method_object=None, object_signature=None, function_args=[], function_kwargs={'from_logits': eval('True'), 'reduction': eval('tf.keras.losses.Reduction.NONE')})
    loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction=tf.keras.losses.Reduction.NONE)

    def compute_loss(labels, predictions, model_losses):
        custom_method(imports='import numpy as np;import tensorflow as tf;import os', function_to_run='obj(*args)', method_object=eval('loss_object'), object_signature=None, function_args=[eval('labels'), eval('predictions')], function_kwargs={}, custom_class=None)
        per_example_loss = loss_object(labels, predictions)
        custom_method(imports='import numpy as np;import tensorflow as tf;import os', function_to_run='tf.nn.compute_average_loss(*args, **kwargs)', method_object=None, object_signature=None, function_args=[eval('per_example_loss')], function_kwargs={'global_batch_size': eval('GLOBAL_BATCH_SIZE')})
        loss = tf.nn.compute_average_loss(per_example_loss, global_batch_size=GLOBAL_BATCH_SIZE)
        if model_losses:
            loss += tf.nn.scale_regularization_loss(tf.add_n(model_losses))
        return loss
with strategy.scope():
    custom_method(imports='import numpy as np;import tensorflow as tf;import os', function_to_run='tf.keras.metrics.Mean(**kwargs)', method_object=None, object_signature=None, function_args=[], function_kwargs={'name': eval("'test_loss'")})
    test_loss = tf.keras.metrics.Mean(name='test_loss')
    custom_method(imports='import numpy as np;import tensorflow as tf;import os', function_to_run='tf.keras.metrics.SparseCategoricalAccuracy(**kwargs)', method_object=None, object_signature=None, function_args=[], function_kwargs={'name': eval("'train_accuracy'")})
    train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')
    custom_method(imports='import numpy as np;import tensorflow as tf;import os', function_to_run='tf.keras.metrics.SparseCategoricalAccuracy(**kwargs)', method_object=None, object_signature=None, function_args=[], function_kwargs={'name': eval("'test_accuracy'")})
    test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')
with strategy.scope():
    model = create_model()
    custom_method(imports='import numpy as np;import tensorflow as tf;import os', function_to_run='tf.keras.optimizers.Adam()', method_object=None, object_signature=None, function_args=[], function_kwargs={})
    optimizer = tf.keras.optimizers.Adam()
    custom_method(imports='import numpy as np;import tensorflow as tf;import os', function_to_run='tf.train.Checkpoint(**kwargs)', method_object=None, object_signature=None, function_args=[], function_kwargs={'optimizer': eval('optimizer'), 'model': eval('model')})
    checkpoint = tf.train.Checkpoint(optimizer=optimizer, model=model)

def train_step(inputs):
    (images, labels) = inputs
    with tf.GradientTape() as tape:
        custom_method(imports='import numpy as np;import tensorflow as tf;import os', function_to_run='obj(*args, **kwargs)', method_object=eval('model'), object_signature=None, function_args=[eval('images')], function_kwargs={'training': eval('True')}, custom_class=None)
        predictions = model(images, training=True)
        loss = compute_loss(labels, predictions, model.losses)
    gradients = tape.gradient(loss, model.trainable_variables)
    custom_method(imports='import numpy as np;import tensorflow as tf;import os', function_to_run='obj.apply_gradients(*args)', method_object=eval('optimizer'), object_signature=None, function_args=[eval('zip(gradients, model.trainable_variables)')], function_kwargs={}, custom_class=None)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    custom_method(imports='import numpy as np;import tensorflow as tf;import os', function_to_run='obj.update_state(*args)', method_object=eval('train_accuracy'), object_signature=None, function_args=[eval('labels'), eval('predictions')], function_kwargs={}, custom_class=None)
    train_accuracy.update_state(labels, predictions)
    return loss

def test_step(inputs):
    (images, labels) = inputs
    custom_method(imports='import numpy as np;import tensorflow as tf;import os', function_to_run='obj(*args, **kwargs)', method_object=eval('model'), object_signature=None, function_args=[eval('images')], function_kwargs={'training': eval('False')}, custom_class=None)
    predictions = model(images, training=False)
    custom_method(imports='import numpy as np;import tensorflow as tf;import os', function_to_run='obj(*args)', method_object=eval('loss_object'), object_signature=None, function_args=[eval('labels'), eval('predictions')], function_kwargs={}, custom_class=None)
    t_loss = loss_object(labels, predictions)
    custom_method(imports='import numpy as np;import tensorflow as tf;import os', function_to_run='obj.update_state(*args)', method_object=eval('test_loss'), object_signature=None, function_args=[eval('t_loss')], function_kwargs={}, custom_class=None)
    test_loss.update_state(t_loss)
    custom_method(imports='import numpy as np;import tensorflow as tf;import os', function_to_run='obj.update_state(*args)', method_object=eval('test_accuracy'), object_signature=None, function_args=[eval('labels'), eval('predictions')], function_kwargs={}, custom_class=None)
    test_accuracy.update_state(labels, predictions)

@tf.function
def distributed_train_step(dataset_inputs):
    custom_method(imports='import numpy as np;import tensorflow as tf;import os', function_to_run='obj.run(*args, **kwargs)', method_object=eval('strategy'), object_signature=None, function_args=[eval('train_step')], function_kwargs={'args': eval('(dataset_inputs,)')}, custom_class=None)
    per_replica_losses = strategy.run(train_step, args=(dataset_inputs,))
    return strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_losses, axis=None)

@tf.function
def distributed_test_step(dataset_inputs):
    return strategy.run(test_step, args=(dataset_inputs,))
for epoch in range(EPOCHS):
    total_loss = 0.0
    num_batches = 0
    for x in train_dist_dataset:
        total_loss += distributed_train_step(x)
        num_batches += 1
    train_loss = total_loss / num_batches
    for x in test_dist_dataset:
        distributed_test_step(x)
    if epoch % 2 == 0:
        custom_method(imports='import numpy as np;import tensorflow as tf;import os', function_to_run='obj.save(*args)', method_object=eval('checkpoint'), object_signature=None, function_args=[eval('checkpoint_prefix')], function_kwargs={}, custom_class=None)
        checkpoint.save(checkpoint_prefix)
    template = 'Epoch {}, Loss: {}, Accuracy: {}, Test Loss: {}, Test Accuracy: {}'
    print(template.format(epoch + 1, train_loss, train_accuracy.result() * 100, test_loss.result(), test_accuracy.result() * 100))
    custom_method(imports='import numpy as np;import tensorflow as tf;import os', function_to_run='obj.reset_states()', method_object=eval('test_loss'), object_signature=None, function_args=[], function_kwargs={}, custom_class=None)
    test_loss.reset_states()
    custom_method(imports='import numpy as np;import tensorflow as tf;import os', function_to_run='obj.reset_states()', method_object=eval('train_accuracy'), object_signature=None, function_args=[], function_kwargs={}, custom_class=None)
    train_accuracy.reset_states()
    custom_method(imports='import numpy as np;import tensorflow as tf;import os', function_to_run='obj.reset_states()', method_object=eval('test_accuracy'), object_signature=None, function_args=[], function_kwargs={}, custom_class=None)
    test_accuracy.reset_states()
custom_method(imports='import numpy as np;import tensorflow as tf;import os', function_to_run='tf.keras.metrics.SparseCategoricalAccuracy(**kwargs)', method_object=None, object_signature=None, function_args=[], function_kwargs={'name': eval("'eval_accuracy'")})
eval_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='eval_accuracy')
new_model = create_model()
custom_method(imports='import numpy as np;import tensorflow as tf;import os', function_to_run='tf.keras.optimizers.Adam()', method_object=None, object_signature=None, function_args=[], function_kwargs={})
new_optimizer = tf.keras.optimizers.Adam()
custom_method(imports='import numpy as np;import tensorflow as tf;import os', function_to_run='tf.data.Dataset.from_tensor_slices((test_images, test_labels)).batch(*args)', method_object=None, object_signature=None, function_args=[eval('GLOBAL_BATCH_SIZE')], function_kwargs={})
test_dataset = tf.data.Dataset.from_tensor_slices((test_images, test_labels)).batch(GLOBAL_BATCH_SIZE)

@tf.function
def eval_step(images, labels):
    predictions = new_model(images, training=False)
    custom_method(imports='import numpy as np;import tensorflow as tf;import os', function_to_run='obj(*args)', method_object=eval('eval_accuracy'), object_signature=None, function_args=[eval('labels'), eval('predictions')], function_kwargs={}, custom_class=None)
    eval_accuracy(labels, predictions)
custom_method(imports='import numpy as np;import tensorflow as tf;import os', function_to_run='tf.train.Checkpoint(**kwargs)', method_object=None, object_signature=None, function_args=[], function_kwargs={'optimizer': eval('new_optimizer'), 'model': eval('new_model')})
checkpoint = tf.train.Checkpoint(optimizer=new_optimizer, model=new_model)
custom_method(imports='import numpy as np;import tensorflow as tf;import os', function_to_run='obj.restore(*args)', method_object=eval('checkpoint'), object_signature=None, function_args=[eval('tf.train.latest_checkpoint(checkpoint_dir)')], function_kwargs={}, custom_class=None)
checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))
for (images, labels) in test_dataset:
    eval_step(images, labels)
print('Accuracy after restoring the saved model without strategy: {}'.format(eval_accuracy.result() * 100))
for _ in range(EPOCHS):
    total_loss = 0.0
    num_batches = 0
    train_iter = iter(train_dist_dataset)
    for _ in range(10):
        total_loss += distributed_train_step(next(train_iter))
        num_batches += 1
    average_train_loss = total_loss / num_batches
    template = 'Epoch {}, Loss: {}, Accuracy: {}'
    print(template.format(epoch + 1, average_train_loss, train_accuracy.result() * 100))
    custom_method(imports='import numpy as np;import tensorflow as tf;import os', function_to_run='obj.reset_states()', method_object=eval('train_accuracy'), object_signature=None, function_args=[], function_kwargs={}, custom_class=None)
    train_accuracy.reset_states()

@tf.function
def distributed_train_epoch(dataset):
    total_loss = 0.0
    num_batches = 0
    for x in dataset:
        custom_method(imports='import numpy as np;import tensorflow as tf;import os', function_to_run='obj.run(*args, **kwargs)', method_object=eval('strategy'), object_signature=None, function_args=[eval('train_step')], function_kwargs={'args': eval('(x,)')}, custom_class=None)
        per_replica_losses = strategy.run(train_step, args=(x,))
        total_loss += strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_losses, axis=None)
        num_batches += 1
    return total_loss / tf.cast(num_batches, dtype=tf.float32)
for epoch in range(EPOCHS):
    train_loss = distributed_train_epoch(train_dist_dataset)
    template = 'Epoch {}, Loss: {}, Accuracy: {}'
    print(template.format(epoch + 1, train_loss, train_accuracy.result() * 100))
    custom_method(imports='import numpy as np;import tensorflow as tf;import os', function_to_run='obj.reset_states()', method_object=eval('train_accuracy'), object_signature=None, function_args=[], function_kwargs={}, custom_class=None)
    train_accuracy.reset_states()
