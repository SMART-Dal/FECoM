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

def custom_method(func, imports: str, function_to_run: str, method_object=None, function_args: list=None, function_kwargs: dict=None, max_wait_secs=MAX_WAIT_S, custom_class=None, wait_after_run_secs=WAIT_AFTER_RUN_S):
    result = send_request(imports=imports, function_to_run=function_to_run, function_args=function_args, function_kwargs=function_kwargs, max_wait_secs=max_wait_secs, wait_after_run_secs=wait_after_run_secs, method_object=method_object, custom_class=custom_class, experiment_file_path=EXPERIMENT_FILE_PATH)
    return func
if __name__ == '__main__':
    print(EXPERIMENT_FILE_PATH)
print(tf.__version__)
fashion_mnist = tf.keras.datasets.fashion_mnist
((train_images, train_labels), (test_images, test_labels)) = fashion_mnist.load_data()
train_images = train_images[..., None]
test_images = test_images[..., None]
train_images = train_images / np.float32(255)
test_images = test_images / np.float32(255)
strategy = custom_method(
tf.distribute.MirroredStrategy(), imports='import tensorflow as tf;import os;import numpy as np', function_to_run='tf.distribute.MirroredStrategy()', method_object=None, function_args=[], function_kwargs={}, max_wait_secs=0)
print('Number of devices: {}'.format(strategy.num_replicas_in_sync))
BUFFER_SIZE = len(train_images)
BATCH_SIZE_PER_REPLICA = 64
GLOBAL_BATCH_SIZE = BATCH_SIZE_PER_REPLICA * strategy.num_replicas_in_sync
EPOCHS = 10
train_dataset = custom_method(
tf.data.Dataset.from_tensor_slices((train_images, train_labels)).shuffle(BUFFER_SIZE).batch(GLOBAL_BATCH_SIZE), imports='import tensorflow as tf;import os;import numpy as np', function_to_run='tf.data.Dataset.from_tensor_slices((train_images, train_labels)).shuffle(BUFFER_SIZE).batch(*args)', method_object=None, function_args=[eval('GLOBAL_BATCH_SIZE')], function_kwargs={}, max_wait_secs=0)
test_dataset = custom_method(
tf.data.Dataset.from_tensor_slices((test_images, test_labels)).batch(GLOBAL_BATCH_SIZE), imports='import tensorflow as tf;import os;import numpy as np', function_to_run='tf.data.Dataset.from_tensor_slices((test_images, test_labels)).batch(*args)', method_object=None, function_args=[eval('GLOBAL_BATCH_SIZE')], function_kwargs={}, max_wait_secs=0)
train_dist_dataset = custom_method(
strategy.experimental_distribute_dataset(train_dataset), imports='import tensorflow as tf;import os;import numpy as np', function_to_run='obj.experimental_distribute_dataset(*args)', method_object=eval('strategy'), function_args=[eval('train_dataset')], function_kwargs={}, max_wait_secs=0, custom_class=None)
test_dist_dataset = custom_method(
strategy.experimental_distribute_dataset(test_dataset), imports='import tensorflow as tf;import os;import numpy as np', function_to_run='obj.experimental_distribute_dataset(*args)', method_object=eval('strategy'), function_args=[eval('test_dataset')], function_kwargs={}, max_wait_secs=0, custom_class=None)

def create_model():
    model = custom_method(
    tf.keras.Sequential([tf.keras.layers.Conv2D(32, 3, activation='relu'), tf.keras.layers.MaxPooling2D(), tf.keras.layers.Conv2D(64, 3, activation='relu'), tf.keras.layers.MaxPooling2D(), tf.keras.layers.Flatten(), tf.keras.layers.Dense(64, activation='relu'), tf.keras.layers.Dense(10)]), imports='import tensorflow as tf;import os;import numpy as np', function_to_run='tf.keras.Sequential(*args)', method_object=None, function_args=[eval("[\n      tf.keras.layers.Conv2D(32, 3, activation='relu'),\n      tf.keras.layers.MaxPooling2D(),\n      tf.keras.layers.Conv2D(64, 3, activation='relu'),\n      tf.keras.layers.MaxPooling2D(),\n      tf.keras.layers.Flatten(),\n      tf.keras.layers.Dense(64, activation='relu'),\n      tf.keras.layers.Dense(10)\n    ]")], function_kwargs={}, max_wait_secs=0)
    return model
checkpoint_dir = './training_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, 'ckpt')
with custom_method(
strategy.scope(), imports='import tensorflow as tf;import os;import numpy as np', function_to_run='obj.scope()', method_object=eval('strategy'), function_args=[], function_kwargs={}, max_wait_secs=0, custom_class=None):
    loss_object = custom_method(
    tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction=tf.keras.losses.Reduction.NONE), imports='import tensorflow as tf;import os;import numpy as np', function_to_run='tf.keras.losses.SparseCategoricalCrossentropy(**kwargs)', method_object=None, function_args=[], function_kwargs={'from_logits': eval('True'), 'reduction': eval('tf.keras.losses.Reduction.NONE')}, max_wait_secs=0)

    def compute_loss(labels, predictions, model_losses):
        per_example_loss = custom_method(
        loss_object(labels, predictions), imports='import tensorflow as tf;import os;import numpy as np', function_to_run='obj(*args)', method_object=eval('loss_object'), function_args=[eval('labels'), eval('predictions')], function_kwargs={}, max_wait_secs=0, custom_class=None)
        loss = custom_method(
        tf.nn.compute_average_loss(per_example_loss, global_batch_size=GLOBAL_BATCH_SIZE), imports='import tensorflow as tf;import os;import numpy as np', function_to_run='tf.nn.compute_average_loss(*args, **kwargs)', method_object=None, function_args=[eval('per_example_loss')], function_kwargs={'global_batch_size': eval('GLOBAL_BATCH_SIZE')}, max_wait_secs=0)
        if model_losses:
            loss += custom_method(
            tf.nn.scale_regularization_loss(tf.add_n(model_losses)), imports='import tensorflow as tf;import os;import numpy as np', function_to_run='tf.nn.scale_regularization_loss(*args)', method_object=None, function_args=[eval('tf.add_n(model_losses)')], function_kwargs={}, max_wait_secs=0)
        return loss
with custom_method(
strategy.scope(), imports='import tensorflow as tf;import os;import numpy as np', function_to_run='obj.scope()', method_object=eval('strategy'), function_args=[], function_kwargs={}, max_wait_secs=0, custom_class=None):
    test_loss = custom_method(
    tf.keras.metrics.Mean(name='test_loss'), imports='import tensorflow as tf;import os;import numpy as np', function_to_run='tf.keras.metrics.Mean(**kwargs)', method_object=None, function_args=[], function_kwargs={'name': eval("'test_loss'")}, max_wait_secs=0)
    train_accuracy = custom_method(
    tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy'), imports='import tensorflow as tf;import os;import numpy as np', function_to_run='tf.keras.metrics.SparseCategoricalAccuracy(**kwargs)', method_object=None, function_args=[], function_kwargs={'name': eval("'train_accuracy'")}, max_wait_secs=0)
    test_accuracy = custom_method(
    tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy'), imports='import tensorflow as tf;import os;import numpy as np', function_to_run='tf.keras.metrics.SparseCategoricalAccuracy(**kwargs)', method_object=None, function_args=[], function_kwargs={'name': eval("'test_accuracy'")}, max_wait_secs=0)
with custom_method(
strategy.scope(), imports='import tensorflow as tf;import os;import numpy as np', function_to_run='obj.scope()', method_object=eval('strategy'), function_args=[], function_kwargs={}, max_wait_secs=0, custom_class=None):
    model = custom_method(
    create_model(), imports='import tensorflow as tf;import os;import numpy as np', function_to_run='obj()', method_object=eval('create_model'), function_args=[], function_kwargs={}, max_wait_secs=0, custom_class=None)
    optimizer = custom_method(
    tf.keras.optimizers.Adam(), imports='import tensorflow as tf;import os;import numpy as np', function_to_run='tf.keras.optimizers.Adam()', method_object=None, function_args=[], function_kwargs={}, max_wait_secs=0)
    checkpoint = custom_method(
    tf.train.Checkpoint(optimizer=optimizer, model=model), imports='import tensorflow as tf;import os;import numpy as np', function_to_run='tf.train.Checkpoint(**kwargs)', method_object=None, function_args=[], function_kwargs={'optimizer': eval('optimizer'), 'model': eval('model')}, max_wait_secs=0)

def train_step(inputs):
    (images, labels) = inputs
    with custom_method(
    tf.GradientTape(), imports='import tensorflow as tf;import os;import numpy as np', function_to_run='tf.GradientTape()', method_object=None, function_args=[], function_kwargs={}, max_wait_secs=0) as tape:
        predictions = custom_method(
        model(images, training=True), imports='import tensorflow as tf;import os;import numpy as np', function_to_run='obj(*args, **kwargs)', method_object=eval('model'), function_args=[eval('images')], function_kwargs={'training': eval('True')}, max_wait_secs=0, custom_class=None)
        loss = custom_method(
        compute_loss(labels, predictions, model.losses), imports='import tensorflow as tf;import os;import numpy as np', function_to_run='obj(*args)', method_object=eval('compute_loss'), function_args=[eval('labels'), eval('predictions'), eval('model.losses')], function_kwargs={}, max_wait_secs=0, custom_class=None)
    gradients = tape.gradient(loss, model.trainable_variables)
    custom_method(
    optimizer.apply_gradients(zip(gradients, model.trainable_variables)), imports='import tensorflow as tf;import os;import numpy as np', function_to_run='obj.apply_gradients(*args)', method_object=eval('optimizer'), function_args=[eval('zip(gradients, model.trainable_variables)')], function_kwargs={}, max_wait_secs=0, custom_class=None)
    custom_method(
    train_accuracy.update_state(labels, predictions), imports='import tensorflow as tf;import os;import numpy as np', function_to_run='obj.update_state(*args)', method_object=eval('train_accuracy'), function_args=[eval('labels'), eval('predictions')], function_kwargs={}, max_wait_secs=0, custom_class=None)
    return loss

def test_step(inputs):
    (images, labels) = inputs
    predictions = custom_method(
    model(images, training=False), imports='import tensorflow as tf;import os;import numpy as np', function_to_run='obj(*args, **kwargs)', method_object=eval('model'), function_args=[eval('images')], function_kwargs={'training': eval('False')}, max_wait_secs=0, custom_class=None)
    t_loss = custom_method(
    loss_object(labels, predictions), imports='import tensorflow as tf;import os;import numpy as np', function_to_run='obj(*args)', method_object=eval('loss_object'), function_args=[eval('labels'), eval('predictions')], function_kwargs={}, max_wait_secs=0, custom_class=None)
    custom_method(
    test_loss.update_state(t_loss), imports='import tensorflow as tf;import os;import numpy as np', function_to_run='obj.update_state(*args)', method_object=eval('test_loss'), function_args=[eval('t_loss')], function_kwargs={}, max_wait_secs=0, custom_class=None)
    custom_method(
    test_accuracy.update_state(labels, predictions), imports='import tensorflow as tf;import os;import numpy as np', function_to_run='obj.update_state(*args)', method_object=eval('test_accuracy'), function_args=[eval('labels'), eval('predictions')], function_kwargs={}, max_wait_secs=0, custom_class=None)

@tf.function
def distributed_train_step(dataset_inputs):
    per_replica_losses = custom_method(
    strategy.run(train_step, args=(dataset_inputs,)), imports='import tensorflow as tf;import os;import numpy as np', function_to_run='obj.run(*args, **kwargs)', method_object=eval('strategy'), function_args=[eval('train_step')], function_kwargs={'args': eval('(dataset_inputs,)')}, max_wait_secs=0, custom_class=None)
    return custom_method(
    strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_losses, axis=None), imports='import tensorflow as tf;import os;import numpy as np', function_to_run='obj.reduce(*args, **kwargs)', method_object=eval('strategy'), function_args=[eval('tf.distribute.ReduceOp.SUM'), eval('per_replica_losses')], function_kwargs={'axis': eval('None')}, max_wait_secs=0, custom_class=None)

@tf.function
def distributed_test_step(dataset_inputs):
    return custom_method(
    strategy.run(test_step, args=(dataset_inputs,)), imports='import tensorflow as tf;import os;import numpy as np', function_to_run='obj.run(*args, **kwargs)', method_object=eval('strategy'), function_args=[eval('test_step')], function_kwargs={'args': eval('(dataset_inputs,)')}, max_wait_secs=0, custom_class=None)
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
        custom_method(
        checkpoint.save(checkpoint_prefix), imports='import tensorflow as tf;import os;import numpy as np', function_to_run='obj.save(*args)', method_object=eval('checkpoint'), function_args=[eval('checkpoint_prefix')], function_kwargs={}, max_wait_secs=0, custom_class=None)
    template = 'Epoch {}, Loss: {}, Accuracy: {}, Test Loss: {}, Test Accuracy: {}'
    print(template.format(epoch + 1, train_loss, train_accuracy.result() * 100, test_loss.result(), test_accuracy.result() * 100))
    custom_method(
    test_loss.reset_states(), imports='import tensorflow as tf;import os;import numpy as np', function_to_run='obj.reset_states()', method_object=eval('test_loss'), function_args=[], function_kwargs={}, max_wait_secs=0, custom_class=None)
    custom_method(
    train_accuracy.reset_states(), imports='import tensorflow as tf;import os;import numpy as np', function_to_run='obj.reset_states()', method_object=eval('train_accuracy'), function_args=[], function_kwargs={}, max_wait_secs=0, custom_class=None)
    custom_method(
    test_accuracy.reset_states(), imports='import tensorflow as tf;import os;import numpy as np', function_to_run='obj.reset_states()', method_object=eval('test_accuracy'), function_args=[], function_kwargs={}, max_wait_secs=0, custom_class=None)
eval_accuracy = custom_method(
tf.keras.metrics.SparseCategoricalAccuracy(name='eval_accuracy'), imports='import tensorflow as tf;import os;import numpy as np', function_to_run='tf.keras.metrics.SparseCategoricalAccuracy(**kwargs)', method_object=None, function_args=[], function_kwargs={'name': eval("'eval_accuracy'")}, max_wait_secs=0)
new_model = custom_method(
create_model(), imports='import tensorflow as tf;import os;import numpy as np', function_to_run='obj()', method_object=eval('create_model'), function_args=[], function_kwargs={}, max_wait_secs=0, custom_class=None)
new_optimizer = custom_method(
tf.keras.optimizers.Adam(), imports='import tensorflow as tf;import os;import numpy as np', function_to_run='tf.keras.optimizers.Adam()', method_object=None, function_args=[], function_kwargs={}, max_wait_secs=0)
test_dataset = custom_method(
tf.data.Dataset.from_tensor_slices((test_images, test_labels)).batch(GLOBAL_BATCH_SIZE), imports='import tensorflow as tf;import os;import numpy as np', function_to_run='tf.data.Dataset.from_tensor_slices((test_images, test_labels)).batch(*args)', method_object=None, function_args=[eval('GLOBAL_BATCH_SIZE')], function_kwargs={}, max_wait_secs=0)

@tf.function
def eval_step(images, labels):
    predictions = custom_method(
    new_model(images, training=False), imports='import tensorflow as tf;import os;import numpy as np', function_to_run='obj(*args, **kwargs)', method_object=eval('new_model'), function_args=[eval('images')], function_kwargs={'training': eval('False')}, max_wait_secs=0, custom_class=None)
    custom_method(
    eval_accuracy(labels, predictions), imports='import tensorflow as tf;import os;import numpy as np', function_to_run='obj(*args)', method_object=eval('eval_accuracy'), function_args=[eval('labels'), eval('predictions')], function_kwargs={}, max_wait_secs=0, custom_class=None)
checkpoint = custom_method(
tf.train.Checkpoint(optimizer=new_optimizer, model=new_model), imports='import tensorflow as tf;import os;import numpy as np', function_to_run='tf.train.Checkpoint(**kwargs)', method_object=None, function_args=[], function_kwargs={'optimizer': eval('new_optimizer'), 'model': eval('new_model')}, max_wait_secs=0)
custom_method(
checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir)), imports='import tensorflow as tf;import os;import numpy as np', function_to_run='obj.restore(*args)', method_object=eval('checkpoint'), function_args=[eval('tf.train.latest_checkpoint(checkpoint_dir)')], function_kwargs={}, max_wait_secs=0, custom_class=None)
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
    custom_method(
    train_accuracy.reset_states(), imports='import tensorflow as tf;import os;import numpy as np', function_to_run='obj.reset_states()', method_object=eval('train_accuracy'), function_args=[], function_kwargs={}, max_wait_secs=0, custom_class=None)

@tf.function
def distributed_train_epoch(dataset):
    total_loss = 0.0
    num_batches = 0
    for x in dataset:
        per_replica_losses = custom_method(
        strategy.run(train_step, args=(x,)), imports='import tensorflow as tf;import os;import numpy as np', function_to_run='obj.run(*args, **kwargs)', method_object=eval('strategy'), function_args=[eval('train_step')], function_kwargs={'args': eval('(x,)')}, max_wait_secs=0, custom_class=None)
        total_loss += custom_method(
        strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_losses, axis=None), imports='import tensorflow as tf;import os;import numpy as np', function_to_run='obj.reduce(*args, **kwargs)', method_object=eval('strategy'), function_args=[eval('tf.distribute.ReduceOp.SUM'), eval('per_replica_losses')], function_kwargs={'axis': eval('None')}, max_wait_secs=0, custom_class=None)
        num_batches += 1
    return total_loss / custom_method(
    tf.cast(num_batches, dtype=tf.float32), imports='import tensorflow as tf;import os;import numpy as np', function_to_run='tf.cast(*args, **kwargs)', method_object=None, function_args=[eval('num_batches')], function_kwargs={'dtype': eval('tf.float32')}, max_wait_secs=0)
for epoch in range(EPOCHS):
    train_loss = distributed_train_epoch(train_dist_dataset)
    template = 'Epoch {}, Loss: {}, Accuracy: {}'
    print(template.format(epoch + 1, train_loss, train_accuracy.result() * 100))
    custom_method(
    train_accuracy.reset_states(), imports='import tensorflow as tf;import os;import numpy as np', function_to_run='obj.reset_states()', method_object=eval('train_accuracy'), function_args=[], function_kwargs={}, max_wait_secs=0, custom_class=None)
