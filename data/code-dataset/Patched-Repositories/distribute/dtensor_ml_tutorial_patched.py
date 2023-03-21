import tempfile
import numpy as np
import tensorflow_datasets as tfds
import tensorflow as tf
from tensorflow.experimental import dtensor
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
print('TensorFlow version:', tf.__version__)

def configure_virtual_cpus(ncpu):
    phy_devices = custom_method(
    tf.config.list_physical_devices('CPU'), imports='import tempfile;from typing import Tuple;from tensorflow.experimental import dtensor;import tensorflow as tf;import numpy as np;import tensorflow_datasets as tfds', function_to_run='tf.config.list_physical_devices(*args)', method_object=None, function_args=[eval("'CPU'")], function_kwargs={}, max_wait_secs=0)
    custom_method(
    tf.config.set_logical_device_configuration(phy_devices[0], [tf.config.LogicalDeviceConfiguration()] * ncpu), imports='import tempfile;from typing import Tuple;from tensorflow.experimental import dtensor;import tensorflow as tf;import numpy as np;import tensorflow_datasets as tfds', function_to_run='tf.config.set_logical_device_configuration(*args)', method_object=None, function_args=[eval('phy_devices[0]'), eval('[\n        tf.config.LogicalDeviceConfiguration(),\n    ] * ncpu')], function_kwargs={}, max_wait_secs=0)
configure_virtual_cpus(8)
DEVICES = [f'CPU:{i}' for i in range(8)]
custom_method(
tf.config.list_logical_devices('CPU'), imports='import tempfile;from typing import Tuple;from tensorflow.experimental import dtensor;import tensorflow as tf;import numpy as np;import tensorflow_datasets as tfds', function_to_run='tf.config.list_logical_devices(*args)', method_object=None, function_args=[eval("'CPU'")], function_kwargs={}, max_wait_secs=0)
train_data = custom_method(
tfds.load('imdb_reviews', split='train', shuffle_files=True, batch_size=64), imports='import tempfile;from typing import Tuple;from tensorflow.experimental import dtensor;import tensorflow as tf;import numpy as np;import tensorflow_datasets as tfds', function_to_run='tfds.load(*args, **kwargs)', method_object=None, function_args=[eval("'imdb_reviews'")], function_kwargs={'split': eval("'train'"), 'shuffle_files': eval('True'), 'batch_size': eval('64')}, max_wait_secs=0)
train_data
text_vectorization = custom_method(
tf.keras.layers.TextVectorization(output_mode='tf_idf', max_tokens=1200, output_sequence_length=None), imports='import tempfile;from typing import Tuple;from tensorflow.experimental import dtensor;import tensorflow as tf;import numpy as np;import tensorflow_datasets as tfds', function_to_run='tf.keras.layers.TextVectorization(**kwargs)', method_object=None, function_args=[], function_kwargs={'output_mode': eval("'tf_idf'"), 'max_tokens': eval('1200'), 'output_sequence_length': eval('None')}, max_wait_secs=0)
custom_method(
text_vectorization.adapt(data=train_data.map(lambda x: x['text'])), imports='import tempfile;from typing import Tuple;from tensorflow.experimental import dtensor;import tensorflow as tf;import numpy as np;import tensorflow_datasets as tfds', function_to_run='obj.adapt(**kwargs)', method_object=eval('text_vectorization'), function_args=[], function_kwargs={'data': eval("train_data.map(lambda x: x['text'])")}, max_wait_secs=0, custom_class=None)

def vectorize(features):
    return (custom_method(
    text_vectorization(features['text']), imports='import tempfile;from typing import Tuple;from tensorflow.experimental import dtensor;import tensorflow as tf;import numpy as np;import tensorflow_datasets as tfds', function_to_run='obj(*args)', method_object=eval('text_vectorization'), function_args=[eval("features['text']")], function_kwargs={}, max_wait_secs=0, custom_class=None), features['label'])
train_data_vec = custom_method(
train_data.map(vectorize), imports='import tempfile;from typing import Tuple;from tensorflow.experimental import dtensor;import tensorflow as tf;import numpy as np;import tensorflow_datasets as tfds', function_to_run='obj.map(*args)', method_object=eval('train_data'), function_args=[eval('vectorize')], function_kwargs={}, max_wait_secs=0, custom_class=None)
train_data_vec

class Dense(tf.Module):

    def __init__(self, input_size, output_size, init_seed, weight_layout, activation=None):
        super().__init__()
        random_normal_initializer = custom_method(
        tf.function(tf.random.stateless_normal), imports='import tempfile;from typing import Tuple;from tensorflow.experimental import dtensor;import tensorflow as tf;import numpy as np;import tensorflow_datasets as tfds', function_to_run='tf.function(*args)', method_object=None, function_args=[eval('tf.random.stateless_normal')], function_kwargs={}, max_wait_secs=0)
        self.weight = custom_method(
        dtensor.DVariable(dtensor.call_with_layout(random_normal_initializer, weight_layout, shape=[input_size, output_size], seed=init_seed)), imports='import tempfile;from typing import Tuple;from tensorflow.experimental import dtensor;import tensorflow as tf;import numpy as np;import tensorflow_datasets as tfds', function_to_run='dtensor.DVariable(*args)', method_object=None, function_args=[eval('dtensor.call_with_layout(\n            random_normal_initializer, weight_layout,\n            shape=[input_size, output_size],\n            seed=init_seed\n            )')], function_kwargs={}, max_wait_secs=0)
        if activation is None:
            activation = lambda x: x
        self.activation = activation
        bias_layout = weight_layout.delete([0])
        self.bias = custom_method(
        dtensor.DVariable(dtensor.call_with_layout(tf.zeros, bias_layout, [output_size])), imports='import tempfile;from typing import Tuple;from tensorflow.experimental import dtensor;import tensorflow as tf;import numpy as np;import tensorflow_datasets as tfds', function_to_run='dtensor.DVariable(*args)', method_object=None, function_args=[eval('dtensor.call_with_layout(tf.zeros, bias_layout, [output_size])')], function_kwargs={}, max_wait_secs=0)

    def __call__(self, x):
        y = custom_method(
        tf.matmul(x, self.weight), imports='import tempfile;from typing import Tuple;from tensorflow.experimental import dtensor;import tensorflow as tf;import numpy as np;import tensorflow_datasets as tfds', function_to_run='tf.matmul(*args)', method_object=None, function_args=[eval('x'), eval('self.weight')], function_kwargs={}, max_wait_secs=0) + self.bias
        y = self.activation(y)
        return y

class BatchNorm(tf.Module):

    def __init__(self):
        super().__init__()

    def __call__(self, x, training=True):
        if not training:
            pass
        (mean, variance) = custom_method(
        tf.nn.moments(x, axes=[0]), imports='import tempfile;from typing import Tuple;from tensorflow.experimental import dtensor;import tensorflow as tf;import numpy as np;import tensorflow_datasets as tfds', function_to_run='tf.nn.moments(*args, **kwargs)', method_object=None, function_args=[eval('x')], function_kwargs={'axes': eval('[0]')}, max_wait_secs=0)
        return custom_method(
        tf.nn.batch_normalization(x, mean, variance, 0.0, 1.0, 1e-05), imports='import tempfile;from typing import Tuple;from tensorflow.experimental import dtensor;import tensorflow as tf;import numpy as np;import tensorflow_datasets as tfds', function_to_run='tf.nn.batch_normalization(*args)', method_object=None, function_args=[eval('x'), eval('mean'), eval('variance'), eval('0.0'), eval('1.0'), eval('1e-5')], function_kwargs={}, max_wait_secs=0)

def make_keras_bn(bn_layout):
    return custom_method(
    tf.keras.layers.BatchNormalization(gamma_layout=bn_layout, beta_layout=bn_layout, moving_mean_layout=bn_layout, moving_variance_layout=bn_layout, fused=False), imports='import tempfile;from typing import Tuple;from tensorflow.experimental import dtensor;import tensorflow as tf;import numpy as np;import tensorflow_datasets as tfds', function_to_run='tf.keras.layers.BatchNormalization(**kwargs)', method_object=None, function_args=[], function_kwargs={'gamma_layout': eval('bn_layout'), 'beta_layout': eval('bn_layout'), 'moving_mean_layout': eval('bn_layout'), 'moving_variance_layout': eval('bn_layout'), 'fused': eval('False')}, max_wait_secs=0)
from typing import Tuple

class MLP(tf.Module):

    def __init__(self, dense_layouts: Tuple[dtensor.Layout, dtensor.Layout]):
        super().__init__()
        self.dense1 = Dense(1200, 48, (1, 2), dense_layouts[0], activation=tf.nn.relu)
        self.bn = BatchNorm()
        self.dense2 = Dense(48, 2, (3, 4), dense_layouts[1])

    def __call__(self, x):
        y = x
        y = self.dense1(y)
        y = self.bn(y)
        y = self.dense2(y)
        return y

class MLPStricter(tf.Module):

    def __init__(self, mesh, input_mesh_dim, inner_mesh_dim1, output_mesh_dim):
        super().__init__()
        self.dense1 = Dense(1200, 48, (1, 2), dtensor.Layout([input_mesh_dim, inner_mesh_dim1], mesh), activation=tf.nn.relu)
        self.bn = BatchNorm()
        self.dense2 = Dense(48, 2, (3, 4), dtensor.Layout([inner_mesh_dim1, output_mesh_dim], mesh))

    def __call__(self, x):
        y = x
        y = self.dense1(y)
        y = self.bn(y)
        y = self.dense2(y)
        return y
WORLD = custom_method(
dtensor.create_mesh([('world', 8)], devices=DEVICES), imports='import tempfile;from typing import Tuple;from tensorflow.experimental import dtensor;import tensorflow as tf;import numpy as np;import tensorflow_datasets as tfds', function_to_run='dtensor.create_mesh(*args, **kwargs)', method_object=None, function_args=[eval('[("world", 8)]')], function_kwargs={'devices': eval('DEVICES')}, max_wait_secs=0)
model = MLP([dtensor.Layout.replicated(WORLD, rank=2), dtensor.Layout.replicated(WORLD, rank=2)])
(sample_x, sample_y) = custom_method(
train_data_vec.take(1).get_single_element(), imports='import tempfile;from typing import Tuple;from tensorflow.experimental import dtensor;import tensorflow as tf;import numpy as np;import tensorflow_datasets as tfds', function_to_run='obj.take(1).get_single_element()', method_object=eval('train_data_vec'), function_args=[], function_kwargs={}, max_wait_secs=0, custom_class=None)
sample_x = custom_method(
dtensor.copy_to_mesh(sample_x, dtensor.Layout.replicated(WORLD, rank=2)), imports='import tempfile;from typing import Tuple;from tensorflow.experimental import dtensor;import tensorflow as tf;import numpy as np;import tensorflow_datasets as tfds', function_to_run='dtensor.copy_to_mesh(*args)', method_object=None, function_args=[eval('sample_x'), eval('dtensor.Layout.replicated(WORLD, rank=2)')], function_kwargs={}, max_wait_secs=0)
print(model(sample_x))

def repack_local_tensor(x, layout):
    """Repacks a local Tensor-like to a DTensor with layout.

  This function assumes a single-client application.
  """
    x = custom_method(
    tf.convert_to_tensor(x), imports='import tempfile;from typing import Tuple;from tensorflow.experimental import dtensor;import tensorflow as tf;import numpy as np;import tensorflow_datasets as tfds', function_to_run='tf.convert_to_tensor(*args)', method_object=None, function_args=[eval('x')], function_kwargs={}, max_wait_secs=0)
    sharded_dims = []
    queue = [x]
    for (axis, dim) in enumerate(layout.sharding_specs):
        if dim == dtensor.UNSHARDED:
            continue
        num_splits = layout.shape[axis]
        queue = custom_method(
        tf.nest.map_structure(lambda x: tf.split(x, num_splits, axis=axis), queue), imports='import tempfile;from typing import Tuple;from tensorflow.experimental import dtensor;import tensorflow as tf;import numpy as np;import tensorflow_datasets as tfds', function_to_run='tf.nest.map_structure(*args)', method_object=None, function_args=[eval('lambda x: tf.split(x, num_splits, axis=axis)'), eval('queue')], function_kwargs={}, max_wait_secs=0)
        sharded_dims.append(dim)
    components = []
    for locations in layout.mesh.local_device_locations():
        t = queue[0]
        for dim in sharded_dims:
            split_index = locations[dim]
            t = t[split_index]
        components.append(t)
    return custom_method(
    dtensor.pack(components, layout), imports='import tempfile;from typing import Tuple;from tensorflow.experimental import dtensor;import tensorflow as tf;import numpy as np;import tensorflow_datasets as tfds', function_to_run='dtensor.pack(*args)', method_object=None, function_args=[eval('components'), eval('layout')], function_kwargs={}, max_wait_secs=0)
mesh = custom_method(
dtensor.create_mesh([('batch', 8)], devices=DEVICES), imports='import tempfile;from typing import Tuple;from tensorflow.experimental import dtensor;import tensorflow as tf;import numpy as np;import tensorflow_datasets as tfds', function_to_run='dtensor.create_mesh(*args, **kwargs)', method_object=None, function_args=[eval('[("batch", 8)]')], function_kwargs={'devices': eval('DEVICES')}, max_wait_secs=0)
model = MLP([dtensor.Layout([dtensor.UNSHARDED, dtensor.UNSHARDED], mesh), dtensor.Layout([dtensor.UNSHARDED, dtensor.UNSHARDED], mesh)])

def repack_batch(x, y, mesh):
    x = repack_local_tensor(x, layout=dtensor.Layout(['batch', dtensor.UNSHARDED], mesh))
    y = repack_local_tensor(y, layout=dtensor.Layout(['batch'], mesh))
    return (x, y)
(sample_x, sample_y) = custom_method(
train_data_vec.take(1).get_single_element(), imports='import tempfile;from typing import Tuple;from tensorflow.experimental import dtensor;import tensorflow as tf;import numpy as np;import tensorflow_datasets as tfds', function_to_run='obj.take(1).get_single_element()', method_object=eval('train_data_vec'), function_args=[], function_kwargs={}, max_wait_secs=0, custom_class=None)
(sample_x, sample_y) = repack_batch(sample_x, sample_y, mesh)
print('x', sample_x[:, 0])
print('y', sample_y)

@tf.function
def train_step(model, x, y, learning_rate=custom_method(
tf.constant(0.0001), imports='import tempfile;from typing import Tuple;from tensorflow.experimental import dtensor;import tensorflow as tf;import numpy as np;import tensorflow_datasets as tfds', function_to_run='tf.constant(*args)', method_object=None, function_args=[eval('1e-4')], function_kwargs={}, max_wait_secs=0)):
    with custom_method(
    tf.GradientTape(), imports='import tempfile;from typing import Tuple;from tensorflow.experimental import dtensor;import tensorflow as tf;import numpy as np;import tensorflow_datasets as tfds', function_to_run='tf.GradientTape()', method_object=None, function_args=[], function_kwargs={}, max_wait_secs=0) as tape:
        logits = custom_method(
        model(x), imports='import tempfile;from typing import Tuple;from tensorflow.experimental import dtensor;import tensorflow as tf;import numpy as np;import tensorflow_datasets as tfds', function_to_run='obj(*args)', method_object=eval('model'), function_args=[eval('x')], function_kwargs={}, max_wait_secs=0, custom_class='class MLP(tf.Module):\n\n  def __init__(self, dense_layouts: Tuple[dtensor.Layout, dtensor.Layout]):\n    super().__init__()\n\n    self.dense1 = Dense(\n        1200, 48, (1, 2), dense_layouts[0], activation=tf.nn.relu)\n    self.bn = BatchNorm()\n    self.dense2 = Dense(48, 2, (3, 4), dense_layouts[1])\n\n  def __call__(self, x):\n    y = x\n    y = self.dense1(y)\n    y = self.bn(y)\n    y = self.dense2(y)\n    return y')
        loss = custom_method(
        tf.reduce_sum(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=y)), imports='import tempfile;from typing import Tuple;from tensorflow.experimental import dtensor;import tensorflow as tf;import numpy as np;import tensorflow_datasets as tfds', function_to_run='tf.reduce_sum(*args)', method_object=None, function_args=[eval('tf.nn.sparse_softmax_cross_entropy_with_logits(\n            logits=logits, labels=y)')], function_kwargs={}, max_wait_secs=0)
    parameters = model.trainable_variables
    gradients = tape.gradient(loss, parameters)
    for (parameter, parameter_gradient) in zip(parameters, gradients):
        parameter.assign_sub(learning_rate * parameter_gradient)
    accuracy = 1.0 - custom_method(
    tf.reduce_sum(tf.cast(tf.argmax(logits, axis=-1, output_type=tf.int64) != y, tf.float32)), imports='import tempfile;from typing import Tuple;from tensorflow.experimental import dtensor;import tensorflow as tf;import numpy as np;import tensorflow_datasets as tfds', function_to_run='tf.reduce_sum(*args)', method_object=None, function_args=[eval('tf.cast(tf.argmax(logits, axis=-1, output_type=tf.int64) != y, tf.float32)')], function_kwargs={}, max_wait_secs=0) / x.shape[0]
    loss_per_sample = loss / len(x)
    return {'loss': loss_per_sample, 'accuracy': accuracy}
CHECKPOINT_DIR = tempfile.mkdtemp()

def start_checkpoint_manager(model):
    ckpt = custom_method(
    tf.train.Checkpoint(root=model), imports='import tempfile;from typing import Tuple;from tensorflow.experimental import dtensor;import tensorflow as tf;import numpy as np;import tensorflow_datasets as tfds', function_to_run='tf.train.Checkpoint(**kwargs)', method_object=None, function_args=[], function_kwargs={'root': eval('model')}, max_wait_secs=0)
    manager = custom_method(
    tf.train.CheckpointManager(ckpt, CHECKPOINT_DIR, max_to_keep=3), imports='import tempfile;from typing import Tuple;from tensorflow.experimental import dtensor;import tensorflow as tf;import numpy as np;import tensorflow_datasets as tfds', function_to_run='tf.train.CheckpointManager(*args, **kwargs)', method_object=None, function_args=[eval('ckpt'), eval('CHECKPOINT_DIR')], function_kwargs={'max_to_keep': eval('3')}, max_wait_secs=0)
    if manager.latest_checkpoint:
        print('Restoring a checkpoint')
        custom_method(
        ckpt.restore(manager.latest_checkpoint).assert_consumed(), imports='import tempfile;from typing import Tuple;from tensorflow.experimental import dtensor;import tensorflow as tf;import numpy as np;import tensorflow_datasets as tfds', function_to_run='ckpt.restore(obj.latest_checkpoint).assert_consumed()', method_object=eval('manager'), function_args=[], function_kwargs={}, max_wait_secs=0, custom_class=None)
    else:
        print('New training')
    return manager
num_epochs = 2
manager = custom_method(
start_checkpoint_manager(model), imports='import tempfile;from typing import Tuple;from tensorflow.experimental import dtensor;import tensorflow as tf;import numpy as np;import tensorflow_datasets as tfds', function_to_run='obj(*args)', method_object=eval('start_checkpoint_manager'), function_args=[eval('model')], function_kwargs={}, max_wait_secs=0, custom_class=None)
for epoch in range(num_epochs):
    step = 0
    pbar = custom_method(
    tf.keras.utils.Progbar(target=int(train_data_vec.cardinality()), stateful_metrics=[]), imports='import tempfile;from typing import Tuple;from tensorflow.experimental import dtensor;import tensorflow as tf;import numpy as np;import tensorflow_datasets as tfds', function_to_run='tf.keras.utils.Progbar(**kwargs)', method_object=None, function_args=[], function_kwargs={'target': eval('int(train_data_vec.cardinality())'), 'stateful_metrics': eval('[]')}, max_wait_secs=0)
    metrics = {'epoch': epoch}
    for (x, y) in train_data_vec:
        (x, y) = repack_batch(x, y, mesh)
        metrics.update(train_step(model, x, y, 0.01))
        custom_method(
        pbar.update(step, values=metrics.items(), finalize=False), imports='import tempfile;from typing import Tuple;from tensorflow.experimental import dtensor;import tensorflow as tf;import numpy as np;import tensorflow_datasets as tfds', function_to_run='obj.update(*args, **kwargs)', method_object=eval('pbar'), function_args=[eval('step')], function_kwargs={'values': eval('metrics.items()'), 'finalize': eval('False')}, max_wait_secs=0, custom_class=None)
        step += 1
    custom_method(
    manager.save(), imports='import tempfile;from typing import Tuple;from tensorflow.experimental import dtensor;import tensorflow as tf;import numpy as np;import tensorflow_datasets as tfds', function_to_run='obj.save()', method_object=eval('manager'), function_args=[], function_kwargs={}, max_wait_secs=0, custom_class=None)
    custom_method(
    pbar.update(step, values=metrics.items(), finalize=True), imports='import tempfile;from typing import Tuple;from tensorflow.experimental import dtensor;import tensorflow as tf;import numpy as np;import tensorflow_datasets as tfds', function_to_run='obj.update(*args, **kwargs)', method_object=eval('pbar'), function_args=[eval('step')], function_kwargs={'values': eval('metrics.items()'), 'finalize': eval('True')}, max_wait_secs=0, custom_class=None)
mesh = custom_method(
dtensor.create_mesh([('batch', 4), ('model', 2)], devices=DEVICES), imports='import tempfile;from typing import Tuple;from tensorflow.experimental import dtensor;import tensorflow as tf;import numpy as np;import tensorflow_datasets as tfds', function_to_run='dtensor.create_mesh(*args, **kwargs)', method_object=None, function_args=[eval('[("batch", 4), ("model", 2)]')], function_kwargs={'devices': eval('DEVICES')}, max_wait_secs=0)
model = MLP([dtensor.Layout([dtensor.UNSHARDED, 'model'], mesh), dtensor.Layout(['model', dtensor.UNSHARDED], mesh)])

def repack_batch(x, y, mesh):
    x = repack_local_tensor(x, layout=dtensor.Layout(['batch', dtensor.UNSHARDED], mesh))
    y = repack_local_tensor(y, layout=dtensor.Layout(['batch'], mesh))
    return (x, y)
num_epochs = 2
manager = custom_method(
start_checkpoint_manager(model), imports='import tempfile;from typing import Tuple;from tensorflow.experimental import dtensor;import tensorflow as tf;import numpy as np;import tensorflow_datasets as tfds', function_to_run='obj(*args)', method_object=eval('start_checkpoint_manager'), function_args=[eval('model')], function_kwargs={}, max_wait_secs=0, custom_class=None)
for epoch in range(num_epochs):
    step = 0
    pbar = custom_method(
    tf.keras.utils.Progbar(target=int(train_data_vec.cardinality())), imports='import tempfile;from typing import Tuple;from tensorflow.experimental import dtensor;import tensorflow as tf;import numpy as np;import tensorflow_datasets as tfds', function_to_run='tf.keras.utils.Progbar(**kwargs)', method_object=None, function_args=[], function_kwargs={'target': eval('int(train_data_vec.cardinality())')}, max_wait_secs=0)
    metrics = {'epoch': epoch}
    for (x, y) in train_data_vec:
        (x, y) = repack_batch(x, y, mesh)
        metrics.update(train_step(model, x, y, 0.01))
        custom_method(
        pbar.update(step, values=metrics.items(), finalize=False), imports='import tempfile;from typing import Tuple;from tensorflow.experimental import dtensor;import tensorflow as tf;import numpy as np;import tensorflow_datasets as tfds', function_to_run='obj.update(*args, **kwargs)', method_object=eval('pbar'), function_args=[eval('step')], function_kwargs={'values': eval('metrics.items()'), 'finalize': eval('False')}, max_wait_secs=0, custom_class=None)
        step += 1
    custom_method(
    manager.save(), imports='import tempfile;from typing import Tuple;from tensorflow.experimental import dtensor;import tensorflow as tf;import numpy as np;import tensorflow_datasets as tfds', function_to_run='obj.save()', method_object=eval('manager'), function_args=[], function_kwargs={}, max_wait_secs=0, custom_class=None)
    custom_method(
    pbar.update(step, values=metrics.items(), finalize=True), imports='import tempfile;from typing import Tuple;from tensorflow.experimental import dtensor;import tensorflow as tf;import numpy as np;import tensorflow_datasets as tfds', function_to_run='obj.update(*args, **kwargs)', method_object=eval('pbar'), function_args=[eval('step')], function_kwargs={'values': eval('metrics.items()'), 'finalize': eval('True')}, max_wait_secs=0, custom_class=None)
mesh = custom_method(
dtensor.create_mesh([('batch', 2), ('feature', 2), ('model', 2)], devices=DEVICES), imports='import tempfile;from typing import Tuple;from tensorflow.experimental import dtensor;import tensorflow as tf;import numpy as np;import tensorflow_datasets as tfds', function_to_run='dtensor.create_mesh(*args, **kwargs)', method_object=None, function_args=[eval('[("batch", 2), ("feature", 2), ("model", 2)]')], function_kwargs={'devices': eval('DEVICES')}, max_wait_secs=0)
model = MLP([dtensor.Layout(['feature', 'model'], mesh), dtensor.Layout(['model', dtensor.UNSHARDED], mesh)])

def repack_batch_for_spt(x, y, mesh):
    x = repack_local_tensor(x, layout=dtensor.Layout(['batch', 'feature'], mesh))
    y = repack_local_tensor(y, layout=dtensor.Layout(['batch'], mesh))
    return (x, y)
num_epochs = 2
manager = custom_method(
start_checkpoint_manager(model), imports='import tempfile;from typing import Tuple;from tensorflow.experimental import dtensor;import tensorflow as tf;import numpy as np;import tensorflow_datasets as tfds', function_to_run='obj(*args)', method_object=eval('start_checkpoint_manager'), function_args=[eval('model')], function_kwargs={}, max_wait_secs=0, custom_class=None)
for epoch in range(num_epochs):
    step = 0
    metrics = {'epoch': epoch}
    pbar = custom_method(
    tf.keras.utils.Progbar(target=int(train_data_vec.cardinality())), imports='import tempfile;from typing import Tuple;from tensorflow.experimental import dtensor;import tensorflow as tf;import numpy as np;import tensorflow_datasets as tfds', function_to_run='tf.keras.utils.Progbar(**kwargs)', method_object=None, function_args=[], function_kwargs={'target': eval('int(train_data_vec.cardinality())')}, max_wait_secs=0)
    for (x, y) in train_data_vec:
        (x, y) = repack_batch_for_spt(x, y, mesh)
        metrics.update(train_step(model, x, y, 0.01))
        custom_method(
        pbar.update(step, values=metrics.items(), finalize=False), imports='import tempfile;from typing import Tuple;from tensorflow.experimental import dtensor;import tensorflow as tf;import numpy as np;import tensorflow_datasets as tfds', function_to_run='obj.update(*args, **kwargs)', method_object=eval('pbar'), function_args=[eval('step')], function_kwargs={'values': eval('metrics.items()'), 'finalize': eval('False')}, max_wait_secs=0, custom_class=None)
        step += 1
    custom_method(
    manager.save(), imports='import tempfile;from typing import Tuple;from tensorflow.experimental import dtensor;import tensorflow as tf;import numpy as np;import tensorflow_datasets as tfds', function_to_run='obj.save()', method_object=eval('manager'), function_args=[], function_kwargs={}, max_wait_secs=0, custom_class=None)
    custom_method(
    pbar.update(step, values=metrics.items(), finalize=True), imports='import tempfile;from typing import Tuple;from tensorflow.experimental import dtensor;import tensorflow as tf;import numpy as np;import tensorflow_datasets as tfds', function_to_run='obj.update(*args, **kwargs)', method_object=eval('pbar'), function_args=[eval('step')], function_kwargs={'values': eval('metrics.items()'), 'finalize': eval('True')}, max_wait_secs=0, custom_class=None)
mesh = custom_method(
dtensor.create_mesh([('world', 1)], devices=DEVICES[:1]), imports='import tempfile;from typing import Tuple;from tensorflow.experimental import dtensor;import tensorflow as tf;import numpy as np;import tensorflow_datasets as tfds', function_to_run='dtensor.create_mesh(*args, **kwargs)', method_object=None, function_args=[eval('[("world", 1)]')], function_kwargs={'devices': eval('DEVICES[:1]')}, max_wait_secs=0)
mlp = MLP([dtensor.Layout([dtensor.UNSHARDED, dtensor.UNSHARDED], mesh), dtensor.Layout([dtensor.UNSHARDED, dtensor.UNSHARDED], mesh)])
manager = custom_method(
start_checkpoint_manager(mlp), imports='import tempfile;from typing import Tuple;from tensorflow.experimental import dtensor;import tensorflow as tf;import numpy as np;import tensorflow_datasets as tfds', function_to_run='obj(*args)', method_object=eval('start_checkpoint_manager'), function_args=[eval('mlp')], function_kwargs={}, max_wait_secs=0, custom_class=None)
model_for_saving = custom_method(
tf.keras.Sequential([text_vectorization, mlp]), imports='import tempfile;from typing import Tuple;from tensorflow.experimental import dtensor;import tensorflow as tf;import numpy as np;import tensorflow_datasets as tfds', function_to_run='tf.keras.Sequential(*args)', method_object=None, function_args=[eval('[\n  text_vectorization,\n  mlp\n]')], function_kwargs={}, max_wait_secs=0)

@custom_method(
tf.function(input_signature=[tf.TensorSpec([None], tf.string)]), imports='import tempfile;from typing import Tuple;from tensorflow.experimental import dtensor;import tensorflow as tf;import numpy as np;import tensorflow_datasets as tfds', function_to_run='tf.function(**kwargs)', method_object=None, function_args=[], function_kwargs={'input_signature': eval('[tf.TensorSpec([None], tf.string)]')}, max_wait_secs=0)
def run(inputs):
    return {'result': custom_method(
    model_for_saving(inputs), imports='import tempfile;from typing import Tuple;from tensorflow.experimental import dtensor;import tensorflow as tf;import numpy as np;import tensorflow_datasets as tfds', function_to_run='obj(*args)', method_object=eval('model_for_saving'), function_args=[eval('inputs')], function_kwargs={}, max_wait_secs=0, custom_class=None)}
custom_method(
tf.saved_model.save(model_for_saving, '/tmp/saved_model', signatures=run), imports='import tempfile;from typing import Tuple;from tensorflow.experimental import dtensor;import tensorflow as tf;import numpy as np;import tensorflow_datasets as tfds', function_to_run='tf.saved_model.save(*args, **kwargs)', method_object=None, function_args=[eval('model_for_saving'), eval('"/tmp/saved_model"')], function_kwargs={'signatures': eval('run')}, max_wait_secs=0)
sample_batch = custom_method(
train_data.take(1).get_single_element(), imports='import tempfile;from typing import Tuple;from tensorflow.experimental import dtensor;import tensorflow as tf;import numpy as np;import tensorflow_datasets as tfds', function_to_run='obj.take(1).get_single_element()', method_object=eval('train_data'), function_args=[], function_kwargs={}, max_wait_secs=0, custom_class=None)
sample_batch
loaded = custom_method(
tf.saved_model.load('/tmp/saved_model'), imports='import tempfile;from typing import Tuple;from tensorflow.experimental import dtensor;import tensorflow as tf;import numpy as np;import tensorflow_datasets as tfds', function_to_run='tf.saved_model.load(*args)', method_object=None, function_args=[eval('"/tmp/saved_model"')], function_kwargs={}, max_wait_secs=0)
run_sig = loaded.signatures['serving_default']
result = run_sig(sample_batch['text'])['result']
np.mean(tf.argmax(result, axis=-1) == sample_batch['label'])
