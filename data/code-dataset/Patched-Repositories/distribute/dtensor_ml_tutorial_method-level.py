import tempfile
import numpy as np
import tensorflow_datasets as tfds
import tensorflow as tf
from tensorflow.experimental import dtensor
import sys
from tool.patching.patching_config import EXPERIMENT_DIR
from tool.measurement.execution import before_execution as before_execution_INSERTED_INTO_SCRIPT
from tool.measurement.execution import after_execution as after_execution_INSERTED_INTO_SCRIPT
from tool.experiment.experiments import ExperimentKinds
experiment_number = sys.argv[1]
experiment_project = sys.argv[2]
EXPERIMENT_FILE_PATH = EXPERIMENT_DIR / ExperimentKinds.METHOD_LEVEL.value / experiment_project / f'experiment-{experiment_number}.json'
print('TensorFlow version:', tf.__version__)

def configure_virtual_cpus(ncpu):
    start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT(experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tensorflow.config.list_physical_devices()')
    phy_devices = tf.config.list_physical_devices('CPU')
    after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tensorflow.config.list_physical_devices()', method_object=None, function_args=['CPU'], function_kwargs=None)
    start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT(experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tensorflow.config.set_logical_device_configuration()')
    tf.config.set_logical_device_configuration(phy_devices[0], [tf.config.LogicalDeviceConfiguration()] * ncpu)
    after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tensorflow.config.set_logical_device_configuration()', method_object=None, function_args=[phy_devices[0], [tf.config.LogicalDeviceConfiguration()] * ncpu], function_kwargs=None)
configure_virtual_cpus(8)
DEVICES = [f'CPU:{i}' for i in range(8)]
start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT(experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tensorflow.config.list_logical_devices()')
tf.config.list_logical_devices('CPU')
after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tensorflow.config.list_logical_devices()', method_object=None, function_args=['CPU'], function_kwargs=None)
train_data = tfds.load('imdb_reviews', split='train', shuffle_files=True, batch_size=64)
train_data
start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT(experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tensorflow.keras.layers.TextVectorization()')
text_vectorization = tf.keras.layers.TextVectorization(output_mode='tf_idf', max_tokens=1200, output_sequence_length=None)
after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tensorflow.keras.layers.TextVectorization()', method_object=None, function_args=None, function_kwargs={'output_mode': 'tf_idf', 'max_tokens': 1200, 'output_sequence_length': None})
start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT(experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tensorflow.keras.layers.TextVectorization.adapt()')
text_vectorization.adapt(data=train_data.map(lambda x: x['text']))
after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tensorflow.keras.layers.TextVectorization.adapt()', method_object=text_vectorization, function_args=None, function_kwargs={'data': train_data.map(lambda x: x['text'])})

def vectorize(features):
    return (text_vectorization(features['text']), features['label'])
train_data_vec = train_data.map(vectorize)
train_data_vec

class Dense(tf.Module):

    def __init__(self, input_size, output_size, init_seed, weight_layout, activation=None):
        super().__init__()
        start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT(experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tensorflow.function()')
        random_normal_initializer = tf.function(tf.random.stateless_normal)
        after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tensorflow.function()', method_object=None, function_args=[tf.random.stateless_normal], function_kwargs=None)
        start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT(experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tensorflow.experimental.dtensor.DVariable()')
        self.weight = dtensor.DVariable(dtensor.call_with_layout(random_normal_initializer, weight_layout, shape=[input_size, output_size], seed=init_seed))
        after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tensorflow.experimental.dtensor.DVariable()', method_object=None, function_args=[dtensor.call_with_layout(random_normal_initializer, weight_layout, shape=[input_size, output_size], seed=init_seed)], function_kwargs=None)
        if activation is None:
            activation = lambda x: x
        self.activation = activation
        bias_layout = weight_layout.delete([0])
        start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT(experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tensorflow.experimental.dtensor.DVariable()')
        self.bias = dtensor.DVariable(dtensor.call_with_layout(tf.zeros, bias_layout, [output_size]))
        after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tensorflow.experimental.dtensor.DVariable()', method_object=None, function_args=[dtensor.call_with_layout(tf.zeros, bias_layout, [output_size])], function_kwargs=None)

    def __call__(self, x):
        y = tf.matmul(x, self.weight) + self.bias
        y = self.activation(y)
        return y

class BatchNorm(tf.Module):

    def __init__(self):
        super().__init__()

    def __call__(self, x, training=True):
        if not training:
            pass
        start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT(experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tensorflow.nn.moments()')
        (mean, variance) = tf.nn.moments(x, axes=[0])
        after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tensorflow.nn.moments()', method_object=None, function_args=[x], function_kwargs={'axes': [0]})
        return tf.nn.batch_normalization(x, mean, variance, 0.0, 1.0, 1e-05)

def make_keras_bn(bn_layout):
    return tf.keras.layers.BatchNormalization(gamma_layout=bn_layout, beta_layout=bn_layout, moving_mean_layout=bn_layout, moving_variance_layout=bn_layout, fused=False)
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
start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT(experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tensorflow.experimental.dtensor.create_mesh()')
WORLD = dtensor.create_mesh([('world', 8)], devices=DEVICES)
after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tensorflow.experimental.dtensor.create_mesh()', method_object=None, function_args=[[('world', 8)]], function_kwargs={'devices': DEVICES})
model = MLP([dtensor.Layout.replicated(WORLD, rank=2), dtensor.Layout.replicated(WORLD, rank=2)])
(sample_x, sample_y) = train_data_vec.take(1).get_single_element()
start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT(experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tensorflow.experimental.dtensor.copy_to_mesh()')
sample_x = dtensor.copy_to_mesh(sample_x, dtensor.Layout.replicated(WORLD, rank=2))
after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tensorflow.experimental.dtensor.copy_to_mesh()', method_object=None, function_args=[sample_x, dtensor.Layout.replicated(WORLD, rank=2)], function_kwargs=None)
print(model(sample_x))

def repack_local_tensor(x, layout):
    """Repacks a local Tensor-like to a DTensor with layout.

  This function assumes a single-client application.
  """
    start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT(experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tensorflow.convert_to_tensor()')
    x = tf.convert_to_tensor(x)
    after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tensorflow.convert_to_tensor()', method_object=None, function_args=[x], function_kwargs=None)
    sharded_dims = []
    queue = [x]
    for (axis, dim) in enumerate(layout.sharding_specs):
        if dim == dtensor.UNSHARDED:
            continue
        num_splits = layout.shape[axis]
        start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT(experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tensorflow.nest.map_structure()')
        queue = tf.nest.map_structure(lambda x: tf.split(x, num_splits, axis=axis), queue)
        after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tensorflow.nest.map_structure()', method_object=None, function_args=[lambda x: tf.split(x, num_splits, axis=axis), queue], function_kwargs=None)
        sharded_dims.append(dim)
    components = []
    for locations in layout.mesh.local_device_locations():
        t = queue[0]
        for dim in sharded_dims:
            split_index = locations[dim]
            t = t[split_index]
        components.append(t)
    return dtensor.pack(components, layout)
start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT(experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tensorflow.experimental.dtensor.create_mesh()')
mesh = dtensor.create_mesh([('batch', 8)], devices=DEVICES)
after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tensorflow.experimental.dtensor.create_mesh()', method_object=None, function_args=[[('batch', 8)]], function_kwargs={'devices': DEVICES})
model = MLP([dtensor.Layout([dtensor.UNSHARDED, dtensor.UNSHARDED], mesh), dtensor.Layout([dtensor.UNSHARDED, dtensor.UNSHARDED], mesh)])

def repack_batch(x, y, mesh):
    x = repack_local_tensor(x, layout=dtensor.Layout(['batch', dtensor.UNSHARDED], mesh))
    y = repack_local_tensor(y, layout=dtensor.Layout(['batch'], mesh))
    return (x, y)
(sample_x, sample_y) = train_data_vec.take(1).get_single_element()
(sample_x, sample_y) = repack_batch(sample_x, sample_y, mesh)
print('x', sample_x[:, 0])
print('y', sample_y)

@tf.function
def train_step(model, x, y, learning_rate=tf.constant(0.0001)):
    with tf.GradientTape() as tape:
        start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT(experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tensorflow.Module()')
        logits = model(x)
        after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tensorflow.Module()', method_object=model, function_args=[x], function_kwargs=None)
        start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT(experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tensorflow.reduce_sum()')
        loss = tf.reduce_sum(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=y))
        after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tensorflow.reduce_sum()', method_object=None, function_args=[tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=y)], function_kwargs=None)
    parameters = model.trainable_variables
    gradients = tape.gradient(loss, parameters)
    for (parameter, parameter_gradient) in zip(parameters, gradients):
        parameter.assign_sub(learning_rate * parameter_gradient)
    accuracy = 1.0 - tf.reduce_sum(tf.cast(tf.argmax(logits, axis=-1, output_type=tf.int64) != y, tf.float32)) / x.shape[0]
    loss_per_sample = loss / len(x)
    return {'loss': loss_per_sample, 'accuracy': accuracy}
CHECKPOINT_DIR = tempfile.mkdtemp()

def start_checkpoint_manager(model):
    start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT(experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tensorflow.train.Checkpoint()')
    ckpt = tf.train.Checkpoint(root=model)
    after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tensorflow.train.Checkpoint()', method_object=None, function_args=None, function_kwargs={'root': model})
    start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT(experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tensorflow.train.CheckpointManager()')
    manager = tf.train.CheckpointManager(ckpt, CHECKPOINT_DIR, max_to_keep=3)
    after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tensorflow.train.CheckpointManager()', method_object=None, function_args=[ckpt, CHECKPOINT_DIR], function_kwargs={'max_to_keep': 3})
    if manager.latest_checkpoint:
        print('Restoring a checkpoint')
        start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT(experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tensorflow.train.Checkpoint.restore(manager.latest_checkpoint).assert_consumed()')
        ckpt.restore(manager.latest_checkpoint).assert_consumed()
        after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tensorflow.train.Checkpoint.restore(manager.latest_checkpoint).assert_consumed()', method_object=ckpt, function_args=None, function_kwargs=None)
    else:
        print('New training')
    return manager
num_epochs = 2
manager = start_checkpoint_manager(model)
for epoch in range(num_epochs):
    step = 0
    start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT(experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tensorflow.keras.utils.Progbar()')
    pbar = tf.keras.utils.Progbar(target=int(train_data_vec.cardinality()), stateful_metrics=[])
    after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tensorflow.keras.utils.Progbar()', method_object=None, function_args=None, function_kwargs={'target': int(train_data_vec.cardinality()), 'stateful_metrics': []})
    metrics = {'epoch': epoch}
    for (x, y) in train_data_vec:
        (x, y) = repack_batch(x, y, mesh)
        metrics.update(train_step(model, x, y, 0.01))
        start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT(experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tensorflow.keras.utils.Progbar.update()')
        pbar.update(step, values=metrics.items(), finalize=False)
        after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tensorflow.keras.utils.Progbar.update()', method_object=pbar, function_args=[step], function_kwargs={'values': metrics.items(), 'finalize': False})
        step += 1
    manager.save()
    start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT(experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tensorflow.keras.utils.Progbar.update()')
    pbar.update(step, values=metrics.items(), finalize=True)
    after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tensorflow.keras.utils.Progbar.update()', method_object=pbar, function_args=[step], function_kwargs={'values': metrics.items(), 'finalize': True})
start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT(experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tensorflow.experimental.dtensor.create_mesh()')
mesh = dtensor.create_mesh([('batch', 4), ('model', 2)], devices=DEVICES)
after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tensorflow.experimental.dtensor.create_mesh()', method_object=None, function_args=[[('batch', 4), ('model', 2)]], function_kwargs={'devices': DEVICES})
model = MLP([dtensor.Layout([dtensor.UNSHARDED, 'model'], mesh), dtensor.Layout(['model', dtensor.UNSHARDED], mesh)])

def repack_batch(x, y, mesh):
    x = repack_local_tensor(x, layout=dtensor.Layout(['batch', dtensor.UNSHARDED], mesh))
    y = repack_local_tensor(y, layout=dtensor.Layout(['batch'], mesh))
    return (x, y)
num_epochs = 2
manager = start_checkpoint_manager(model)
for epoch in range(num_epochs):
    step = 0
    start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT(experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tensorflow.keras.utils.Progbar()')
    pbar = tf.keras.utils.Progbar(target=int(train_data_vec.cardinality()))
    after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tensorflow.keras.utils.Progbar()', method_object=None, function_args=None, function_kwargs={'target': int(train_data_vec.cardinality())})
    metrics = {'epoch': epoch}
    for (x, y) in train_data_vec:
        (x, y) = repack_batch(x, y, mesh)
        metrics.update(train_step(model, x, y, 0.01))
        start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT(experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tensorflow.keras.utils.Progbar.update()')
        pbar.update(step, values=metrics.items(), finalize=False)
        after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tensorflow.keras.utils.Progbar.update()', method_object=pbar, function_args=[step], function_kwargs={'values': metrics.items(), 'finalize': False})
        step += 1
    manager.save()
    start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT(experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tensorflow.keras.utils.Progbar.update()')
    pbar.update(step, values=metrics.items(), finalize=True)
    after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tensorflow.keras.utils.Progbar.update()', method_object=pbar, function_args=[step], function_kwargs={'values': metrics.items(), 'finalize': True})
start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT(experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tensorflow.experimental.dtensor.create_mesh()')
mesh = dtensor.create_mesh([('batch', 2), ('feature', 2), ('model', 2)], devices=DEVICES)
after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tensorflow.experimental.dtensor.create_mesh()', method_object=None, function_args=[[('batch', 2), ('feature', 2), ('model', 2)]], function_kwargs={'devices': DEVICES})
model = MLP([dtensor.Layout(['feature', 'model'], mesh), dtensor.Layout(['model', dtensor.UNSHARDED], mesh)])

def repack_batch_for_spt(x, y, mesh):
    x = repack_local_tensor(x, layout=dtensor.Layout(['batch', 'feature'], mesh))
    y = repack_local_tensor(y, layout=dtensor.Layout(['batch'], mesh))
    return (x, y)
num_epochs = 2
manager = start_checkpoint_manager(model)
for epoch in range(num_epochs):
    step = 0
    metrics = {'epoch': epoch}
    start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT(experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tensorflow.keras.utils.Progbar()')
    pbar = tf.keras.utils.Progbar(target=int(train_data_vec.cardinality()))
    after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tensorflow.keras.utils.Progbar()', method_object=None, function_args=None, function_kwargs={'target': int(train_data_vec.cardinality())})
    for (x, y) in train_data_vec:
        (x, y) = repack_batch_for_spt(x, y, mesh)
        metrics.update(train_step(model, x, y, 0.01))
        start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT(experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tensorflow.keras.utils.Progbar.update()')
        pbar.update(step, values=metrics.items(), finalize=False)
        after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tensorflow.keras.utils.Progbar.update()', method_object=pbar, function_args=[step], function_kwargs={'values': metrics.items(), 'finalize': False})
        step += 1
    manager.save()
    start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT(experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tensorflow.keras.utils.Progbar.update()')
    pbar.update(step, values=metrics.items(), finalize=True)
    after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tensorflow.keras.utils.Progbar.update()', method_object=pbar, function_args=[step], function_kwargs={'values': metrics.items(), 'finalize': True})
start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT(experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tensorflow.experimental.dtensor.create_mesh()')
mesh = dtensor.create_mesh([('world', 1)], devices=DEVICES[:1])
after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tensorflow.experimental.dtensor.create_mesh()', method_object=None, function_args=[[('world', 1)]], function_kwargs={'devices': DEVICES[:1]})
mlp = MLP([dtensor.Layout([dtensor.UNSHARDED, dtensor.UNSHARDED], mesh), dtensor.Layout([dtensor.UNSHARDED, dtensor.UNSHARDED], mesh)])
manager = start_checkpoint_manager(mlp)
start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT(experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tensorflow.keras.Sequential()')
model_for_saving = tf.keras.Sequential([text_vectorization, mlp])
after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tensorflow.keras.Sequential()', method_object=None, function_args=[[text_vectorization, mlp]], function_kwargs=None)

@tf.function(input_signature=[tf.TensorSpec([None], tf.string)])
def run(inputs):
    return {'result': model_for_saving(inputs)}
start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT(experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tensorflow.saved_model.save()')
tf.saved_model.save(model_for_saving, '/tmp/saved_model', signatures=run)
after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tensorflow.saved_model.save()', method_object=None, function_args=[model_for_saving, '/tmp/saved_model'], function_kwargs={'signatures': run})
sample_batch = train_data.take(1).get_single_element()
sample_batch
start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT(experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tensorflow.saved_model.load()')
loaded = tf.saved_model.load('/tmp/saved_model')
after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tensorflow.saved_model.load()', method_object=None, function_args=['/tmp/saved_model'], function_kwargs=None)
run_sig = loaded.signatures['serving_default']
result = run_sig(sample_batch['text'])['result']
np.mean(tf.argmax(result, axis=-1) == sample_batch['label'])
