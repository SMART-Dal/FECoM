import json
import os
import sys
import sys
from tool.patching.patching_config import EXPERIMENT_DIR
from tool.measurement.execution import before_execution as before_execution_INSERTED_INTO_SCRIPT
from tool.measurement.execution import after_execution as after_execution_INSERTED_INTO_SCRIPT
from tool.experiment.experiments import ExperimentKinds
experiment_number = sys.argv[1]
experiment_project = sys.argv[2]
EXPERIMENT_FILE_PATH = EXPERIMENT_DIR / ExperimentKinds.METHOD_LEVEL.value / experiment_project / f'experiment-{experiment_number}.json'
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ.pop('TF_CONFIG', None)
if '.' not in sys.path:
    sys.path.insert(0, '.')
import tensorflow as tf
import os
import tensorflow as tf
import numpy as np

def mnist_dataset(batch_size):
    start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT()
    ((x_train, y_train), _) = tf.keras.datasets.mnist.load_data()
    after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tensorflow.keras.datasets.mnist.load_data()', method_object=None, function_args=None, function_kwargs=None)
    x_train = x_train / np.float32(255)
    y_train = y_train.astype(np.int64)
    start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT()
    train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train)).shuffle(60000)
    after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tensorflow.data.Dataset.from_tensor_slices((x_train, y_train)).shuffle()', method_object=None, function_args=[60000], function_kwargs=None)
    return train_dataset

def dataset_fn(global_batch_size, input_context):
    batch_size = input_context.get_per_replica_batch_size(global_batch_size)
    dataset = mnist_dataset(batch_size)
    dataset = dataset.shard(input_context.num_input_pipelines, input_context.input_pipeline_id)
    dataset = dataset.batch(batch_size)
    return dataset

def build_cnn_model():
    return tf.keras.Sequential([tf.keras.Input(shape=(28, 28)), tf.keras.layers.Reshape(target_shape=(28, 28, 1)), tf.keras.layers.Conv2D(32, 3, activation='relu'), tf.keras.layers.Flatten(), tf.keras.layers.Dense(128, activation='relu'), tf.keras.layers.Dense(10)])
tf_config = {'cluster': {'worker': ['localhost:12345', 'localhost:23456']}, 'task': {'type': 'worker', 'index': 0}}
json.dumps(tf_config)
os.environ['GREETINGS'] = 'Hello TensorFlow!'
start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT()
strategy = tf.distribute.MultiWorkerMirroredStrategy()
after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tensorflow.distribute.MultiWorkerMirroredStrategy()', method_object=None, function_args=None, function_kwargs=None)
import mnist
with strategy.scope():
    multi_worker_model = mnist.build_cnn_model()
per_worker_batch_size = 64
num_workers = len(tf_config['cluster']['worker'])
global_batch_size = per_worker_batch_size * num_workers
with strategy.scope():
    start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT()
    multi_worker_dataset = strategy.distribute_datasets_from_function(lambda input_context: mnist.dataset_fn(global_batch_size, input_context))
    after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tensorflow.distribute.MultiWorkerMirroredStrategy.distribute_datasets_from_function()', method_object=strategy, function_args=[lambda input_context: mnist.dataset_fn(global_batch_size, input_context)], function_kwargs=None)
with strategy.scope():
    start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT()
    optimizer = tf.keras.optimizers.RMSprop(learning_rate=0.001)
    after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tensorflow.keras.optimizers.RMSprop()', method_object=None, function_args=None, function_kwargs={'learning_rate': 0.001})
    start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT()
    train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')
    after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tensorflow.keras.metrics.SparseCategoricalAccuracy()', method_object=None, function_args=None, function_kwargs={'name': 'train_accuracy'})

@tf.function
def train_step(iterator):
    """Training step function."""

    def step_fn(inputs):
        """Per-Replica step function."""
        (x, y) = inputs
        with tf.GradientTape() as tape:
            predictions = multi_worker_model(x, training=True)
            start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT()
            per_batch_loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction=tf.keras.losses.Reduction.NONE)(y, predictions)
            after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tensorflow.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction=tf.keras.losses.Reduction.NONE)()', method_object=None, function_args=[y, predictions], function_kwargs=None)
            start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT()
            loss = tf.nn.compute_average_loss(per_batch_loss, global_batch_size=global_batch_size)
            after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tensorflow.nn.compute_average_loss()', method_object=None, function_args=[per_batch_loss], function_kwargs={'global_batch_size': global_batch_size})
        grads = tape.gradient(loss, multi_worker_model.trainable_variables)
        start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT()
        optimizer.apply_gradients(zip(grads, multi_worker_model.trainable_variables))
        after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tensorflow.keras.optimizers.RMSprop.apply_gradients()', method_object=optimizer, function_args=[zip(grads, multi_worker_model.trainable_variables)], function_kwargs=None)
        start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT()
        train_accuracy.update_state(y, predictions)
        after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tensorflow.keras.metrics.SparseCategoricalAccuracy.update_state()', method_object=train_accuracy, function_args=[y, predictions], function_kwargs=None)
        return loss
    start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT()
    per_replica_losses = strategy.run(step_fn, args=(next(iterator),))
    after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tensorflow.distribute.MultiWorkerMirroredStrategy.run()', method_object=strategy, function_args=[step_fn], function_kwargs={'args': (next(iterator),)})
    return strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_losses, axis=None)
from multiprocessing import util
checkpoint_dir = os.path.join(util.get_temp_dir(), 'ckpt')

def _is_chief(task_type, task_id, cluster_spec):
    return task_type is None or task_type == 'chief' or (task_type == 'worker' and task_id == 0 and ('chief' not in cluster_spec.as_dict()))

def _get_temp_dir(dirpath, task_id):
    base_dirpath = 'workertemp_' + str(task_id)
    temp_dir = os.path.join(dirpath, base_dirpath)
    start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT()
    tf.io.gfile.makedirs(temp_dir)
    after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tensorflow.io.gfile.makedirs()', method_object=None, function_args=[temp_dir], function_kwargs=None)
    return temp_dir

def write_filepath(filepath, task_type, task_id, cluster_spec):
    dirpath = os.path.dirname(filepath)
    base = os.path.basename(filepath)
    if not _is_chief(task_type, task_id, cluster_spec):
        dirpath = _get_temp_dir(dirpath, task_id)
    return os.path.join(dirpath, base)
start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT()
epoch = tf.Variable(initial_value=tf.constant(0, dtype=tf.dtypes.int64), name='epoch')
after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tensorflow.Variable()', method_object=None, function_args=None, function_kwargs={'initial_value': tf.constant(0, dtype=tf.dtypes.int64), 'name': 'epoch'})
start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT()
step_in_epoch = tf.Variable(initial_value=tf.constant(0, dtype=tf.dtypes.int64), name='step_in_epoch')
after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tensorflow.Variable()', method_object=None, function_args=None, function_kwargs={'initial_value': tf.constant(0, dtype=tf.dtypes.int64), 'name': 'step_in_epoch'})
(task_type, task_id) = (strategy.cluster_resolver.task_type, strategy.cluster_resolver.task_id)
start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT()
cluster_spec = tf.train.ClusterSpec(tf_config['cluster'])
after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tensorflow.train.ClusterSpec()', method_object=None, function_args=[tf_config['cluster']], function_kwargs=None)
start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT()
checkpoint = tf.train.Checkpoint(model=multi_worker_model, epoch=epoch, step_in_epoch=step_in_epoch)
after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tensorflow.train.Checkpoint()', method_object=None, function_args=None, function_kwargs={'model': multi_worker_model, 'epoch': epoch, 'step_in_epoch': step_in_epoch})
write_checkpoint_dir = write_filepath(checkpoint_dir, task_type, task_id, cluster_spec)
start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT()
checkpoint_manager = tf.train.CheckpointManager(checkpoint, directory=write_checkpoint_dir, max_to_keep=1)
after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tensorflow.train.CheckpointManager()', method_object=None, function_args=[checkpoint], function_kwargs={'directory': write_checkpoint_dir, 'max_to_keep': 1})
start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT()
latest_checkpoint = tf.train.latest_checkpoint(checkpoint_dir)
after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tensorflow.train.latest_checkpoint()', method_object=None, function_args=[checkpoint_dir], function_kwargs=None)
if latest_checkpoint:
    start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT()
    checkpoint.restore(latest_checkpoint)
    after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tensorflow.train.Checkpoint.restore()', method_object=checkpoint, function_args=[latest_checkpoint], function_kwargs=None)
num_epochs = 3
num_steps_per_epoch = 70
while epoch.numpy() < num_epochs:
    iterator = iter(multi_worker_dataset)
    total_loss = 0.0
    num_batches = 0
    while step_in_epoch.numpy() < num_steps_per_epoch:
        total_loss += train_step(iterator)
        num_batches += 1
        start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT()
        step_in_epoch.assign_add(1)
        after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tensorflow.Variable.assign_add()', method_object=step_in_epoch, function_args=[1], function_kwargs=None)
    train_loss = total_loss / num_batches
    print('Epoch: %d, accuracy: %f, train_loss: %f.' % (epoch.numpy(), train_accuracy.result(), train_loss))
    start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT()
    train_accuracy.reset_states()
    after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tensorflow.keras.metrics.SparseCategoricalAccuracy.reset_states()', method_object=train_accuracy, function_args=None, function_kwargs=None)
    start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT()
    checkpoint_manager.save()
    after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tensorflow.train.CheckpointManager.save()', method_object=checkpoint_manager, function_args=None, function_kwargs=None)
    if not _is_chief(task_type, task_id, cluster_spec):
        start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT()
        tf.io.gfile.rmtree(write_checkpoint_dir)
        after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tensorflow.io.gfile.rmtree()', method_object=None, function_args=[write_checkpoint_dir], function_kwargs=None)
    start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT()
    epoch.assign_add(1)
    after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tensorflow.Variable.assign_add()', method_object=epoch, function_args=[1], function_kwargs=None)
    start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT()
    step_in_epoch.assign(0)
    after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tensorflow.Variable.assign()', method_object=step_in_epoch, function_args=[0], function_kwargs=None)
import os
import json
import tensorflow as tf
import mnist
from multiprocessing import util
per_worker_batch_size = 64
tf_config = json.loads(os.environ['TF_CONFIG'])
num_workers = len(tf_config['cluster']['worker'])
global_batch_size = per_worker_batch_size * num_workers
num_epochs = 3
num_steps_per_epoch = 70

def _is_chief(task_type, task_id, cluster_spec):
    return task_type is None or task_type == 'chief' or (task_type == 'worker' and task_id == 0 and ('chief' not in cluster_spec.as_dict()))

def _get_temp_dir(dirpath, task_id):
    base_dirpath = 'workertemp_' + str(task_id)
    temp_dir = os.path.join(dirpath, base_dirpath)
    start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT()
    tf.io.gfile.makedirs(temp_dir)
    after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tensorflow.io.gfile.makedirs()', method_object=None, function_args=[temp_dir], function_kwargs=None)
    return temp_dir

def write_filepath(filepath, task_type, task_id, cluster_spec):
    dirpath = os.path.dirname(filepath)
    base = os.path.basename(filepath)
    if not _is_chief(task_type, task_id, cluster_spec):
        dirpath = _get_temp_dir(dirpath, task_id)
    return os.path.join(dirpath, base)
checkpoint_dir = os.path.join(util.get_temp_dir(), 'ckpt')
start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT()
strategy = tf.distribute.MultiWorkerMirroredStrategy()
after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tensorflow.distribute.MultiWorkerMirroredStrategy()', method_object=None, function_args=None, function_kwargs=None)
with strategy.scope():
    multi_worker_model = mnist.build_cnn_model()
    start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT()
    multi_worker_dataset = strategy.distribute_datasets_from_function(lambda input_context: mnist.dataset_fn(global_batch_size, input_context))
    after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tensorflow.distribute.MultiWorkerMirroredStrategy.distribute_datasets_from_function()', method_object=strategy, function_args=[lambda input_context: mnist.dataset_fn(global_batch_size, input_context)], function_kwargs=None)
    start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT()
    optimizer = tf.keras.optimizers.RMSprop(learning_rate=0.001)
    after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tensorflow.keras.optimizers.RMSprop()', method_object=None, function_args=None, function_kwargs={'learning_rate': 0.001})
    start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT()
    train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')
    after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tensorflow.keras.metrics.SparseCategoricalAccuracy()', method_object=None, function_args=None, function_kwargs={'name': 'train_accuracy'})

@tf.function
def train_step(iterator):
    """Training step function."""

    def step_fn(inputs):
        """Per-Replica step function."""
        (x, y) = inputs
        with tf.GradientTape() as tape:
            predictions = multi_worker_model(x, training=True)
            start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT()
            per_batch_loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction=tf.keras.losses.Reduction.NONE)(y, predictions)
            after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tensorflow.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction=tf.keras.losses.Reduction.NONE)()', method_object=None, function_args=[y, predictions], function_kwargs=None)
            start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT()
            loss = tf.nn.compute_average_loss(per_batch_loss, global_batch_size=global_batch_size)
            after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tensorflow.nn.compute_average_loss()', method_object=None, function_args=[per_batch_loss], function_kwargs={'global_batch_size': global_batch_size})
        grads = tape.gradient(loss, multi_worker_model.trainable_variables)
        start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT()
        optimizer.apply_gradients(zip(grads, multi_worker_model.trainable_variables))
        after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tensorflow.keras.optimizers.RMSprop.apply_gradients()', method_object=optimizer, function_args=[zip(grads, multi_worker_model.trainable_variables)], function_kwargs=None)
        start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT()
        train_accuracy.update_state(y, predictions)
        after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tensorflow.keras.metrics.SparseCategoricalAccuracy.update_state()', method_object=train_accuracy, function_args=[y, predictions], function_kwargs=None)
        return loss
    start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT()
    per_replica_losses = strategy.run(step_fn, args=(next(iterator),))
    after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tensorflow.distribute.MultiWorkerMirroredStrategy.run()', method_object=strategy, function_args=[step_fn], function_kwargs={'args': (next(iterator),)})
    return strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_losses, axis=None)
start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT()
epoch = tf.Variable(initial_value=tf.constant(0, dtype=tf.dtypes.int64), name='epoch')
after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tensorflow.Variable()', method_object=None, function_args=None, function_kwargs={'initial_value': tf.constant(0, dtype=tf.dtypes.int64), 'name': 'epoch'})
start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT()
step_in_epoch = tf.Variable(initial_value=tf.constant(0, dtype=tf.dtypes.int64), name='step_in_epoch')
after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tensorflow.Variable()', method_object=None, function_args=None, function_kwargs={'initial_value': tf.constant(0, dtype=tf.dtypes.int64), 'name': 'step_in_epoch'})
(task_type, task_id, cluster_spec) = (strategy.cluster_resolver.task_type, strategy.cluster_resolver.task_id, strategy.cluster_resolver.cluster_spec())
start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT()
checkpoint = tf.train.Checkpoint(model=multi_worker_model, epoch=epoch, step_in_epoch=step_in_epoch)
after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tensorflow.train.Checkpoint()', method_object=None, function_args=None, function_kwargs={'model': multi_worker_model, 'epoch': epoch, 'step_in_epoch': step_in_epoch})
write_checkpoint_dir = write_filepath(checkpoint_dir, task_type, task_id, cluster_spec)
start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT()
checkpoint_manager = tf.train.CheckpointManager(checkpoint, directory=write_checkpoint_dir, max_to_keep=1)
after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tensorflow.train.CheckpointManager()', method_object=None, function_args=[checkpoint], function_kwargs={'directory': write_checkpoint_dir, 'max_to_keep': 1})
start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT()
latest_checkpoint = tf.train.latest_checkpoint(checkpoint_dir)
after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tensorflow.train.latest_checkpoint()', method_object=None, function_args=[checkpoint_dir], function_kwargs=None)
if latest_checkpoint:
    start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT()
    checkpoint.restore(latest_checkpoint)
    after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tensorflow.train.Checkpoint.restore()', method_object=checkpoint, function_args=[latest_checkpoint], function_kwargs=None)
while epoch.numpy() < num_epochs:
    iterator = iter(multi_worker_dataset)
    total_loss = 0.0
    num_batches = 0
    while step_in_epoch.numpy() < num_steps_per_epoch:
        total_loss += train_step(iterator)
        num_batches += 1
        start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT()
        step_in_epoch.assign_add(1)
        after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tensorflow.Variable.assign_add()', method_object=step_in_epoch, function_args=[1], function_kwargs=None)
    train_loss = total_loss / num_batches
    print('Epoch: %d, accuracy: %f, train_loss: %f.' % (epoch.numpy(), train_accuracy.result(), train_loss))
    start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT()
    train_accuracy.reset_states()
    after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tensorflow.keras.metrics.SparseCategoricalAccuracy.reset_states()', method_object=train_accuracy, function_args=None, function_kwargs=None)
    start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT()
    checkpoint_manager.save()
    after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tensorflow.train.CheckpointManager.save()', method_object=checkpoint_manager, function_args=None, function_kwargs=None)
    if not _is_chief(task_type, task_id, cluster_spec):
        start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT()
        tf.io.gfile.rmtree(write_checkpoint_dir)
        after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tensorflow.io.gfile.rmtree()', method_object=None, function_args=[write_checkpoint_dir], function_kwargs=None)
    start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT()
    epoch.assign_add(1)
    after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tensorflow.Variable.assign_add()', method_object=epoch, function_args=[1], function_kwargs=None)
    start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT()
    step_in_epoch.assign(0)
    after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tensorflow.Variable.assign()', method_object=step_in_epoch, function_args=[0], function_kwargs=None)
os.environ['TF_CONFIG'] = json.dumps(tf_config)
import time
time.sleep(20)
tf_config['task']['index'] = 1
os.environ['TF_CONFIG'] = json.dumps(tf_config)
os.environ.pop('TF_CONFIG', None)
