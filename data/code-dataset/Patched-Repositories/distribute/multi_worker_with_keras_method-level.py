import json
import os
import sys
import sys
from fecom.patching.patching_config import EXPERIMENT_DIR
from fecom.measurement.execution import before_execution as before_execution_INSERTED_INTO_SCRIPT
from fecom.measurement.execution import after_execution as after_execution_INSERTED_INTO_SCRIPT
from fecom.experiment.experiment_kinds import ExperimentKinds
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
    start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT(experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tensorflow.keras.datasets.mnist.load_data()')
    ((x_train, y_train), _) = tf.keras.datasets.mnist.load_data()
    after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tensorflow.keras.datasets.mnist.load_data()', method_object=None, function_args=None, function_kwargs=None)
    x_train = x_train / np.float32(255)
    y_train = y_train.astype(np.int64)
    start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT(experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tensorflow.data.Dataset.from_tensor_slices((x_train, y_train)).shuffle(60000).repeat().batch()')
    train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train)).shuffle(60000).repeat().batch(batch_size)
    after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tensorflow.data.Dataset.from_tensor_slices((x_train, y_train)).shuffle(60000).repeat().batch()', method_object=None, function_args=[batch_size], function_kwargs=None)
    return train_dataset

def build_and_compile_cnn_model():
    start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT(experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tensorflow.keras.Sequential()')
    model = tf.keras.Sequential([tf.keras.layers.InputLayer(input_shape=(28, 28)), tf.keras.layers.Reshape(target_shape=(28, 28, 1)), tf.keras.layers.Conv2D(32, 3, activation='relu'), tf.keras.layers.Flatten(), tf.keras.layers.Dense(128, activation='relu'), tf.keras.layers.Dense(10)])
    after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tensorflow.keras.Sequential()', method_object=None, function_args=[[tf.keras.layers.InputLayer(input_shape=(28, 28)), tf.keras.layers.Reshape(target_shape=(28, 28, 1)), tf.keras.layers.Conv2D(32, 3, activation='relu'), tf.keras.layers.Flatten(), tf.keras.layers.Dense(128, activation='relu'), tf.keras.layers.Dense(10)]], function_kwargs=None)
    start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT(experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tensorflow.keras.Sequential.compile()')
    model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), optimizer=tf.keras.optimizers.SGD(learning_rate=0.001), metrics=['accuracy'])
    after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tensorflow.keras.Sequential.compile()', method_object=model, function_args=None, function_kwargs={'loss': tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), 'optimizer': tf.keras.optimizers.SGD(learning_rate=0.001), 'metrics': ['accuracy']})
    return model
import mnist_setup
batch_size = 64
single_worker_dataset = mnist_setup.mnist_dataset(batch_size)
single_worker_model = mnist_setup.build_and_compile_cnn_model()
single_worker_model.fit(single_worker_dataset, epochs=3, steps_per_epoch=70)
tf_config = {'cluster': {'worker': ['localhost:12345', 'localhost:23456']}, 'task': {'type': 'worker', 'index': 0}}
json.dumps(tf_config)
os.environ['GREETINGS'] = 'Hello TensorFlow!'
start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT(experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tensorflow.distribute.MultiWorkerMirroredStrategy()')
strategy = tf.distribute.MultiWorkerMirroredStrategy()
after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tensorflow.distribute.MultiWorkerMirroredStrategy()', method_object=None, function_args=None, function_kwargs=None)
with strategy.scope():
    multi_worker_model = mnist_setup.build_and_compile_cnn_model()
import os
import json
import tensorflow as tf
import mnist_setup
per_worker_batch_size = 64
tf_config = json.loads(os.environ['TF_CONFIG'])
num_workers = len(tf_config['cluster']['worker'])
start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT(experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tensorflow.distribute.MultiWorkerMirroredStrategy()')
strategy = tf.distribute.MultiWorkerMirroredStrategy()
after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tensorflow.distribute.MultiWorkerMirroredStrategy()', method_object=None, function_args=None, function_kwargs=None)
global_batch_size = per_worker_batch_size * num_workers
multi_worker_dataset = mnist_setup.mnist_dataset(global_batch_size)
with strategy.scope():
    multi_worker_model = mnist_setup.build_and_compile_cnn_model()
multi_worker_model.fit(multi_worker_dataset, epochs=3, steps_per_epoch=70)
os.environ['TF_CONFIG'] = json.dumps(tf_config)
import time
time.sleep(10)
tf_config['task']['index'] = 1
os.environ['TF_CONFIG'] = json.dumps(tf_config)
os.environ.pop('TF_CONFIG', None)
start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT(experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tensorflow.data.Options()')
options = tf.data.Options()
after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tensorflow.data.Options()', method_object=None, function_args=None, function_kwargs=None)
options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.OFF
global_batch_size = 64
multi_worker_dataset = mnist_setup.mnist_dataset(batch_size=64)
dataset_no_auto_shard = multi_worker_dataset.with_options(options)
model_path = '/tmp/keras-model'

def _is_chief(task_type, task_id):
    return task_type == 'worker' and task_id == 0 or task_type is None

def _get_temp_dir(dirpath, task_id):
    base_dirpath = 'workertemp_' + str(task_id)
    temp_dir = os.path.join(dirpath, base_dirpath)
    start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT(experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tensorflow.io.gfile.makedirs()')
    tf.io.gfile.makedirs(temp_dir)
    after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tensorflow.io.gfile.makedirs()', method_object=None, function_args=[temp_dir], function_kwargs=None)
    return temp_dir

def write_filepath(filepath, task_type, task_id):
    dirpath = os.path.dirname(filepath)
    base = os.path.basename(filepath)
    if not _is_chief(task_type, task_id):
        dirpath = _get_temp_dir(dirpath, task_id)
    return os.path.join(dirpath, base)
(task_type, task_id) = (strategy.cluster_resolver.task_type, strategy.cluster_resolver.task_id)
write_model_path = write_filepath(model_path, task_type, task_id)
multi_worker_model.save(write_model_path)
if not _is_chief(task_type, task_id):
    start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT(experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tensorflow.io.gfile.rmtree()')
    tf.io.gfile.rmtree(os.path.dirname(write_model_path))
    after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tensorflow.io.gfile.rmtree()', method_object=None, function_args=[os.path.dirname(write_model_path)], function_kwargs=None)
start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT(experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tensorflow.keras.models.load_model()')
loaded_model = tf.keras.models.load_model(model_path)
after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tensorflow.keras.models.load_model()', method_object=None, function_args=[model_path], function_kwargs=None)
start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT(experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tensorflow.keras.models.load_model.fit()')
loaded_model.fit(single_worker_dataset, epochs=2, steps_per_epoch=20)
after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tensorflow.keras.models.load_model.fit()', method_object=loaded_model, function_args=[single_worker_dataset], function_kwargs={'epochs': 2, 'steps_per_epoch': 20})
checkpoint_dir = '/tmp/ckpt'
start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT(experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tensorflow.train.Checkpoint()')
checkpoint = tf.train.Checkpoint(model=multi_worker_model)
after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tensorflow.train.Checkpoint()', method_object=None, function_args=None, function_kwargs={'model': multi_worker_model})
write_checkpoint_dir = write_filepath(checkpoint_dir, task_type, task_id)
start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT(experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tensorflow.train.CheckpointManager()')
checkpoint_manager = tf.train.CheckpointManager(checkpoint, directory=write_checkpoint_dir, max_to_keep=1)
after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tensorflow.train.CheckpointManager()', method_object=None, function_args=[checkpoint], function_kwargs={'directory': write_checkpoint_dir, 'max_to_keep': 1})
start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT(experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tensorflow.train.CheckpointManager.save()')
checkpoint_manager.save()
after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tensorflow.train.CheckpointManager.save()', method_object=checkpoint_manager, function_args=None, function_kwargs=None)
if not _is_chief(task_type, task_id):
    start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT(experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tensorflow.io.gfile.rmtree()')
    tf.io.gfile.rmtree(write_checkpoint_dir)
    after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tensorflow.io.gfile.rmtree()', method_object=None, function_args=[write_checkpoint_dir], function_kwargs=None)
start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT(experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tensorflow.train.latest_checkpoint()')
latest_checkpoint = tf.train.latest_checkpoint(checkpoint_dir)
after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tensorflow.train.latest_checkpoint()', method_object=None, function_args=[checkpoint_dir], function_kwargs=None)
start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT(experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tensorflow.train.Checkpoint.restore()')
checkpoint.restore(latest_checkpoint)
after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tensorflow.train.Checkpoint.restore()', method_object=checkpoint, function_args=[latest_checkpoint], function_kwargs=None)
multi_worker_model.fit(multi_worker_dataset, epochs=2, steps_per_epoch=20)
callbacks = [tf.keras.callbacks.BackupAndRestore(backup_dir='/tmp/backup')]
with strategy.scope():
    multi_worker_model = mnist_setup.build_and_compile_cnn_model()
multi_worker_model.fit(multi_worker_dataset, epochs=3, steps_per_epoch=70, callbacks=callbacks)
callbacks = [tf.keras.callbacks.BackupAndRestore(backup_dir='/tmp/backup')]
with strategy.scope():
    multi_worker_model = mnist_setup.build_and_compile_cnn_model()
multi_worker_model.fit(multi_worker_dataset, epochs=3, steps_per_epoch=70, callbacks=callbacks)
callbacks = [tf.keras.callbacks.BackupAndRestore(backup_dir='/tmp/backup', save_freq=30)]
with strategy.scope():
    multi_worker_model = mnist_setup.build_and_compile_cnn_model()
multi_worker_model.fit(multi_worker_dataset, epochs=3, steps_per_epoch=70, callbacks=callbacks)
