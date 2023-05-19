import json
import os
import sys
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

def custom_method(imports: str, function_to_run: str, method_object=None, object_signature=None, function_args: list=None, function_kwargs: dict=None, custom_class=None):
    result = send_request(imports=imports, function_to_run=function_to_run, function_args=function_args, function_kwargs=function_kwargs, max_wait_secs=MAX_WAIT_S, wait_after_run_secs=WAIT_AFTER_RUN_S, method_object=method_object, object_signature=object_signature, custom_class=custom_class, experiment_file_path=EXPERIMENT_FILE_PATH)
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ.pop('TF_CONFIG', None)
if '.' not in sys.path:
    sys.path.insert(0, '.')
import tensorflow as tf
import os
import tensorflow as tf
import numpy as np

def mnist_dataset(batch_size):
    custom_method(imports='import tensorflow as tf;import time;import numpy as np;import mnist_setup;import sys;import os;import json', function_to_run='tf.keras.datasets.mnist.load_data()', method_object=None, object_signature=None, function_args=[], function_kwargs={})
    ((x_train, y_train), _) = tf.keras.datasets.mnist.load_data()
    x_train = x_train / np.float32(255)
    y_train = y_train.astype(np.int64)
    custom_method(imports='import tensorflow as tf;import time;import numpy as np;import mnist_setup;import sys;import os;import json', function_to_run='tf.data.Dataset.from_tensor_slices((x_train, y_train)).shuffle(60000).repeat().batch(*args)', method_object=None, object_signature=None, function_args=[eval('batch_size')], function_kwargs={})
    train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train)).shuffle(60000).repeat().batch(batch_size)
    return train_dataset

def build_and_compile_cnn_model():
    custom_method(imports='import tensorflow as tf;import time;import numpy as np;import mnist_setup;import sys;import os;import json', function_to_run='tf.keras.Sequential(*args)', method_object=None, object_signature=None, function_args=[eval("[\n      tf.keras.layers.InputLayer(input_shape=(28, 28)),\n      tf.keras.layers.Reshape(target_shape=(28, 28, 1)),\n      tf.keras.layers.Conv2D(32, 3, activation='relu'),\n      tf.keras.layers.Flatten(),\n      tf.keras.layers.Dense(128, activation='relu'),\n      tf.keras.layers.Dense(10)\n  ]")], function_kwargs={})
    model = tf.keras.Sequential([tf.keras.layers.InputLayer(input_shape=(28, 28)), tf.keras.layers.Reshape(target_shape=(28, 28, 1)), tf.keras.layers.Conv2D(32, 3, activation='relu'), tf.keras.layers.Flatten(), tf.keras.layers.Dense(128, activation='relu'), tf.keras.layers.Dense(10)])
    custom_method(imports='import tensorflow as tf;import time;import numpy as np;import mnist_setup;import sys;import os;import json', function_to_run='obj.compile(**kwargs)', method_object=eval('model'), object_signature=None, function_args=[], function_kwargs={'loss': eval('tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)'), 'optimizer': eval('tf.keras.optimizers.SGD(learning_rate=0.001)'), 'metrics': eval("['accuracy']")}, custom_class=None)
    model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), optimizer=tf.keras.optimizers.SGD(learning_rate=0.001), metrics=['accuracy'])
    return model
import mnist_setup
batch_size = 64
single_worker_dataset = mnist_setup.mnist_dataset(batch_size)
single_worker_model = mnist_setup.build_and_compile_cnn_model()
single_worker_model.fit(single_worker_dataset, epochs=3, steps_per_epoch=70)
tf_config = {'cluster': {'worker': ['localhost:12345', 'localhost:23456']}, 'task': {'type': 'worker', 'index': 0}}
json.dumps(tf_config)
os.environ['GREETINGS'] = 'Hello TensorFlow!'
custom_method(imports='import tensorflow as tf;import time;import numpy as np;import mnist_setup;import sys;import os;import json', function_to_run='tf.distribute.MultiWorkerMirroredStrategy()', method_object=None, object_signature=None, function_args=[], function_kwargs={})
strategy = tf.distribute.MultiWorkerMirroredStrategy()
with strategy.scope():
    multi_worker_model = mnist_setup.build_and_compile_cnn_model()
import os
import json
import tensorflow as tf
import mnist_setup
per_worker_batch_size = 64
tf_config = json.loads(os.environ['TF_CONFIG'])
num_workers = len(tf_config['cluster']['worker'])
custom_method(imports='import tensorflow as tf;import time;import numpy as np;import mnist_setup;import sys;import os;import json', function_to_run='tf.distribute.MultiWorkerMirroredStrategy()', method_object=None, object_signature=None, function_args=[], function_kwargs={})
strategy = tf.distribute.MultiWorkerMirroredStrategy()
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
custom_method(imports='import tensorflow as tf;import time;import numpy as np;import mnist_setup;import sys;import os;import json', function_to_run='tf.data.Options()', method_object=None, object_signature=None, function_args=[], function_kwargs={})
options = tf.data.Options()
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
    custom_method(imports='import tensorflow as tf;import time;import numpy as np;import mnist_setup;import sys;import os;import json', function_to_run='tf.io.gfile.makedirs(*args)', method_object=None, object_signature=None, function_args=[eval('temp_dir')], function_kwargs={})
    tf.io.gfile.makedirs(temp_dir)
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
    custom_method(imports='import tensorflow as tf;import time;import numpy as np;import mnist_setup;import sys;import os;import json', function_to_run='tf.io.gfile.rmtree(*args)', method_object=None, object_signature=None, function_args=[eval('os.path.dirname(write_model_path)')], function_kwargs={})
    tf.io.gfile.rmtree(os.path.dirname(write_model_path))
custom_method(imports='import tensorflow as tf;import time;import numpy as np;import mnist_setup;import sys;import os;import json', function_to_run='tf.keras.models.load_model(*args)', method_object=None, object_signature=None, function_args=[eval('model_path')], function_kwargs={})
loaded_model = tf.keras.models.load_model(model_path)
custom_method(imports='import tensorflow as tf;import time;import numpy as np;import mnist_setup;import sys;import os;import json', function_to_run='obj.fit(*args, **kwargs)', method_object=eval('loaded_model'), object_signature=None, function_args=[eval('single_worker_dataset')], function_kwargs={'epochs': eval('2'), 'steps_per_epoch': eval('20')}, custom_class=None)
loaded_model.fit(single_worker_dataset, epochs=2, steps_per_epoch=20)
checkpoint_dir = '/tmp/ckpt'
custom_method(imports='import tensorflow as tf;import time;import numpy as np;import mnist_setup;import sys;import os;import json', function_to_run='tf.train.Checkpoint(**kwargs)', method_object=None, object_signature=None, function_args=[], function_kwargs={'model': eval('multi_worker_model')})
checkpoint = tf.train.Checkpoint(model=multi_worker_model)
write_checkpoint_dir = write_filepath(checkpoint_dir, task_type, task_id)
custom_method(imports='import tensorflow as tf;import time;import numpy as np;import mnist_setup;import sys;import os;import json', function_to_run='tf.train.CheckpointManager(*args, **kwargs)', method_object=None, object_signature=None, function_args=[eval('checkpoint')], function_kwargs={'directory': eval('write_checkpoint_dir'), 'max_to_keep': eval('1')})
checkpoint_manager = tf.train.CheckpointManager(checkpoint, directory=write_checkpoint_dir, max_to_keep=1)
custom_method(imports='import tensorflow as tf;import time;import numpy as np;import mnist_setup;import sys;import os;import json', function_to_run='obj.save()', method_object=eval('checkpoint_manager'), object_signature=None, function_args=[], function_kwargs={}, custom_class=None)
checkpoint_manager.save()
if not _is_chief(task_type, task_id):
    custom_method(imports='import tensorflow as tf;import time;import numpy as np;import mnist_setup;import sys;import os;import json', function_to_run='tf.io.gfile.rmtree(*args)', method_object=None, object_signature=None, function_args=[eval('write_checkpoint_dir')], function_kwargs={})
    tf.io.gfile.rmtree(write_checkpoint_dir)
custom_method(imports='import tensorflow as tf;import time;import numpy as np;import mnist_setup;import sys;import os;import json', function_to_run='tf.train.latest_checkpoint(*args)', method_object=None, object_signature=None, function_args=[eval('checkpoint_dir')], function_kwargs={})
latest_checkpoint = tf.train.latest_checkpoint(checkpoint_dir)
custom_method(imports='import tensorflow as tf;import time;import numpy as np;import mnist_setup;import sys;import os;import json', function_to_run='obj.restore(*args)', method_object=eval('checkpoint'), object_signature=None, function_args=[eval('latest_checkpoint')], function_kwargs={}, custom_class=None)
checkpoint.restore(latest_checkpoint)
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
