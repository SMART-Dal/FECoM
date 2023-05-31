import tensorflow_datasets as tfds
import tensorflow as tf
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
(datasets, info) = tfds.load(name='mnist', with_info=True, as_supervised=True)
(mnist_train, mnist_test) = (datasets['train'], datasets['test'])
custom_method(imports='import os;import tensorflow as tf;import tensorflow_datasets as tfds', function_to_run='tf.distribute.MirroredStrategy()', method_object=None, object_signature=None, function_args=[], function_kwargs={})
strategy = tf.distribute.MirroredStrategy()
print('Number of devices: {}'.format(strategy.num_replicas_in_sync))
num_train_examples = info.splits['train'].num_examples
num_test_examples = info.splits['test'].num_examples
BUFFER_SIZE = 10000
BATCH_SIZE_PER_REPLICA = 64
BATCH_SIZE = BATCH_SIZE_PER_REPLICA * strategy.num_replicas_in_sync

def scale(image, label):
    custom_method(imports='import os;import tensorflow as tf;import tensorflow_datasets as tfds', function_to_run='tf.cast(*args)', method_object=None, object_signature=None, function_args=[eval('image'), eval('tf.float32')], function_kwargs={})
    image = tf.cast(image, tf.float32)
    image /= 255
    return (image, label)
train_dataset = mnist_train.map(scale).cache().shuffle(BUFFER_SIZE).batch(BATCH_SIZE)
eval_dataset = mnist_test.map(scale).batch(BATCH_SIZE)
with strategy.scope():
    custom_method(imports='import os;import tensorflow as tf;import tensorflow_datasets as tfds', function_to_run='tf.keras.Sequential(*args)', method_object=None, object_signature=None, function_args=[eval("[\n      tf.keras.layers.Conv2D(32, 3, activation='relu', input_shape=(28, 28, 1)),\n      tf.keras.layers.MaxPooling2D(),\n      tf.keras.layers.Flatten(),\n      tf.keras.layers.Dense(64, activation='relu'),\n      tf.keras.layers.Dense(10)\n  ]")], function_kwargs={})
    model = tf.keras.Sequential([tf.keras.layers.Conv2D(32, 3, activation='relu', input_shape=(28, 28, 1)), tf.keras.layers.MaxPooling2D(), tf.keras.layers.Flatten(), tf.keras.layers.Dense(64, activation='relu'), tf.keras.layers.Dense(10)])
    custom_method(imports='import os;import tensorflow as tf;import tensorflow_datasets as tfds', function_to_run='obj.compile(**kwargs)', method_object=eval('model'), object_signature=None, function_args=[], function_kwargs={'loss': eval('tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)'), 'optimizer': eval('tf.keras.optimizers.Adam()'), 'metrics': eval("['accuracy']")}, custom_class=None)
    model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), optimizer=tf.keras.optimizers.Adam(), metrics=['accuracy'])
checkpoint_dir = './training_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, 'ckpt_{epoch}')

def decay(epoch):
    if epoch < 3:
        return 0.001
    elif epoch >= 3 and epoch < 7:
        return 0.0001
    else:
        return 1e-05

class PrintLR(tf.keras.callbacks.Callback):

    def on_epoch_end(self, epoch, logs=None):
        print('\nLearning rate for epoch {} is {}'.format(epoch + 1, model.optimizer.lr.numpy()))
callbacks = [tf.keras.callbacks.TensorBoard(log_dir='./logs'), tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_prefix, save_weights_only=True), tf.keras.callbacks.LearningRateScheduler(decay), PrintLR()]
EPOCHS = 12
custom_method(imports='import os;import tensorflow as tf;import tensorflow_datasets as tfds', function_to_run='obj.fit(*args, **kwargs)', method_object=eval('model'), object_signature=None, function_args=[eval('train_dataset')], function_kwargs={'epochs': eval('EPOCHS'), 'callbacks': eval('callbacks')}, custom_class=None)
model.fit(train_dataset, epochs=EPOCHS, callbacks=callbacks)
custom_method(imports='import os;import tensorflow as tf;import tensorflow_datasets as tfds', function_to_run='obj.load_weights(*args)', method_object=eval('model'), object_signature=None, function_args=[eval('tf.train.latest_checkpoint(checkpoint_dir)')], function_kwargs={}, custom_class=None)
model.load_weights(tf.train.latest_checkpoint(checkpoint_dir))
custom_method(imports='import os;import tensorflow as tf;import tensorflow_datasets as tfds', function_to_run='obj.evaluate(*args)', method_object=eval('model'), object_signature=None, function_args=[eval('eval_dataset')], function_kwargs={}, custom_class=None)
(eval_loss, eval_acc) = model.evaluate(eval_dataset)
print('Eval loss: {}, Eval accuracy: {}'.format(eval_loss, eval_acc))
path = 'saved_model/'
custom_method(imports='import os;import tensorflow as tf;import tensorflow_datasets as tfds', function_to_run='obj.save(*args, **kwargs)', method_object=eval('model'), object_signature=None, function_args=[eval('path')], function_kwargs={'save_format': eval("'tf'")}, custom_class=None)
model.save(path, save_format='tf')
custom_method(imports='import os;import tensorflow as tf;import tensorflow_datasets as tfds', function_to_run='tf.keras.models.load_model(*args)', method_object=None, object_signature=None, function_args=[eval('path')], function_kwargs={})
unreplicated_model = tf.keras.models.load_model(path)
custom_method(imports='import os;import tensorflow as tf;import tensorflow_datasets as tfds', function_to_run='obj.compile(**kwargs)', method_object=eval('unreplicated_model'), object_signature=None, function_args=[], function_kwargs={'loss': eval('tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)'), 'optimizer': eval('tf.keras.optimizers.Adam()'), 'metrics': eval("['accuracy']")}, custom_class=None)
unreplicated_model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), optimizer=tf.keras.optimizers.Adam(), metrics=['accuracy'])
custom_method(imports='import os;import tensorflow as tf;import tensorflow_datasets as tfds', function_to_run='obj.evaluate(*args)', method_object=eval('unreplicated_model'), object_signature=None, function_args=[eval('eval_dataset')], function_kwargs={}, custom_class=None)
(eval_loss, eval_acc) = unreplicated_model.evaluate(eval_dataset)
print('Eval loss: {}, Eval Accuracy: {}'.format(eval_loss, eval_acc))
with strategy.scope():
    custom_method(imports='import os;import tensorflow as tf;import tensorflow_datasets as tfds', function_to_run='tf.keras.models.load_model(*args)', method_object=None, object_signature=None, function_args=[eval('path')], function_kwargs={})
    replicated_model = tf.keras.models.load_model(path)
    custom_method(imports='import os;import tensorflow as tf;import tensorflow_datasets as tfds', function_to_run='obj.compile(**kwargs)', method_object=eval('replicated_model'), object_signature=None, function_args=[], function_kwargs={'loss': eval('tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)'), 'optimizer': eval('tf.keras.optimizers.Adam()'), 'metrics': eval("['accuracy']")}, custom_class=None)
    replicated_model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), optimizer=tf.keras.optimizers.Adam(), metrics=['accuracy'])
    custom_method(imports='import os;import tensorflow as tf;import tensorflow_datasets as tfds', function_to_run='obj.evaluate(*args)', method_object=eval('replicated_model'), object_signature=None, function_args=[eval('eval_dataset')], function_kwargs={}, custom_class=None)
    (eval_loss, eval_acc) = replicated_model.evaluate(eval_dataset)
    print('Eval loss: {}, Eval Accuracy: {}'.format(eval_loss, eval_acc))
