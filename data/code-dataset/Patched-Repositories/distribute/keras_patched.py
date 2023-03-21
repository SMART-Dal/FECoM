import tensorflow_datasets as tfds
import tensorflow as tf
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

def custom_method(func, imports: str, function_to_run: str, method_object=None, object_signature=None, function_args: list=None, function_kwargs: dict=None, max_wait_secs=MAX_WAIT_S, custom_class=None, wait_after_run_secs=WAIT_AFTER_RUN_S):
    result = send_request(imports=imports, function_to_run=function_to_run, function_args=function_args, function_kwargs=function_kwargs, max_wait_secs=max_wait_secs, wait_after_run_secs=wait_after_run_secs, method_object=method_object, object_signature=object_signature, custom_class=custom_class, experiment_file_path=EXPERIMENT_FILE_PATH)
    return func
print(tf.__version__)
(datasets, info) = custom_method(
tfds.load(name='mnist', with_info=True, as_supervised=True), imports='import tensorflow_datasets as tfds;import os;import tensorflow as tf', function_to_run='tfds.load(**kwargs)', method_object=None, object_signature=None, function_args=[], function_kwargs={'name': eval("'mnist'"), 'with_info': eval('True'), 'as_supervised': eval('True')}, max_wait_secs=0)
(mnist_train, mnist_test) = (datasets['train'], datasets['test'])
strategy = custom_method(
tf.distribute.MirroredStrategy(), imports='import tensorflow_datasets as tfds;import os;import tensorflow as tf', function_to_run='tf.distribute.MirroredStrategy()', method_object=None, object_signature=None, function_args=[], function_kwargs={}, max_wait_secs=0)
print('Number of devices: {}'.format(strategy.num_replicas_in_sync))
num_train_examples = info.splits['train'].num_examples
num_test_examples = info.splits['test'].num_examples
BUFFER_SIZE = 10000
BATCH_SIZE_PER_REPLICA = 64
BATCH_SIZE = BATCH_SIZE_PER_REPLICA * strategy.num_replicas_in_sync

def scale(image, label):
    image = custom_method(
    tf.cast(image, tf.float32), imports='import tensorflow_datasets as tfds;import os;import tensorflow as tf', function_to_run='tf.cast(*args)', method_object=None, object_signature=None, function_args=[eval('image'), eval('tf.float32')], function_kwargs={}, max_wait_secs=0)
    image /= 255
    return (image, label)
train_dataset = mnist_train.map(scale).cache().shuffle(BUFFER_SIZE).batch(BATCH_SIZE)
eval_dataset = mnist_test.map(scale).batch(BATCH_SIZE)
with custom_method(
strategy.scope(), imports='import tensorflow_datasets as tfds;import os;import tensorflow as tf', function_to_run='obj.scope()', method_object=eval('strategy'), object_signature='tf.distribute.MirroredStrategy', function_args=[], function_kwargs={}, max_wait_secs=0, custom_class=None):
    model = custom_method(
    tf.keras.Sequential([tf.keras.layers.Conv2D(32, 3, activation='relu', input_shape=(28, 28, 1)), tf.keras.layers.MaxPooling2D(), tf.keras.layers.Flatten(), tf.keras.layers.Dense(64, activation='relu'), tf.keras.layers.Dense(10)]), imports='import tensorflow_datasets as tfds;import os;import tensorflow as tf', function_to_run='tf.keras.Sequential(*args)', method_object=None, object_signature=None, function_args=[eval("[\n      tf.keras.layers.Conv2D(32, 3, activation='relu', input_shape=(28, 28, 1)),\n      tf.keras.layers.MaxPooling2D(),\n      tf.keras.layers.Flatten(),\n      tf.keras.layers.Dense(64, activation='relu'),\n      tf.keras.layers.Dense(10)\n  ]")], function_kwargs={}, max_wait_secs=0)
    custom_method(
    model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), optimizer=tf.keras.optimizers.Adam(), metrics=['accuracy']), imports='import tensorflow_datasets as tfds;import os;import tensorflow as tf', function_to_run='obj.compile(**kwargs)', method_object=eval('model'), object_signature='tf.keras.Sequential', function_args=[], function_kwargs={'loss': eval('tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)'), 'optimizer': eval('tf.keras.optimizers.Adam()'), 'metrics': eval("['accuracy']")}, max_wait_secs=0, custom_class=None)
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
callbacks = [custom_method(
tf.keras.callbacks.TensorBoard(log_dir='./logs'), imports='import tensorflow_datasets as tfds;import os;import tensorflow as tf', function_to_run='tf.keras.callbacks.TensorBoard(**kwargs)', method_object=None, object_signature=None, function_args=[], function_kwargs={'log_dir': eval("'./logs'")}, max_wait_secs=0), custom_method(
tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_prefix, save_weights_only=True), imports='import tensorflow_datasets as tfds;import os;import tensorflow as tf', function_to_run='tf.keras.callbacks.ModelCheckpoint(**kwargs)', method_object=None, object_signature=None, function_args=[], function_kwargs={'filepath': eval('checkpoint_prefix'), 'save_weights_only': eval('True')}, max_wait_secs=0), custom_method(
tf.keras.callbacks.LearningRateScheduler(decay), imports='import tensorflow_datasets as tfds;import os;import tensorflow as tf', function_to_run='tf.keras.callbacks.LearningRateScheduler(*args)', method_object=None, object_signature=None, function_args=[eval('decay')], function_kwargs={}, max_wait_secs=0), PrintLR()]
EPOCHS = 12
custom_method(
model.fit(train_dataset, epochs=EPOCHS, callbacks=callbacks), imports='import tensorflow_datasets as tfds;import os;import tensorflow as tf', function_to_run='obj.fit(*args, **kwargs)', method_object=eval('model'), object_signature='tf.keras.Sequential', function_args=[eval('train_dataset')], function_kwargs={'epochs': eval('EPOCHS'), 'callbacks': eval('callbacks')}, max_wait_secs=0, custom_class=None)
custom_method(
model.load_weights(tf.train.latest_checkpoint(checkpoint_dir)), imports='import tensorflow_datasets as tfds;import os;import tensorflow as tf', function_to_run='obj.load_weights(*args)', method_object=eval('model'), object_signature='tf.keras.Sequential', function_args=[eval('tf.train.latest_checkpoint(checkpoint_dir)')], function_kwargs={}, max_wait_secs=0, custom_class=None)
(eval_loss, eval_acc) = custom_method(
model.evaluate(eval_dataset), imports='import tensorflow_datasets as tfds;import os;import tensorflow as tf', function_to_run='obj.evaluate(*args)', method_object=eval('model'), object_signature='tf.keras.Sequential', function_args=[eval('eval_dataset')], function_kwargs={}, max_wait_secs=0, custom_class=None)
print('Eval loss: {}, Eval accuracy: {}'.format(eval_loss, eval_acc))
path = 'saved_model/'
custom_method(
model.save(path, save_format='tf'), imports='import tensorflow_datasets as tfds;import os;import tensorflow as tf', function_to_run='obj.save(*args, **kwargs)', method_object=eval('model'), object_signature='tf.keras.Sequential', function_args=[eval('path')], function_kwargs={'save_format': eval("'tf'")}, max_wait_secs=0, custom_class=None)
unreplicated_model = custom_method(
tf.keras.models.load_model(path), imports='import tensorflow_datasets as tfds;import os;import tensorflow as tf', function_to_run='tf.keras.models.load_model(*args)', method_object=None, object_signature=None, function_args=[eval('path')], function_kwargs={}, max_wait_secs=0)
custom_method(
unreplicated_model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), optimizer=tf.keras.optimizers.Adam(), metrics=['accuracy']), imports='import tensorflow_datasets as tfds;import os;import tensorflow as tf', function_to_run='obj.compile(**kwargs)', method_object=eval('unreplicated_model'), object_signature='tf.keras.Sequential', function_args=[], function_kwargs={'loss': eval('tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)'), 'optimizer': eval('tf.keras.optimizers.Adam()'), 'metrics': eval("['accuracy']")}, max_wait_secs=0, custom_class=None)
(eval_loss, eval_acc) = custom_method(
unreplicated_model.evaluate(eval_dataset), imports='import tensorflow_datasets as tfds;import os;import tensorflow as tf', function_to_run='obj.evaluate(*args)', method_object=eval('unreplicated_model'), object_signature='tf.keras.Sequential', function_args=[eval('eval_dataset')], function_kwargs={}, max_wait_secs=0, custom_class=None)
print('Eval loss: {}, Eval Accuracy: {}'.format(eval_loss, eval_acc))
with custom_method(
strategy.scope(), imports='import tensorflow_datasets as tfds;import os;import tensorflow as tf', function_to_run='obj.scope()', method_object=eval('strategy'), object_signature='tf.distribute.MirroredStrategy', function_args=[], function_kwargs={}, max_wait_secs=0, custom_class=None):
    replicated_model = custom_method(
    tf.keras.models.load_model(path), imports='import tensorflow_datasets as tfds;import os;import tensorflow as tf', function_to_run='tf.keras.models.load_model(*args)', method_object=None, object_signature=None, function_args=[eval('path')], function_kwargs={}, max_wait_secs=0)
    custom_method(
    replicated_model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), optimizer=tf.keras.optimizers.Adam(), metrics=['accuracy']), imports='import tensorflow_datasets as tfds;import os;import tensorflow as tf', function_to_run='obj.compile(**kwargs)', method_object=eval('replicated_model'), object_signature='tf.keras.Sequential', function_args=[], function_kwargs={'loss': eval('tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)'), 'optimizer': eval('tf.keras.optimizers.Adam()'), 'metrics': eval("['accuracy']")}, max_wait_secs=0, custom_class=None)
    (eval_loss, eval_acc) = custom_method(
    replicated_model.evaluate(eval_dataset), imports='import tensorflow_datasets as tfds;import os;import tensorflow as tf', function_to_run='obj.evaluate(*args)', method_object=eval('replicated_model'), object_signature='tf.keras.Sequential', function_args=[eval('eval_dataset')], function_kwargs={}, max_wait_secs=0, custom_class=None)
    print('Eval loss: {}, Eval Accuracy: {}'.format(eval_loss, eval_acc))
