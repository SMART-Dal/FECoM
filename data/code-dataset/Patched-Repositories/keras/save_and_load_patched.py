import os
import tensorflow as tf
from tensorflow import keras
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

def custom_method(func, imports: str, function_to_run: str, method_object=None, object_signature=None, function_args: list=None, function_kwargs: dict=None, custom_class=None):
    result = send_request(imports=imports, function_to_run=function_to_run, function_args=function_args, function_kwargs=function_kwargs, max_wait_secs=MAX_WAIT_S, wait_after_run_secs=WAIT_AFTER_RUN_S, method_object=method_object, object_signature=object_signature, custom_class=custom_class, experiment_file_path=EXPERIMENT_FILE_PATH)
    return func
print(tf.version.VERSION)
((train_images, train_labels), (test_images, test_labels)) = custom_method(
tf.keras.datasets.mnist.load_data(), imports='import tensorflow as tf;from tensorflow import keras;import os', function_to_run='tf.keras.datasets.mnist.load_data()', method_object=None, object_signature=None, function_args=[], function_kwargs={})
train_labels = train_labels[:1000]
test_labels = test_labels[:1000]
train_images = train_images[:1000].reshape(-1, 28 * 28) / 255.0
test_images = test_images[:1000].reshape(-1, 28 * 28) / 255.0

def create_model():
    model = custom_method(
    tf.keras.Sequential([keras.layers.Dense(512, activation='relu', input_shape=(784,)), keras.layers.Dropout(0.2), keras.layers.Dense(10)]), imports='import tensorflow as tf;from tensorflow import keras;import os', function_to_run='tf.keras.Sequential(*args)', method_object=None, object_signature=None, function_args=[eval("[\n    keras.layers.Dense(512, activation='relu', input_shape=(784,)),\n    keras.layers.Dropout(0.2),\n    keras.layers.Dense(10)\n  ]")], function_kwargs={})
    custom_method(
    model.compile(optimizer='adam', loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=[tf.keras.metrics.SparseCategoricalAccuracy()]), imports='import tensorflow as tf;from tensorflow import keras;import os', function_to_run='obj.compile(**kwargs)', method_object=eval('model'), object_signature=None, function_args=[], function_kwargs={'optimizer': eval("'adam'"), 'loss': eval('tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)'), 'metrics': eval('[tf.keras.metrics.SparseCategoricalAccuracy()]')}, custom_class=None)
    return model
model = create_model()
custom_method(
model.summary(), imports='import tensorflow as tf;from tensorflow import keras;import os', function_to_run='obj.summary()', method_object=eval('model'), object_signature=None, function_args=[], function_kwargs={}, custom_class=None)
checkpoint_path = 'training_1/cp.ckpt'
checkpoint_dir = os.path.dirname(checkpoint_path)
cp_callback = custom_method(
tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path, save_weights_only=True, verbose=1), imports='import tensorflow as tf;from tensorflow import keras;import os', function_to_run='tf.keras.callbacks.ModelCheckpoint(**kwargs)', method_object=None, object_signature=None, function_args=[], function_kwargs={'filepath': eval('checkpoint_path'), 'save_weights_only': eval('True'), 'verbose': eval('1')})
custom_method(
model.fit(train_images, train_labels, epochs=10, validation_data=(test_images, test_labels), callbacks=[cp_callback]), imports='import tensorflow as tf;from tensorflow import keras;import os', function_to_run='obj.fit(*args, **kwargs)', method_object=eval('model'), object_signature=None, function_args=[eval('train_images'), eval('train_labels')], function_kwargs={'epochs': eval('10'), 'validation_data': eval('(test_images, test_labels)'), 'callbacks': eval('[cp_callback]')}, custom_class=None)
os.listdir(checkpoint_dir)
model = create_model()
(loss, acc) = custom_method(
model.evaluate(test_images, test_labels, verbose=2), imports='import tensorflow as tf;from tensorflow import keras;import os', function_to_run='obj.evaluate(*args, **kwargs)', method_object=eval('model'), object_signature=None, function_args=[eval('test_images'), eval('test_labels')], function_kwargs={'verbose': eval('2')}, custom_class=None)
print('Untrained model, accuracy: {:5.2f}%'.format(100 * acc))
custom_method(
model.load_weights(checkpoint_path), imports='import tensorflow as tf;from tensorflow import keras;import os', function_to_run='obj.load_weights(*args)', method_object=eval('model'), object_signature=None, function_args=[eval('checkpoint_path')], function_kwargs={}, custom_class=None)
(loss, acc) = custom_method(
model.evaluate(test_images, test_labels, verbose=2), imports='import tensorflow as tf;from tensorflow import keras;import os', function_to_run='obj.evaluate(*args, **kwargs)', method_object=eval('model'), object_signature=None, function_args=[eval('test_images'), eval('test_labels')], function_kwargs={'verbose': eval('2')}, custom_class=None)
print('Restored model, accuracy: {:5.2f}%'.format(100 * acc))
checkpoint_path = 'training_2/cp-{epoch:04d}.ckpt'
checkpoint_dir = os.path.dirname(checkpoint_path)
batch_size = 32
cp_callback = custom_method(
tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path, verbose=1, save_weights_only=True, save_freq=5 * batch_size), imports='import tensorflow as tf;from tensorflow import keras;import os', function_to_run='tf.keras.callbacks.ModelCheckpoint(**kwargs)', method_object=None, object_signature=None, function_args=[], function_kwargs={'filepath': eval('checkpoint_path'), 'verbose': eval('1'), 'save_weights_only': eval('True'), 'save_freq': eval('5*batch_size')})
model = create_model()
custom_method(
model.save_weights(checkpoint_path.format(epoch=0)), imports='import tensorflow as tf;from tensorflow import keras;import os', function_to_run='obj.save_weights(*args)', method_object=eval('model'), object_signature=None, function_args=[eval('checkpoint_path.format(epoch=0)')], function_kwargs={}, custom_class=None)
custom_method(
model.fit(train_images, train_labels, epochs=50, batch_size=batch_size, callbacks=[cp_callback], validation_data=(test_images, test_labels), verbose=0), imports='import tensorflow as tf;from tensorflow import keras;import os', function_to_run='obj.fit(*args, **kwargs)', method_object=eval('model'), object_signature=None, function_args=[eval('train_images'), eval('train_labels')], function_kwargs={'epochs': eval('50'), 'batch_size': eval('batch_size'), 'callbacks': eval('[cp_callback]'), 'validation_data': eval('(test_images, test_labels)'), 'verbose': eval('0')}, custom_class=None)
os.listdir(checkpoint_dir)
latest = custom_method(
tf.train.latest_checkpoint(checkpoint_dir), imports='import tensorflow as tf;from tensorflow import keras;import os', function_to_run='tf.train.latest_checkpoint(*args)', method_object=None, object_signature=None, function_args=[eval('checkpoint_dir')], function_kwargs={})
latest
model = create_model()
custom_method(
model.load_weights(latest), imports='import tensorflow as tf;from tensorflow import keras;import os', function_to_run='obj.load_weights(*args)', method_object=eval('model'), object_signature=None, function_args=[eval('latest')], function_kwargs={}, custom_class=None)
(loss, acc) = custom_method(
model.evaluate(test_images, test_labels, verbose=2), imports='import tensorflow as tf;from tensorflow import keras;import os', function_to_run='obj.evaluate(*args, **kwargs)', method_object=eval('model'), object_signature=None, function_args=[eval('test_images'), eval('test_labels')], function_kwargs={'verbose': eval('2')}, custom_class=None)
print('Restored model, accuracy: {:5.2f}%'.format(100 * acc))
custom_method(
model.save_weights('./checkpoints/my_checkpoint'), imports='import tensorflow as tf;from tensorflow import keras;import os', function_to_run='obj.save_weights(*args)', method_object=eval('model'), object_signature=None, function_args=[eval("'./checkpoints/my_checkpoint'")], function_kwargs={}, custom_class=None)
model = create_model()
custom_method(
model.load_weights('./checkpoints/my_checkpoint'), imports='import tensorflow as tf;from tensorflow import keras;import os', function_to_run='obj.load_weights(*args)', method_object=eval('model'), object_signature=None, function_args=[eval("'./checkpoints/my_checkpoint'")], function_kwargs={}, custom_class=None)
(loss, acc) = custom_method(
model.evaluate(test_images, test_labels, verbose=2), imports='import tensorflow as tf;from tensorflow import keras;import os', function_to_run='obj.evaluate(*args, **kwargs)', method_object=eval('model'), object_signature=None, function_args=[eval('test_images'), eval('test_labels')], function_kwargs={'verbose': eval('2')}, custom_class=None)
print('Restored model, accuracy: {:5.2f}%'.format(100 * acc))
model = create_model()
custom_method(
model.fit(train_images, train_labels, epochs=5), imports='import tensorflow as tf;from tensorflow import keras;import os', function_to_run='obj.fit(*args, **kwargs)', method_object=eval('model'), object_signature=None, function_args=[eval('train_images'), eval('train_labels')], function_kwargs={'epochs': eval('5')}, custom_class=None)
custom_method(
model.save('saved_model/my_model'), imports='import tensorflow as tf;from tensorflow import keras;import os', function_to_run='obj.save(*args)', method_object=eval('model'), object_signature=None, function_args=[eval("'saved_model/my_model'")], function_kwargs={}, custom_class=None)
new_model = custom_method(
tf.keras.models.load_model('saved_model/my_model'), imports='import tensorflow as tf;from tensorflow import keras;import os', function_to_run='tf.keras.models.load_model(*args)', method_object=None, object_signature=None, function_args=[eval("'saved_model/my_model'")], function_kwargs={})
custom_method(
new_model.summary(), imports='import tensorflow as tf;from tensorflow import keras;import os', function_to_run='obj.summary()', method_object=eval('new_model'), object_signature=None, function_args=[], function_kwargs={}, custom_class=None)
(loss, acc) = custom_method(
new_model.evaluate(test_images, test_labels, verbose=2), imports='import tensorflow as tf;from tensorflow import keras;import os', function_to_run='obj.evaluate(*args, **kwargs)', method_object=eval('new_model'), object_signature=None, function_args=[eval('test_images'), eval('test_labels')], function_kwargs={'verbose': eval('2')}, custom_class=None)
print('Restored model, accuracy: {:5.2f}%'.format(100 * acc))
print(new_model.predict(test_images).shape)
model = create_model()
custom_method(
model.fit(train_images, train_labels, epochs=5), imports='import tensorflow as tf;from tensorflow import keras;import os', function_to_run='obj.fit(*args, **kwargs)', method_object=eval('model'), object_signature=None, function_args=[eval('train_images'), eval('train_labels')], function_kwargs={'epochs': eval('5')}, custom_class=None)
custom_method(
model.save('my_model.h5'), imports='import tensorflow as tf;from tensorflow import keras;import os', function_to_run='obj.save(*args)', method_object=eval('model'), object_signature=None, function_args=[eval("'my_model.h5'")], function_kwargs={}, custom_class=None)
new_model = custom_method(
tf.keras.models.load_model('my_model.h5'), imports='import tensorflow as tf;from tensorflow import keras;import os', function_to_run='tf.keras.models.load_model(*args)', method_object=None, object_signature=None, function_args=[eval("'my_model.h5'")], function_kwargs={})
custom_method(
new_model.summary(), imports='import tensorflow as tf;from tensorflow import keras;import os', function_to_run='obj.summary()', method_object=eval('new_model'), object_signature=None, function_args=[], function_kwargs={}, custom_class=None)
(loss, acc) = custom_method(
new_model.evaluate(test_images, test_labels, verbose=2), imports='import tensorflow as tf;from tensorflow import keras;import os', function_to_run='obj.evaluate(*args, **kwargs)', method_object=eval('new_model'), object_signature=None, function_args=[eval('test_images'), eval('test_labels')], function_kwargs={'verbose': eval('2')}, custom_class=None)
print('Restored model, accuracy: {:5.2f}%'.format(100 * acc))
