import os
import tensorflow as tf
from tensorflow import keras
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
print(tf.version.VERSION)
custom_method(imports='import os;import tensorflow as tf;from tensorflow import keras', function_to_run='tf.keras.datasets.mnist.load_data()', method_object=None, object_signature=None, function_args=[], function_kwargs={})
((train_images, train_labels), (test_images, test_labels)) = tf.keras.datasets.mnist.load_data()
train_labels = train_labels[:1000]
test_labels = test_labels[:1000]
train_images = train_images[:1000].reshape(-1, 28 * 28) / 255.0
test_images = test_images[:1000].reshape(-1, 28 * 28) / 255.0

def create_model():
    custom_method(imports='import os;import tensorflow as tf;from tensorflow import keras', function_to_run='tf.keras.Sequential(*args)', method_object=None, object_signature=None, function_args=[eval("[\n    keras.layers.Dense(512, activation='relu', input_shape=(784,)),\n    keras.layers.Dropout(0.2),\n    keras.layers.Dense(10)\n  ]")], function_kwargs={})
    model = tf.keras.Sequential([keras.layers.Dense(512, activation='relu', input_shape=(784,)), keras.layers.Dropout(0.2), keras.layers.Dense(10)])
    custom_method(imports='import os;import tensorflow as tf;from tensorflow import keras', function_to_run='obj.compile(**kwargs)', method_object=eval('model'), object_signature=None, function_args=[], function_kwargs={'optimizer': eval("'adam'"), 'loss': eval('tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)'), 'metrics': eval('[tf.keras.metrics.SparseCategoricalAccuracy()]')}, custom_class=None)
    model.compile(optimizer='adam', loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=[tf.keras.metrics.SparseCategoricalAccuracy()])
    return model
model = create_model()
custom_method(imports='import os;import tensorflow as tf;from tensorflow import keras', function_to_run='obj.summary()', method_object=eval('model'), object_signature=None, function_args=[], function_kwargs={}, custom_class=None)
model.summary()
checkpoint_path = 'training_1/cp.ckpt'
checkpoint_dir = os.path.dirname(checkpoint_path)
custom_method(imports='import os;import tensorflow as tf;from tensorflow import keras', function_to_run='tf.keras.callbacks.ModelCheckpoint(**kwargs)', method_object=None, object_signature=None, function_args=[], function_kwargs={'filepath': eval('checkpoint_path'), 'save_weights_only': eval('True'), 'verbose': eval('1')})
cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path, save_weights_only=True, verbose=1)
custom_method(imports='import os;import tensorflow as tf;from tensorflow import keras', function_to_run='obj.fit(*args, **kwargs)', method_object=eval('model'), object_signature=None, function_args=[eval('train_images'), eval('train_labels')], function_kwargs={'epochs': eval('10'), 'validation_data': eval('(test_images, test_labels)'), 'callbacks': eval('[cp_callback]')}, custom_class=None)
model.fit(train_images, train_labels, epochs=10, validation_data=(test_images, test_labels), callbacks=[cp_callback])
os.listdir(checkpoint_dir)
model = create_model()
custom_method(imports='import os;import tensorflow as tf;from tensorflow import keras', function_to_run='obj.evaluate(*args, **kwargs)', method_object=eval('model'), object_signature=None, function_args=[eval('test_images'), eval('test_labels')], function_kwargs={'verbose': eval('2')}, custom_class=None)
(loss, acc) = model.evaluate(test_images, test_labels, verbose=2)
print('Untrained model, accuracy: {:5.2f}%'.format(100 * acc))
custom_method(imports='import os;import tensorflow as tf;from tensorflow import keras', function_to_run='obj.load_weights(*args)', method_object=eval('model'), object_signature=None, function_args=[eval('checkpoint_path')], function_kwargs={}, custom_class=None)
model.load_weights(checkpoint_path)
custom_method(imports='import os;import tensorflow as tf;from tensorflow import keras', function_to_run='obj.evaluate(*args, **kwargs)', method_object=eval('model'), object_signature=None, function_args=[eval('test_images'), eval('test_labels')], function_kwargs={'verbose': eval('2')}, custom_class=None)
(loss, acc) = model.evaluate(test_images, test_labels, verbose=2)
print('Restored model, accuracy: {:5.2f}%'.format(100 * acc))
checkpoint_path = 'training_2/cp-{epoch:04d}.ckpt'
checkpoint_dir = os.path.dirname(checkpoint_path)
batch_size = 32
custom_method(imports='import os;import tensorflow as tf;from tensorflow import keras', function_to_run='tf.keras.callbacks.ModelCheckpoint(**kwargs)', method_object=None, object_signature=None, function_args=[], function_kwargs={'filepath': eval('checkpoint_path'), 'verbose': eval('1'), 'save_weights_only': eval('True'), 'save_freq': eval('5*batch_size')})
cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path, verbose=1, save_weights_only=True, save_freq=5 * batch_size)
model = create_model()
custom_method(imports='import os;import tensorflow as tf;from tensorflow import keras', function_to_run='obj.save_weights(*args)', method_object=eval('model'), object_signature=None, function_args=[eval('checkpoint_path.format(epoch=0)')], function_kwargs={}, custom_class=None)
model.save_weights(checkpoint_path.format(epoch=0))
custom_method(imports='import os;import tensorflow as tf;from tensorflow import keras', function_to_run='obj.fit(*args, **kwargs)', method_object=eval('model'), object_signature=None, function_args=[eval('train_images'), eval('train_labels')], function_kwargs={'epochs': eval('50'), 'batch_size': eval('batch_size'), 'callbacks': eval('[cp_callback]'), 'validation_data': eval('(test_images, test_labels)'), 'verbose': eval('0')}, custom_class=None)
model.fit(train_images, train_labels, epochs=50, batch_size=batch_size, callbacks=[cp_callback], validation_data=(test_images, test_labels), verbose=0)
os.listdir(checkpoint_dir)
custom_method(imports='import os;import tensorflow as tf;from tensorflow import keras', function_to_run='tf.train.latest_checkpoint(*args)', method_object=None, object_signature=None, function_args=[eval('checkpoint_dir')], function_kwargs={})
latest = tf.train.latest_checkpoint(checkpoint_dir)
latest
model = create_model()
custom_method(imports='import os;import tensorflow as tf;from tensorflow import keras', function_to_run='obj.load_weights(*args)', method_object=eval('model'), object_signature=None, function_args=[eval('latest')], function_kwargs={}, custom_class=None)
model.load_weights(latest)
custom_method(imports='import os;import tensorflow as tf;from tensorflow import keras', function_to_run='obj.evaluate(*args, **kwargs)', method_object=eval('model'), object_signature=None, function_args=[eval('test_images'), eval('test_labels')], function_kwargs={'verbose': eval('2')}, custom_class=None)
(loss, acc) = model.evaluate(test_images, test_labels, verbose=2)
print('Restored model, accuracy: {:5.2f}%'.format(100 * acc))
custom_method(imports='import os;import tensorflow as tf;from tensorflow import keras', function_to_run='obj.save_weights(*args)', method_object=eval('model'), object_signature=None, function_args=[eval("'./checkpoints/my_checkpoint'")], function_kwargs={}, custom_class=None)
model.save_weights('./checkpoints/my_checkpoint')
model = create_model()
custom_method(imports='import os;import tensorflow as tf;from tensorflow import keras', function_to_run='obj.load_weights(*args)', method_object=eval('model'), object_signature=None, function_args=[eval("'./checkpoints/my_checkpoint'")], function_kwargs={}, custom_class=None)
model.load_weights('./checkpoints/my_checkpoint')
custom_method(imports='import os;import tensorflow as tf;from tensorflow import keras', function_to_run='obj.evaluate(*args, **kwargs)', method_object=eval('model'), object_signature=None, function_args=[eval('test_images'), eval('test_labels')], function_kwargs={'verbose': eval('2')}, custom_class=None)
(loss, acc) = model.evaluate(test_images, test_labels, verbose=2)
print('Restored model, accuracy: {:5.2f}%'.format(100 * acc))
model = create_model()
custom_method(imports='import os;import tensorflow as tf;from tensorflow import keras', function_to_run='obj.fit(*args, **kwargs)', method_object=eval('model'), object_signature=None, function_args=[eval('train_images'), eval('train_labels')], function_kwargs={'epochs': eval('5')}, custom_class=None)
model.fit(train_images, train_labels, epochs=5)
custom_method(imports='import os;import tensorflow as tf;from tensorflow import keras', function_to_run='obj.save(*args)', method_object=eval('model'), object_signature=None, function_args=[eval("'saved_model/my_model'")], function_kwargs={}, custom_class=None)
model.save('saved_model/my_model')
custom_method(imports='import os;import tensorflow as tf;from tensorflow import keras', function_to_run='tf.keras.models.load_model(*args)', method_object=None, object_signature=None, function_args=[eval("'saved_model/my_model'")], function_kwargs={})
new_model = tf.keras.models.load_model('saved_model/my_model')
custom_method(imports='import os;import tensorflow as tf;from tensorflow import keras', function_to_run='obj.summary()', method_object=eval('new_model'), object_signature=None, function_args=[], function_kwargs={}, custom_class=None)
new_model.summary()
custom_method(imports='import os;import tensorflow as tf;from tensorflow import keras', function_to_run='obj.evaluate(*args, **kwargs)', method_object=eval('new_model'), object_signature=None, function_args=[eval('test_images'), eval('test_labels')], function_kwargs={'verbose': eval('2')}, custom_class=None)
(loss, acc) = new_model.evaluate(test_images, test_labels, verbose=2)
print('Restored model, accuracy: {:5.2f}%'.format(100 * acc))
print(new_model.predict(test_images).shape)
model = create_model()
custom_method(imports='import os;import tensorflow as tf;from tensorflow import keras', function_to_run='obj.fit(*args, **kwargs)', method_object=eval('model'), object_signature=None, function_args=[eval('train_images'), eval('train_labels')], function_kwargs={'epochs': eval('5')}, custom_class=None)
model.fit(train_images, train_labels, epochs=5)
custom_method(imports='import os;import tensorflow as tf;from tensorflow import keras', function_to_run='obj.save(*args)', method_object=eval('model'), object_signature=None, function_args=[eval("'my_model.h5'")], function_kwargs={}, custom_class=None)
model.save('my_model.h5')
custom_method(imports='import os;import tensorflow as tf;from tensorflow import keras', function_to_run='tf.keras.models.load_model(*args)', method_object=None, object_signature=None, function_args=[eval("'my_model.h5'")], function_kwargs={})
new_model = tf.keras.models.load_model('my_model.h5')
custom_method(imports='import os;import tensorflow as tf;from tensorflow import keras', function_to_run='obj.summary()', method_object=eval('new_model'), object_signature=None, function_args=[], function_kwargs={}, custom_class=None)
new_model.summary()
custom_method(imports='import os;import tensorflow as tf;from tensorflow import keras', function_to_run='obj.evaluate(*args, **kwargs)', method_object=eval('new_model'), object_signature=None, function_args=[eval('test_images'), eval('test_labels')], function_kwargs={'verbose': eval('2')}, custom_class=None)
(loss, acc) = new_model.evaluate(test_images, test_labels, verbose=2)
print('Restored model, accuracy: {:5.2f}%'.format(100 * acc))
