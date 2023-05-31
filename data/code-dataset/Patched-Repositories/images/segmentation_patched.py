import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow_examples.models.pix2pix import pix2pix
from IPython.display import clear_output
import matplotlib.pyplot as plt
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
(dataset, info) = tfds.load('oxford_iiit_pet:3.*.*', with_info=True)

def normalize(input_image, input_mask):
    input_image = tf.cast(input_image, tf.float32) / 255.0
    input_mask -= 1
    return (input_image, input_mask)

def load_image(datapoint):
    custom_method(imports='import tensorflow_datasets as tfds;from IPython.display import clear_output;import tensorflow as tf;import matplotlib.pyplot as plt;from tensorflow_examples.models.pix2pix import pix2pix', function_to_run='tf.image.resize(*args)', method_object=None, object_signature=None, function_args=[eval("datapoint['image']"), eval('(128, 128)')], function_kwargs={})
    input_image = tf.image.resize(datapoint['image'], (128, 128))
    custom_method(imports='import tensorflow_datasets as tfds;from IPython.display import clear_output;import tensorflow as tf;import matplotlib.pyplot as plt;from tensorflow_examples.models.pix2pix import pix2pix', function_to_run='tf.image.resize(*args, **kwargs)', method_object=None, object_signature=None, function_args=[eval("datapoint['segmentation_mask']"), eval('(128, 128)')], function_kwargs={'method': eval('tf.image.ResizeMethod.NEAREST_NEIGHBOR')})
    input_mask = tf.image.resize(datapoint['segmentation_mask'], (128, 128), method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    (input_image, input_mask) = normalize(input_image, input_mask)
    return (input_image, input_mask)
TRAIN_LENGTH = info.splits['train'].num_examples
BATCH_SIZE = 64
BUFFER_SIZE = 1000
STEPS_PER_EPOCH = TRAIN_LENGTH // BATCH_SIZE
train_images = dataset['train'].map(load_image, num_parallel_calls=tf.data.AUTOTUNE)
test_images = dataset['test'].map(load_image, num_parallel_calls=tf.data.AUTOTUNE)

class Augment(tf.keras.layers.Layer):

    def __init__(self, seed=42):
        super().__init__()
        custom_method(imports='import tensorflow_datasets as tfds;from IPython.display import clear_output;import tensorflow as tf;import matplotlib.pyplot as plt;from tensorflow_examples.models.pix2pix import pix2pix', function_to_run='tf.keras.layers.RandomFlip(**kwargs)', method_object=None, object_signature=None, function_args=[], function_kwargs={'mode': eval('"horizontal"'), 'seed': eval('seed')})
        self.augment_inputs = tf.keras.layers.RandomFlip(mode='horizontal', seed=seed)
        custom_method(imports='import tensorflow_datasets as tfds;from IPython.display import clear_output;import tensorflow as tf;import matplotlib.pyplot as plt;from tensorflow_examples.models.pix2pix import pix2pix', function_to_run='tf.keras.layers.RandomFlip(**kwargs)', method_object=None, object_signature=None, function_args=[], function_kwargs={'mode': eval('"horizontal"'), 'seed': eval('seed')})
        self.augment_labels = tf.keras.layers.RandomFlip(mode='horizontal', seed=seed)

    def call(self, inputs, labels):
        inputs = self.augment_inputs(inputs)
        labels = self.augment_labels(labels)
        return (inputs, labels)
train_batches = train_images.cache().shuffle(BUFFER_SIZE).batch(BATCH_SIZE).repeat().map(Augment()).prefetch(buffer_size=tf.data.AUTOTUNE)
test_batches = test_images.batch(BATCH_SIZE)

def display(display_list):
    plt.figure(figsize=(15, 15))
    title = ['Input Image', 'True Mask', 'Predicted Mask']
    for i in range(len(display_list)):
        plt.subplot(1, len(display_list), i + 1)
        plt.title(title[i])
        plt.imshow(tf.keras.utils.array_to_img(display_list[i]))
        plt.axis('off')
    plt.show()
for (images, masks) in train_batches.take(2):
    (sample_image, sample_mask) = (images[0], masks[0])
    display([sample_image, sample_mask])
custom_method(imports='import tensorflow_datasets as tfds;from IPython.display import clear_output;import tensorflow as tf;import matplotlib.pyplot as plt;from tensorflow_examples.models.pix2pix import pix2pix', function_to_run='tf.keras.applications.MobileNetV2(**kwargs)', method_object=None, object_signature=None, function_args=[], function_kwargs={'input_shape': eval('[128, 128, 3]'), 'include_top': eval('False')})
base_model = tf.keras.applications.MobileNetV2(input_shape=[128, 128, 3], include_top=False)
layer_names = ['block_1_expand_relu', 'block_3_expand_relu', 'block_6_expand_relu', 'block_13_expand_relu', 'block_16_project']
base_model_outputs = [base_model.get_layer(name).output for name in layer_names]
custom_method(imports='import tensorflow_datasets as tfds;from IPython.display import clear_output;import tensorflow as tf;import matplotlib.pyplot as plt;from tensorflow_examples.models.pix2pix import pix2pix', function_to_run='tf.keras.Model(**kwargs)', method_object=None, object_signature=None, function_args=[], function_kwargs={'inputs': eval('base_model.input'), 'outputs': eval('base_model_outputs')})
down_stack = tf.keras.Model(inputs=base_model.input, outputs=base_model_outputs)
down_stack.trainable = False
up_stack = [pix2pix.upsample(512, 3), pix2pix.upsample(256, 3), pix2pix.upsample(128, 3), pix2pix.upsample(64, 3)]

def unet_model(output_channels: int):
    custom_method(imports='import tensorflow_datasets as tfds;from IPython.display import clear_output;import tensorflow as tf;import matplotlib.pyplot as plt;from tensorflow_examples.models.pix2pix import pix2pix', function_to_run='tf.keras.layers.Input(**kwargs)', method_object=None, object_signature=None, function_args=[], function_kwargs={'shape': eval('[128, 128, 3]')})
    inputs = tf.keras.layers.Input(shape=[128, 128, 3])
    custom_method(imports='import tensorflow_datasets as tfds;from IPython.display import clear_output;import tensorflow as tf;import matplotlib.pyplot as plt;from tensorflow_examples.models.pix2pix import pix2pix', function_to_run='obj(*args)', method_object=eval('down_stack'), object_signature=None, function_args=[eval('inputs')], function_kwargs={}, custom_class=None)
    skips = down_stack(inputs)
    x = skips[-1]
    skips = reversed(skips[:-1])
    for (up, skip) in zip(up_stack, skips):
        x = up(x)
        custom_method(imports='import tensorflow_datasets as tfds;from IPython.display import clear_output;import tensorflow as tf;import matplotlib.pyplot as plt;from tensorflow_examples.models.pix2pix import pix2pix', function_to_run='tf.keras.layers.Concatenate()', method_object=None, object_signature=None, function_args=[], function_kwargs={})
        concat = tf.keras.layers.Concatenate()
        custom_method(imports='import tensorflow_datasets as tfds;from IPython.display import clear_output;import tensorflow as tf;import matplotlib.pyplot as plt;from tensorflow_examples.models.pix2pix import pix2pix', function_to_run='obj(*args)', method_object=eval('concat'), object_signature=None, function_args=[eval('[x, skip]')], function_kwargs={}, custom_class=None)
        x = concat([x, skip])
    custom_method(imports='import tensorflow_datasets as tfds;from IPython.display import clear_output;import tensorflow as tf;import matplotlib.pyplot as plt;from tensorflow_examples.models.pix2pix import pix2pix', function_to_run='tf.keras.layers.Conv2DTranspose(**kwargs)', method_object=None, object_signature=None, function_args=[], function_kwargs={'filters': eval('output_channels'), 'kernel_size': eval('3'), 'strides': eval('2'), 'padding': eval("'same'")})
    last = tf.keras.layers.Conv2DTranspose(filters=output_channels, kernel_size=3, strides=2, padding='same')
    custom_method(imports='import tensorflow_datasets as tfds;from IPython.display import clear_output;import tensorflow as tf;import matplotlib.pyplot as plt;from tensorflow_examples.models.pix2pix import pix2pix', function_to_run='obj(*args)', method_object=eval('last'), object_signature=None, function_args=[eval('x')], function_kwargs={}, custom_class=None)
    x = last(x)
    return tf.keras.Model(inputs=inputs, outputs=x)
OUTPUT_CLASSES = 3
model = unet_model(output_channels=OUTPUT_CLASSES)
model.compile(optimizer='adam', loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=['accuracy'])
custom_method(imports='import tensorflow_datasets as tfds;from IPython.display import clear_output;import tensorflow as tf;import matplotlib.pyplot as plt;from tensorflow_examples.models.pix2pix import pix2pix', function_to_run='tf.keras.utils.plot_model(*args, **kwargs)', method_object=None, object_signature=None, function_args=[eval('model')], function_kwargs={'show_shapes': eval('True')})
tf.keras.utils.plot_model(model, show_shapes=True)

def create_mask(pred_mask):
    custom_method(imports='import tensorflow_datasets as tfds;from IPython.display import clear_output;import tensorflow as tf;import matplotlib.pyplot as plt;from tensorflow_examples.models.pix2pix import pix2pix', function_to_run='tf.math.argmax(*args, **kwargs)', method_object=None, object_signature=None, function_args=[eval('pred_mask')], function_kwargs={'axis': eval('-1')})
    pred_mask = tf.math.argmax(pred_mask, axis=-1)
    pred_mask = pred_mask[..., tf.newaxis]
    return pred_mask[0]

def show_predictions(dataset=None, num=1):
    if dataset:
        for (image, mask) in dataset.take(num):
            pred_mask = model.predict(image)
            display([image[0], mask[0], create_mask(pred_mask)])
    else:
        display([sample_image, sample_mask, create_mask(model.predict(sample_image[tf.newaxis, ...]))])
show_predictions()

class DisplayCallback(tf.keras.callbacks.Callback):

    def on_epoch_end(self, epoch, logs=None):
        clear_output(wait=True)
        show_predictions()
        print('\nSample Prediction after epoch {}\n'.format(epoch + 1))
EPOCHS = 20
VAL_SUBSPLITS = 5
VALIDATION_STEPS = info.splits['test'].num_examples // BATCH_SIZE // VAL_SUBSPLITS
model_history = model.fit(train_batches, epochs=EPOCHS, steps_per_epoch=STEPS_PER_EPOCH, validation_steps=VALIDATION_STEPS, validation_data=test_batches, callbacks=[DisplayCallback()])
loss = model_history.history['loss']
val_loss = model_history.history['val_loss']
plt.figure()
plt.plot(model_history.epoch, loss, 'r', label='Training loss')
plt.plot(model_history.epoch, val_loss, 'bo', label='Validation loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss Value')
plt.ylim([0, 1])
plt.legend()
plt.show()
show_predictions(test_batches, 3)
try:
    model_history = model.fit(train_batches, epochs=EPOCHS, steps_per_epoch=STEPS_PER_EPOCH, class_weight={0: 2.0, 1: 2.0, 2: 1.0})
    assert False
except Exception as e:
    print(f'Expected {type(e).__name__}: {e}')
label = [0, 0]
prediction = [[-3.0, 0], [-3, 0]]
sample_weight = [1, 10]
custom_method(imports='import tensorflow_datasets as tfds;from IPython.display import clear_output;import tensorflow as tf;import matplotlib.pyplot as plt;from tensorflow_examples.models.pix2pix import pix2pix', function_to_run='tf.keras.losses.SparseCategoricalCrossentropy(**kwargs)', method_object=None, object_signature=None, function_args=[], function_kwargs={'from_logits': eval('True'), 'reduction': eval('tf.keras.losses.Reduction.NONE')})
loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction=tf.keras.losses.Reduction.NONE)
custom_method(imports='import tensorflow_datasets as tfds;from IPython.display import clear_output;import tensorflow as tf;import matplotlib.pyplot as plt;from tensorflow_examples.models.pix2pix import pix2pix', function_to_run='obj(label, prediction, sample_weight).numpy()', method_object=eval('loss'), object_signature=None, function_args=[], function_kwargs={}, custom_class=None)
loss(label, prediction, sample_weight).numpy()

def add_sample_weights(image, label):
    custom_method(imports='import tensorflow_datasets as tfds;from IPython.display import clear_output;import tensorflow as tf;import matplotlib.pyplot as plt;from tensorflow_examples.models.pix2pix import pix2pix', function_to_run='tf.constant(*args)', method_object=None, object_signature=None, function_args=[eval('[2.0, 2.0, 1.0]')], function_kwargs={})
    class_weights = tf.constant([2.0, 2.0, 1.0])
    class_weights = class_weights / tf.reduce_sum(class_weights)
    custom_method(imports='import tensorflow_datasets as tfds;from IPython.display import clear_output;import tensorflow as tf;import matplotlib.pyplot as plt;from tensorflow_examples.models.pix2pix import pix2pix', function_to_run='tf.gather(*args, **kwargs)', method_object=None, object_signature=None, function_args=[eval('class_weights')], function_kwargs={'indices': eval('tf.cast(label, tf.int32)')})
    sample_weights = tf.gather(class_weights, indices=tf.cast(label, tf.int32))
    return (image, label, sample_weights)
train_batches.map(add_sample_weights).element_spec
weighted_model = unet_model(OUTPUT_CLASSES)
weighted_model.compile(optimizer='adam', loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=['accuracy'])
weighted_model.fit(train_batches.map(add_sample_weights), epochs=1, steps_per_epoch=10)
