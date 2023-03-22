import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow_examples.models.pix2pix import pix2pix
from IPython.display import clear_output
import matplotlib.pyplot as plt
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
(dataset, info) = custom_method(
tfds.load('oxford_iiit_pet:3.*.*', with_info=True), imports='import tensorflow as tf;import matplotlib.pyplot as plt;from IPython.display import clear_output;from tensorflow_examples.models.pix2pix import pix2pix;import tensorflow_datasets as tfds', function_to_run='tfds.load(*args, **kwargs)', method_object=None, object_signature=None, function_args=[eval("'oxford_iiit_pet:3.*.*'")], function_kwargs={'with_info': eval('True')}, max_wait_secs=0)

def normalize(input_image, input_mask):
    input_image = custom_method(
    tf.cast(input_image, tf.float32), imports='import tensorflow as tf;import matplotlib.pyplot as plt;from IPython.display import clear_output;from tensorflow_examples.models.pix2pix import pix2pix;import tensorflow_datasets as tfds', function_to_run='tf.cast(*args)', method_object=None, object_signature=None, function_args=[eval('input_image'), eval('tf.float32')], function_kwargs={}, max_wait_secs=0) / 255.0
    input_mask -= 1
    return (input_image, input_mask)

def load_image(datapoint):
    input_image = custom_method(
    tf.image.resize(datapoint['image'], (128, 128)), imports='import tensorflow as tf;import matplotlib.pyplot as plt;from IPython.display import clear_output;from tensorflow_examples.models.pix2pix import pix2pix;import tensorflow_datasets as tfds', function_to_run='tf.image.resize(*args)', method_object=None, object_signature=None, function_args=[eval("datapoint['image']"), eval('(128, 128)')], function_kwargs={}, max_wait_secs=0)
    input_mask = custom_method(
    tf.image.resize(datapoint['segmentation_mask'], (128, 128), method=tf.image.ResizeMethod.NEAREST_NEIGHBOR), imports='import tensorflow as tf;import matplotlib.pyplot as plt;from IPython.display import clear_output;from tensorflow_examples.models.pix2pix import pix2pix;import tensorflow_datasets as tfds', function_to_run='tf.image.resize(*args, **kwargs)', method_object=None, object_signature=None, function_args=[eval("datapoint['segmentation_mask']"), eval('(128, 128)')], function_kwargs={'method': eval('tf.image.ResizeMethod.NEAREST_NEIGHBOR')}, max_wait_secs=0)
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
        self.augment_inputs = custom_method(
        tf.keras.layers.RandomFlip(mode='horizontal', seed=seed), imports='import tensorflow as tf;import matplotlib.pyplot as plt;from IPython.display import clear_output;from tensorflow_examples.models.pix2pix import pix2pix;import tensorflow_datasets as tfds', function_to_run='tf.keras.layers.RandomFlip(**kwargs)', method_object=None, object_signature=None, function_args=[], function_kwargs={'mode': eval('"horizontal"'), 'seed': eval('seed')}, max_wait_secs=0)
        self.augment_labels = custom_method(
        tf.keras.layers.RandomFlip(mode='horizontal', seed=seed), imports='import tensorflow as tf;import matplotlib.pyplot as plt;from IPython.display import clear_output;from tensorflow_examples.models.pix2pix import pix2pix;import tensorflow_datasets as tfds', function_to_run='tf.keras.layers.RandomFlip(**kwargs)', method_object=None, object_signature=None, function_args=[], function_kwargs={'mode': eval('"horizontal"'), 'seed': eval('seed')}, max_wait_secs=0)

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
for (images, masks) in custom_method(
train_batches.take(2), imports='import tensorflow as tf;import matplotlib.pyplot as plt;from IPython.display import clear_output;from tensorflow_examples.models.pix2pix import pix2pix;import tensorflow_datasets as tfds', function_to_run='obj.take(*args)', method_object=eval('train_batches'), object_signature='train_images\n    .cache()\n    .shuffle(BUFFER_SIZE)\n    .batch(BATCH_SIZE)\n    .repeat()\n    .map(Augment())\n    .prefetch', function_args=[eval('2')], function_kwargs={}, max_wait_secs=0, custom_class='class Augment(tf.keras.layers.Layer):\n  def __init__(self, seed=42):\n    super().__init__()\n    self.augment_inputs = tf.keras.layers.RandomFlip(mode="horizontal", seed=seed)\n    self.augment_labels = tf.keras.layers.RandomFlip(mode="horizontal", seed=seed)\n  \n  def call(self, inputs, labels):\n    inputs = self.augment_inputs(inputs)\n    labels = self.augment_labels(labels)\n    return inputs, labels'):
    (sample_image, sample_mask) = (images[0], masks[0])
    display([sample_image, sample_mask])
base_model = custom_method(
tf.keras.applications.MobileNetV2(input_shape=[128, 128, 3], include_top=False), imports='import tensorflow as tf;import matplotlib.pyplot as plt;from IPython.display import clear_output;from tensorflow_examples.models.pix2pix import pix2pix;import tensorflow_datasets as tfds', function_to_run='tf.keras.applications.MobileNetV2(**kwargs)', method_object=None, object_signature=None, function_args=[], function_kwargs={'input_shape': eval('[128, 128, 3]'), 'include_top': eval('False')}, max_wait_secs=0)
layer_names = ['block_1_expand_relu', 'block_3_expand_relu', 'block_6_expand_relu', 'block_13_expand_relu', 'block_16_project']
base_model_outputs = [custom_method(
base_model.get_layer(name), imports='import tensorflow as tf;import matplotlib.pyplot as plt;from IPython.display import clear_output;from tensorflow_examples.models.pix2pix import pix2pix;import tensorflow_datasets as tfds', function_to_run='obj.get_layer(*args)', method_object=eval('base_model'), object_signature='tf.keras.applications.MobileNetV2', function_args=[eval('name')], function_kwargs={}, max_wait_secs=0, custom_class=None).output for name in layer_names]
down_stack = custom_method(
tf.keras.Model(inputs=base_model.input, outputs=base_model_outputs), imports='import tensorflow as tf;import matplotlib.pyplot as plt;from IPython.display import clear_output;from tensorflow_examples.models.pix2pix import pix2pix;import tensorflow_datasets as tfds', function_to_run='tf.keras.Model(**kwargs)', method_object=None, object_signature=None, function_args=[], function_kwargs={'inputs': eval('base_model.input'), 'outputs': eval('base_model_outputs')}, max_wait_secs=0)
down_stack.trainable = False
up_stack = [pix2pix.upsample(512, 3), pix2pix.upsample(256, 3), pix2pix.upsample(128, 3), pix2pix.upsample(64, 3)]

def unet_model(output_channels: int):
    inputs = custom_method(
    tf.keras.layers.Input(shape=[128, 128, 3]), imports='import tensorflow as tf;import matplotlib.pyplot as plt;from IPython.display import clear_output;from tensorflow_examples.models.pix2pix import pix2pix;import tensorflow_datasets as tfds', function_to_run='tf.keras.layers.Input(**kwargs)', method_object=None, object_signature=None, function_args=[], function_kwargs={'shape': eval('[128, 128, 3]')}, max_wait_secs=0)
    skips = custom_method(
    down_stack(inputs), imports='import tensorflow as tf;import matplotlib.pyplot as plt;from IPython.display import clear_output;from tensorflow_examples.models.pix2pix import pix2pix;import tensorflow_datasets as tfds', function_to_run='obj(*args)', method_object=eval('down_stack'), object_signature='tf.keras.Model', function_args=[eval('inputs')], function_kwargs={}, max_wait_secs=0, custom_class=None)
    x = skips[-1]
    skips = reversed(skips[:-1])
    for (up, skip) in zip(up_stack, skips):
        x = up(x)
        concat = custom_method(
        tf.keras.layers.Concatenate(), imports='import tensorflow as tf;import matplotlib.pyplot as plt;from IPython.display import clear_output;from tensorflow_examples.models.pix2pix import pix2pix;import tensorflow_datasets as tfds', function_to_run='tf.keras.layers.Concatenate()', method_object=None, object_signature=None, function_args=[], function_kwargs={}, max_wait_secs=0)
        x = custom_method(
        concat([x, skip]), imports='import tensorflow as tf;import matplotlib.pyplot as plt;from IPython.display import clear_output;from tensorflow_examples.models.pix2pix import pix2pix;import tensorflow_datasets as tfds', function_to_run='obj(*args)', method_object=eval('concat'), object_signature='tf.keras.layers.Concatenate', function_args=[eval('[x, skip]')], function_kwargs={}, max_wait_secs=0, custom_class=None)
    last = custom_method(
    tf.keras.layers.Conv2DTranspose(filters=output_channels, kernel_size=3, strides=2, padding='same'), imports='import tensorflow as tf;import matplotlib.pyplot as plt;from IPython.display import clear_output;from tensorflow_examples.models.pix2pix import pix2pix;import tensorflow_datasets as tfds', function_to_run='tf.keras.layers.Conv2DTranspose(**kwargs)', method_object=None, object_signature=None, function_args=[], function_kwargs={'filters': eval('output_channels'), 'kernel_size': eval('3'), 'strides': eval('2'), 'padding': eval("'same'")}, max_wait_secs=0)
    x = custom_method(
    last(x), imports='import tensorflow as tf;import matplotlib.pyplot as plt;from IPython.display import clear_output;from tensorflow_examples.models.pix2pix import pix2pix;import tensorflow_datasets as tfds', function_to_run='obj(*args)', method_object=eval('last'), object_signature='tf.keras.layers.Conv2DTranspose', function_args=[eval('x')], function_kwargs={}, max_wait_secs=0, custom_class=None)
    return custom_method(
    tf.keras.Model(inputs=inputs, outputs=x), imports='import tensorflow as tf;import matplotlib.pyplot as plt;from IPython.display import clear_output;from tensorflow_examples.models.pix2pix import pix2pix;import tensorflow_datasets as tfds', function_to_run='tf.keras.Model(**kwargs)', method_object=None, object_signature=None, function_args=[], function_kwargs={'inputs': eval('inputs'), 'outputs': eval('x')}, max_wait_secs=0)
OUTPUT_CLASSES = 3
model = unet_model(output_channels=OUTPUT_CLASSES)
model.compile(optimizer='adam', loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=['accuracy'])
custom_method(
tf.keras.utils.plot_model(model, show_shapes=True), imports='import tensorflow as tf;import matplotlib.pyplot as plt;from IPython.display import clear_output;from tensorflow_examples.models.pix2pix import pix2pix;import tensorflow_datasets as tfds', function_to_run='tf.keras.utils.plot_model(*args, **kwargs)', method_object=None, object_signature=None, function_args=[eval('model')], function_kwargs={'show_shapes': eval('True')}, max_wait_secs=0)

def create_mask(pred_mask):
    pred_mask = custom_method(
    tf.math.argmax(pred_mask, axis=-1), imports='import tensorflow as tf;import matplotlib.pyplot as plt;from IPython.display import clear_output;from tensorflow_examples.models.pix2pix import pix2pix;import tensorflow_datasets as tfds', function_to_run='tf.math.argmax(*args, **kwargs)', method_object=None, object_signature=None, function_args=[eval('pred_mask')], function_kwargs={'axis': eval('-1')}, max_wait_secs=0)
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
loss = custom_method(
tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction=tf.keras.losses.Reduction.NONE), imports='import tensorflow as tf;import matplotlib.pyplot as plt;from IPython.display import clear_output;from tensorflow_examples.models.pix2pix import pix2pix;import tensorflow_datasets as tfds', function_to_run='tf.keras.losses.SparseCategoricalCrossentropy(**kwargs)', method_object=None, object_signature=None, function_args=[], function_kwargs={'from_logits': eval('True'), 'reduction': eval('tf.keras.losses.Reduction.NONE')}, max_wait_secs=0)
loss(label, prediction, sample_weight).numpy()

def add_sample_weights(image, label):
    class_weights = custom_method(
    tf.constant([2.0, 2.0, 1.0]), imports='import tensorflow as tf;import matplotlib.pyplot as plt;from IPython.display import clear_output;from tensorflow_examples.models.pix2pix import pix2pix;import tensorflow_datasets as tfds', function_to_run='tf.constant(*args)', method_object=None, object_signature=None, function_args=[eval('[2.0, 2.0, 1.0]')], function_kwargs={}, max_wait_secs=0)
    class_weights = class_weights / custom_method(
    tf.reduce_sum(class_weights), imports='import tensorflow as tf;import matplotlib.pyplot as plt;from IPython.display import clear_output;from tensorflow_examples.models.pix2pix import pix2pix;import tensorflow_datasets as tfds', function_to_run='tf.reduce_sum(*args)', method_object=None, object_signature=None, function_args=[eval('class_weights')], function_kwargs={}, max_wait_secs=0)
    sample_weights = custom_method(
    tf.gather(class_weights, indices=tf.cast(label, tf.int32)), imports='import tensorflow as tf;import matplotlib.pyplot as plt;from IPython.display import clear_output;from tensorflow_examples.models.pix2pix import pix2pix;import tensorflow_datasets as tfds', function_to_run='tf.gather(*args, **kwargs)', method_object=None, object_signature=None, function_args=[eval('class_weights')], function_kwargs={'indices': eval('tf.cast(label, tf.int32)')}, max_wait_secs=0)
    return (image, label, sample_weights)
custom_method(
train_batches.map(add_sample_weights), imports='import tensorflow as tf;import matplotlib.pyplot as plt;from IPython.display import clear_output;from tensorflow_examples.models.pix2pix import pix2pix;import tensorflow_datasets as tfds', function_to_run='obj.map(*args)', method_object=eval('train_batches'), object_signature='train_images\n    .cache()\n    .shuffle(BUFFER_SIZE)\n    .batch(BATCH_SIZE)\n    .repeat()\n    .map(Augment())\n    .prefetch', function_args=[eval('add_sample_weights')], function_kwargs={}, max_wait_secs=0, custom_class='class Augment(tf.keras.layers.Layer):\n  def __init__(self, seed=42):\n    super().__init__()\n    self.augment_inputs = tf.keras.layers.RandomFlip(mode="horizontal", seed=seed)\n    self.augment_labels = tf.keras.layers.RandomFlip(mode="horizontal", seed=seed)\n  \n  def call(self, inputs, labels):\n    inputs = self.augment_inputs(inputs)\n    labels = self.augment_labels(labels)\n    return inputs, labels').element_spec
weighted_model = unet_model(OUTPUT_CLASSES)
weighted_model.compile(optimizer='adam', loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=['accuracy'])
weighted_model.fit(train_batches.map(add_sample_weights), epochs=1, steps_per_epoch=10)
