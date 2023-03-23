import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow.keras import layers
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
    result = send_request(imports=imports, function_to_run=function_to_run, function_args=function_args, function_kwargs=function_kwargs, max_wait_secs=max_wait_secs, wait_after_run_secs=wait_after_run_secs, method_object=method_object, object_signature=object_signature, custom_class=custom_class, experiment_file_path=EXPERIMENT_FILE_PATH)
    return func
((train_ds, val_ds, test_ds), metadata) = custom_method(
tfds.load('tf_flowers', split=['train[:80%]', 'train[80%:90%]', 'train[90%:]'], with_info=True, as_supervised=True), imports='from tensorflow.keras import layers;import numpy as np;import tensorflow as tf;import tensorflow_datasets as tfds;import matplotlib.pyplot as plt', function_to_run='tfds.load(*args, **kwargs)', method_object=None, object_signature=None, function_args=[eval("'tf_flowers'")], function_kwargs={'split': eval("['train[:80%]', 'train[80%:90%]', 'train[90%:]']"), 'with_info': eval('True'), 'as_supervised': eval('True')})
num_classes = metadata.features['label'].num_classes
print(num_classes)
get_label_name = metadata.features['label'].int2str
(image, label) = next(iter(train_ds))
_ = plt.imshow(image)
_ = plt.title(get_label_name(label))
IMG_SIZE = 180
resize_and_rescale = custom_method(
tf.keras.Sequential([layers.Resizing(IMG_SIZE, IMG_SIZE), layers.Rescaling(1.0 / 255)]), imports='from tensorflow.keras import layers;import numpy as np;import tensorflow as tf;import tensorflow_datasets as tfds;import matplotlib.pyplot as plt', function_to_run='tf.keras.Sequential(*args)', method_object=None, object_signature=None, function_args=[eval('[\n  layers.Resizing(IMG_SIZE, IMG_SIZE),\n  layers.Rescaling(1./255)\n]')], function_kwargs={})
result = custom_method(
resize_and_rescale(image), imports='from tensorflow.keras import layers;import numpy as np;import tensorflow as tf;import tensorflow_datasets as tfds;import matplotlib.pyplot as plt', function_to_run='obj(*args)', method_object=eval('resize_and_rescale'), object_signature='tf.keras.Sequential', function_args=[eval('image')], function_kwargs={}, custom_class=None)
_ = plt.imshow(result)
print('Min and max pixel values:', result.numpy().min(), result.numpy().max())
data_augmentation = custom_method(
tf.keras.Sequential([layers.RandomFlip('horizontal_and_vertical'), layers.RandomRotation(0.2)]), imports='from tensorflow.keras import layers;import numpy as np;import tensorflow as tf;import tensorflow_datasets as tfds;import matplotlib.pyplot as plt', function_to_run='tf.keras.Sequential(*args)', method_object=None, object_signature=None, function_args=[eval('[\n  layers.RandomFlip("horizontal_and_vertical"),\n  layers.RandomRotation(0.2),\n]')], function_kwargs={})
image = custom_method(
tf.cast(tf.expand_dims(image, 0), tf.float32), imports='from tensorflow.keras import layers;import numpy as np;import tensorflow as tf;import tensorflow_datasets as tfds;import matplotlib.pyplot as plt', function_to_run='tf.cast(*args)', method_object=None, object_signature=None, function_args=[eval('tf.expand_dims(image, 0)'), eval('tf.float32')], function_kwargs={})
plt.figure(figsize=(10, 10))
for i in range(9):
    augmented_image = custom_method(
    data_augmentation(image), imports='from tensorflow.keras import layers;import numpy as np;import tensorflow as tf;import tensorflow_datasets as tfds;import matplotlib.pyplot as plt', function_to_run='obj(*args)', method_object=eval('data_augmentation'), object_signature='tf.keras.Sequential', function_args=[eval('image')], function_kwargs={}, custom_class=None)
    ax = plt.subplot(3, 3, i + 1)
    plt.imshow(augmented_image[0])
    plt.axis('off')
model = custom_method(
tf.keras.Sequential([resize_and_rescale, data_augmentation, layers.Conv2D(16, 3, padding='same', activation='relu'), layers.MaxPooling2D()]), imports='from tensorflow.keras import layers;import numpy as np;import tensorflow as tf;import tensorflow_datasets as tfds;import matplotlib.pyplot as plt', function_to_run='tf.keras.Sequential(*args)', method_object=None, object_signature=None, function_args=[eval("[\n  resize_and_rescale,\n  data_augmentation,\n  layers.Conv2D(16, 3, padding='same', activation='relu'),\n  layers.MaxPooling2D(),\n]")], function_kwargs={})
aug_ds = custom_method(
train_ds.map(lambda x, y: (resize_and_rescale(x, training=True), y)), imports='from tensorflow.keras import layers;import numpy as np;import tensorflow as tf;import tensorflow_datasets as tfds;import matplotlib.pyplot as plt', function_to_run='obj.map(*args)', method_object=eval('train_ds'), object_signature='tf.data.Dataset.zip', function_args=[eval('lambda x, y: (resize_and_rescale(x, training=True), y)')], function_kwargs={}, custom_class=None)
batch_size = 32
AUTOTUNE = tf.data.AUTOTUNE

def prepare(ds, shuffle=False, augment=False):
    ds = ds.map(lambda x, y: (resize_and_rescale(x), y), num_parallel_calls=AUTOTUNE)
    if shuffle:
        ds = ds.shuffle(1000)
    ds = ds.batch(batch_size)
    if augment:
        ds = ds.map(lambda x, y: (data_augmentation(x, training=True), y), num_parallel_calls=AUTOTUNE)
    return ds.prefetch(buffer_size=AUTOTUNE)
train_ds = prepare(train_ds, shuffle=True, augment=True)
val_ds = prepare(val_ds)
test_ds = prepare(test_ds)
model = custom_method(
tf.keras.Sequential([layers.Conv2D(16, 3, padding='same', activation='relu'), layers.MaxPooling2D(), layers.Conv2D(32, 3, padding='same', activation='relu'), layers.MaxPooling2D(), layers.Conv2D(64, 3, padding='same', activation='relu'), layers.MaxPooling2D(), layers.Flatten(), layers.Dense(128, activation='relu'), layers.Dense(num_classes)]), imports='from tensorflow.keras import layers;import numpy as np;import tensorflow as tf;import tensorflow_datasets as tfds;import matplotlib.pyplot as plt', function_to_run='tf.keras.Sequential(*args)', method_object=None, object_signature=None, function_args=[eval("[\n  layers.Conv2D(16, 3, padding='same', activation='relu'),\n  layers.MaxPooling2D(),\n  layers.Conv2D(32, 3, padding='same', activation='relu'),\n  layers.MaxPooling2D(),\n  layers.Conv2D(64, 3, padding='same', activation='relu'),\n  layers.MaxPooling2D(),\n  layers.Flatten(),\n  layers.Dense(128, activation='relu'),\n  layers.Dense(num_classes)\n]")], function_kwargs={})
custom_method(
model.compile(optimizer='adam', loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=['accuracy']), imports='from tensorflow.keras import layers;import numpy as np;import tensorflow as tf;import tensorflow_datasets as tfds;import matplotlib.pyplot as plt', function_to_run='obj.compile(**kwargs)', method_object=eval('model'), object_signature='tf.keras.Sequential', function_args=[], function_kwargs={'optimizer': eval("'adam'"), 'loss': eval('tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)'), 'metrics': eval("['accuracy']")}, custom_class=None)
epochs = 5
history = custom_method(
model.fit(train_ds, validation_data=val_ds, epochs=epochs), imports='from tensorflow.keras import layers;import numpy as np;import tensorflow as tf;import tensorflow_datasets as tfds;import matplotlib.pyplot as plt', function_to_run='obj.fit(*args, **kwargs)', method_object=eval('model'), object_signature='tf.keras.Sequential', function_args=[eval('train_ds')], function_kwargs={'validation_data': eval('val_ds'), 'epochs': eval('epochs')}, custom_class=None)
(loss, acc) = custom_method(
model.evaluate(test_ds), imports='from tensorflow.keras import layers;import numpy as np;import tensorflow as tf;import tensorflow_datasets as tfds;import matplotlib.pyplot as plt', function_to_run='obj.evaluate(*args)', method_object=eval('model'), object_signature='tf.keras.Sequential', function_args=[eval('test_ds')], function_kwargs={}, custom_class=None)
print('Accuracy', acc)

def random_invert_img(x, p=0.5):
    if custom_method(
    tf.random.uniform([]), imports='from tensorflow.keras import layers;import numpy as np;import tensorflow as tf;import tensorflow_datasets as tfds;import matplotlib.pyplot as plt', function_to_run='tf.random.uniform(*args)', method_object=None, object_signature=None, function_args=[eval('[]')], function_kwargs={}) < p:
        x = 255 - x
    else:
        x
    return x

def random_invert(factor=0.5):
    return custom_method(
    layers.Lambda(lambda x: random_invert_img(x, factor)), imports='from tensorflow.keras import layers;import numpy as np;import tensorflow as tf;import tensorflow_datasets as tfds;import matplotlib.pyplot as plt', function_to_run='layers.Lambda(*args)', method_object=None, object_signature=None, function_args=[eval('lambda x: random_invert_img(x, factor)')], function_kwargs={})
random_invert = random_invert()
plt.figure(figsize=(10, 10))
for i in range(9):
    augmented_image = random_invert(image)
    ax = plt.subplot(3, 3, i + 1)
    plt.imshow(augmented_image[0].numpy().astype('uint8'))
    plt.axis('off')

class RandomInvert(layers.Layer):

    def __init__(self, factor=0.5, **kwargs):
        super().__init__(**kwargs)
        self.factor = factor

    def call(self, x):
        return random_invert_img(x)
_ = plt.imshow(RandomInvert()(image)[0])
((train_ds, val_ds, test_ds), metadata) = custom_method(
tfds.load('tf_flowers', split=['train[:80%]', 'train[80%:90%]', 'train[90%:]'], with_info=True, as_supervised=True), imports='from tensorflow.keras import layers;import numpy as np;import tensorflow as tf;import tensorflow_datasets as tfds;import matplotlib.pyplot as plt', function_to_run='tfds.load(*args, **kwargs)', method_object=None, object_signature=None, function_args=[eval("'tf_flowers'")], function_kwargs={'split': eval("['train[:80%]', 'train[80%:90%]', 'train[90%:]']"), 'with_info': eval('True'), 'as_supervised': eval('True')})
(image, label) = next(iter(train_ds))
_ = plt.imshow(image)
_ = plt.title(get_label_name(label))

def visualize(original, augmented):
    fig = plt.figure()
    plt.subplot(1, 2, 1)
    plt.title('Original image')
    plt.imshow(original)
    plt.subplot(1, 2, 2)
    plt.title('Augmented image')
    plt.imshow(augmented)
flipped = custom_method(
tf.image.flip_left_right(image), imports='from tensorflow.keras import layers;import numpy as np;import tensorflow as tf;import tensorflow_datasets as tfds;import matplotlib.pyplot as plt', function_to_run='tf.image.flip_left_right(*args)', method_object=None, object_signature=None, function_args=[eval('image')], function_kwargs={})
visualize(image, flipped)
grayscaled = custom_method(
tf.image.rgb_to_grayscale(image), imports='from tensorflow.keras import layers;import numpy as np;import tensorflow as tf;import tensorflow_datasets as tfds;import matplotlib.pyplot as plt', function_to_run='tf.image.rgb_to_grayscale(*args)', method_object=None, object_signature=None, function_args=[eval('image')], function_kwargs={})
visualize(image, tf.squeeze(grayscaled))
_ = plt.colorbar()
saturated = custom_method(
tf.image.adjust_saturation(image, 3), imports='from tensorflow.keras import layers;import numpy as np;import tensorflow as tf;import tensorflow_datasets as tfds;import matplotlib.pyplot as plt', function_to_run='tf.image.adjust_saturation(*args)', method_object=None, object_signature=None, function_args=[eval('image'), eval('3')], function_kwargs={})
visualize(image, saturated)
bright = custom_method(
tf.image.adjust_brightness(image, 0.4), imports='from tensorflow.keras import layers;import numpy as np;import tensorflow as tf;import tensorflow_datasets as tfds;import matplotlib.pyplot as plt', function_to_run='tf.image.adjust_brightness(*args)', method_object=None, object_signature=None, function_args=[eval('image'), eval('0.4')], function_kwargs={})
visualize(image, bright)
cropped = custom_method(
tf.image.central_crop(image, central_fraction=0.5), imports='from tensorflow.keras import layers;import numpy as np;import tensorflow as tf;import tensorflow_datasets as tfds;import matplotlib.pyplot as plt', function_to_run='tf.image.central_crop(*args, **kwargs)', method_object=None, object_signature=None, function_args=[eval('image')], function_kwargs={'central_fraction': eval('0.5')})
visualize(image, cropped)
rotated = custom_method(
tf.image.rot90(image), imports='from tensorflow.keras import layers;import numpy as np;import tensorflow as tf;import tensorflow_datasets as tfds;import matplotlib.pyplot as plt', function_to_run='tf.image.rot90(*args)', method_object=None, object_signature=None, function_args=[eval('image')], function_kwargs={})
visualize(image, rotated)
for i in range(3):
    seed = (i, 0)
    stateless_random_brightness = custom_method(
    tf.image.stateless_random_brightness(image, max_delta=0.95, seed=seed), imports='from tensorflow.keras import layers;import numpy as np;import tensorflow as tf;import tensorflow_datasets as tfds;import matplotlib.pyplot as plt', function_to_run='tf.image.stateless_random_brightness(*args, **kwargs)', method_object=None, object_signature=None, function_args=[eval('image')], function_kwargs={'max_delta': eval('0.95'), 'seed': eval('seed')})
    visualize(image, stateless_random_brightness)
for i in range(3):
    seed = (i, 0)
    stateless_random_contrast = custom_method(
    tf.image.stateless_random_contrast(image, lower=0.1, upper=0.9, seed=seed), imports='from tensorflow.keras import layers;import numpy as np;import tensorflow as tf;import tensorflow_datasets as tfds;import matplotlib.pyplot as plt', function_to_run='tf.image.stateless_random_contrast(*args, **kwargs)', method_object=None, object_signature=None, function_args=[eval('image')], function_kwargs={'lower': eval('0.1'), 'upper': eval('0.9'), 'seed': eval('seed')})
    visualize(image, stateless_random_contrast)
for i in range(3):
    seed = (i, 0)
    stateless_random_crop = custom_method(
    tf.image.stateless_random_crop(image, size=[210, 300, 3], seed=seed), imports='from tensorflow.keras import layers;import numpy as np;import tensorflow as tf;import tensorflow_datasets as tfds;import matplotlib.pyplot as plt', function_to_run='tf.image.stateless_random_crop(*args, **kwargs)', method_object=None, object_signature=None, function_args=[eval('image')], function_kwargs={'size': eval('[210, 300, 3]'), 'seed': eval('seed')})
    visualize(image, stateless_random_crop)
((train_datasets, val_ds, test_ds), metadata) = custom_method(
tfds.load('tf_flowers', split=['train[:80%]', 'train[80%:90%]', 'train[90%:]'], with_info=True, as_supervised=True), imports='from tensorflow.keras import layers;import numpy as np;import tensorflow as tf;import tensorflow_datasets as tfds;import matplotlib.pyplot as plt', function_to_run='tfds.load(*args, **kwargs)', method_object=None, object_signature=None, function_args=[eval("'tf_flowers'")], function_kwargs={'split': eval("['train[:80%]', 'train[80%:90%]', 'train[90%:]']"), 'with_info': eval('True'), 'as_supervised': eval('True')})

def resize_and_rescale(image, label):
    image = custom_method(
    tf.cast(image, tf.float32), imports='from tensorflow.keras import layers;import numpy as np;import tensorflow as tf;import tensorflow_datasets as tfds;import matplotlib.pyplot as plt', function_to_run='tf.cast(*args)', method_object=None, object_signature=None, function_args=[eval('image'), eval('tf.float32')], function_kwargs={})
    image = custom_method(
    tf.image.resize(image, [IMG_SIZE, IMG_SIZE]), imports='from tensorflow.keras import layers;import numpy as np;import tensorflow as tf;import tensorflow_datasets as tfds;import matplotlib.pyplot as plt', function_to_run='tf.image.resize(*args)', method_object=None, object_signature=None, function_args=[eval('image'), eval('[IMG_SIZE, IMG_SIZE]')], function_kwargs={})
    image = image / 255.0
    return (image, label)

def augment(image_label, seed):
    (image, label) = image_label
    (image, label) = custom_method(
    resize_and_rescale(image, label), imports='from tensorflow.keras import layers;import numpy as np;import tensorflow as tf;import tensorflow_datasets as tfds;import matplotlib.pyplot as plt', function_to_run='obj(*args)', method_object=eval('resize_and_rescale'), object_signature='tf.keras.Sequential', function_args=[eval('image'), eval('label')], function_kwargs={}, custom_class=None)
    image = custom_method(
    tf.image.resize_with_crop_or_pad(image, IMG_SIZE + 6, IMG_SIZE + 6), imports='from tensorflow.keras import layers;import numpy as np;import tensorflow as tf;import tensorflow_datasets as tfds;import matplotlib.pyplot as plt', function_to_run='tf.image.resize_with_crop_or_pad(*args)', method_object=None, object_signature=None, function_args=[eval('image'), eval('IMG_SIZE + 6'), eval('IMG_SIZE + 6')], function_kwargs={})
    new_seed = custom_method(
    tf.random.split(seed, num=1), imports='from tensorflow.keras import layers;import numpy as np;import tensorflow as tf;import tensorflow_datasets as tfds;import matplotlib.pyplot as plt', function_to_run='tf.random.split(*args, **kwargs)', method_object=None, object_signature=None, function_args=[eval('seed')], function_kwargs={'num': eval('1')})[0, :]
    image = custom_method(
    tf.image.stateless_random_crop(image, size=[IMG_SIZE, IMG_SIZE, 3], seed=seed), imports='from tensorflow.keras import layers;import numpy as np;import tensorflow as tf;import tensorflow_datasets as tfds;import matplotlib.pyplot as plt', function_to_run='tf.image.stateless_random_crop(*args, **kwargs)', method_object=None, object_signature=None, function_args=[eval('image')], function_kwargs={'size': eval('[IMG_SIZE, IMG_SIZE, 3]'), 'seed': eval('seed')})
    image = custom_method(
    tf.image.stateless_random_brightness(image, max_delta=0.5, seed=new_seed), imports='from tensorflow.keras import layers;import numpy as np;import tensorflow as tf;import tensorflow_datasets as tfds;import matplotlib.pyplot as plt', function_to_run='tf.image.stateless_random_brightness(*args, **kwargs)', method_object=None, object_signature=None, function_args=[eval('image')], function_kwargs={'max_delta': eval('0.5'), 'seed': eval('new_seed')})
    image = custom_method(
    tf.clip_by_value(image, 0, 1), imports='from tensorflow.keras import layers;import numpy as np;import tensorflow as tf;import tensorflow_datasets as tfds;import matplotlib.pyplot as plt', function_to_run='tf.clip_by_value(*args)', method_object=None, object_signature=None, function_args=[eval('image'), eval('0'), eval('1')], function_kwargs={})
    return (image, label)
counter = custom_method(
tf.data.experimental.Counter(), imports='from tensorflow.keras import layers;import numpy as np;import tensorflow as tf;import tensorflow_datasets as tfds;import matplotlib.pyplot as plt', function_to_run='tf.data.experimental.Counter()', method_object=None, object_signature=None, function_args=[], function_kwargs={})
train_ds = custom_method(
tf.data.Dataset.zip((train_datasets, (counter, counter))), imports='from tensorflow.keras import layers;import numpy as np;import tensorflow as tf;import tensorflow_datasets as tfds;import matplotlib.pyplot as plt', function_to_run='tf.data.Dataset.zip(*args)', method_object=None, object_signature=None, function_args=[eval('(train_datasets, (counter, counter))')], function_kwargs={})
train_ds = train_ds.shuffle(1000).map(augment, num_parallel_calls=AUTOTUNE).batch(batch_size).prefetch(AUTOTUNE)
val_ds = val_ds.map(resize_and_rescale, num_parallel_calls=AUTOTUNE).batch(batch_size).prefetch(AUTOTUNE)
test_ds = test_ds.map(resize_and_rescale, num_parallel_calls=AUTOTUNE).batch(batch_size).prefetch(AUTOTUNE)
rng = custom_method(
tf.random.Generator.from_seed(123, alg='philox'), imports='from tensorflow.keras import layers;import numpy as np;import tensorflow as tf;import tensorflow_datasets as tfds;import matplotlib.pyplot as plt', function_to_run='tf.random.Generator.from_seed(*args, **kwargs)', method_object=None, object_signature=None, function_args=[eval('123')], function_kwargs={'alg': eval("'philox'")})

def f(x, y):
    seed = custom_method(
    rng.make_seeds(2), imports='from tensorflow.keras import layers;import numpy as np;import tensorflow as tf;import tensorflow_datasets as tfds;import matplotlib.pyplot as plt', function_to_run='obj.make_seeds(*args)', method_object=eval('rng'), object_signature='tf.random.Generator.from_seed', function_args=[eval('2')], function_kwargs={}, custom_class=None)[0]
    (image, label) = augment((x, y), seed)
    return (image, label)
train_ds = train_datasets.shuffle(1000).map(f, num_parallel_calls=AUTOTUNE).batch(batch_size).prefetch(AUTOTUNE)
val_ds = val_ds.map(resize_and_rescale, num_parallel_calls=AUTOTUNE).batch(batch_size).prefetch(AUTOTUNE)
test_ds = test_ds.map(resize_and_rescale, num_parallel_calls=AUTOTUNE).batch(batch_size).prefetch(AUTOTUNE)
