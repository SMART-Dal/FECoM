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

def custom_method(imports: str, function_to_run: str, method_object=None, object_signature=None, function_args: list=None, function_kwargs: dict=None, custom_class=None):
    result = send_request(imports=imports, function_to_run=function_to_run, function_args=function_args, function_kwargs=function_kwargs, max_wait_secs=MAX_WAIT_S, wait_after_run_secs=WAIT_AFTER_RUN_S, method_object=method_object, object_signature=object_signature, custom_class=custom_class, experiment_file_path=EXPERIMENT_FILE_PATH)
((train_ds, val_ds, test_ds), metadata) = tfds.load('tf_flowers', split=['train[:80%]', 'train[80%:90%]', 'train[90%:]'], with_info=True, as_supervised=True)
num_classes = metadata.features['label'].num_classes
print(num_classes)
get_label_name = metadata.features['label'].int2str
(image, label) = next(iter(train_ds))
_ = plt.imshow(image)
_ = plt.title(get_label_name(label))
IMG_SIZE = 180
custom_method(imports='from tensorflow.keras import layers;import tensorflow_datasets as tfds;import matplotlib.pyplot as plt;import numpy as np;import tensorflow as tf', function_to_run='tf.keras.Sequential(*args)', method_object=None, object_signature=None, function_args=[eval('[\n  layers.Resizing(IMG_SIZE, IMG_SIZE),\n  layers.Rescaling(1./255)\n]')], function_kwargs={})
resize_and_rescale = tf.keras.Sequential([layers.Resizing(IMG_SIZE, IMG_SIZE), layers.Rescaling(1.0 / 255)])
custom_method(imports='from tensorflow.keras import layers;import tensorflow_datasets as tfds;import matplotlib.pyplot as plt;import numpy as np;import tensorflow as tf', function_to_run='obj(*args)', method_object=eval('resize_and_rescale'), object_signature=None, function_args=[eval('image')], function_kwargs={}, custom_class=None)
result = resize_and_rescale(image)
_ = plt.imshow(result)
print('Min and max pixel values:', result.numpy().min(), result.numpy().max())
custom_method(imports='from tensorflow.keras import layers;import tensorflow_datasets as tfds;import matplotlib.pyplot as plt;import numpy as np;import tensorflow as tf', function_to_run='tf.keras.Sequential(*args)', method_object=None, object_signature=None, function_args=[eval('[\n  layers.RandomFlip("horizontal_and_vertical"),\n  layers.RandomRotation(0.2),\n]')], function_kwargs={})
data_augmentation = tf.keras.Sequential([layers.RandomFlip('horizontal_and_vertical'), layers.RandomRotation(0.2)])
custom_method(imports='from tensorflow.keras import layers;import tensorflow_datasets as tfds;import matplotlib.pyplot as plt;import numpy as np;import tensorflow as tf', function_to_run='tf.cast(*args)', method_object=None, object_signature=None, function_args=[eval('tf.expand_dims(image, 0)'), eval('tf.float32')], function_kwargs={})
image = tf.cast(tf.expand_dims(image, 0), tf.float32)
plt.figure(figsize=(10, 10))
for i in range(9):
    custom_method(imports='from tensorflow.keras import layers;import tensorflow_datasets as tfds;import matplotlib.pyplot as plt;import numpy as np;import tensorflow as tf', function_to_run='obj(*args)', method_object=eval('data_augmentation'), object_signature=None, function_args=[eval('image')], function_kwargs={}, custom_class=None)
    augmented_image = data_augmentation(image)
    ax = plt.subplot(3, 3, i + 1)
    plt.imshow(augmented_image[0])
    plt.axis('off')
custom_method(imports='from tensorflow.keras import layers;import tensorflow_datasets as tfds;import matplotlib.pyplot as plt;import numpy as np;import tensorflow as tf', function_to_run='tf.keras.Sequential(*args)', method_object=None, object_signature=None, function_args=[eval("[\n  resize_and_rescale,\n  data_augmentation,\n  layers.Conv2D(16, 3, padding='same', activation='relu'),\n  layers.MaxPooling2D(),\n]")], function_kwargs={})
model = tf.keras.Sequential([resize_and_rescale, data_augmentation, layers.Conv2D(16, 3, padding='same', activation='relu'), layers.MaxPooling2D()])
custom_method(imports='from tensorflow.keras import layers;import tensorflow_datasets as tfds;import matplotlib.pyplot as plt;import numpy as np;import tensorflow as tf', function_to_run='obj.map(*args)', method_object=eval('train_ds'), object_signature=None, function_args=[eval('lambda x, y: (resize_and_rescale(x, training=True), y)')], function_kwargs={}, custom_class=None)
aug_ds = train_ds.map(lambda x, y: (resize_and_rescale(x, training=True), y))
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
custom_method(imports='from tensorflow.keras import layers;import tensorflow_datasets as tfds;import matplotlib.pyplot as plt;import numpy as np;import tensorflow as tf', function_to_run='tf.keras.Sequential(*args)', method_object=None, object_signature=None, function_args=[eval("[\n  layers.Conv2D(16, 3, padding='same', activation='relu'),\n  layers.MaxPooling2D(),\n  layers.Conv2D(32, 3, padding='same', activation='relu'),\n  layers.MaxPooling2D(),\n  layers.Conv2D(64, 3, padding='same', activation='relu'),\n  layers.MaxPooling2D(),\n  layers.Flatten(),\n  layers.Dense(128, activation='relu'),\n  layers.Dense(num_classes)\n]")], function_kwargs={})
model = tf.keras.Sequential([layers.Conv2D(16, 3, padding='same', activation='relu'), layers.MaxPooling2D(), layers.Conv2D(32, 3, padding='same', activation='relu'), layers.MaxPooling2D(), layers.Conv2D(64, 3, padding='same', activation='relu'), layers.MaxPooling2D(), layers.Flatten(), layers.Dense(128, activation='relu'), layers.Dense(num_classes)])
custom_method(imports='from tensorflow.keras import layers;import tensorflow_datasets as tfds;import matplotlib.pyplot as plt;import numpy as np;import tensorflow as tf', function_to_run='obj.compile(**kwargs)', method_object=eval('model'), object_signature=None, function_args=[], function_kwargs={'optimizer': eval("'adam'"), 'loss': eval('tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)'), 'metrics': eval("['accuracy']")}, custom_class=None)
model.compile(optimizer='adam', loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=['accuracy'])
epochs = 5
custom_method(imports='from tensorflow.keras import layers;import tensorflow_datasets as tfds;import matplotlib.pyplot as plt;import numpy as np;import tensorflow as tf', function_to_run='obj.fit(*args, **kwargs)', method_object=eval('model'), object_signature=None, function_args=[eval('train_ds')], function_kwargs={'validation_data': eval('val_ds'), 'epochs': eval('epochs')}, custom_class=None)
history = model.fit(train_ds, validation_data=val_ds, epochs=epochs)
custom_method(imports='from tensorflow.keras import layers;import tensorflow_datasets as tfds;import matplotlib.pyplot as plt;import numpy as np;import tensorflow as tf', function_to_run='obj.evaluate(*args)', method_object=eval('model'), object_signature=None, function_args=[eval('test_ds')], function_kwargs={}, custom_class=None)
(loss, acc) = model.evaluate(test_ds)
print('Accuracy', acc)

def random_invert_img(x, p=0.5):
    if tf.random.uniform([]) < p:
        x = 255 - x
    else:
        x
    return x

def random_invert(factor=0.5):
    return layers.Lambda(lambda x: random_invert_img(x, factor))
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
((train_ds, val_ds, test_ds), metadata) = tfds.load('tf_flowers', split=['train[:80%]', 'train[80%:90%]', 'train[90%:]'], with_info=True, as_supervised=True)
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
custom_method(imports='from tensorflow.keras import layers;import tensorflow_datasets as tfds;import matplotlib.pyplot as plt;import numpy as np;import tensorflow as tf', function_to_run='tf.image.flip_left_right(*args)', method_object=None, object_signature=None, function_args=[eval('image')], function_kwargs={})
flipped = tf.image.flip_left_right(image)
visualize(image, flipped)
custom_method(imports='from tensorflow.keras import layers;import tensorflow_datasets as tfds;import matplotlib.pyplot as plt;import numpy as np;import tensorflow as tf', function_to_run='tf.image.rgb_to_grayscale(*args)', method_object=None, object_signature=None, function_args=[eval('image')], function_kwargs={})
grayscaled = tf.image.rgb_to_grayscale(image)
visualize(image, tf.squeeze(grayscaled))
_ = plt.colorbar()
custom_method(imports='from tensorflow.keras import layers;import tensorflow_datasets as tfds;import matplotlib.pyplot as plt;import numpy as np;import tensorflow as tf', function_to_run='tf.image.adjust_saturation(*args)', method_object=None, object_signature=None, function_args=[eval('image'), eval('3')], function_kwargs={})
saturated = tf.image.adjust_saturation(image, 3)
visualize(image, saturated)
custom_method(imports='from tensorflow.keras import layers;import tensorflow_datasets as tfds;import matplotlib.pyplot as plt;import numpy as np;import tensorflow as tf', function_to_run='tf.image.adjust_brightness(*args)', method_object=None, object_signature=None, function_args=[eval('image'), eval('0.4')], function_kwargs={})
bright = tf.image.adjust_brightness(image, 0.4)
visualize(image, bright)
custom_method(imports='from tensorflow.keras import layers;import tensorflow_datasets as tfds;import matplotlib.pyplot as plt;import numpy as np;import tensorflow as tf', function_to_run='tf.image.central_crop(*args, **kwargs)', method_object=None, object_signature=None, function_args=[eval('image')], function_kwargs={'central_fraction': eval('0.5')})
cropped = tf.image.central_crop(image, central_fraction=0.5)
visualize(image, cropped)
custom_method(imports='from tensorflow.keras import layers;import tensorflow_datasets as tfds;import matplotlib.pyplot as plt;import numpy as np;import tensorflow as tf', function_to_run='tf.image.rot90(*args)', method_object=None, object_signature=None, function_args=[eval('image')], function_kwargs={})
rotated = tf.image.rot90(image)
visualize(image, rotated)
for i in range(3):
    seed = (i, 0)
    custom_method(imports='from tensorflow.keras import layers;import tensorflow_datasets as tfds;import matplotlib.pyplot as plt;import numpy as np;import tensorflow as tf', function_to_run='tf.image.stateless_random_brightness(*args, **kwargs)', method_object=None, object_signature=None, function_args=[eval('image')], function_kwargs={'max_delta': eval('0.95'), 'seed': eval('seed')})
    stateless_random_brightness = tf.image.stateless_random_brightness(image, max_delta=0.95, seed=seed)
    visualize(image, stateless_random_brightness)
for i in range(3):
    seed = (i, 0)
    custom_method(imports='from tensorflow.keras import layers;import tensorflow_datasets as tfds;import matplotlib.pyplot as plt;import numpy as np;import tensorflow as tf', function_to_run='tf.image.stateless_random_contrast(*args, **kwargs)', method_object=None, object_signature=None, function_args=[eval('image')], function_kwargs={'lower': eval('0.1'), 'upper': eval('0.9'), 'seed': eval('seed')})
    stateless_random_contrast = tf.image.stateless_random_contrast(image, lower=0.1, upper=0.9, seed=seed)
    visualize(image, stateless_random_contrast)
for i in range(3):
    seed = (i, 0)
    custom_method(imports='from tensorflow.keras import layers;import tensorflow_datasets as tfds;import matplotlib.pyplot as plt;import numpy as np;import tensorflow as tf', function_to_run='tf.image.stateless_random_crop(*args, **kwargs)', method_object=None, object_signature=None, function_args=[eval('image')], function_kwargs={'size': eval('[210, 300, 3]'), 'seed': eval('seed')})
    stateless_random_crop = tf.image.stateless_random_crop(image, size=[210, 300, 3], seed=seed)
    visualize(image, stateless_random_crop)
((train_datasets, val_ds, test_ds), metadata) = tfds.load('tf_flowers', split=['train[:80%]', 'train[80%:90%]', 'train[90%:]'], with_info=True, as_supervised=True)

def resize_and_rescale(image, label):
    custom_method(imports='from tensorflow.keras import layers;import tensorflow_datasets as tfds;import matplotlib.pyplot as plt;import numpy as np;import tensorflow as tf', function_to_run='tf.cast(*args)', method_object=None, object_signature=None, function_args=[eval('image'), eval('tf.float32')], function_kwargs={})
    image = tf.cast(image, tf.float32)
    custom_method(imports='from tensorflow.keras import layers;import tensorflow_datasets as tfds;import matplotlib.pyplot as plt;import numpy as np;import tensorflow as tf', function_to_run='tf.image.resize(*args)', method_object=None, object_signature=None, function_args=[eval('image'), eval('[IMG_SIZE, IMG_SIZE]')], function_kwargs={})
    image = tf.image.resize(image, [IMG_SIZE, IMG_SIZE])
    image = image / 255.0
    return (image, label)

def augment(image_label, seed):
    (image, label) = image_label
    custom_method(imports='from tensorflow.keras import layers;import tensorflow_datasets as tfds;import matplotlib.pyplot as plt;import numpy as np;import tensorflow as tf', function_to_run='obj(*args)', method_object=eval('resize_and_rescale'), object_signature=None, function_args=[eval('image'), eval('label')], function_kwargs={}, custom_class=None)
    (image, label) = resize_and_rescale(image, label)
    custom_method(imports='from tensorflow.keras import layers;import tensorflow_datasets as tfds;import matplotlib.pyplot as plt;import numpy as np;import tensorflow as tf', function_to_run='tf.image.resize_with_crop_or_pad(*args)', method_object=None, object_signature=None, function_args=[eval('image'), eval('IMG_SIZE + 6'), eval('IMG_SIZE + 6')], function_kwargs={})
    image = tf.image.resize_with_crop_or_pad(image, IMG_SIZE + 6, IMG_SIZE + 6)
    new_seed = tf.random.split(seed, num=1)[0, :]
    custom_method(imports='from tensorflow.keras import layers;import tensorflow_datasets as tfds;import matplotlib.pyplot as plt;import numpy as np;import tensorflow as tf', function_to_run='tf.image.stateless_random_crop(*args, **kwargs)', method_object=None, object_signature=None, function_args=[eval('image')], function_kwargs={'size': eval('[IMG_SIZE, IMG_SIZE, 3]'), 'seed': eval('seed')})
    image = tf.image.stateless_random_crop(image, size=[IMG_SIZE, IMG_SIZE, 3], seed=seed)
    custom_method(imports='from tensorflow.keras import layers;import tensorflow_datasets as tfds;import matplotlib.pyplot as plt;import numpy as np;import tensorflow as tf', function_to_run='tf.image.stateless_random_brightness(*args, **kwargs)', method_object=None, object_signature=None, function_args=[eval('image')], function_kwargs={'max_delta': eval('0.5'), 'seed': eval('new_seed')})
    image = tf.image.stateless_random_brightness(image, max_delta=0.5, seed=new_seed)
    custom_method(imports='from tensorflow.keras import layers;import tensorflow_datasets as tfds;import matplotlib.pyplot as plt;import numpy as np;import tensorflow as tf', function_to_run='tf.clip_by_value(*args)', method_object=None, object_signature=None, function_args=[eval('image'), eval('0'), eval('1')], function_kwargs={})
    image = tf.clip_by_value(image, 0, 1)
    return (image, label)
custom_method(imports='from tensorflow.keras import layers;import tensorflow_datasets as tfds;import matplotlib.pyplot as plt;import numpy as np;import tensorflow as tf', function_to_run='tf.data.experimental.Counter()', method_object=None, object_signature=None, function_args=[], function_kwargs={})
counter = tf.data.experimental.Counter()
custom_method(imports='from tensorflow.keras import layers;import tensorflow_datasets as tfds;import matplotlib.pyplot as plt;import numpy as np;import tensorflow as tf', function_to_run='tf.data.Dataset.zip(*args)', method_object=None, object_signature=None, function_args=[eval('(train_datasets, (counter, counter))')], function_kwargs={})
train_ds = tf.data.Dataset.zip((train_datasets, (counter, counter)))
custom_method(imports='from tensorflow.keras import layers;import tensorflow_datasets as tfds;import matplotlib.pyplot as plt;import numpy as np;import tensorflow as tf', function_to_run='obj.shuffle(1000).map(augment, num_parallel_calls=AUTOTUNE).batch(batch_size).prefetch(*args)', method_object=eval('train_ds'), object_signature=None, function_args=[eval('AUTOTUNE')], function_kwargs={}, custom_class=None)
train_ds = train_ds.shuffle(1000).map(augment, num_parallel_calls=AUTOTUNE).batch(batch_size).prefetch(AUTOTUNE)
val_ds = val_ds.map(resize_and_rescale, num_parallel_calls=AUTOTUNE).batch(batch_size).prefetch(AUTOTUNE)
test_ds = test_ds.map(resize_and_rescale, num_parallel_calls=AUTOTUNE).batch(batch_size).prefetch(AUTOTUNE)
custom_method(imports='from tensorflow.keras import layers;import tensorflow_datasets as tfds;import matplotlib.pyplot as plt;import numpy as np;import tensorflow as tf', function_to_run='tf.random.Generator.from_seed(*args, **kwargs)', method_object=None, object_signature=None, function_args=[eval('123')], function_kwargs={'alg': eval("'philox'")})
rng = tf.random.Generator.from_seed(123, alg='philox')

def f(x, y):
    seed = rng.make_seeds(2)[0]
    (image, label) = augment((x, y), seed)
    return (image, label)
train_ds = train_datasets.shuffle(1000).map(f, num_parallel_calls=AUTOTUNE).batch(batch_size).prefetch(AUTOTUNE)
val_ds = val_ds.map(resize_and_rescale, num_parallel_calls=AUTOTUNE).batch(batch_size).prefetch(AUTOTUNE)
test_ds = test_ds.map(resize_and_rescale, num_parallel_calls=AUTOTUNE).batch(batch_size).prefetch(AUTOTUNE)
