import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow.keras import layers
import sys
from fecom.patching.patching_config import EXPERIMENT_DIR
from fecom.measurement.execution import before_execution as before_execution_INSERTED_INTO_SCRIPT
from fecom.measurement.execution import after_execution as after_execution_INSERTED_INTO_SCRIPT
from fecom.experiment.experiment_kinds import ExperimentKinds
experiment_number = sys.argv[1]
experiment_project = sys.argv[2]
EXPERIMENT_FILE_PATH = EXPERIMENT_DIR / ExperimentKinds.METHOD_LEVEL.value / experiment_project / f'experiment-{experiment_number}.json'
((train_ds, val_ds, test_ds), metadata) = tfds.load('tf_flowers', split=['train[:80%]', 'train[80%:90%]', 'train[90%:]'], with_info=True, as_supervised=True)
num_classes = metadata.features['label'].num_classes
print(num_classes)
get_label_name = metadata.features['label'].int2str
(image, label) = next(iter(train_ds))
_ = plt.imshow(image)
_ = plt.title(get_label_name(label))
IMG_SIZE = 180
start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT(experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tensorflow.keras.Sequential()')
resize_and_rescale = tf.keras.Sequential([layers.Resizing(IMG_SIZE, IMG_SIZE), layers.Rescaling(1.0 / 255)])
after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tensorflow.keras.Sequential()', method_object=None, function_args=[[layers.Resizing(IMG_SIZE, IMG_SIZE), layers.Rescaling(1.0 / 255)]], function_kwargs=None)
start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT(experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tensorflow.keras.Sequential()')
result = resize_and_rescale(image)
after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tensorflow.keras.Sequential()', method_object=resize_and_rescale, function_args=[image], function_kwargs=None)
_ = plt.imshow(result)
print('Min and max pixel values:', result.numpy().min(), result.numpy().max())
start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT(experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tensorflow.keras.Sequential()')
data_augmentation = tf.keras.Sequential([layers.RandomFlip('horizontal_and_vertical'), layers.RandomRotation(0.2)])
after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tensorflow.keras.Sequential()', method_object=None, function_args=[[layers.RandomFlip('horizontal_and_vertical'), layers.RandomRotation(0.2)]], function_kwargs=None)
start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT(experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tensorflow.cast()')
image = tf.cast(tf.expand_dims(image, 0), tf.float32)
after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tensorflow.cast()', method_object=None, function_args=[tf.expand_dims(image, 0), tf.float32], function_kwargs=None)
plt.figure(figsize=(10, 10))
for i in range(9):
    start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT(experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tensorflow.keras.Sequential()')
    augmented_image = data_augmentation(image)
    after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tensorflow.keras.Sequential()', method_object=data_augmentation, function_args=[image], function_kwargs=None)
    ax = plt.subplot(3, 3, i + 1)
    plt.imshow(augmented_image[0])
    plt.axis('off')
start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT(experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tensorflow.keras.Sequential()')
model = tf.keras.Sequential([resize_and_rescale, data_augmentation, layers.Conv2D(16, 3, padding='same', activation='relu'), layers.MaxPooling2D()])
after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tensorflow.keras.Sequential()', method_object=None, function_args=[[resize_and_rescale, data_augmentation, layers.Conv2D(16, 3, padding='same', activation='relu'), layers.MaxPooling2D()]], function_kwargs=None)
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
start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT(experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tensorflow.keras.Sequential()')
model = tf.keras.Sequential([layers.Conv2D(16, 3, padding='same', activation='relu'), layers.MaxPooling2D(), layers.Conv2D(32, 3, padding='same', activation='relu'), layers.MaxPooling2D(), layers.Conv2D(64, 3, padding='same', activation='relu'), layers.MaxPooling2D(), layers.Flatten(), layers.Dense(128, activation='relu'), layers.Dense(num_classes)])
after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tensorflow.keras.Sequential()', method_object=None, function_args=[[layers.Conv2D(16, 3, padding='same', activation='relu'), layers.MaxPooling2D(), layers.Conv2D(32, 3, padding='same', activation='relu'), layers.MaxPooling2D(), layers.Conv2D(64, 3, padding='same', activation='relu'), layers.MaxPooling2D(), layers.Flatten(), layers.Dense(128, activation='relu'), layers.Dense(num_classes)]], function_kwargs=None)
start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT(experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tensorflow.keras.Sequential.compile()')
model.compile(optimizer='adam', loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=['accuracy'])
after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tensorflow.keras.Sequential.compile()', method_object=model, function_args=None, function_kwargs={'optimizer': 'adam', 'loss': tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), 'metrics': ['accuracy']})
epochs = 5
start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT(experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tensorflow.keras.Sequential.fit()')
history = model.fit(train_ds, validation_data=val_ds, epochs=epochs)
after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tensorflow.keras.Sequential.fit()', method_object=model, function_args=[train_ds], function_kwargs={'validation_data': val_ds, 'epochs': epochs})
start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT(experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tensorflow.keras.Sequential.evaluate()')
(loss, acc) = model.evaluate(test_ds)
after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tensorflow.keras.Sequential.evaluate()', method_object=model, function_args=[test_ds], function_kwargs=None)
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
start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT(experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tensorflow.image.flip_left_right()')
flipped = tf.image.flip_left_right(image)
after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tensorflow.image.flip_left_right()', method_object=None, function_args=[image], function_kwargs=None)
visualize(image, flipped)
start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT(experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tensorflow.image.rgb_to_grayscale()')
grayscaled = tf.image.rgb_to_grayscale(image)
after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tensorflow.image.rgb_to_grayscale()', method_object=None, function_args=[image], function_kwargs=None)
visualize(image, tf.squeeze(grayscaled))
_ = plt.colorbar()
start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT(experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tensorflow.image.adjust_saturation()')
saturated = tf.image.adjust_saturation(image, 3)
after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tensorflow.image.adjust_saturation()', method_object=None, function_args=[image, 3], function_kwargs=None)
visualize(image, saturated)
start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT(experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tensorflow.image.adjust_brightness()')
bright = tf.image.adjust_brightness(image, 0.4)
after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tensorflow.image.adjust_brightness()', method_object=None, function_args=[image, 0.4], function_kwargs=None)
visualize(image, bright)
start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT(experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tensorflow.image.central_crop()')
cropped = tf.image.central_crop(image, central_fraction=0.5)
after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tensorflow.image.central_crop()', method_object=None, function_args=[image], function_kwargs={'central_fraction': 0.5})
visualize(image, cropped)
start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT(experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tensorflow.image.rot90()')
rotated = tf.image.rot90(image)
after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tensorflow.image.rot90()', method_object=None, function_args=[image], function_kwargs=None)
visualize(image, rotated)
for i in range(3):
    seed = (i, 0)
    start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT(experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tensorflow.image.stateless_random_brightness()')
    stateless_random_brightness = tf.image.stateless_random_brightness(image, max_delta=0.95, seed=seed)
    after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tensorflow.image.stateless_random_brightness()', method_object=None, function_args=[image], function_kwargs={'max_delta': 0.95, 'seed': seed})
    visualize(image, stateless_random_brightness)
for i in range(3):
    seed = (i, 0)
    start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT(experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tensorflow.image.stateless_random_contrast()')
    stateless_random_contrast = tf.image.stateless_random_contrast(image, lower=0.1, upper=0.9, seed=seed)
    after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tensorflow.image.stateless_random_contrast()', method_object=None, function_args=[image], function_kwargs={'lower': 0.1, 'upper': 0.9, 'seed': seed})
    visualize(image, stateless_random_contrast)
for i in range(3):
    seed = (i, 0)
    start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT(experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tensorflow.image.stateless_random_crop()')
    stateless_random_crop = tf.image.stateless_random_crop(image, size=[210, 300, 3], seed=seed)
    after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tensorflow.image.stateless_random_crop()', method_object=None, function_args=[image], function_kwargs={'size': [210, 300, 3], 'seed': seed})
    visualize(image, stateless_random_crop)
((train_datasets, val_ds, test_ds), metadata) = tfds.load('tf_flowers', split=['train[:80%]', 'train[80%:90%]', 'train[90%:]'], with_info=True, as_supervised=True)

def resize_and_rescale(image, label):
    start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT(experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tensorflow.cast()')
    image = tf.cast(image, tf.float32)
    after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tensorflow.cast()', method_object=None, function_args=[image, tf.float32], function_kwargs=None)
    start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT(experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tensorflow.image.resize()')
    image = tf.image.resize(image, [IMG_SIZE, IMG_SIZE])
    after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tensorflow.image.resize()', method_object=None, function_args=[image, [IMG_SIZE, IMG_SIZE]], function_kwargs=None)
    image = image / 255.0
    return (image, label)

def augment(image_label, seed):
    (image, label) = image_label
    start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT(experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tensorflow.keras.Sequential()')
    (image, label) = resize_and_rescale(image, label)
    after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tensorflow.keras.Sequential()', method_object=resize_and_rescale, function_args=[image, label], function_kwargs=None)
    start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT(experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tensorflow.image.resize_with_crop_or_pad()')
    image = tf.image.resize_with_crop_or_pad(image, IMG_SIZE + 6, IMG_SIZE + 6)
    after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tensorflow.image.resize_with_crop_or_pad()', method_object=None, function_args=[image, IMG_SIZE + 6, IMG_SIZE + 6], function_kwargs=None)
    new_seed = tf.random.split(seed, num=1)[0, :]
    start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT(experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tensorflow.image.stateless_random_crop()')
    image = tf.image.stateless_random_crop(image, size=[IMG_SIZE, IMG_SIZE, 3], seed=seed)
    after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tensorflow.image.stateless_random_crop()', method_object=None, function_args=[image], function_kwargs={'size': [IMG_SIZE, IMG_SIZE, 3], 'seed': seed})
    start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT(experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tensorflow.image.stateless_random_brightness()')
    image = tf.image.stateless_random_brightness(image, max_delta=0.5, seed=new_seed)
    after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tensorflow.image.stateless_random_brightness()', method_object=None, function_args=[image], function_kwargs={'max_delta': 0.5, 'seed': new_seed})
    start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT(experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tensorflow.clip_by_value()')
    image = tf.clip_by_value(image, 0, 1)
    after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tensorflow.clip_by_value()', method_object=None, function_args=[image, 0, 1], function_kwargs=None)
    return (image, label)
start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT(experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tensorflow.data.experimental.Counter()')
counter = tf.data.experimental.Counter()
after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tensorflow.data.experimental.Counter()', method_object=None, function_args=None, function_kwargs=None)
start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT(experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tensorflow.data.Dataset.zip()')
train_ds = tf.data.Dataset.zip((train_datasets, (counter, counter)))
after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tensorflow.data.Dataset.zip()', method_object=None, function_args=[(train_datasets, (counter, counter))], function_kwargs=None)
start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT(experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tensorflow.data.Dataset.zip.shuffle(1000).map(augment, num_parallel_calls=AUTOTUNE).batch(batch_size).prefetch()')
train_ds = train_ds.shuffle(1000).map(augment, num_parallel_calls=AUTOTUNE).batch(batch_size).prefetch(AUTOTUNE)
after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tensorflow.data.Dataset.zip.shuffle(1000).map(augment, num_parallel_calls=AUTOTUNE).batch(batch_size).prefetch()', method_object=train_ds, function_args=[AUTOTUNE], function_kwargs=None)
val_ds = val_ds.map(resize_and_rescale, num_parallel_calls=AUTOTUNE).batch(batch_size).prefetch(AUTOTUNE)
test_ds = test_ds.map(resize_and_rescale, num_parallel_calls=AUTOTUNE).batch(batch_size).prefetch(AUTOTUNE)
start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT(experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tensorflow.random.Generator.from_seed()')
rng = tf.random.Generator.from_seed(123, alg='philox')
after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tensorflow.random.Generator.from_seed()', method_object=None, function_args=[123], function_kwargs={'alg': 'philox'})

def f(x, y):
    seed = rng.make_seeds(2)[0]
    (image, label) = augment((x, y), seed)
    return (image, label)
train_ds = train_datasets.shuffle(1000).map(f, num_parallel_calls=AUTOTUNE).batch(batch_size).prefetch(AUTOTUNE)
val_ds = val_ds.map(resize_and_rescale, num_parallel_calls=AUTOTUNE).batch(batch_size).prefetch(AUTOTUNE)
test_ds = test_ds.map(resize_and_rescale, num_parallel_calls=AUTOTUNE).batch(batch_size).prefetch(AUTOTUNE)
