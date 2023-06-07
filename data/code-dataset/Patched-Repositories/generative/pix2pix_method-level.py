import tensorflow as tf
import os
import pathlib
import time
import datetime
from matplotlib import pyplot as plt
from IPython import display
import sys
from tool.patching.patching_config import EXPERIMENT_DIR
from tool.measurement.execution import before_execution as before_execution_INSERTED_INTO_SCRIPT
from tool.measurement.execution import after_execution as after_execution_INSERTED_INTO_SCRIPT
from tool.experiment.experiments import ExperimentKinds
experiment_number = sys.argv[1]
experiment_project = sys.argv[2]
EXPERIMENT_FILE_PATH = EXPERIMENT_DIR / ExperimentKinds.METHOD_LEVEL.value / experiment_project / f'experiment-{experiment_number}.json'
dataset_name = 'facades'
_URL = f'http://efrosgans.eecs.berkeley.edu/pix2pix/datasets/{dataset_name}.tar.gz'
start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT()
path_to_zip = tf.keras.utils.get_file(fname=f'{dataset_name}.tar.gz', origin=_URL, extract=True)
after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tensorflow.keras.utils.get_file()', method_object=None, function_args=None, function_kwargs={'fname': f'{dataset_name}.tar.gz', 'origin': _URL, 'extract': True})
path_to_zip = pathlib.Path(path_to_zip)
PATH = path_to_zip.parent / dataset_name
list(PATH.parent.iterdir())
start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT()
sample_image = tf.io.read_file(str(PATH / 'train/1.jpg'))
after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tensorflow.io.read_file()', method_object=None, function_args=[str(PATH / 'train/1.jpg')], function_kwargs=None)
start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT()
sample_image = tf.io.decode_jpeg(sample_image)
after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tensorflow.io.decode_jpeg()', method_object=None, function_args=[sample_image], function_kwargs=None)
print(sample_image.shape)
plt.figure()
plt.imshow(sample_image)

def load(image_file):
    start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT()
    image = tf.io.read_file(image_file)
    after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tensorflow.io.read_file()', method_object=None, function_args=[image_file], function_kwargs=None)
    start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT()
    image = tf.io.decode_jpeg(image)
    after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tensorflow.io.decode_jpeg()', method_object=None, function_args=[image], function_kwargs=None)
    w = tf.shape(image)[1]
    w = w // 2
    input_image = image[:, w:, :]
    real_image = image[:, :w, :]
    start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT()
    input_image = tf.cast(input_image, tf.float32)
    after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tensorflow.cast()', method_object=None, function_args=[input_image, tf.float32], function_kwargs=None)
    start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT()
    real_image = tf.cast(real_image, tf.float32)
    after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tensorflow.cast()', method_object=None, function_args=[real_image, tf.float32], function_kwargs=None)
    return (input_image, real_image)
(inp, re) = load(str(PATH / 'train/100.jpg'))
plt.figure()
plt.imshow(inp / 255.0)
plt.figure()
plt.imshow(re / 255.0)
BUFFER_SIZE = 400
BATCH_SIZE = 1
IMG_WIDTH = 256
IMG_HEIGHT = 256

def resize(input_image, real_image, height, width):
    start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT()
    input_image = tf.image.resize(input_image, [height, width], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tensorflow.image.resize()', method_object=None, function_args=[input_image, [height, width]], function_kwargs={'method': tf.image.ResizeMethod.NEAREST_NEIGHBOR})
    start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT()
    real_image = tf.image.resize(real_image, [height, width], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tensorflow.image.resize()', method_object=None, function_args=[real_image, [height, width]], function_kwargs={'method': tf.image.ResizeMethod.NEAREST_NEIGHBOR})
    return (input_image, real_image)

def random_crop(input_image, real_image):
    start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT()
    stacked_image = tf.stack([input_image, real_image], axis=0)
    after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tensorflow.stack()', method_object=None, function_args=[[input_image, real_image]], function_kwargs={'axis': 0})
    start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT()
    cropped_image = tf.image.random_crop(stacked_image, size=[2, IMG_HEIGHT, IMG_WIDTH, 3])
    after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tensorflow.image.random_crop()', method_object=None, function_args=[stacked_image], function_kwargs={'size': [2, IMG_HEIGHT, IMG_WIDTH, 3]})
    return (cropped_image[0], cropped_image[1])

def normalize(input_image, real_image):
    input_image = input_image / 127.5 - 1
    real_image = real_image / 127.5 - 1
    return (input_image, real_image)

@tf.function()
def random_jitter(input_image, real_image):
    (input_image, real_image) = resize(input_image, real_image, 286, 286)
    (input_image, real_image) = random_crop(input_image, real_image)
    if tf.random.uniform(()) > 0.5:
        start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT()
        input_image = tf.image.flip_left_right(input_image)
        after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tensorflow.image.flip_left_right()', method_object=None, function_args=[input_image], function_kwargs=None)
        start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT()
        real_image = tf.image.flip_left_right(real_image)
        after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tensorflow.image.flip_left_right()', method_object=None, function_args=[real_image], function_kwargs=None)
    return (input_image, real_image)
plt.figure(figsize=(6, 6))
for i in range(4):
    (rj_inp, rj_re) = random_jitter(inp, re)
    plt.subplot(2, 2, i + 1)
    plt.imshow(rj_inp / 255.0)
    plt.axis('off')
plt.show()

def load_image_train(image_file):
    (input_image, real_image) = load(image_file)
    (input_image, real_image) = random_jitter(input_image, real_image)
    (input_image, real_image) = normalize(input_image, real_image)
    return (input_image, real_image)

def load_image_test(image_file):
    (input_image, real_image) = load(image_file)
    (input_image, real_image) = resize(input_image, real_image, IMG_HEIGHT, IMG_WIDTH)
    (input_image, real_image) = normalize(input_image, real_image)
    return (input_image, real_image)
start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT()
train_dataset = tf.data.Dataset.list_files(str(PATH / 'train/*.jpg'))
after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tensorflow.data.Dataset.list_files()', method_object=None, function_args=[str(PATH / 'train/*.jpg')], function_kwargs=None)
start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT()
train_dataset = train_dataset.map(load_image_train, num_parallel_calls=tf.data.AUTOTUNE)
after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tensorflow.data.Dataset.list_files.map()', method_object=train_dataset, function_args=[load_image_train], function_kwargs={'num_parallel_calls': tf.data.AUTOTUNE})
start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT()
train_dataset = train_dataset.shuffle(BUFFER_SIZE)
after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tensorflow.data.Dataset.list_files.map.shuffle()', method_object=train_dataset, function_args=[BUFFER_SIZE], function_kwargs=None)
start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT()
train_dataset = train_dataset.batch(BATCH_SIZE)
after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tensorflow.data.Dataset.list_files.map.shuffle.batch()', method_object=train_dataset, function_args=[BATCH_SIZE], function_kwargs=None)
try:
    start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT()
    test_dataset = tf.data.Dataset.list_files(str(PATH / 'test/*.jpg'))
    after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tensorflow.data.Dataset.list_files()', method_object=None, function_args=[str(PATH / 'test/*.jpg')], function_kwargs=None)
except tf.errors.InvalidArgumentError:
    start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT()
    test_dataset = tf.data.Dataset.list_files(str(PATH / 'val/*.jpg'))
    after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tensorflow.data.Dataset.list_files()', method_object=None, function_args=[str(PATH / 'val/*.jpg')], function_kwargs=None)
start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT()
test_dataset = test_dataset.map(load_image_test)
after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tensorflow.data.Dataset.list_files.map()', method_object=test_dataset, function_args=[load_image_test], function_kwargs=None)
start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT()
test_dataset = test_dataset.batch(BATCH_SIZE)
after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tensorflow.data.Dataset.list_files.map.batch()', method_object=test_dataset, function_args=[BATCH_SIZE], function_kwargs=None)
OUTPUT_CHANNELS = 3

def downsample(filters, size, apply_batchnorm=True):
    start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT()
    initializer = tf.random_normal_initializer(0.0, 0.02)
    after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tensorflow.random_normal_initializer()', method_object=None, function_args=[0.0, 0.02], function_kwargs=None)
    start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT()
    result = tf.keras.Sequential()
    after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tensorflow.keras.Sequential()', method_object=None, function_args=None, function_kwargs=None)
    start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT()
    result.add(tf.keras.layers.Conv2D(filters, size, strides=2, padding='same', kernel_initializer=initializer, use_bias=False))
    after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tensorflow.keras.Sequential.add()', method_object=result, function_args=[tf.keras.layers.Conv2D(filters, size, strides=2, padding='same', kernel_initializer=initializer, use_bias=False)], function_kwargs=None)
    if apply_batchnorm:
        start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT()
        result.add(tf.keras.layers.BatchNormalization())
        after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tensorflow.keras.Sequential.add()', method_object=result, function_args=[tf.keras.layers.BatchNormalization()], function_kwargs=None)
    start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT()
    result.add(tf.keras.layers.LeakyReLU())
    after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tensorflow.keras.Sequential.add()', method_object=result, function_args=[tf.keras.layers.LeakyReLU()], function_kwargs=None)
    return result
down_model = downsample(3, 4)
down_result = down_model(tf.expand_dims(inp, 0))
print(down_result.shape)

def upsample(filters, size, apply_dropout=False):
    start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT()
    initializer = tf.random_normal_initializer(0.0, 0.02)
    after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tensorflow.random_normal_initializer()', method_object=None, function_args=[0.0, 0.02], function_kwargs=None)
    start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT()
    result = tf.keras.Sequential()
    after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tensorflow.keras.Sequential()', method_object=None, function_args=None, function_kwargs=None)
    start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT()
    result.add(tf.keras.layers.Conv2DTranspose(filters, size, strides=2, padding='same', kernel_initializer=initializer, use_bias=False))
    after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tensorflow.keras.Sequential.add()', method_object=result, function_args=[tf.keras.layers.Conv2DTranspose(filters, size, strides=2, padding='same', kernel_initializer=initializer, use_bias=False)], function_kwargs=None)
    start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT()
    result.add(tf.keras.layers.BatchNormalization())
    after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tensorflow.keras.Sequential.add()', method_object=result, function_args=[tf.keras.layers.BatchNormalization()], function_kwargs=None)
    if apply_dropout:
        start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT()
        result.add(tf.keras.layers.Dropout(0.5))
        after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tensorflow.keras.Sequential.add()', method_object=result, function_args=[tf.keras.layers.Dropout(0.5)], function_kwargs=None)
    start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT()
    result.add(tf.keras.layers.ReLU())
    after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tensorflow.keras.Sequential.add()', method_object=result, function_args=[tf.keras.layers.ReLU()], function_kwargs=None)
    return result
up_model = upsample(3, 4)
up_result = up_model(down_result)
print(up_result.shape)

def Generator():
    start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT()
    inputs = tf.keras.layers.Input(shape=[256, 256, 3])
    after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tensorflow.keras.layers.Input()', method_object=None, function_args=None, function_kwargs={'shape': [256, 256, 3]})
    down_stack = [downsample(64, 4, apply_batchnorm=False), downsample(128, 4), downsample(256, 4), downsample(512, 4), downsample(512, 4), downsample(512, 4), downsample(512, 4), downsample(512, 4)]
    up_stack = [upsample(512, 4, apply_dropout=True), upsample(512, 4, apply_dropout=True), upsample(512, 4, apply_dropout=True), upsample(512, 4), upsample(256, 4), upsample(128, 4), upsample(64, 4)]
    start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT()
    initializer = tf.random_normal_initializer(0.0, 0.02)
    after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tensorflow.random_normal_initializer()', method_object=None, function_args=[0.0, 0.02], function_kwargs=None)
    start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT()
    last = tf.keras.layers.Conv2DTranspose(OUTPUT_CHANNELS, 4, strides=2, padding='same', kernel_initializer=initializer, activation='tanh')
    after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tensorflow.keras.layers.Conv2DTranspose()', method_object=None, function_args=[OUTPUT_CHANNELS, 4], function_kwargs={'strides': 2, 'padding': 'same', 'kernel_initializer': initializer, 'activation': 'tanh'})
    x = inputs
    skips = []
    for down in down_stack:
        x = down(x)
        skips.append(x)
    skips = reversed(skips[:-1])
    for (up, skip) in zip(up_stack, skips):
        x = up(x)
        start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT()
        x = tf.keras.layers.Concatenate()([x, skip])
        after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tensorflow.keras.layers.Concatenate()()', method_object=None, function_args=[[x, skip]], function_kwargs=None)
    start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT()
    x = last(x)
    after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tensorflow.keras.layers.Conv2DTranspose()', method_object=last, function_args=[x], function_kwargs=None)
    return tf.keras.Model(inputs=inputs, outputs=x)
generator = Generator()
start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT()
tf.keras.utils.plot_model(generator, show_shapes=True, dpi=64)
after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tensorflow.keras.utils.plot_model()', method_object=None, function_args=[generator], function_kwargs={'show_shapes': True, 'dpi': 64})
gen_output = generator(inp[tf.newaxis, ...], training=False)
plt.imshow(gen_output[0, ...])
LAMBDA = 100
start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT()
loss_object = tf.keras.losses.BinaryCrossentropy(from_logits=True)
after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tensorflow.keras.losses.BinaryCrossentropy()', method_object=None, function_args=None, function_kwargs={'from_logits': True})

def generator_loss(disc_generated_output, gen_output, target):
    start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT()
    gan_loss = loss_object(tf.ones_like(disc_generated_output), disc_generated_output)
    after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tensorflow.keras.losses.BinaryCrossentropy()', method_object=loss_object, function_args=[tf.ones_like(disc_generated_output), disc_generated_output], function_kwargs=None)
    start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT()
    l1_loss = tf.reduce_mean(tf.abs(target - gen_output))
    after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tensorflow.reduce_mean()', method_object=None, function_args=[tf.abs(target - gen_output)], function_kwargs=None)
    total_gen_loss = gan_loss + LAMBDA * l1_loss
    return (total_gen_loss, gan_loss, l1_loss)

def Discriminator():
    start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT()
    initializer = tf.random_normal_initializer(0.0, 0.02)
    after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tensorflow.random_normal_initializer()', method_object=None, function_args=[0.0, 0.02], function_kwargs=None)
    start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT()
    inp = tf.keras.layers.Input(shape=[256, 256, 3], name='input_image')
    after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tensorflow.keras.layers.Input()', method_object=None, function_args=None, function_kwargs={'shape': [256, 256, 3], 'name': 'input_image'})
    start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT()
    tar = tf.keras.layers.Input(shape=[256, 256, 3], name='target_image')
    after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tensorflow.keras.layers.Input()', method_object=None, function_args=None, function_kwargs={'shape': [256, 256, 3], 'name': 'target_image'})
    start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT()
    x = tf.keras.layers.concatenate([inp, tar])
    after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tensorflow.keras.layers.concatenate()', method_object=None, function_args=[[inp, tar]], function_kwargs=None)
    down1 = downsample(64, 4, False)(x)
    down2 = downsample(128, 4)(down1)
    down3 = downsample(256, 4)(down2)
    start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT()
    zero_pad1 = tf.keras.layers.ZeroPadding2D()(down3)
    after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tensorflow.keras.layers.ZeroPadding2D()()', method_object=None, function_args=[down3], function_kwargs=None)
    start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT()
    conv = tf.keras.layers.Conv2D(512, 4, strides=1, kernel_initializer=initializer, use_bias=False)(zero_pad1)
    after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tensorflow.keras.layers.Conv2D(512, 4, strides=1, kernel_initializer=initializer, use_bias=False)()', method_object=None, function_args=[zero_pad1], function_kwargs=None)
    start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT()
    batchnorm1 = tf.keras.layers.BatchNormalization()(conv)
    after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tensorflow.keras.layers.BatchNormalization()()', method_object=None, function_args=[conv], function_kwargs=None)
    start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT()
    leaky_relu = tf.keras.layers.LeakyReLU()(batchnorm1)
    after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tensorflow.keras.layers.LeakyReLU()()', method_object=None, function_args=[batchnorm1], function_kwargs=None)
    start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT()
    zero_pad2 = tf.keras.layers.ZeroPadding2D()(leaky_relu)
    after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tensorflow.keras.layers.ZeroPadding2D()()', method_object=None, function_args=[leaky_relu], function_kwargs=None)
    start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT()
    last = tf.keras.layers.Conv2D(1, 4, strides=1, kernel_initializer=initializer)(zero_pad2)
    after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tensorflow.keras.layers.Conv2D(1, 4, strides=1, kernel_initializer=initializer)()', method_object=None, function_args=[zero_pad2], function_kwargs=None)
    return tf.keras.Model(inputs=[inp, tar], outputs=last)
discriminator = Discriminator()
start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT()
tf.keras.utils.plot_model(discriminator, show_shapes=True, dpi=64)
after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tensorflow.keras.utils.plot_model()', method_object=None, function_args=[discriminator], function_kwargs={'show_shapes': True, 'dpi': 64})
disc_out = discriminator([inp[tf.newaxis, ...], gen_output], training=False)
plt.imshow(disc_out[0, ..., -1], vmin=-20, vmax=20, cmap='RdBu_r')
plt.colorbar()

def discriminator_loss(disc_real_output, disc_generated_output):
    start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT()
    real_loss = loss_object(tf.ones_like(disc_real_output), disc_real_output)
    after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tensorflow.keras.losses.BinaryCrossentropy()', method_object=loss_object, function_args=[tf.ones_like(disc_real_output), disc_real_output], function_kwargs=None)
    start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT()
    generated_loss = loss_object(tf.zeros_like(disc_generated_output), disc_generated_output)
    after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tensorflow.keras.losses.BinaryCrossentropy()', method_object=loss_object, function_args=[tf.zeros_like(disc_generated_output), disc_generated_output], function_kwargs=None)
    total_disc_loss = real_loss + generated_loss
    return total_disc_loss
start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT()
generator_optimizer = tf.keras.optimizers.Adam(0.0002, beta_1=0.5)
after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tensorflow.keras.optimizers.Adam()', method_object=None, function_args=[0.0002], function_kwargs={'beta_1': 0.5})
start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT()
discriminator_optimizer = tf.keras.optimizers.Adam(0.0002, beta_1=0.5)
after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tensorflow.keras.optimizers.Adam()', method_object=None, function_args=[0.0002], function_kwargs={'beta_1': 0.5})
checkpoint_dir = './training_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, 'ckpt')
start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT()
checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer, discriminator_optimizer=discriminator_optimizer, generator=generator, discriminator=discriminator)
after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tensorflow.train.Checkpoint()', method_object=None, function_args=None, function_kwargs={'generator_optimizer': generator_optimizer, 'discriminator_optimizer': discriminator_optimizer, 'generator': generator, 'discriminator': discriminator})

def generate_images(model, test_input, tar):
    prediction = model(test_input, training=True)
    plt.figure(figsize=(15, 15))
    display_list = [test_input[0], tar[0], prediction[0]]
    title = ['Input Image', 'Ground Truth', 'Predicted Image']
    for i in range(3):
        plt.subplot(1, 3, i + 1)
        plt.title(title[i])
        plt.imshow(display_list[i] * 0.5 + 0.5)
        plt.axis('off')
    plt.show()
for (example_input, example_target) in test_dataset.take(1):
    generate_images(generator, example_input, example_target)
log_dir = 'logs/'
start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT()
summary_writer = tf.summary.create_file_writer(log_dir + 'fit/' + datetime.datetime.now().strftime('%Y%m%d-%H%M%S'))
after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tensorflow.summary.create_file_writer()', method_object=None, function_args=[log_dir + 'fit/' + datetime.datetime.now().strftime('%Y%m%d-%H%M%S')], function_kwargs=None)

@tf.function
def train_step(input_image, target, step):
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        gen_output = generator(input_image, training=True)
        disc_real_output = discriminator([input_image, target], training=True)
        disc_generated_output = discriminator([input_image, gen_output], training=True)
        (gen_total_loss, gen_gan_loss, gen_l1_loss) = generator_loss(disc_generated_output, gen_output, target)
        disc_loss = discriminator_loss(disc_real_output, disc_generated_output)
    generator_gradients = gen_tape.gradient(gen_total_loss, generator.trainable_variables)
    discriminator_gradients = disc_tape.gradient(disc_loss, discriminator.trainable_variables)
    start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT()
    generator_optimizer.apply_gradients(zip(generator_gradients, generator.trainable_variables))
    after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tensorflow.keras.optimizers.Adam.apply_gradients()', method_object=generator_optimizer, function_args=[zip(generator_gradients, generator.trainable_variables)], function_kwargs=None)
    start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT()
    discriminator_optimizer.apply_gradients(zip(discriminator_gradients, discriminator.trainable_variables))
    after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tensorflow.keras.optimizers.Adam.apply_gradients()', method_object=discriminator_optimizer, function_args=[zip(discriminator_gradients, discriminator.trainable_variables)], function_kwargs=None)
    with summary_writer.as_default():
        start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT()
        tf.summary.scalar('gen_total_loss', gen_total_loss, step=step // 1000)
        after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tensorflow.summary.scalar()', method_object=None, function_args=['gen_total_loss', gen_total_loss], function_kwargs={'step': step // 1000})
        start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT()
        tf.summary.scalar('gen_gan_loss', gen_gan_loss, step=step // 1000)
        after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tensorflow.summary.scalar()', method_object=None, function_args=['gen_gan_loss', gen_gan_loss], function_kwargs={'step': step // 1000})
        start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT()
        tf.summary.scalar('gen_l1_loss', gen_l1_loss, step=step // 1000)
        after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tensorflow.summary.scalar()', method_object=None, function_args=['gen_l1_loss', gen_l1_loss], function_kwargs={'step': step // 1000})
        start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT()
        tf.summary.scalar('disc_loss', disc_loss, step=step // 1000)
        after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tensorflow.summary.scalar()', method_object=None, function_args=['disc_loss', disc_loss], function_kwargs={'step': step // 1000})

def fit(train_ds, test_ds, steps):
    (example_input, example_target) = next(iter(test_ds.take(1)))
    start = time.time()
    for (step, (input_image, target)) in train_ds.repeat().take(steps).enumerate():
        if step % 1000 == 0:
            display.clear_output(wait=True)
            if step != 0:
                print(f'Time taken for 1000 steps: {time.time() - start:.2f} sec\n')
            start = time.time()
            generate_images(generator, example_input, example_target)
            print(f'Step: {step // 1000}k')
        train_step(input_image, target, step)
        if (step + 1) % 10 == 0:
            print('.', end='', flush=True)
        if (step + 1) % 5000 == 0:
            start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT()
            checkpoint.save(file_prefix=checkpoint_prefix)
            after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tensorflow.train.Checkpoint.save()', method_object=checkpoint, function_args=None, function_kwargs={'file_prefix': checkpoint_prefix})
fit(train_dataset, test_dataset, steps=40000)
start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT()
checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))
after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tensorflow.train.Checkpoint.restore()', method_object=checkpoint, function_args=[tf.train.latest_checkpoint(checkpoint_dir)], function_kwargs=None)
for (inp, tar) in test_dataset.take(5):
    generate_images(generator, inp, tar)
