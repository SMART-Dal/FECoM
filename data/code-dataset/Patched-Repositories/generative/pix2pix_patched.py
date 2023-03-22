import tensorflow as tf
import os
import pathlib
import time
import datetime
from matplotlib import pyplot as plt
from IPython import display
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
dataset_name = 'facades'
_URL = f'http://efrosgans.eecs.berkeley.edu/pix2pix/datasets/{dataset_name}.tar.gz'
path_to_zip = custom_method(
tf.keras.utils.get_file(fname=f'{dataset_name}.tar.gz', origin=_URL, extract=True), imports='from IPython import display;import tensorflow as tf;from matplotlib import pyplot as plt;import datetime;import os;import pathlib;import time', function_to_run='tf.keras.utils.get_file(**kwargs)', method_object=None, object_signature=None, function_args=[], function_kwargs={'fname': eval('f"{dataset_name}.tar.gz"'), 'origin': eval('_URL'), 'extract': eval('True')}, max_wait_secs=0)
path_to_zip = pathlib.Path(path_to_zip)
PATH = path_to_zip.parent / dataset_name
list(PATH.parent.iterdir())
sample_image = custom_method(
tf.io.read_file(str(PATH / 'train/1.jpg')), imports='from IPython import display;import tensorflow as tf;from matplotlib import pyplot as plt;import datetime;import os;import pathlib;import time', function_to_run='tf.io.read_file(*args)', method_object=None, object_signature=None, function_args=[eval("str(PATH / 'train/1.jpg')")], function_kwargs={}, max_wait_secs=0)
sample_image = custom_method(
tf.io.decode_jpeg(sample_image), imports='from IPython import display;import tensorflow as tf;from matplotlib import pyplot as plt;import datetime;import os;import pathlib;import time', function_to_run='tf.io.decode_jpeg(*args)', method_object=None, object_signature=None, function_args=[eval('sample_image')], function_kwargs={}, max_wait_secs=0)
print(sample_image.shape)
plt.figure()
plt.imshow(sample_image)

def load(image_file):
    image = custom_method(
    tf.io.read_file(image_file), imports='from IPython import display;import tensorflow as tf;from matplotlib import pyplot as plt;import datetime;import os;import pathlib;import time', function_to_run='tf.io.read_file(*args)', method_object=None, object_signature=None, function_args=[eval('image_file')], function_kwargs={}, max_wait_secs=0)
    image = custom_method(
    tf.io.decode_jpeg(image), imports='from IPython import display;import tensorflow as tf;from matplotlib import pyplot as plt;import datetime;import os;import pathlib;import time', function_to_run='tf.io.decode_jpeg(*args)', method_object=None, object_signature=None, function_args=[eval('image')], function_kwargs={}, max_wait_secs=0)
    w = custom_method(
    tf.shape(image), imports='from IPython import display;import tensorflow as tf;from matplotlib import pyplot as plt;import datetime;import os;import pathlib;import time', function_to_run='tf.shape(*args)', method_object=None, object_signature=None, function_args=[eval('image')], function_kwargs={}, max_wait_secs=0)[1]
    w = w // 2
    input_image = image[:, w:, :]
    real_image = image[:, :w, :]
    input_image = custom_method(
    tf.cast(input_image, tf.float32), imports='from IPython import display;import tensorflow as tf;from matplotlib import pyplot as plt;import datetime;import os;import pathlib;import time', function_to_run='tf.cast(*args)', method_object=None, object_signature=None, function_args=[eval('input_image'), eval('tf.float32')], function_kwargs={}, max_wait_secs=0)
    real_image = custom_method(
    tf.cast(real_image, tf.float32), imports='from IPython import display;import tensorflow as tf;from matplotlib import pyplot as plt;import datetime;import os;import pathlib;import time', function_to_run='tf.cast(*args)', method_object=None, object_signature=None, function_args=[eval('real_image'), eval('tf.float32')], function_kwargs={}, max_wait_secs=0)
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
    input_image = custom_method(
    tf.image.resize(input_image, [height, width], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR), imports='from IPython import display;import tensorflow as tf;from matplotlib import pyplot as plt;import datetime;import os;import pathlib;import time', function_to_run='tf.image.resize(*args, **kwargs)', method_object=None, object_signature=None, function_args=[eval('input_image'), eval('[height, width]')], function_kwargs={'method': eval('tf.image.ResizeMethod.NEAREST_NEIGHBOR')}, max_wait_secs=0)
    real_image = custom_method(
    tf.image.resize(real_image, [height, width], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR), imports='from IPython import display;import tensorflow as tf;from matplotlib import pyplot as plt;import datetime;import os;import pathlib;import time', function_to_run='tf.image.resize(*args, **kwargs)', method_object=None, object_signature=None, function_args=[eval('real_image'), eval('[height, width]')], function_kwargs={'method': eval('tf.image.ResizeMethod.NEAREST_NEIGHBOR')}, max_wait_secs=0)
    return (input_image, real_image)

def random_crop(input_image, real_image):
    stacked_image = custom_method(
    tf.stack([input_image, real_image], axis=0), imports='from IPython import display;import tensorflow as tf;from matplotlib import pyplot as plt;import datetime;import os;import pathlib;import time', function_to_run='tf.stack(*args, **kwargs)', method_object=None, object_signature=None, function_args=[eval('[input_image, real_image]')], function_kwargs={'axis': eval('0')}, max_wait_secs=0)
    cropped_image = custom_method(
    tf.image.random_crop(stacked_image, size=[2, IMG_HEIGHT, IMG_WIDTH, 3]), imports='from IPython import display;import tensorflow as tf;from matplotlib import pyplot as plt;import datetime;import os;import pathlib;import time', function_to_run='tf.image.random_crop(*args, **kwargs)', method_object=None, object_signature=None, function_args=[eval('stacked_image')], function_kwargs={'size': eval('[2, IMG_HEIGHT, IMG_WIDTH, 3]')}, max_wait_secs=0)
    return (cropped_image[0], cropped_image[1])

def normalize(input_image, real_image):
    input_image = input_image / 127.5 - 1
    real_image = real_image / 127.5 - 1
    return (input_image, real_image)

@custom_method(
tf.function(), imports='from IPython import display;import tensorflow as tf;from matplotlib import pyplot as plt;import datetime;import os;import pathlib;import time', function_to_run='tf.function()', method_object=None, object_signature=None, function_args=[], function_kwargs={}, max_wait_secs=0)
def random_jitter(input_image, real_image):
    (input_image, real_image) = resize(input_image, real_image, 286, 286)
    (input_image, real_image) = random_crop(input_image, real_image)
    if custom_method(
    tf.random.uniform(()), imports='from IPython import display;import tensorflow as tf;from matplotlib import pyplot as plt;import datetime;import os;import pathlib;import time', function_to_run='tf.random.uniform(*args)', method_object=None, object_signature=None, function_args=[eval('()')], function_kwargs={}, max_wait_secs=0) > 0.5:
        input_image = custom_method(
        tf.image.flip_left_right(input_image), imports='from IPython import display;import tensorflow as tf;from matplotlib import pyplot as plt;import datetime;import os;import pathlib;import time', function_to_run='tf.image.flip_left_right(*args)', method_object=None, object_signature=None, function_args=[eval('input_image')], function_kwargs={}, max_wait_secs=0)
        real_image = custom_method(
        tf.image.flip_left_right(real_image), imports='from IPython import display;import tensorflow as tf;from matplotlib import pyplot as plt;import datetime;import os;import pathlib;import time', function_to_run='tf.image.flip_left_right(*args)', method_object=None, object_signature=None, function_args=[eval('real_image')], function_kwargs={}, max_wait_secs=0)
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
train_dataset = custom_method(
tf.data.Dataset.list_files(str(PATH / 'train/*.jpg')), imports='from IPython import display;import tensorflow as tf;from matplotlib import pyplot as plt;import datetime;import os;import pathlib;import time', function_to_run='tf.data.Dataset.list_files(*args)', method_object=None, object_signature=None, function_args=[eval("str(PATH / 'train/*.jpg')")], function_kwargs={}, max_wait_secs=0)
train_dataset = custom_method(
train_dataset.map(load_image_train, num_parallel_calls=tf.data.AUTOTUNE), imports='from IPython import display;import tensorflow as tf;from matplotlib import pyplot as plt;import datetime;import os;import pathlib;import time', function_to_run='obj.map(*args, **kwargs)', method_object=eval('train_dataset'), object_signature='tf.data.Dataset.list_files', function_args=[eval('load_image_train')], function_kwargs={'num_parallel_calls': eval('tf.data.AUTOTUNE')}, max_wait_secs=0, custom_class=None)
train_dataset = custom_method(
train_dataset.shuffle(BUFFER_SIZE), imports='from IPython import display;import tensorflow as tf;from matplotlib import pyplot as plt;import datetime;import os;import pathlib;import time', function_to_run='obj.shuffle(*args)', method_object=eval('train_dataset'), object_signature='tf.data.Dataset.list_files', function_args=[eval('BUFFER_SIZE')], function_kwargs={}, max_wait_secs=0, custom_class=None)
train_dataset = custom_method(
train_dataset.batch(BATCH_SIZE), imports='from IPython import display;import tensorflow as tf;from matplotlib import pyplot as plt;import datetime;import os;import pathlib;import time', function_to_run='obj.batch(*args)', method_object=eval('train_dataset'), object_signature='tf.data.Dataset.list_files', function_args=[eval('BATCH_SIZE')], function_kwargs={}, max_wait_secs=0, custom_class=None)
try:
    test_dataset = custom_method(
    tf.data.Dataset.list_files(str(PATH / 'test/*.jpg')), imports='from IPython import display;import tensorflow as tf;from matplotlib import pyplot as plt;import datetime;import os;import pathlib;import time', function_to_run='tf.data.Dataset.list_files(*args)', method_object=None, object_signature=None, function_args=[eval("str(PATH / 'test/*.jpg')")], function_kwargs={}, max_wait_secs=0)
except tf.errors.InvalidArgumentError:
    test_dataset = custom_method(
    tf.data.Dataset.list_files(str(PATH / 'val/*.jpg')), imports='from IPython import display;import tensorflow as tf;from matplotlib import pyplot as plt;import datetime;import os;import pathlib;import time', function_to_run='tf.data.Dataset.list_files(*args)', method_object=None, object_signature=None, function_args=[eval("str(PATH / 'val/*.jpg')")], function_kwargs={}, max_wait_secs=0)
test_dataset = custom_method(
test_dataset.map(load_image_test), imports='from IPython import display;import tensorflow as tf;from matplotlib import pyplot as plt;import datetime;import os;import pathlib;import time', function_to_run='obj.map(*args)', method_object=eval('test_dataset'), object_signature='tf.data.Dataset.list_files', function_args=[eval('load_image_test')], function_kwargs={}, max_wait_secs=0, custom_class=None)
test_dataset = custom_method(
test_dataset.batch(BATCH_SIZE), imports='from IPython import display;import tensorflow as tf;from matplotlib import pyplot as plt;import datetime;import os;import pathlib;import time', function_to_run='obj.batch(*args)', method_object=eval('test_dataset'), object_signature='tf.data.Dataset.list_files', function_args=[eval('BATCH_SIZE')], function_kwargs={}, max_wait_secs=0, custom_class=None)
OUTPUT_CHANNELS = 3

def downsample(filters, size, apply_batchnorm=True):
    initializer = custom_method(
    tf.random_normal_initializer(0.0, 0.02), imports='from IPython import display;import tensorflow as tf;from matplotlib import pyplot as plt;import datetime;import os;import pathlib;import time', function_to_run='tf.random_normal_initializer(*args)', method_object=None, object_signature=None, function_args=[eval('0.'), eval('0.02')], function_kwargs={}, max_wait_secs=0)
    result = custom_method(
    tf.keras.Sequential(), imports='from IPython import display;import tensorflow as tf;from matplotlib import pyplot as plt;import datetime;import os;import pathlib;import time', function_to_run='tf.keras.Sequential()', method_object=None, object_signature=None, function_args=[], function_kwargs={}, max_wait_secs=0)
    custom_method(
    result.add(tf.keras.layers.Conv2D(filters, size, strides=2, padding='same', kernel_initializer=initializer, use_bias=False)), imports='from IPython import display;import tensorflow as tf;from matplotlib import pyplot as plt;import datetime;import os;import pathlib;import time', function_to_run='obj.add(*args)', method_object=eval('result'), object_signature='tf.keras.Sequential', function_args=[eval("tf.keras.layers.Conv2D(filters, size, strides=2, padding='same',\n                             kernel_initializer=initializer, use_bias=False)")], function_kwargs={}, max_wait_secs=0, custom_class=None)
    if apply_batchnorm:
        custom_method(
        result.add(tf.keras.layers.BatchNormalization()), imports='from IPython import display;import tensorflow as tf;from matplotlib import pyplot as plt;import datetime;import os;import pathlib;import time', function_to_run='obj.add(*args)', method_object=eval('result'), object_signature='tf.keras.Sequential', function_args=[eval('tf.keras.layers.BatchNormalization()')], function_kwargs={}, max_wait_secs=0, custom_class=None)
    custom_method(
    result.add(tf.keras.layers.LeakyReLU()), imports='from IPython import display;import tensorflow as tf;from matplotlib import pyplot as plt;import datetime;import os;import pathlib;import time', function_to_run='obj.add(*args)', method_object=eval('result'), object_signature='tf.keras.Sequential', function_args=[eval('tf.keras.layers.LeakyReLU()')], function_kwargs={}, max_wait_secs=0, custom_class=None)
    return result
down_model = downsample(3, 4)
down_result = down_model(tf.expand_dims(inp, 0))
print(down_result.shape)

def upsample(filters, size, apply_dropout=False):
    initializer = custom_method(
    tf.random_normal_initializer(0.0, 0.02), imports='from IPython import display;import tensorflow as tf;from matplotlib import pyplot as plt;import datetime;import os;import pathlib;import time', function_to_run='tf.random_normal_initializer(*args)', method_object=None, object_signature=None, function_args=[eval('0.'), eval('0.02')], function_kwargs={}, max_wait_secs=0)
    result = custom_method(
    tf.keras.Sequential(), imports='from IPython import display;import tensorflow as tf;from matplotlib import pyplot as plt;import datetime;import os;import pathlib;import time', function_to_run='tf.keras.Sequential()', method_object=None, object_signature=None, function_args=[], function_kwargs={}, max_wait_secs=0)
    custom_method(
    result.add(tf.keras.layers.Conv2DTranspose(filters, size, strides=2, padding='same', kernel_initializer=initializer, use_bias=False)), imports='from IPython import display;import tensorflow as tf;from matplotlib import pyplot as plt;import datetime;import os;import pathlib;import time', function_to_run='obj.add(*args)', method_object=eval('result'), object_signature='tf.keras.Sequential', function_args=[eval("tf.keras.layers.Conv2DTranspose(filters, size, strides=2,\n                                    padding='same',\n                                    kernel_initializer=initializer,\n                                    use_bias=False)")], function_kwargs={}, max_wait_secs=0, custom_class=None)
    custom_method(
    result.add(tf.keras.layers.BatchNormalization()), imports='from IPython import display;import tensorflow as tf;from matplotlib import pyplot as plt;import datetime;import os;import pathlib;import time', function_to_run='obj.add(*args)', method_object=eval('result'), object_signature='tf.keras.Sequential', function_args=[eval('tf.keras.layers.BatchNormalization()')], function_kwargs={}, max_wait_secs=0, custom_class=None)
    if apply_dropout:
        custom_method(
        result.add(tf.keras.layers.Dropout(0.5)), imports='from IPython import display;import tensorflow as tf;from matplotlib import pyplot as plt;import datetime;import os;import pathlib;import time', function_to_run='obj.add(*args)', method_object=eval('result'), object_signature='tf.keras.Sequential', function_args=[eval('tf.keras.layers.Dropout(0.5)')], function_kwargs={}, max_wait_secs=0, custom_class=None)
    custom_method(
    result.add(tf.keras.layers.ReLU()), imports='from IPython import display;import tensorflow as tf;from matplotlib import pyplot as plt;import datetime;import os;import pathlib;import time', function_to_run='obj.add(*args)', method_object=eval('result'), object_signature='tf.keras.Sequential', function_args=[eval('tf.keras.layers.ReLU()')], function_kwargs={}, max_wait_secs=0, custom_class=None)
    return result
up_model = upsample(3, 4)
up_result = up_model(down_result)
print(up_result.shape)

def Generator():
    inputs = custom_method(
    tf.keras.layers.Input(shape=[256, 256, 3]), imports='from IPython import display;import tensorflow as tf;from matplotlib import pyplot as plt;import datetime;import os;import pathlib;import time', function_to_run='tf.keras.layers.Input(**kwargs)', method_object=None, object_signature=None, function_args=[], function_kwargs={'shape': eval('[256, 256, 3]')}, max_wait_secs=0)
    down_stack = [downsample(64, 4, apply_batchnorm=False), downsample(128, 4), downsample(256, 4), downsample(512, 4), downsample(512, 4), downsample(512, 4), downsample(512, 4), downsample(512, 4)]
    up_stack = [upsample(512, 4, apply_dropout=True), upsample(512, 4, apply_dropout=True), upsample(512, 4, apply_dropout=True), upsample(512, 4), upsample(256, 4), upsample(128, 4), upsample(64, 4)]
    initializer = custom_method(
    tf.random_normal_initializer(0.0, 0.02), imports='from IPython import display;import tensorflow as tf;from matplotlib import pyplot as plt;import datetime;import os;import pathlib;import time', function_to_run='tf.random_normal_initializer(*args)', method_object=None, object_signature=None, function_args=[eval('0.'), eval('0.02')], function_kwargs={}, max_wait_secs=0)
    last = custom_method(
    tf.keras.layers.Conv2DTranspose(OUTPUT_CHANNELS, 4, strides=2, padding='same', kernel_initializer=initializer, activation='tanh'), imports='from IPython import display;import tensorflow as tf;from matplotlib import pyplot as plt;import datetime;import os;import pathlib;import time', function_to_run='tf.keras.layers.Conv2DTranspose(*args, **kwargs)', method_object=None, object_signature=None, function_args=[eval('OUTPUT_CHANNELS'), eval('4')], function_kwargs={'strides': eval('2'), 'padding': eval("'same'"), 'kernel_initializer': eval('initializer'), 'activation': eval("'tanh'")}, max_wait_secs=0)
    x = inputs
    skips = []
    for down in down_stack:
        x = down(x)
        skips.append(x)
    skips = reversed(skips[:-1])
    for (up, skip) in zip(up_stack, skips):
        x = up(x)
        x = custom_method(
        tf.keras.layers.Concatenate()([x, skip]), imports='from IPython import display;import tensorflow as tf;from matplotlib import pyplot as plt;import datetime;import os;import pathlib;import time', function_to_run='tf.keras.layers.Concatenate()(*args)', method_object=None, object_signature=None, function_args=[eval('[x, skip]')], function_kwargs={}, max_wait_secs=0)
    x = custom_method(
    last(x), imports='from IPython import display;import tensorflow as tf;from matplotlib import pyplot as plt;import datetime;import os;import pathlib;import time', function_to_run='obj(*args)', method_object=eval('last'), object_signature='tf.keras.layers.Conv2DTranspose', function_args=[eval('x')], function_kwargs={}, max_wait_secs=0, custom_class=None)
    return custom_method(
    tf.keras.Model(inputs=inputs, outputs=x), imports='from IPython import display;import tensorflow as tf;from matplotlib import pyplot as plt;import datetime;import os;import pathlib;import time', function_to_run='tf.keras.Model(**kwargs)', method_object=None, object_signature=None, function_args=[], function_kwargs={'inputs': eval('inputs'), 'outputs': eval('x')}, max_wait_secs=0)
generator = Generator()
custom_method(
tf.keras.utils.plot_model(generator, show_shapes=True, dpi=64), imports='from IPython import display;import tensorflow as tf;from matplotlib import pyplot as plt;import datetime;import os;import pathlib;import time', function_to_run='tf.keras.utils.plot_model(*args, **kwargs)', method_object=None, object_signature=None, function_args=[eval('generator')], function_kwargs={'show_shapes': eval('True'), 'dpi': eval('64')}, max_wait_secs=0)
gen_output = generator(inp[tf.newaxis, ...], training=False)
plt.imshow(gen_output[0, ...])
LAMBDA = 100
loss_object = custom_method(
tf.keras.losses.BinaryCrossentropy(from_logits=True), imports='from IPython import display;import tensorflow as tf;from matplotlib import pyplot as plt;import datetime;import os;import pathlib;import time', function_to_run='tf.keras.losses.BinaryCrossentropy(**kwargs)', method_object=None, object_signature=None, function_args=[], function_kwargs={'from_logits': eval('True')}, max_wait_secs=0)

def generator_loss(disc_generated_output, gen_output, target):
    gan_loss = custom_method(
    loss_object(tf.ones_like(disc_generated_output), disc_generated_output), imports='from IPython import display;import tensorflow as tf;from matplotlib import pyplot as plt;import datetime;import os;import pathlib;import time', function_to_run='obj(*args)', method_object=eval('loss_object'), object_signature='tf.keras.losses.BinaryCrossentropy', function_args=[eval('tf.ones_like(disc_generated_output)'), eval('disc_generated_output')], function_kwargs={}, max_wait_secs=0, custom_class=None)
    l1_loss = custom_method(
    tf.reduce_mean(tf.abs(target - gen_output)), imports='from IPython import display;import tensorflow as tf;from matplotlib import pyplot as plt;import datetime;import os;import pathlib;import time', function_to_run='tf.reduce_mean(*args)', method_object=None, object_signature=None, function_args=[eval('tf.abs(target - gen_output)')], function_kwargs={}, max_wait_secs=0)
    total_gen_loss = gan_loss + LAMBDA * l1_loss
    return (total_gen_loss, gan_loss, l1_loss)

def Discriminator():
    initializer = custom_method(
    tf.random_normal_initializer(0.0, 0.02), imports='from IPython import display;import tensorflow as tf;from matplotlib import pyplot as plt;import datetime;import os;import pathlib;import time', function_to_run='tf.random_normal_initializer(*args)', method_object=None, object_signature=None, function_args=[eval('0.'), eval('0.02')], function_kwargs={}, max_wait_secs=0)
    inp = custom_method(
    tf.keras.layers.Input(shape=[256, 256, 3], name='input_image'), imports='from IPython import display;import tensorflow as tf;from matplotlib import pyplot as plt;import datetime;import os;import pathlib;import time', function_to_run='tf.keras.layers.Input(**kwargs)', method_object=None, object_signature=None, function_args=[], function_kwargs={'shape': eval('[256, 256, 3]'), 'name': eval("'input_image'")}, max_wait_secs=0)
    tar = custom_method(
    tf.keras.layers.Input(shape=[256, 256, 3], name='target_image'), imports='from IPython import display;import tensorflow as tf;from matplotlib import pyplot as plt;import datetime;import os;import pathlib;import time', function_to_run='tf.keras.layers.Input(**kwargs)', method_object=None, object_signature=None, function_args=[], function_kwargs={'shape': eval('[256, 256, 3]'), 'name': eval("'target_image'")}, max_wait_secs=0)
    x = custom_method(
    tf.keras.layers.concatenate([inp, tar]), imports='from IPython import display;import tensorflow as tf;from matplotlib import pyplot as plt;import datetime;import os;import pathlib;import time', function_to_run='tf.keras.layers.concatenate(*args)', method_object=None, object_signature=None, function_args=[eval('[inp, tar]')], function_kwargs={}, max_wait_secs=0)
    down1 = downsample(64, 4, False)(x)
    down2 = downsample(128, 4)(down1)
    down3 = downsample(256, 4)(down2)
    zero_pad1 = custom_method(
    tf.keras.layers.ZeroPadding2D()(down3), imports='from IPython import display;import tensorflow as tf;from matplotlib import pyplot as plt;import datetime;import os;import pathlib;import time', function_to_run='tf.keras.layers.ZeroPadding2D()(*args)', method_object=None, object_signature=None, function_args=[eval('down3')], function_kwargs={}, max_wait_secs=0)
    conv = custom_method(
    tf.keras.layers.Conv2D(512, 4, strides=1, kernel_initializer=initializer, use_bias=False)(zero_pad1), imports='from IPython import display;import tensorflow as tf;from matplotlib import pyplot as plt;import datetime;import os;import pathlib;import time', function_to_run='tf.keras.layers.Conv2D(512, 4, strides=1, kernel_initializer=initializer, use_bias=False)(*args)', method_object=None, object_signature=None, function_args=[eval('zero_pad1')], function_kwargs={}, max_wait_secs=0)
    batchnorm1 = custom_method(
    tf.keras.layers.BatchNormalization()(conv), imports='from IPython import display;import tensorflow as tf;from matplotlib import pyplot as plt;import datetime;import os;import pathlib;import time', function_to_run='tf.keras.layers.BatchNormalization()(*args)', method_object=None, object_signature=None, function_args=[eval('conv')], function_kwargs={}, max_wait_secs=0)
    leaky_relu = custom_method(
    tf.keras.layers.LeakyReLU()(batchnorm1), imports='from IPython import display;import tensorflow as tf;from matplotlib import pyplot as plt;import datetime;import os;import pathlib;import time', function_to_run='tf.keras.layers.LeakyReLU()(*args)', method_object=None, object_signature=None, function_args=[eval('batchnorm1')], function_kwargs={}, max_wait_secs=0)
    zero_pad2 = custom_method(
    tf.keras.layers.ZeroPadding2D()(leaky_relu), imports='from IPython import display;import tensorflow as tf;from matplotlib import pyplot as plt;import datetime;import os;import pathlib;import time', function_to_run='tf.keras.layers.ZeroPadding2D()(*args)', method_object=None, object_signature=None, function_args=[eval('leaky_relu')], function_kwargs={}, max_wait_secs=0)
    last = custom_method(
    tf.keras.layers.Conv2D(1, 4, strides=1, kernel_initializer=initializer)(zero_pad2), imports='from IPython import display;import tensorflow as tf;from matplotlib import pyplot as plt;import datetime;import os;import pathlib;import time', function_to_run='tf.keras.layers.Conv2D(1, 4, strides=1, kernel_initializer=initializer)(*args)', method_object=None, object_signature=None, function_args=[eval('zero_pad2')], function_kwargs={}, max_wait_secs=0)
    return custom_method(
    tf.keras.Model(inputs=[inp, tar], outputs=last), imports='from IPython import display;import tensorflow as tf;from matplotlib import pyplot as plt;import datetime;import os;import pathlib;import time', function_to_run='tf.keras.Model(**kwargs)', method_object=None, object_signature=None, function_args=[], function_kwargs={'inputs': eval('[inp, tar]'), 'outputs': eval('last')}, max_wait_secs=0)
discriminator = Discriminator()
custom_method(
tf.keras.utils.plot_model(discriminator, show_shapes=True, dpi=64), imports='from IPython import display;import tensorflow as tf;from matplotlib import pyplot as plt;import datetime;import os;import pathlib;import time', function_to_run='tf.keras.utils.plot_model(*args, **kwargs)', method_object=None, object_signature=None, function_args=[eval('discriminator')], function_kwargs={'show_shapes': eval('True'), 'dpi': eval('64')}, max_wait_secs=0)
disc_out = discriminator([inp[tf.newaxis, ...], gen_output], training=False)
plt.imshow(disc_out[0, ..., -1], vmin=-20, vmax=20, cmap='RdBu_r')
plt.colorbar()

def discriminator_loss(disc_real_output, disc_generated_output):
    real_loss = custom_method(
    loss_object(tf.ones_like(disc_real_output), disc_real_output), imports='from IPython import display;import tensorflow as tf;from matplotlib import pyplot as plt;import datetime;import os;import pathlib;import time', function_to_run='obj(*args)', method_object=eval('loss_object'), object_signature='tf.keras.losses.BinaryCrossentropy', function_args=[eval('tf.ones_like(disc_real_output)'), eval('disc_real_output')], function_kwargs={}, max_wait_secs=0, custom_class=None)
    generated_loss = custom_method(
    loss_object(tf.zeros_like(disc_generated_output), disc_generated_output), imports='from IPython import display;import tensorflow as tf;from matplotlib import pyplot as plt;import datetime;import os;import pathlib;import time', function_to_run='obj(*args)', method_object=eval('loss_object'), object_signature='tf.keras.losses.BinaryCrossentropy', function_args=[eval('tf.zeros_like(disc_generated_output)'), eval('disc_generated_output')], function_kwargs={}, max_wait_secs=0, custom_class=None)
    total_disc_loss = real_loss + generated_loss
    return total_disc_loss
generator_optimizer = custom_method(
tf.keras.optimizers.Adam(0.0002, beta_1=0.5), imports='from IPython import display;import tensorflow as tf;from matplotlib import pyplot as plt;import datetime;import os;import pathlib;import time', function_to_run='tf.keras.optimizers.Adam(*args, **kwargs)', method_object=None, object_signature=None, function_args=[eval('2e-4')], function_kwargs={'beta_1': eval('0.5')}, max_wait_secs=0)
discriminator_optimizer = custom_method(
tf.keras.optimizers.Adam(0.0002, beta_1=0.5), imports='from IPython import display;import tensorflow as tf;from matplotlib import pyplot as plt;import datetime;import os;import pathlib;import time', function_to_run='tf.keras.optimizers.Adam(*args, **kwargs)', method_object=None, object_signature=None, function_args=[eval('2e-4')], function_kwargs={'beta_1': eval('0.5')}, max_wait_secs=0)
checkpoint_dir = './training_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, 'ckpt')
checkpoint = custom_method(
tf.train.Checkpoint(generator_optimizer=generator_optimizer, discriminator_optimizer=discriminator_optimizer, generator=generator, discriminator=discriminator), imports='from IPython import display;import tensorflow as tf;from matplotlib import pyplot as plt;import datetime;import os;import pathlib;import time', function_to_run='tf.train.Checkpoint(**kwargs)', method_object=None, object_signature=None, function_args=[], function_kwargs={'generator_optimizer': eval('generator_optimizer'), 'discriminator_optimizer': eval('discriminator_optimizer'), 'generator': eval('generator'), 'discriminator': eval('discriminator')}, max_wait_secs=0)

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
for (example_input, example_target) in custom_method(
test_dataset.take(1), imports='from IPython import display;import tensorflow as tf;from matplotlib import pyplot as plt;import datetime;import os;import pathlib;import time', function_to_run='obj.take(*args)', method_object=eval('test_dataset'), object_signature='tf.data.Dataset.list_files', function_args=[eval('1')], function_kwargs={}, max_wait_secs=0, custom_class=None):
    generate_images(generator, example_input, example_target)
log_dir = 'logs/'
summary_writer = custom_method(
tf.summary.create_file_writer(log_dir + 'fit/' + datetime.datetime.now().strftime('%Y%m%d-%H%M%S')), imports='from IPython import display;import tensorflow as tf;from matplotlib import pyplot as plt;import datetime;import os;import pathlib;import time', function_to_run='tf.summary.create_file_writer(*args)', method_object=None, object_signature=None, function_args=[eval('log_dir + "fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")')], function_kwargs={}, max_wait_secs=0)

@tf.function
def train_step(input_image, target, step):
    with custom_method(
    tf.GradientTape(), imports='from IPython import display;import tensorflow as tf;from matplotlib import pyplot as plt;import datetime;import os;import pathlib;import time', function_to_run='tf.GradientTape()', method_object=None, object_signature=None, function_args=[], function_kwargs={}, max_wait_secs=0) as gen_tape, custom_method(
    tf.GradientTape(), imports='from IPython import display;import tensorflow as tf;from matplotlib import pyplot as plt;import datetime;import os;import pathlib;import time', function_to_run='tf.GradientTape()', method_object=None, object_signature=None, function_args=[], function_kwargs={}, max_wait_secs=0) as disc_tape:
        gen_output = generator(input_image, training=True)
        disc_real_output = discriminator([input_image, target], training=True)
        disc_generated_output = discriminator([input_image, gen_output], training=True)
        (gen_total_loss, gen_gan_loss, gen_l1_loss) = generator_loss(disc_generated_output, gen_output, target)
        disc_loss = discriminator_loss(disc_real_output, disc_generated_output)
    generator_gradients = gen_tape.gradient(gen_total_loss, generator.trainable_variables)
    discriminator_gradients = disc_tape.gradient(disc_loss, discriminator.trainable_variables)
    custom_method(
    generator_optimizer.apply_gradients(zip(generator_gradients, generator.trainable_variables)), imports='from IPython import display;import tensorflow as tf;from matplotlib import pyplot as plt;import datetime;import os;import pathlib;import time', function_to_run='obj.apply_gradients(*args)', method_object=eval('generator_optimizer'), object_signature='tf.keras.optimizers.Adam', function_args=[eval('zip(generator_gradients,\n                                          generator.trainable_variables)')], function_kwargs={}, max_wait_secs=0, custom_class=None)
    custom_method(
    discriminator_optimizer.apply_gradients(zip(discriminator_gradients, discriminator.trainable_variables)), imports='from IPython import display;import tensorflow as tf;from matplotlib import pyplot as plt;import datetime;import os;import pathlib;import time', function_to_run='obj.apply_gradients(*args)', method_object=eval('discriminator_optimizer'), object_signature='tf.keras.optimizers.Adam', function_args=[eval('zip(discriminator_gradients,\n                                              discriminator.trainable_variables)')], function_kwargs={}, max_wait_secs=0, custom_class=None)
    with custom_method(
    summary_writer.as_default(), imports='from IPython import display;import tensorflow as tf;from matplotlib import pyplot as plt;import datetime;import os;import pathlib;import time', function_to_run='obj.as_default()', method_object=eval('summary_writer'), object_signature='tf.summary.create_file_writer', function_args=[], function_kwargs={}, max_wait_secs=0, custom_class=None):
        custom_method(
        tf.summary.scalar('gen_total_loss', gen_total_loss, step=step // 1000), imports='from IPython import display;import tensorflow as tf;from matplotlib import pyplot as plt;import datetime;import os;import pathlib;import time', function_to_run='tf.summary.scalar(*args, **kwargs)', method_object=None, object_signature=None, function_args=[eval("'gen_total_loss'"), eval('gen_total_loss')], function_kwargs={'step': eval('step//1000')}, max_wait_secs=0)
        custom_method(
        tf.summary.scalar('gen_gan_loss', gen_gan_loss, step=step // 1000), imports='from IPython import display;import tensorflow as tf;from matplotlib import pyplot as plt;import datetime;import os;import pathlib;import time', function_to_run='tf.summary.scalar(*args, **kwargs)', method_object=None, object_signature=None, function_args=[eval("'gen_gan_loss'"), eval('gen_gan_loss')], function_kwargs={'step': eval('step//1000')}, max_wait_secs=0)
        custom_method(
        tf.summary.scalar('gen_l1_loss', gen_l1_loss, step=step // 1000), imports='from IPython import display;import tensorflow as tf;from matplotlib import pyplot as plt;import datetime;import os;import pathlib;import time', function_to_run='tf.summary.scalar(*args, **kwargs)', method_object=None, object_signature=None, function_args=[eval("'gen_l1_loss'"), eval('gen_l1_loss')], function_kwargs={'step': eval('step//1000')}, max_wait_secs=0)
        custom_method(
        tf.summary.scalar('disc_loss', disc_loss, step=step // 1000), imports='from IPython import display;import tensorflow as tf;from matplotlib import pyplot as plt;import datetime;import os;import pathlib;import time', function_to_run='tf.summary.scalar(*args, **kwargs)', method_object=None, object_signature=None, function_args=[eval("'disc_loss'"), eval('disc_loss')], function_kwargs={'step': eval('step//1000')}, max_wait_secs=0)

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
            custom_method(
            checkpoint.save(file_prefix=checkpoint_prefix), imports='from IPython import display;import tensorflow as tf;from matplotlib import pyplot as plt;import datetime;import os;import pathlib;import time', function_to_run='obj.save(**kwargs)', method_object=eval('checkpoint'), object_signature='tf.train.Checkpoint', function_args=[], function_kwargs={'file_prefix': eval('checkpoint_prefix')}, max_wait_secs=0, custom_class=None)
fit(train_dataset, test_dataset, steps=40000)
custom_method(
checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir)), imports='from IPython import display;import tensorflow as tf;from matplotlib import pyplot as plt;import datetime;import os;import pathlib;import time', function_to_run='obj.restore(*args)', method_object=eval('checkpoint'), object_signature='tf.train.Checkpoint', function_args=[eval('tf.train.latest_checkpoint(checkpoint_dir)')], function_kwargs={}, max_wait_secs=0, custom_class=None)
for (inp, tar) in custom_method(
test_dataset.take(5), imports='from IPython import display;import tensorflow as tf;from matplotlib import pyplot as plt;import datetime;import os;import pathlib;import time', function_to_run='obj.take(*args)', method_object=eval('test_dataset'), object_signature='tf.data.Dataset.list_files', function_args=[eval('5')], function_kwargs={}, max_wait_secs=0, custom_class=None):
    generate_images(generator, inp, tar)
