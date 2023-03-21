import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow_examples.models.pix2pix import pix2pix
import os
import time
import matplotlib.pyplot as plt
from IPython.display import clear_output
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
AUTOTUNE = tf.data.AUTOTUNE
(dataset, metadata) = custom_method(
tfds.load('cycle_gan/horse2zebra', with_info=True, as_supervised=True), imports='import tensorflow as tf;import tensorflow_datasets as tfds;import matplotlib.pyplot as plt;from IPython.display import clear_output;import time;import os;from tensorflow_examples.models.pix2pix import pix2pix', function_to_run='tfds.load(*args, **kwargs)', method_object=None, object_signature=None, function_args=[eval("'cycle_gan/horse2zebra'")], function_kwargs={'with_info': eval('True'), 'as_supervised': eval('True')}, max_wait_secs=0)
(train_horses, train_zebras) = (dataset['trainA'], dataset['trainB'])
(test_horses, test_zebras) = (dataset['testA'], dataset['testB'])
BUFFER_SIZE = 1000
BATCH_SIZE = 1
IMG_WIDTH = 256
IMG_HEIGHT = 256

def random_crop(image):
    cropped_image = custom_method(
    tf.image.random_crop(image, size=[IMG_HEIGHT, IMG_WIDTH, 3]), imports='import tensorflow as tf;import tensorflow_datasets as tfds;import matplotlib.pyplot as plt;from IPython.display import clear_output;import time;import os;from tensorflow_examples.models.pix2pix import pix2pix', function_to_run='tf.image.random_crop(*args, **kwargs)', method_object=None, object_signature=None, function_args=[eval('image')], function_kwargs={'size': eval('[IMG_HEIGHT, IMG_WIDTH, 3]')}, max_wait_secs=0)
    return cropped_image

def normalize(image):
    image = custom_method(
    tf.cast(image, tf.float32), imports='import tensorflow as tf;import tensorflow_datasets as tfds;import matplotlib.pyplot as plt;from IPython.display import clear_output;import time;import os;from tensorflow_examples.models.pix2pix import pix2pix', function_to_run='tf.cast(*args)', method_object=None, object_signature=None, function_args=[eval('image'), eval('tf.float32')], function_kwargs={}, max_wait_secs=0)
    image = image / 127.5 - 1
    return image

def random_jitter(image):
    image = custom_method(
    tf.image.resize(image, [286, 286], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR), imports='import tensorflow as tf;import tensorflow_datasets as tfds;import matplotlib.pyplot as plt;from IPython.display import clear_output;import time;import os;from tensorflow_examples.models.pix2pix import pix2pix', function_to_run='tf.image.resize(*args, **kwargs)', method_object=None, object_signature=None, function_args=[eval('image'), eval('[286, 286]')], function_kwargs={'method': eval('tf.image.ResizeMethod.NEAREST_NEIGHBOR')}, max_wait_secs=0)
    image = random_crop(image)
    image = custom_method(
    tf.image.random_flip_left_right(image), imports='import tensorflow as tf;import tensorflow_datasets as tfds;import matplotlib.pyplot as plt;from IPython.display import clear_output;import time;import os;from tensorflow_examples.models.pix2pix import pix2pix', function_to_run='tf.image.random_flip_left_right(*args)', method_object=None, object_signature=None, function_args=[eval('image')], function_kwargs={}, max_wait_secs=0)
    return image

def preprocess_image_train(image, label):
    image = random_jitter(image)
    image = normalize(image)
    return image

def preprocess_image_test(image, label):
    image = normalize(image)
    return image
train_horses = train_horses.cache().map(preprocess_image_train, num_parallel_calls=AUTOTUNE).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)
train_zebras = train_zebras.cache().map(preprocess_image_train, num_parallel_calls=AUTOTUNE).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)
test_horses = test_horses.map(preprocess_image_test, num_parallel_calls=AUTOTUNE).cache().shuffle(BUFFER_SIZE).batch(BATCH_SIZE)
test_zebras = test_zebras.map(preprocess_image_test, num_parallel_calls=AUTOTUNE).cache().shuffle(BUFFER_SIZE).batch(BATCH_SIZE)
sample_horse = next(iter(train_horses))
sample_zebra = next(iter(train_zebras))
plt.subplot(121)
plt.title('Horse')
plt.imshow(sample_horse[0] * 0.5 + 0.5)
plt.subplot(122)
plt.title('Horse with random jitter')
plt.imshow(random_jitter(sample_horse[0]) * 0.5 + 0.5)
plt.subplot(121)
plt.title('Zebra')
plt.imshow(sample_zebra[0] * 0.5 + 0.5)
plt.subplot(122)
plt.title('Zebra with random jitter')
plt.imshow(random_jitter(sample_zebra[0]) * 0.5 + 0.5)
OUTPUT_CHANNELS = 3
generator_g = pix2pix.unet_generator(OUTPUT_CHANNELS, norm_type='instancenorm')
generator_f = pix2pix.unet_generator(OUTPUT_CHANNELS, norm_type='instancenorm')
discriminator_x = pix2pix.discriminator(norm_type='instancenorm', target=False)
discriminator_y = pix2pix.discriminator(norm_type='instancenorm', target=False)
to_zebra = generator_g(sample_horse)
to_horse = generator_f(sample_zebra)
plt.figure(figsize=(8, 8))
contrast = 8
imgs = [sample_horse, to_zebra, sample_zebra, to_horse]
title = ['Horse', 'To Zebra', 'Zebra', 'To Horse']
for i in range(len(imgs)):
    plt.subplot(2, 2, i + 1)
    plt.title(title[i])
    if i % 2 == 0:
        plt.imshow(imgs[i][0] * 0.5 + 0.5)
    else:
        plt.imshow(imgs[i][0] * 0.5 * contrast + 0.5)
plt.show()
plt.figure(figsize=(8, 8))
plt.subplot(121)
plt.title('Is a real zebra?')
plt.imshow(discriminator_y(sample_zebra)[0, ..., -1], cmap='RdBu_r')
plt.subplot(122)
plt.title('Is a real horse?')
plt.imshow(discriminator_x(sample_horse)[0, ..., -1], cmap='RdBu_r')
plt.show()
LAMBDA = 10
loss_obj = custom_method(
tf.keras.losses.BinaryCrossentropy(from_logits=True), imports='import tensorflow as tf;import tensorflow_datasets as tfds;import matplotlib.pyplot as plt;from IPython.display import clear_output;import time;import os;from tensorflow_examples.models.pix2pix import pix2pix', function_to_run='tf.keras.losses.BinaryCrossentropy(**kwargs)', method_object=None, object_signature=None, function_args=[], function_kwargs={'from_logits': eval('True')}, max_wait_secs=0)

def discriminator_loss(real, generated):
    real_loss = custom_method(
    loss_obj(tf.ones_like(real), real), imports='import tensorflow as tf;import tensorflow_datasets as tfds;import matplotlib.pyplot as plt;from IPython.display import clear_output;import time;import os;from tensorflow_examples.models.pix2pix import pix2pix', function_to_run='obj(*args)', method_object=eval('loss_obj'), object_signature='tf.keras.losses.BinaryCrossentropy', function_args=[eval('tf.ones_like(real)'), eval('real')], function_kwargs={}, max_wait_secs=0, custom_class=None)
    generated_loss = custom_method(
    loss_obj(tf.zeros_like(generated), generated), imports='import tensorflow as tf;import tensorflow_datasets as tfds;import matplotlib.pyplot as plt;from IPython.display import clear_output;import time;import os;from tensorflow_examples.models.pix2pix import pix2pix', function_to_run='obj(*args)', method_object=eval('loss_obj'), object_signature='tf.keras.losses.BinaryCrossentropy', function_args=[eval('tf.zeros_like(generated)'), eval('generated')], function_kwargs={}, max_wait_secs=0, custom_class=None)
    total_disc_loss = real_loss + generated_loss
    return total_disc_loss * 0.5

def generator_loss(generated):
    return custom_method(
    loss_obj(tf.ones_like(generated), generated), imports='import tensorflow as tf;import tensorflow_datasets as tfds;import matplotlib.pyplot as plt;from IPython.display import clear_output;import time;import os;from tensorflow_examples.models.pix2pix import pix2pix', function_to_run='obj(*args)', method_object=eval('loss_obj'), object_signature='tf.keras.losses.BinaryCrossentropy', function_args=[eval('tf.ones_like(generated)'), eval('generated')], function_kwargs={}, max_wait_secs=0, custom_class=None)

def calc_cycle_loss(real_image, cycled_image):
    loss1 = custom_method(
    tf.reduce_mean(tf.abs(real_image - cycled_image)), imports='import tensorflow as tf;import tensorflow_datasets as tfds;import matplotlib.pyplot as plt;from IPython.display import clear_output;import time;import os;from tensorflow_examples.models.pix2pix import pix2pix', function_to_run='tf.reduce_mean(*args)', method_object=None, object_signature=None, function_args=[eval('tf.abs(real_image - cycled_image)')], function_kwargs={}, max_wait_secs=0)
    return LAMBDA * loss1

def identity_loss(real_image, same_image):
    loss = custom_method(
    tf.reduce_mean(tf.abs(real_image - same_image)), imports='import tensorflow as tf;import tensorflow_datasets as tfds;import matplotlib.pyplot as plt;from IPython.display import clear_output;import time;import os;from tensorflow_examples.models.pix2pix import pix2pix', function_to_run='tf.reduce_mean(*args)', method_object=None, object_signature=None, function_args=[eval('tf.abs(real_image - same_image)')], function_kwargs={}, max_wait_secs=0)
    return LAMBDA * 0.5 * loss
generator_g_optimizer = custom_method(
tf.keras.optimizers.Adam(0.0002, beta_1=0.5), imports='import tensorflow as tf;import tensorflow_datasets as tfds;import matplotlib.pyplot as plt;from IPython.display import clear_output;import time;import os;from tensorflow_examples.models.pix2pix import pix2pix', function_to_run='tf.keras.optimizers.Adam(*args, **kwargs)', method_object=None, object_signature=None, function_args=[eval('2e-4')], function_kwargs={'beta_1': eval('0.5')}, max_wait_secs=0)
generator_f_optimizer = custom_method(
tf.keras.optimizers.Adam(0.0002, beta_1=0.5), imports='import tensorflow as tf;import tensorflow_datasets as tfds;import matplotlib.pyplot as plt;from IPython.display import clear_output;import time;import os;from tensorflow_examples.models.pix2pix import pix2pix', function_to_run='tf.keras.optimizers.Adam(*args, **kwargs)', method_object=None, object_signature=None, function_args=[eval('2e-4')], function_kwargs={'beta_1': eval('0.5')}, max_wait_secs=0)
discriminator_x_optimizer = custom_method(
tf.keras.optimizers.Adam(0.0002, beta_1=0.5), imports='import tensorflow as tf;import tensorflow_datasets as tfds;import matplotlib.pyplot as plt;from IPython.display import clear_output;import time;import os;from tensorflow_examples.models.pix2pix import pix2pix', function_to_run='tf.keras.optimizers.Adam(*args, **kwargs)', method_object=None, object_signature=None, function_args=[eval('2e-4')], function_kwargs={'beta_1': eval('0.5')}, max_wait_secs=0)
discriminator_y_optimizer = custom_method(
tf.keras.optimizers.Adam(0.0002, beta_1=0.5), imports='import tensorflow as tf;import tensorflow_datasets as tfds;import matplotlib.pyplot as plt;from IPython.display import clear_output;import time;import os;from tensorflow_examples.models.pix2pix import pix2pix', function_to_run='tf.keras.optimizers.Adam(*args, **kwargs)', method_object=None, object_signature=None, function_args=[eval('2e-4')], function_kwargs={'beta_1': eval('0.5')}, max_wait_secs=0)
checkpoint_path = './checkpoints/train'
ckpt = custom_method(
tf.train.Checkpoint(generator_g=generator_g, generator_f=generator_f, discriminator_x=discriminator_x, discriminator_y=discriminator_y, generator_g_optimizer=generator_g_optimizer, generator_f_optimizer=generator_f_optimizer, discriminator_x_optimizer=discriminator_x_optimizer, discriminator_y_optimizer=discriminator_y_optimizer), imports='import tensorflow as tf;import tensorflow_datasets as tfds;import matplotlib.pyplot as plt;from IPython.display import clear_output;import time;import os;from tensorflow_examples.models.pix2pix import pix2pix', function_to_run='tf.train.Checkpoint(**kwargs)', method_object=None, object_signature=None, function_args=[], function_kwargs={'generator_g': eval('generator_g'), 'generator_f': eval('generator_f'), 'discriminator_x': eval('discriminator_x'), 'discriminator_y': eval('discriminator_y'), 'generator_g_optimizer': eval('generator_g_optimizer'), 'generator_f_optimizer': eval('generator_f_optimizer'), 'discriminator_x_optimizer': eval('discriminator_x_optimizer'), 'discriminator_y_optimizer': eval('discriminator_y_optimizer')}, max_wait_secs=0)
ckpt_manager = custom_method(
tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=5), imports='import tensorflow as tf;import tensorflow_datasets as tfds;import matplotlib.pyplot as plt;from IPython.display import clear_output;import time;import os;from tensorflow_examples.models.pix2pix import pix2pix', function_to_run='tf.train.CheckpointManager(*args, **kwargs)', method_object=None, object_signature=None, function_args=[eval('ckpt'), eval('checkpoint_path')], function_kwargs={'max_to_keep': eval('5')}, max_wait_secs=0)
if ckpt_manager.latest_checkpoint:
    custom_method(
    ckpt.restore(ckpt_manager.latest_checkpoint), imports='import tensorflow as tf;import tensorflow_datasets as tfds;import matplotlib.pyplot as plt;from IPython.display import clear_output;import time;import os;from tensorflow_examples.models.pix2pix import pix2pix', function_to_run='obj.restore(*args)', method_object=eval('ckpt'), object_signature='tf.train.Checkpoint', function_args=[eval('ckpt_manager.latest_checkpoint')], function_kwargs={}, max_wait_secs=0, custom_class=None)
    print('Latest checkpoint restored!!')
EPOCHS = 10

def generate_images(model, test_input):
    prediction = model(test_input)
    plt.figure(figsize=(12, 12))
    display_list = [test_input[0], prediction[0]]
    title = ['Input Image', 'Predicted Image']
    for i in range(2):
        plt.subplot(1, 2, i + 1)
        plt.title(title[i])
        plt.imshow(display_list[i] * 0.5 + 0.5)
        plt.axis('off')
    plt.show()

@tf.function
def train_step(real_x, real_y):
    with custom_method(
    tf.GradientTape(persistent=True), imports='import tensorflow as tf;import tensorflow_datasets as tfds;import matplotlib.pyplot as plt;from IPython.display import clear_output;import time;import os;from tensorflow_examples.models.pix2pix import pix2pix', function_to_run='tf.GradientTape(**kwargs)', method_object=None, object_signature=None, function_args=[], function_kwargs={'persistent': eval('True')}, max_wait_secs=0) as tape:
        fake_y = generator_g(real_x, training=True)
        cycled_x = generator_f(fake_y, training=True)
        fake_x = generator_f(real_y, training=True)
        cycled_y = generator_g(fake_x, training=True)
        same_x = generator_f(real_x, training=True)
        same_y = generator_g(real_y, training=True)
        disc_real_x = discriminator_x(real_x, training=True)
        disc_real_y = discriminator_y(real_y, training=True)
        disc_fake_x = discriminator_x(fake_x, training=True)
        disc_fake_y = discriminator_y(fake_y, training=True)
        gen_g_loss = custom_method(
        generator_loss(disc_fake_y), imports='import tensorflow as tf;import tensorflow_datasets as tfds;import matplotlib.pyplot as plt;from IPython.display import clear_output;import time;import os;from tensorflow_examples.models.pix2pix import pix2pix', function_to_run='obj(*args)', method_object=eval('generator_loss'), object_signature='tf.reduce_mean', function_args=[eval('disc_fake_y')], function_kwargs={}, max_wait_secs=0, custom_class=None)
        gen_f_loss = custom_method(
        generator_loss(disc_fake_x), imports='import tensorflow as tf;import tensorflow_datasets as tfds;import matplotlib.pyplot as plt;from IPython.display import clear_output;import time;import os;from tensorflow_examples.models.pix2pix import pix2pix', function_to_run='obj(*args)', method_object=eval('generator_loss'), object_signature='tf.reduce_mean', function_args=[eval('disc_fake_x')], function_kwargs={}, max_wait_secs=0, custom_class=None)
        total_cycle_loss = custom_method(
        calc_cycle_loss(real_x, cycled_x), imports='import tensorflow as tf;import tensorflow_datasets as tfds;import matplotlib.pyplot as plt;from IPython.display import clear_output;import time;import os;from tensorflow_examples.models.pix2pix import pix2pix', function_to_run='obj(*args)', method_object=eval('calc_cycle_loss'), object_signature='tf.reduce_mean', function_args=[eval('real_x'), eval('cycled_x')], function_kwargs={}, max_wait_secs=0, custom_class=None) + custom_method(
        calc_cycle_loss(real_y, cycled_y), imports='import tensorflow as tf;import tensorflow_datasets as tfds;import matplotlib.pyplot as plt;from IPython.display import clear_output;import time;import os;from tensorflow_examples.models.pix2pix import pix2pix', function_to_run='obj(*args)', method_object=eval('calc_cycle_loss'), object_signature='tf.reduce_mean', function_args=[eval('real_y'), eval('cycled_y')], function_kwargs={}, max_wait_secs=0, custom_class=None)
        total_gen_g_loss = gen_g_loss + total_cycle_loss + custom_method(
        identity_loss(real_y, same_y), imports='import tensorflow as tf;import tensorflow_datasets as tfds;import matplotlib.pyplot as plt;from IPython.display import clear_output;import time;import os;from tensorflow_examples.models.pix2pix import pix2pix', function_to_run='obj(*args)', method_object=eval('identity_loss'), object_signature='tf.reduce_mean', function_args=[eval('real_y'), eval('same_y')], function_kwargs={}, max_wait_secs=0, custom_class=None)
        total_gen_f_loss = gen_f_loss + total_cycle_loss + custom_method(
        identity_loss(real_x, same_x), imports='import tensorflow as tf;import tensorflow_datasets as tfds;import matplotlib.pyplot as plt;from IPython.display import clear_output;import time;import os;from tensorflow_examples.models.pix2pix import pix2pix', function_to_run='obj(*args)', method_object=eval('identity_loss'), object_signature='tf.reduce_mean', function_args=[eval('real_x'), eval('same_x')], function_kwargs={}, max_wait_secs=0, custom_class=None)
        disc_x_loss = custom_method(
        discriminator_loss(disc_real_x, disc_fake_x), imports='import tensorflow as tf;import tensorflow_datasets as tfds;import matplotlib.pyplot as plt;from IPython.display import clear_output;import time;import os;from tensorflow_examples.models.pix2pix import pix2pix', function_to_run='obj(*args)', method_object=eval('discriminator_loss'), object_signature='tf.reduce_mean', function_args=[eval('disc_real_x'), eval('disc_fake_x')], function_kwargs={}, max_wait_secs=0, custom_class=None)
        disc_y_loss = custom_method(
        discriminator_loss(disc_real_y, disc_fake_y), imports='import tensorflow as tf;import tensorflow_datasets as tfds;import matplotlib.pyplot as plt;from IPython.display import clear_output;import time;import os;from tensorflow_examples.models.pix2pix import pix2pix', function_to_run='obj(*args)', method_object=eval('discriminator_loss'), object_signature='tf.reduce_mean', function_args=[eval('disc_real_y'), eval('disc_fake_y')], function_kwargs={}, max_wait_secs=0, custom_class=None)
    generator_g_gradients = tape.gradient(total_gen_g_loss, generator_g.trainable_variables)
    generator_f_gradients = tape.gradient(total_gen_f_loss, generator_f.trainable_variables)
    discriminator_x_gradients = tape.gradient(disc_x_loss, discriminator_x.trainable_variables)
    discriminator_y_gradients = tape.gradient(disc_y_loss, discriminator_y.trainable_variables)
    custom_method(
    generator_g_optimizer.apply_gradients(zip(generator_g_gradients, generator_g.trainable_variables)), imports='import tensorflow as tf;import tensorflow_datasets as tfds;import matplotlib.pyplot as plt;from IPython.display import clear_output;import time;import os;from tensorflow_examples.models.pix2pix import pix2pix', function_to_run='obj.apply_gradients(*args)', method_object=eval('generator_g_optimizer'), object_signature='tf.keras.optimizers.Adam', function_args=[eval('zip(generator_g_gradients, \n                                            generator_g.trainable_variables)')], function_kwargs={}, max_wait_secs=0, custom_class=None)
    custom_method(
    generator_f_optimizer.apply_gradients(zip(generator_f_gradients, generator_f.trainable_variables)), imports='import tensorflow as tf;import tensorflow_datasets as tfds;import matplotlib.pyplot as plt;from IPython.display import clear_output;import time;import os;from tensorflow_examples.models.pix2pix import pix2pix', function_to_run='obj.apply_gradients(*args)', method_object=eval('generator_f_optimizer'), object_signature='tf.keras.optimizers.Adam', function_args=[eval('zip(generator_f_gradients, \n                                            generator_f.trainable_variables)')], function_kwargs={}, max_wait_secs=0, custom_class=None)
    custom_method(
    discriminator_x_optimizer.apply_gradients(zip(discriminator_x_gradients, discriminator_x.trainable_variables)), imports='import tensorflow as tf;import tensorflow_datasets as tfds;import matplotlib.pyplot as plt;from IPython.display import clear_output;import time;import os;from tensorflow_examples.models.pix2pix import pix2pix', function_to_run='obj.apply_gradients(*args)', method_object=eval('discriminator_x_optimizer'), object_signature='tf.keras.optimizers.Adam', function_args=[eval('zip(discriminator_x_gradients,\n                                                discriminator_x.trainable_variables)')], function_kwargs={}, max_wait_secs=0, custom_class=None)
    custom_method(
    discriminator_y_optimizer.apply_gradients(zip(discriminator_y_gradients, discriminator_y.trainable_variables)), imports='import tensorflow as tf;import tensorflow_datasets as tfds;import matplotlib.pyplot as plt;from IPython.display import clear_output;import time;import os;from tensorflow_examples.models.pix2pix import pix2pix', function_to_run='obj.apply_gradients(*args)', method_object=eval('discriminator_y_optimizer'), object_signature='tf.keras.optimizers.Adam', function_args=[eval('zip(discriminator_y_gradients,\n                                                discriminator_y.trainable_variables)')], function_kwargs={}, max_wait_secs=0, custom_class=None)
for epoch in range(EPOCHS):
    start = time.time()
    n = 0
    for (image_x, image_y) in custom_method(
    tf.data.Dataset.zip((train_horses, train_zebras)), imports='import tensorflow as tf;import tensorflow_datasets as tfds;import matplotlib.pyplot as plt;from IPython.display import clear_output;import time;import os;from tensorflow_examples.models.pix2pix import pix2pix', function_to_run='tf.data.Dataset.zip(*args)', method_object=None, object_signature=None, function_args=[eval('(train_horses, train_zebras)')], function_kwargs={}, max_wait_secs=0):
        train_step(image_x, image_y)
        if n % 10 == 0:
            print('.', end='')
        n += 1
    clear_output(wait=True)
    custom_method(
    generate_images(generator_g, sample_horse), imports='import tensorflow as tf;import tensorflow_datasets as tfds;import matplotlib.pyplot as plt;from IPython.display import clear_output;import time;import os;from tensorflow_examples.models.pix2pix import pix2pix', function_to_run='obj(*args)', method_object=eval('generate_images'), object_signature='tf.cast', function_args=[eval('generator_g'), eval('sample_horse')], function_kwargs={}, max_wait_secs=0, custom_class=None)
    if (epoch + 1) % 5 == 0:
        ckpt_save_path = custom_method(
        ckpt_manager.save(), imports='import tensorflow as tf;import tensorflow_datasets as tfds;import matplotlib.pyplot as plt;from IPython.display import clear_output;import time;import os;from tensorflow_examples.models.pix2pix import pix2pix', function_to_run='obj.save()', method_object=eval('ckpt_manager'), object_signature='tf.train.Checkpoint', function_args=[], function_kwargs={}, max_wait_secs=0, custom_class=None)
        print('Saving checkpoint for epoch {} at {}'.format(epoch + 1, ckpt_save_path))
    print('Time taken for epoch {} is {} sec\n'.format(epoch + 1, time.time() - start))
for inp in test_horses.take(5):
    custom_method(
    generate_images(generator_g, inp), imports='import tensorflow as tf;import tensorflow_datasets as tfds;import matplotlib.pyplot as plt;from IPython.display import clear_output;import time;import os;from tensorflow_examples.models.pix2pix import pix2pix', function_to_run='obj(*args)', method_object=eval('generate_images'), object_signature='tf.cast', function_args=[eval('generator_g'), eval('inp')], function_kwargs={}, max_wait_secs=0, custom_class=None)
