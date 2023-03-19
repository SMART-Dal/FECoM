import tensorflow as tf
import os
from pathlib import Path
import dill as pickle
from tool.client.client_config import EXPERIMENT_DIR, EXPERIMENT_TAG, MAX_WAIT_S, WAIT_AFTER_RUN_S
from tool.server.send_request import send_request
from tool.server.function_details import FunctionDetails
current_path = os.path.abspath(__file__)
(immediate_folder, file_name) = os.path.split(current_path)
immediate_folder = os.path.basename(immediate_folder)
experiment_file_name = os.path.splitext(file_name)[0]
EXPERIMENT_FILE_PATH = EXPERIMENT_DIR / immediate_folder / EXPERIMENT_TAG / (experiment_file_name + '-energy.json')

def custom_method(func, imports: str, function_to_run: str, method_object=None, function_args: list=None, function_kwargs: dict=None, max_wait_secs=MAX_WAIT_S, custom_class=None, wait_after_run_secs=WAIT_AFTER_RUN_S):
    result = send_request(imports=imports, function_to_run=function_to_run, function_args=function_args, function_kwargs=function_kwargs, max_wait_secs=max_wait_secs, method_object=method_object, custom_class=custom_class, experiment_file_path=EXPERIMENT_FILE_PATH)
    return func
if __name__ == '__main__':
    print(EXPERIMENT_FILE_PATH)
tf.__version__
import glob
import imageio
import matplotlib.pyplot as plt
import numpy as np
import os
import PIL
from tensorflow.keras import layers
import time
from IPython import display
((train_images, train_labels), (_, _)) = custom_method(
tf.keras.datasets.mnist.load_data(), imports='from tensorflow.keras import layers;import imageio;import glob;import tensorflow as tf;import numpy as np;import PIL;import time;import os;from IPython import display;import tensorflow_docs.vis.embed as embed;import matplotlib.pyplot as plt', function_to_run='tf.keras.datasets.mnist.load_data()', method_object=None, function_args=[], function_kwargs={}, max_wait_secs=0)
train_images = train_images.reshape(train_images.shape[0], 28, 28, 1).astype('float32')
train_images = (train_images - 127.5) / 127.5
BUFFER_SIZE = 60000
BATCH_SIZE = 256
train_dataset = custom_method(
tf.data.Dataset.from_tensor_slices(train_images).shuffle(BUFFER_SIZE).batch(BATCH_SIZE), imports='from tensorflow.keras import layers;import imageio;import glob;import tensorflow as tf;import numpy as np;import PIL;import time;import os;from IPython import display;import tensorflow_docs.vis.embed as embed;import matplotlib.pyplot as plt', function_to_run='tf.data.Dataset.from_tensor_slices(train_images).shuffle(BUFFER_SIZE).batch(*args)', method_object=None, function_args=[eval('BATCH_SIZE')], function_kwargs={}, max_wait_secs=0)

def make_generator_model():
    model = custom_method(
    tf.keras.Sequential(), imports='from tensorflow.keras import layers;import imageio;import glob;import tensorflow as tf;import numpy as np;import PIL;import time;import os;from IPython import display;import tensorflow_docs.vis.embed as embed;import matplotlib.pyplot as plt', function_to_run='tf.keras.Sequential()', method_object=None, function_args=[], function_kwargs={}, max_wait_secs=0)
    custom_method(
    model.add(layers.Dense(7 * 7 * 256, use_bias=False, input_shape=(100,))), imports='from tensorflow.keras import layers;import imageio;import glob;import tensorflow as tf;import numpy as np;import PIL;import time;import os;from IPython import display;import tensorflow_docs.vis.embed as embed;import matplotlib.pyplot as plt', function_to_run='obj.add(*args)', method_object=eval('model'), function_args=[eval('layers.Dense(7*7*256, use_bias=False, input_shape=(100,))')], function_kwargs={}, max_wait_secs=0, custom_class=None)
    custom_method(
    model.add(layers.BatchNormalization()), imports='from tensorflow.keras import layers;import imageio;import glob;import tensorflow as tf;import numpy as np;import PIL;import time;import os;from IPython import display;import tensorflow_docs.vis.embed as embed;import matplotlib.pyplot as plt', function_to_run='obj.add(*args)', method_object=eval('model'), function_args=[eval('layers.BatchNormalization()')], function_kwargs={}, max_wait_secs=0, custom_class=None)
    custom_method(
    model.add(layers.LeakyReLU()), imports='from tensorflow.keras import layers;import imageio;import glob;import tensorflow as tf;import numpy as np;import PIL;import time;import os;from IPython import display;import tensorflow_docs.vis.embed as embed;import matplotlib.pyplot as plt', function_to_run='obj.add(*args)', method_object=eval('model'), function_args=[eval('layers.LeakyReLU()')], function_kwargs={}, max_wait_secs=0, custom_class=None)
    custom_method(
    model.add(layers.Reshape((7, 7, 256))), imports='from tensorflow.keras import layers;import imageio;import glob;import tensorflow as tf;import numpy as np;import PIL;import time;import os;from IPython import display;import tensorflow_docs.vis.embed as embed;import matplotlib.pyplot as plt', function_to_run='obj.add(*args)', method_object=eval('model'), function_args=[eval('layers.Reshape((7, 7, 256))')], function_kwargs={}, max_wait_secs=0, custom_class=None)
    assert model.output_shape == (None, 7, 7, 256)
    custom_method(
    model.add(layers.Conv2DTranspose(128, (5, 5), strides=(1, 1), padding='same', use_bias=False)), imports='from tensorflow.keras import layers;import imageio;import glob;import tensorflow as tf;import numpy as np;import PIL;import time;import os;from IPython import display;import tensorflow_docs.vis.embed as embed;import matplotlib.pyplot as plt', function_to_run='obj.add(*args)', method_object=eval('model'), function_args=[eval("layers.Conv2DTranspose(128, (5, 5), strides=(1, 1), padding='same', use_bias=False)")], function_kwargs={}, max_wait_secs=0, custom_class=None)
    assert model.output_shape == (None, 7, 7, 128)
    custom_method(
    model.add(layers.BatchNormalization()), imports='from tensorflow.keras import layers;import imageio;import glob;import tensorflow as tf;import numpy as np;import PIL;import time;import os;from IPython import display;import tensorflow_docs.vis.embed as embed;import matplotlib.pyplot as plt', function_to_run='obj.add(*args)', method_object=eval('model'), function_args=[eval('layers.BatchNormalization()')], function_kwargs={}, max_wait_secs=0, custom_class=None)
    custom_method(
    model.add(layers.LeakyReLU()), imports='from tensorflow.keras import layers;import imageio;import glob;import tensorflow as tf;import numpy as np;import PIL;import time;import os;from IPython import display;import tensorflow_docs.vis.embed as embed;import matplotlib.pyplot as plt', function_to_run='obj.add(*args)', method_object=eval('model'), function_args=[eval('layers.LeakyReLU()')], function_kwargs={}, max_wait_secs=0, custom_class=None)
    custom_method(
    model.add(layers.Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same', use_bias=False)), imports='from tensorflow.keras import layers;import imageio;import glob;import tensorflow as tf;import numpy as np;import PIL;import time;import os;from IPython import display;import tensorflow_docs.vis.embed as embed;import matplotlib.pyplot as plt', function_to_run='obj.add(*args)', method_object=eval('model'), function_args=[eval("layers.Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same', use_bias=False)")], function_kwargs={}, max_wait_secs=0, custom_class=None)
    assert model.output_shape == (None, 14, 14, 64)
    custom_method(
    model.add(layers.BatchNormalization()), imports='from tensorflow.keras import layers;import imageio;import glob;import tensorflow as tf;import numpy as np;import PIL;import time;import os;from IPython import display;import tensorflow_docs.vis.embed as embed;import matplotlib.pyplot as plt', function_to_run='obj.add(*args)', method_object=eval('model'), function_args=[eval('layers.BatchNormalization()')], function_kwargs={}, max_wait_secs=0, custom_class=None)
    custom_method(
    model.add(layers.LeakyReLU()), imports='from tensorflow.keras import layers;import imageio;import glob;import tensorflow as tf;import numpy as np;import PIL;import time;import os;from IPython import display;import tensorflow_docs.vis.embed as embed;import matplotlib.pyplot as plt', function_to_run='obj.add(*args)', method_object=eval('model'), function_args=[eval('layers.LeakyReLU()')], function_kwargs={}, max_wait_secs=0, custom_class=None)
    custom_method(
    model.add(layers.Conv2DTranspose(1, (5, 5), strides=(2, 2), padding='same', use_bias=False, activation='tanh')), imports='from tensorflow.keras import layers;import imageio;import glob;import tensorflow as tf;import numpy as np;import PIL;import time;import os;from IPython import display;import tensorflow_docs.vis.embed as embed;import matplotlib.pyplot as plt', function_to_run='obj.add(*args)', method_object=eval('model'), function_args=[eval("layers.Conv2DTranspose(1, (5, 5), strides=(2, 2), padding='same', use_bias=False, activation='tanh')")], function_kwargs={}, max_wait_secs=0, custom_class=None)
    assert model.output_shape == (None, 28, 28, 1)
    return model
generator = custom_method(
make_generator_model(), imports='from tensorflow.keras import layers;import imageio;import glob;import tensorflow as tf;import numpy as np;import PIL;import time;import os;from IPython import display;import tensorflow_docs.vis.embed as embed;import matplotlib.pyplot as plt', function_to_run='obj()', method_object=eval('make_generator_model'), function_args=[], function_kwargs={}, max_wait_secs=0, custom_class=None)
noise = custom_method(
tf.random.normal([1, 100]), imports='from tensorflow.keras import layers;import imageio;import glob;import tensorflow as tf;import numpy as np;import PIL;import time;import os;from IPython import display;import tensorflow_docs.vis.embed as embed;import matplotlib.pyplot as plt', function_to_run='tf.random.normal(*args)', method_object=None, function_args=[eval('[1, 100]')], function_kwargs={}, max_wait_secs=0)
generated_image = generator(noise, training=False)
plt.imshow(generated_image[0, :, :, 0], cmap='gray')

def make_discriminator_model():
    model = custom_method(
    tf.keras.Sequential(), imports='from tensorflow.keras import layers;import imageio;import glob;import tensorflow as tf;import numpy as np;import PIL;import time;import os;from IPython import display;import tensorflow_docs.vis.embed as embed;import matplotlib.pyplot as plt', function_to_run='tf.keras.Sequential()', method_object=None, function_args=[], function_kwargs={}, max_wait_secs=0)
    custom_method(
    model.add(layers.Conv2D(64, (5, 5), strides=(2, 2), padding='same', input_shape=[28, 28, 1])), imports='from tensorflow.keras import layers;import imageio;import glob;import tensorflow as tf;import numpy as np;import PIL;import time;import os;from IPython import display;import tensorflow_docs.vis.embed as embed;import matplotlib.pyplot as plt', function_to_run='obj.add(*args)', method_object=eval('model'), function_args=[eval("layers.Conv2D(64, (5, 5), strides=(2, 2), padding='same',\n                                     input_shape=[28, 28, 1])")], function_kwargs={}, max_wait_secs=0, custom_class=None)
    custom_method(
    model.add(layers.LeakyReLU()), imports='from tensorflow.keras import layers;import imageio;import glob;import tensorflow as tf;import numpy as np;import PIL;import time;import os;from IPython import display;import tensorflow_docs.vis.embed as embed;import matplotlib.pyplot as plt', function_to_run='obj.add(*args)', method_object=eval('model'), function_args=[eval('layers.LeakyReLU()')], function_kwargs={}, max_wait_secs=0, custom_class=None)
    custom_method(
    model.add(layers.Dropout(0.3)), imports='from tensorflow.keras import layers;import imageio;import glob;import tensorflow as tf;import numpy as np;import PIL;import time;import os;from IPython import display;import tensorflow_docs.vis.embed as embed;import matplotlib.pyplot as plt', function_to_run='obj.add(*args)', method_object=eval('model'), function_args=[eval('layers.Dropout(0.3)')], function_kwargs={}, max_wait_secs=0, custom_class=None)
    custom_method(
    model.add(layers.Conv2D(128, (5, 5), strides=(2, 2), padding='same')), imports='from tensorflow.keras import layers;import imageio;import glob;import tensorflow as tf;import numpy as np;import PIL;import time;import os;from IPython import display;import tensorflow_docs.vis.embed as embed;import matplotlib.pyplot as plt', function_to_run='obj.add(*args)', method_object=eval('model'), function_args=[eval("layers.Conv2D(128, (5, 5), strides=(2, 2), padding='same')")], function_kwargs={}, max_wait_secs=0, custom_class=None)
    custom_method(
    model.add(layers.LeakyReLU()), imports='from tensorflow.keras import layers;import imageio;import glob;import tensorflow as tf;import numpy as np;import PIL;import time;import os;from IPython import display;import tensorflow_docs.vis.embed as embed;import matplotlib.pyplot as plt', function_to_run='obj.add(*args)', method_object=eval('model'), function_args=[eval('layers.LeakyReLU()')], function_kwargs={}, max_wait_secs=0, custom_class=None)
    custom_method(
    model.add(layers.Dropout(0.3)), imports='from tensorflow.keras import layers;import imageio;import glob;import tensorflow as tf;import numpy as np;import PIL;import time;import os;from IPython import display;import tensorflow_docs.vis.embed as embed;import matplotlib.pyplot as plt', function_to_run='obj.add(*args)', method_object=eval('model'), function_args=[eval('layers.Dropout(0.3)')], function_kwargs={}, max_wait_secs=0, custom_class=None)
    custom_method(
    model.add(layers.Flatten()), imports='from tensorflow.keras import layers;import imageio;import glob;import tensorflow as tf;import numpy as np;import PIL;import time;import os;from IPython import display;import tensorflow_docs.vis.embed as embed;import matplotlib.pyplot as plt', function_to_run='obj.add(*args)', method_object=eval('model'), function_args=[eval('layers.Flatten()')], function_kwargs={}, max_wait_secs=0, custom_class=None)
    custom_method(
    model.add(layers.Dense(1)), imports='from tensorflow.keras import layers;import imageio;import glob;import tensorflow as tf;import numpy as np;import PIL;import time;import os;from IPython import display;import tensorflow_docs.vis.embed as embed;import matplotlib.pyplot as plt', function_to_run='obj.add(*args)', method_object=eval('model'), function_args=[eval('layers.Dense(1)')], function_kwargs={}, max_wait_secs=0, custom_class=None)
    return model
discriminator = custom_method(
make_discriminator_model(), imports='from tensorflow.keras import layers;import imageio;import glob;import tensorflow as tf;import numpy as np;import PIL;import time;import os;from IPython import display;import tensorflow_docs.vis.embed as embed;import matplotlib.pyplot as plt', function_to_run='obj()', method_object=eval('make_discriminator_model'), function_args=[], function_kwargs={}, max_wait_secs=0, custom_class=None)
decision = discriminator(generated_image)
print(decision)
cross_entropy = custom_method(
tf.keras.losses.BinaryCrossentropy(from_logits=True), imports='from tensorflow.keras import layers;import imageio;import glob;import tensorflow as tf;import numpy as np;import PIL;import time;import os;from IPython import display;import tensorflow_docs.vis.embed as embed;import matplotlib.pyplot as plt', function_to_run='tf.keras.losses.BinaryCrossentropy(**kwargs)', method_object=None, function_args=[], function_kwargs={'from_logits': eval('True')}, max_wait_secs=0)

def discriminator_loss(real_output, fake_output):
    real_loss = custom_method(
    cross_entropy(tf.ones_like(real_output), real_output), imports='from tensorflow.keras import layers;import imageio;import glob;import tensorflow as tf;import numpy as np;import PIL;import time;import os;from IPython import display;import tensorflow_docs.vis.embed as embed;import matplotlib.pyplot as plt', function_to_run='obj(*args)', method_object=eval('cross_entropy'), function_args=[eval('tf.ones_like(real_output)'), eval('real_output')], function_kwargs={}, max_wait_secs=0, custom_class=None)
    fake_loss = custom_method(
    cross_entropy(tf.zeros_like(fake_output), fake_output), imports='from tensorflow.keras import layers;import imageio;import glob;import tensorflow as tf;import numpy as np;import PIL;import time;import os;from IPython import display;import tensorflow_docs.vis.embed as embed;import matplotlib.pyplot as plt', function_to_run='obj(*args)', method_object=eval('cross_entropy'), function_args=[eval('tf.zeros_like(fake_output)'), eval('fake_output')], function_kwargs={}, max_wait_secs=0, custom_class=None)
    total_loss = real_loss + fake_loss
    return total_loss

def generator_loss(fake_output):
    return custom_method(
    cross_entropy(tf.ones_like(fake_output), fake_output), imports='from tensorflow.keras import layers;import imageio;import glob;import tensorflow as tf;import numpy as np;import PIL;import time;import os;from IPython import display;import tensorflow_docs.vis.embed as embed;import matplotlib.pyplot as plt', function_to_run='obj(*args)', method_object=eval('cross_entropy'), function_args=[eval('tf.ones_like(fake_output)'), eval('fake_output')], function_kwargs={}, max_wait_secs=0, custom_class=None)
generator_optimizer = custom_method(
tf.keras.optimizers.Adam(0.0001), imports='from tensorflow.keras import layers;import imageio;import glob;import tensorflow as tf;import numpy as np;import PIL;import time;import os;from IPython import display;import tensorflow_docs.vis.embed as embed;import matplotlib.pyplot as plt', function_to_run='tf.keras.optimizers.Adam(*args)', method_object=None, function_args=[eval('1e-4')], function_kwargs={}, max_wait_secs=0)
discriminator_optimizer = custom_method(
tf.keras.optimizers.Adam(0.0001), imports='from tensorflow.keras import layers;import imageio;import glob;import tensorflow as tf;import numpy as np;import PIL;import time;import os;from IPython import display;import tensorflow_docs.vis.embed as embed;import matplotlib.pyplot as plt', function_to_run='tf.keras.optimizers.Adam(*args)', method_object=None, function_args=[eval('1e-4')], function_kwargs={}, max_wait_secs=0)
checkpoint_dir = './training_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, 'ckpt')
checkpoint = custom_method(
tf.train.Checkpoint(generator_optimizer=generator_optimizer, discriminator_optimizer=discriminator_optimizer, generator=generator, discriminator=discriminator), imports='from tensorflow.keras import layers;import imageio;import glob;import tensorflow as tf;import numpy as np;import PIL;import time;import os;from IPython import display;import tensorflow_docs.vis.embed as embed;import matplotlib.pyplot as plt', function_to_run='tf.train.Checkpoint(**kwargs)', method_object=None, function_args=[], function_kwargs={'generator_optimizer': eval('generator_optimizer'), 'discriminator_optimizer': eval('discriminator_optimizer'), 'generator': eval('generator'), 'discriminator': eval('discriminator')}, max_wait_secs=0)
EPOCHS = 50
noise_dim = 100
num_examples_to_generate = 16
seed = custom_method(
tf.random.normal([num_examples_to_generate, noise_dim]), imports='from tensorflow.keras import layers;import imageio;import glob;import tensorflow as tf;import numpy as np;import PIL;import time;import os;from IPython import display;import tensorflow_docs.vis.embed as embed;import matplotlib.pyplot as plt', function_to_run='tf.random.normal(*args)', method_object=None, function_args=[eval('[num_examples_to_generate, noise_dim]')], function_kwargs={}, max_wait_secs=0)

@tf.function
def train_step(images):
    noise = custom_method(
    tf.random.normal([BATCH_SIZE, noise_dim]), imports='from tensorflow.keras import layers;import imageio;import glob;import tensorflow as tf;import numpy as np;import PIL;import time;import os;from IPython import display;import tensorflow_docs.vis.embed as embed;import matplotlib.pyplot as plt', function_to_run='tf.random.normal(*args)', method_object=None, function_args=[eval('[BATCH_SIZE, noise_dim]')], function_kwargs={}, max_wait_secs=0)
    with custom_method(
    tf.GradientTape(), imports='from tensorflow.keras import layers;import imageio;import glob;import tensorflow as tf;import numpy as np;import PIL;import time;import os;from IPython import display;import tensorflow_docs.vis.embed as embed;import matplotlib.pyplot as plt', function_to_run='tf.GradientTape()', method_object=None, function_args=[], function_kwargs={}, max_wait_secs=0) as gen_tape, custom_method(
    tf.GradientTape(), imports='from tensorflow.keras import layers;import imageio;import glob;import tensorflow as tf;import numpy as np;import PIL;import time;import os;from IPython import display;import tensorflow_docs.vis.embed as embed;import matplotlib.pyplot as plt', function_to_run='tf.GradientTape()', method_object=None, function_args=[], function_kwargs={}, max_wait_secs=0) as disc_tape:
        generated_images = generator(noise, training=True)
        real_output = discriminator(images, training=True)
        fake_output = discriminator(generated_images, training=True)
        gen_loss = generator_loss(fake_output)
        disc_loss = discriminator_loss(real_output, fake_output)
    gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
    gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)
    custom_method(
    generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables)), imports='from tensorflow.keras import layers;import imageio;import glob;import tensorflow as tf;import numpy as np;import PIL;import time;import os;from IPython import display;import tensorflow_docs.vis.embed as embed;import matplotlib.pyplot as plt', function_to_run='obj.apply_gradients(*args)', method_object=eval('generator_optimizer'), function_args=[eval('zip(gradients_of_generator, generator.trainable_variables)')], function_kwargs={}, max_wait_secs=0, custom_class=None)
    custom_method(
    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables)), imports='from tensorflow.keras import layers;import imageio;import glob;import tensorflow as tf;import numpy as np;import PIL;import time;import os;from IPython import display;import tensorflow_docs.vis.embed as embed;import matplotlib.pyplot as plt', function_to_run='obj.apply_gradients(*args)', method_object=eval('discriminator_optimizer'), function_args=[eval('zip(gradients_of_discriminator, discriminator.trainable_variables)')], function_kwargs={}, max_wait_secs=0, custom_class=None)

def train(dataset, epochs):
    for epoch in range(epochs):
        start = time.time()
        for image_batch in dataset:
            train_step(image_batch)
        display.clear_output(wait=True)
        generate_and_save_images(generator, epoch + 1, seed)
        if (epoch + 1) % 15 == 0:
            custom_method(
            checkpoint.save(file_prefix=checkpoint_prefix), imports='from tensorflow.keras import layers;import imageio;import glob;import tensorflow as tf;import numpy as np;import PIL;import time;import os;from IPython import display;import tensorflow_docs.vis.embed as embed;import matplotlib.pyplot as plt', function_to_run='obj.save(**kwargs)', method_object=eval('checkpoint'), function_args=[], function_kwargs={'file_prefix': eval('checkpoint_prefix')}, max_wait_secs=0, custom_class=None)
        print('Time for epoch {} is {} sec'.format(epoch + 1, time.time() - start))
    display.clear_output(wait=True)
    generate_and_save_images(generator, epochs, seed)

def generate_and_save_images(model, epoch, test_input):
    predictions = custom_method(
    model(test_input, training=False), imports='from tensorflow.keras import layers;import imageio;import glob;import tensorflow as tf;import numpy as np;import PIL;import time;import os;from IPython import display;import tensorflow_docs.vis.embed as embed;import matplotlib.pyplot as plt', function_to_run='obj(*args, **kwargs)', method_object=eval('model'), function_args=[eval('test_input')], function_kwargs={'training': eval('False')}, max_wait_secs=0, custom_class=None)
    fig = plt.figure(figsize=(4, 4))
    for i in range(predictions.shape[0]):
        plt.subplot(4, 4, i + 1)
        plt.imshow(predictions[i, :, :, 0] * 127.5 + 127.5, cmap='gray')
        plt.axis('off')
    plt.savefig('image_at_epoch_{:04d}.png'.format(epoch))
    plt.show()
train(train_dataset, EPOCHS)
custom_method(
checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir)), imports='from tensorflow.keras import layers;import imageio;import glob;import tensorflow as tf;import numpy as np;import PIL;import time;import os;from IPython import display;import tensorflow_docs.vis.embed as embed;import matplotlib.pyplot as plt', function_to_run='obj.restore(*args)', method_object=eval('checkpoint'), function_args=[eval('tf.train.latest_checkpoint(checkpoint_dir)')], function_kwargs={}, max_wait_secs=0, custom_class=None)

def display_image(epoch_no):
    return PIL.Image.open('image_at_epoch_{:04d}.png'.format(epoch_no))
display_image(EPOCHS)
anim_file = 'dcgan.gif'
with imageio.get_writer(anim_file, mode='I') as writer:
    filenames = glob.glob('image*.png')
    filenames = sorted(filenames)
    for filename in filenames:
        image = imageio.imread(filename)
        writer.append_data(image)
    image = imageio.imread(filename)
    writer.append_data(image)
import tensorflow_docs.vis.embed as embed
embed.embed_file(anim_file)