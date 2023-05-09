import tensorflow as tf
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
    result = send_request(imports=imports, function_to_run=function_to_run, function_args=function_args, function_kwargs=function_kwargs, max_wait_secs=MAX_WAIT_S, wait_after_run_secs=WAIT_AFTER_RUN_S, method_object=method_object, object_signature=object_signature, custom_class=custom_class, experiment_file_path=EXPERIMENT_FILE_PATH)
    return func
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
tf.keras.datasets.mnist.load_data(), imports='import tensorflow_docs.vis.embed as embed;import os;import numpy as np;import matplotlib.pyplot as plt;from tensorflow.keras import layers;import tensorflow as tf;import imageio;import PIL;import glob;import time;from IPython import display', function_to_run='tf.keras.datasets.mnist.load_data()', method_object=None, object_signature=None, function_args=[], function_kwargs={})
train_images = train_images.reshape(train_images.shape[0], 28, 28, 1).astype('float32')
train_images = (train_images - 127.5) / 127.5
BUFFER_SIZE = 60000
BATCH_SIZE = 256
train_dataset = custom_method(
tf.data.Dataset.from_tensor_slices(train_images).shuffle(BUFFER_SIZE).batch(BATCH_SIZE), imports='import tensorflow_docs.vis.embed as embed;import os;import numpy as np;import matplotlib.pyplot as plt;from tensorflow.keras import layers;import tensorflow as tf;import imageio;import PIL;import glob;import time;from IPython import display', function_to_run='tf.data.Dataset.from_tensor_slices(train_images).shuffle(BUFFER_SIZE).batch(*args)', method_object=None, object_signature=None, function_args=[eval('BATCH_SIZE')], function_kwargs={})

def make_generator_model():
    model = custom_method(
    tf.keras.Sequential(), imports='import tensorflow_docs.vis.embed as embed;import os;import numpy as np;import matplotlib.pyplot as plt;from tensorflow.keras import layers;import tensorflow as tf;import imageio;import PIL;import glob;import time;from IPython import display', function_to_run='tf.keras.Sequential()', method_object=None, object_signature=None, function_args=[], function_kwargs={})
    model.add(layers.Dense(7 * 7 * 256, use_bias=False, input_shape=(100,)))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())
    model.add(layers.Reshape((7, 7, 256)))
    assert model.output_shape == (None, 7, 7, 256)
    model.add(layers.Conv2DTranspose(128, (5, 5), strides=(1, 1), padding='same', use_bias=False))
    assert model.output_shape == (None, 7, 7, 128)
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())
    model.add(layers.Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same', use_bias=False))
    assert model.output_shape == (None, 14, 14, 64)
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())
    model.add(layers.Conv2DTranspose(1, (5, 5), strides=(2, 2), padding='same', use_bias=False, activation='tanh'))
    assert model.output_shape == (None, 28, 28, 1)
    return model
generator = make_generator_model()
noise = custom_method(
tf.random.normal([1, 100]), imports='import tensorflow_docs.vis.embed as embed;import os;import numpy as np;import matplotlib.pyplot as plt;from tensorflow.keras import layers;import tensorflow as tf;import imageio;import PIL;import glob;import time;from IPython import display', function_to_run='tf.random.normal(*args)', method_object=None, object_signature=None, function_args=[eval('[1, 100]')], function_kwargs={})
generated_image = generator(noise, training=False)
plt.imshow(generated_image[0, :, :, 0], cmap='gray')

def make_discriminator_model():
    model = custom_method(
    tf.keras.Sequential(), imports='import tensorflow_docs.vis.embed as embed;import os;import numpy as np;import matplotlib.pyplot as plt;from tensorflow.keras import layers;import tensorflow as tf;import imageio;import PIL;import glob;import time;from IPython import display', function_to_run='tf.keras.Sequential()', method_object=None, object_signature=None, function_args=[], function_kwargs={})
    model.add(layers.Conv2D(64, (5, 5), strides=(2, 2), padding='same', input_shape=[28, 28, 1]))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))
    model.add(layers.Conv2D(128, (5, 5), strides=(2, 2), padding='same'))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))
    model.add(layers.Flatten())
    model.add(layers.Dense(1))
    return model
discriminator = make_discriminator_model()
decision = discriminator(generated_image)
print(decision)
cross_entropy = custom_method(
tf.keras.losses.BinaryCrossentropy(from_logits=True), imports='import tensorflow_docs.vis.embed as embed;import os;import numpy as np;import matplotlib.pyplot as plt;from tensorflow.keras import layers;import tensorflow as tf;import imageio;import PIL;import glob;import time;from IPython import display', function_to_run='tf.keras.losses.BinaryCrossentropy(**kwargs)', method_object=None, object_signature=None, function_args=[], function_kwargs={'from_logits': eval('True')})

def discriminator_loss(real_output, fake_output):
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    total_loss = real_loss + fake_loss
    return total_loss

def generator_loss(fake_output):
    return cross_entropy(tf.ones_like(fake_output), fake_output)
generator_optimizer = custom_method(
tf.keras.optimizers.Adam(0.0001), imports='import tensorflow_docs.vis.embed as embed;import os;import numpy as np;import matplotlib.pyplot as plt;from tensorflow.keras import layers;import tensorflow as tf;import imageio;import PIL;import glob;import time;from IPython import display', function_to_run='tf.keras.optimizers.Adam(*args)', method_object=None, object_signature=None, function_args=[eval('1e-4')], function_kwargs={})
discriminator_optimizer = custom_method(
tf.keras.optimizers.Adam(0.0001), imports='import tensorflow_docs.vis.embed as embed;import os;import numpy as np;import matplotlib.pyplot as plt;from tensorflow.keras import layers;import tensorflow as tf;import imageio;import PIL;import glob;import time;from IPython import display', function_to_run='tf.keras.optimizers.Adam(*args)', method_object=None, object_signature=None, function_args=[eval('1e-4')], function_kwargs={})
checkpoint_dir = './training_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, 'ckpt')
checkpoint = custom_method(
tf.train.Checkpoint(generator_optimizer=generator_optimizer, discriminator_optimizer=discriminator_optimizer, generator=generator, discriminator=discriminator), imports='import tensorflow_docs.vis.embed as embed;import os;import numpy as np;import matplotlib.pyplot as plt;from tensorflow.keras import layers;import tensorflow as tf;import imageio;import PIL;import glob;import time;from IPython import display', function_to_run='tf.train.Checkpoint(**kwargs)', method_object=None, object_signature=None, function_args=[], function_kwargs={'generator_optimizer': eval('generator_optimizer'), 'discriminator_optimizer': eval('discriminator_optimizer'), 'generator': eval('generator'), 'discriminator': eval('discriminator')})
EPOCHS = 50
noise_dim = 100
num_examples_to_generate = 16
seed = custom_method(
tf.random.normal([num_examples_to_generate, noise_dim]), imports='import tensorflow_docs.vis.embed as embed;import os;import numpy as np;import matplotlib.pyplot as plt;from tensorflow.keras import layers;import tensorflow as tf;import imageio;import PIL;import glob;import time;from IPython import display', function_to_run='tf.random.normal(*args)', method_object=None, object_signature=None, function_args=[eval('[num_examples_to_generate, noise_dim]')], function_kwargs={})

@tf.function
def train_step(images):
    noise = custom_method(
    tf.random.normal([BATCH_SIZE, noise_dim]), imports='import tensorflow_docs.vis.embed as embed;import os;import numpy as np;import matplotlib.pyplot as plt;from tensorflow.keras import layers;import tensorflow as tf;import imageio;import PIL;import glob;import time;from IPython import display', function_to_run='tf.random.normal(*args)', method_object=None, object_signature=None, function_args=[eval('[BATCH_SIZE, noise_dim]')], function_kwargs={})
    with custom_method(
    tf.GradientTape(), imports='import tensorflow_docs.vis.embed as embed;import os;import numpy as np;import matplotlib.pyplot as plt;from tensorflow.keras import layers;import tensorflow as tf;import imageio;import PIL;import glob;import time;from IPython import display', function_to_run='tf.GradientTape()', method_object=None, object_signature=None, function_args=[], function_kwargs={}) as gen_tape, custom_method(
    tf.GradientTape(), imports='import tensorflow_docs.vis.embed as embed;import os;import numpy as np;import matplotlib.pyplot as plt;from tensorflow.keras import layers;import tensorflow as tf;import imageio;import PIL;import glob;import time;from IPython import display', function_to_run='tf.GradientTape()', method_object=None, object_signature=None, function_args=[], function_kwargs={}) as disc_tape:
        generated_images = generator(noise, training=True)
        real_output = discriminator(images, training=True)
        fake_output = discriminator(generated_images, training=True)
        gen_loss = generator_loss(fake_output)
        disc_loss = discriminator_loss(real_output, fake_output)
    gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
    gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)
    generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))

def train(dataset, epochs):
    for epoch in range(epochs):
        start = time.time()
        for image_batch in dataset:
            train_step(image_batch)
        display.clear_output(wait=True)
        generate_and_save_images(generator, epoch + 1, seed)
        if (epoch + 1) % 15 == 0:
            checkpoint.save(file_prefix=checkpoint_prefix)
        print('Time for epoch {} is {} sec'.format(epoch + 1, time.time() - start))
    display.clear_output(wait=True)
    generate_and_save_images(generator, epochs, seed)

def generate_and_save_images(model, epoch, test_input):
    predictions = model(test_input, training=False)
    fig = plt.figure(figsize=(4, 4))
    for i in range(predictions.shape[0]):
        plt.subplot(4, 4, i + 1)
        plt.imshow(predictions[i, :, :, 0] * 127.5 + 127.5, cmap='gray')
        plt.axis('off')
    plt.savefig('image_at_epoch_{:04d}.png'.format(epoch))
    plt.show()
train(train_dataset, EPOCHS)
checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))

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
