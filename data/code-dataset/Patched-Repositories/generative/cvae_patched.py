from IPython import display
import glob
import imageio
import matplotlib.pyplot as plt
import numpy as np
import PIL
import tensorflow as tf
import tensorflow_probability as tfp
import time
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
((train_images, _), (test_images, _)) = custom_method(
tf.keras.datasets.mnist.load_data(), imports='import imageio;import time;import matplotlib.pyplot as plt;import tensorflow as tf;import numpy as np;import tensorflow_probability as tfp;import tensorflow_docs.vis.embed as embed;import glob;import PIL;from IPython import display', function_to_run='tf.keras.datasets.mnist.load_data()', method_object=None, object_signature=None, function_args=[], function_kwargs={})

def preprocess_images(images):
    images = images.reshape((images.shape[0], 28, 28, 1)) / 255.0
    return np.where(images > 0.5, 1.0, 0.0).astype('float32')
train_images = preprocess_images(train_images)
test_images = preprocess_images(test_images)
train_size = 60000
batch_size = 32
test_size = 10000
train_dataset = custom_method(
tf.data.Dataset.from_tensor_slices(train_images).shuffle(train_size).batch(batch_size), imports='import imageio;import time;import matplotlib.pyplot as plt;import tensorflow as tf;import numpy as np;import tensorflow_probability as tfp;import tensorflow_docs.vis.embed as embed;import glob;import PIL;from IPython import display', function_to_run='tf.data.Dataset.from_tensor_slices(train_images).shuffle(train_size).batch(*args)', method_object=None, object_signature=None, function_args=[eval('batch_size')], function_kwargs={})
test_dataset = custom_method(
tf.data.Dataset.from_tensor_slices(test_images).shuffle(test_size).batch(batch_size), imports='import imageio;import time;import matplotlib.pyplot as plt;import tensorflow as tf;import numpy as np;import tensorflow_probability as tfp;import tensorflow_docs.vis.embed as embed;import glob;import PIL;from IPython import display', function_to_run='tf.data.Dataset.from_tensor_slices(test_images).shuffle(test_size).batch(*args)', method_object=None, object_signature=None, function_args=[eval('batch_size')], function_kwargs={})

class CVAE(tf.keras.Model):
    """Convolutional variational autoencoder."""

    def __init__(self, latent_dim):
        super(CVAE, self).__init__()
        self.latent_dim = latent_dim
        self.encoder = custom_method(
        tf.keras.Sequential([tf.keras.layers.InputLayer(input_shape=(28, 28, 1)), tf.keras.layers.Conv2D(filters=32, kernel_size=3, strides=(2, 2), activation='relu'), tf.keras.layers.Conv2D(filters=64, kernel_size=3, strides=(2, 2), activation='relu'), tf.keras.layers.Flatten(), tf.keras.layers.Dense(latent_dim + latent_dim)]), imports='import imageio;import time;import matplotlib.pyplot as plt;import tensorflow as tf;import numpy as np;import tensorflow_probability as tfp;import tensorflow_docs.vis.embed as embed;import glob;import PIL;from IPython import display', function_to_run='tf.keras.Sequential(*args)', method_object=None, object_signature=None, function_args=[eval("[\n            tf.keras.layers.InputLayer(input_shape=(28, 28, 1)),\n            tf.keras.layers.Conv2D(\n                filters=32, kernel_size=3, strides=(2, 2), activation='relu'),\n            tf.keras.layers.Conv2D(\n                filters=64, kernel_size=3, strides=(2, 2), activation='relu'),\n            tf.keras.layers.Flatten(),\n            tf.keras.layers.Dense(latent_dim + latent_dim),\n        ]")], function_kwargs={})
        self.decoder = custom_method(
        tf.keras.Sequential([tf.keras.layers.InputLayer(input_shape=(latent_dim,)), tf.keras.layers.Dense(units=7 * 7 * 32, activation=tf.nn.relu), tf.keras.layers.Reshape(target_shape=(7, 7, 32)), tf.keras.layers.Conv2DTranspose(filters=64, kernel_size=3, strides=2, padding='same', activation='relu'), tf.keras.layers.Conv2DTranspose(filters=32, kernel_size=3, strides=2, padding='same', activation='relu'), tf.keras.layers.Conv2DTranspose(filters=1, kernel_size=3, strides=1, padding='same')]), imports='import imageio;import time;import matplotlib.pyplot as plt;import tensorflow as tf;import numpy as np;import tensorflow_probability as tfp;import tensorflow_docs.vis.embed as embed;import glob;import PIL;from IPython import display', function_to_run='tf.keras.Sequential(*args)', method_object=None, object_signature=None, function_args=[eval("[\n            tf.keras.layers.InputLayer(input_shape=(latent_dim,)),\n            tf.keras.layers.Dense(units=7*7*32, activation=tf.nn.relu),\n            tf.keras.layers.Reshape(target_shape=(7, 7, 32)),\n            tf.keras.layers.Conv2DTranspose(\n                filters=64, kernel_size=3, strides=2, padding='same',\n                activation='relu'),\n            tf.keras.layers.Conv2DTranspose(\n                filters=32, kernel_size=3, strides=2, padding='same',\n                activation='relu'),\n            tf.keras.layers.Conv2DTranspose(\n                filters=1, kernel_size=3, strides=1, padding='same'),\n        ]")], function_kwargs={})

    @tf.function
    def sample(self, eps=None):
        if eps is None:
            eps = custom_method(
            tf.random.normal(shape=(100, self.latent_dim)), imports='import imageio;import time;import matplotlib.pyplot as plt;import tensorflow as tf;import numpy as np;import tensorflow_probability as tfp;import tensorflow_docs.vis.embed as embed;import glob;import PIL;from IPython import display', function_to_run='tf.random.normal(**kwargs)', method_object=None, object_signature=None, function_args=[], function_kwargs={'shape': eval('(100, self.latent_dim)')})
        return self.decode(eps, apply_sigmoid=True)

    def encode(self, x):
        (mean, logvar) = custom_method(
        tf.split(self.encoder(x), num_or_size_splits=2, axis=1), imports='import imageio;import time;import matplotlib.pyplot as plt;import tensorflow as tf;import numpy as np;import tensorflow_probability as tfp;import tensorflow_docs.vis.embed as embed;import glob;import PIL;from IPython import display', function_to_run='tf.split(*args, **kwargs)', method_object=None, object_signature=None, function_args=[eval('self.encoder(x)')], function_kwargs={'num_or_size_splits': eval('2'), 'axis': eval('1')})
        return (mean, logvar)

    def reparameterize(self, mean, logvar):
        eps = custom_method(
        tf.random.normal(shape=mean.shape), imports='import imageio;import time;import matplotlib.pyplot as plt;import tensorflow as tf;import numpy as np;import tensorflow_probability as tfp;import tensorflow_docs.vis.embed as embed;import glob;import PIL;from IPython import display', function_to_run='tf.random.normal(**kwargs)', method_object=None, object_signature=None, function_args=[], function_kwargs={'shape': eval('mean.shape')})
        return eps * custom_method(
        tf.exp(logvar * 0.5), imports='import imageio;import time;import matplotlib.pyplot as plt;import tensorflow as tf;import numpy as np;import tensorflow_probability as tfp;import tensorflow_docs.vis.embed as embed;import glob;import PIL;from IPython import display', function_to_run='tf.exp(*args)', method_object=None, object_signature=None, function_args=[eval('logvar * .5')], function_kwargs={}) + mean

    def decode(self, z, apply_sigmoid=False):
        logits = self.decoder(z)
        if apply_sigmoid:
            probs = custom_method(
            tf.sigmoid(logits), imports='import imageio;import time;import matplotlib.pyplot as plt;import tensorflow as tf;import numpy as np;import tensorflow_probability as tfp;import tensorflow_docs.vis.embed as embed;import glob;import PIL;from IPython import display', function_to_run='tf.sigmoid(*args)', method_object=None, object_signature=None, function_args=[eval('logits')], function_kwargs={})
            return probs
        return logits
optimizer = custom_method(
tf.keras.optimizers.Adam(0.0001), imports='import imageio;import time;import matplotlib.pyplot as plt;import tensorflow as tf;import numpy as np;import tensorflow_probability as tfp;import tensorflow_docs.vis.embed as embed;import glob;import PIL;from IPython import display', function_to_run='tf.keras.optimizers.Adam(*args)', method_object=None, object_signature=None, function_args=[eval('1e-4')], function_kwargs={})

def log_normal_pdf(sample, mean, logvar, raxis=1):
    log2pi = custom_method(
    tf.math.log(2.0 * np.pi), imports='import imageio;import time;import matplotlib.pyplot as plt;import tensorflow as tf;import numpy as np;import tensorflow_probability as tfp;import tensorflow_docs.vis.embed as embed;import glob;import PIL;from IPython import display', function_to_run='tf.math.log(*args)', method_object=None, object_signature=None, function_args=[eval('2. * np.pi')], function_kwargs={})
    return custom_method(
    tf.reduce_sum(-0.5 * ((sample - mean) ** 2.0 * tf.exp(-logvar) + logvar + log2pi), axis=raxis), imports='import imageio;import time;import matplotlib.pyplot as plt;import tensorflow as tf;import numpy as np;import tensorflow_probability as tfp;import tensorflow_docs.vis.embed as embed;import glob;import PIL;from IPython import display', function_to_run='tf.reduce_sum(*args, **kwargs)', method_object=None, object_signature=None, function_args=[eval('-.5 * ((sample - mean) ** 2. * tf.exp(-logvar) + logvar + log2pi)')], function_kwargs={'axis': eval('raxis')})

def compute_loss(model, x):
    (mean, logvar) = model.encode(x)
    z = model.reparameterize(mean, logvar)
    x_logit = model.decode(z)
    cross_ent = custom_method(
    tf.nn.sigmoid_cross_entropy_with_logits(logits=x_logit, labels=x), imports='import imageio;import time;import matplotlib.pyplot as plt;import tensorflow as tf;import numpy as np;import tensorflow_probability as tfp;import tensorflow_docs.vis.embed as embed;import glob;import PIL;from IPython import display', function_to_run='tf.nn.sigmoid_cross_entropy_with_logits(**kwargs)', method_object=None, object_signature=None, function_args=[], function_kwargs={'logits': eval('x_logit'), 'labels': eval('x')})
    logpx_z = -tf.reduce_sum(cross_ent, axis=[1, 2, 3])
    logpz = log_normal_pdf(z, 0.0, 0.0)
    logqz_x = log_normal_pdf(z, mean, logvar)
    return -custom_method(
    tf.reduce_mean(logpx_z + logpz - logqz_x), imports='import imageio;import time;import matplotlib.pyplot as plt;import tensorflow as tf;import numpy as np;import tensorflow_probability as tfp;import tensorflow_docs.vis.embed as embed;import glob;import PIL;from IPython import display', function_to_run='tf.reduce_mean(*args)', method_object=None, object_signature=None, function_args=[eval('logpx_z + logpz - logqz_x')], function_kwargs={})

@tf.function
def train_step(model, x, optimizer):
    """Executes one training step and returns the loss.

  This function computes the loss and gradients, and uses the latter to
  update the model's parameters.
  """
    with custom_method(
    tf.GradientTape(), imports='import imageio;import time;import matplotlib.pyplot as plt;import tensorflow as tf;import numpy as np;import tensorflow_probability as tfp;import tensorflow_docs.vis.embed as embed;import glob;import PIL;from IPython import display', function_to_run='tf.GradientTape()', method_object=None, object_signature=None, function_args=[], function_kwargs={}) as tape:
        loss = compute_loss(model, x)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
epochs = 10
latent_dim = 2
num_examples_to_generate = 16
random_vector_for_generation = custom_method(
tf.random.normal(shape=[num_examples_to_generate, latent_dim]), imports='import imageio;import time;import matplotlib.pyplot as plt;import tensorflow as tf;import numpy as np;import tensorflow_probability as tfp;import tensorflow_docs.vis.embed as embed;import glob;import PIL;from IPython import display', function_to_run='tf.random.normal(**kwargs)', method_object=None, object_signature=None, function_args=[], function_kwargs={'shape': eval('[num_examples_to_generate, latent_dim]')})
model = CVAE(latent_dim)

def generate_and_save_images(model, epoch, test_sample):
    (mean, logvar) = custom_method(
    model.encode(test_sample), imports='import imageio;import time;import matplotlib.pyplot as plt;import tensorflow as tf;import numpy as np;import tensorflow_probability as tfp;import tensorflow_docs.vis.embed as embed;import glob;import PIL;from IPython import display', function_to_run='obj.encode(*args)', method_object=eval('model'), object_signature=None, function_args=[eval('test_sample')], function_kwargs={}, custom_class='class CVAE(tf.keras.Model):\n  """Convolutional variational autoencoder."""\n\n  def __init__(self, latent_dim):\n    super(CVAE, self).__init__()\n    self.latent_dim = latent_dim\n    self.encoder = tf.keras.Sequential(\n        [\n            tf.keras.layers.InputLayer(input_shape=(28, 28, 1)),\n            tf.keras.layers.Conv2D(\n                filters=32, kernel_size=3, strides=(2, 2), activation=\'relu\'),\n            tf.keras.layers.Conv2D(\n                filters=64, kernel_size=3, strides=(2, 2), activation=\'relu\'),\n            tf.keras.layers.Flatten(),\n            tf.keras.layers.Dense(latent_dim + latent_dim),\n        ]\n    )\n\n    self.decoder = tf.keras.Sequential(\n        [\n            tf.keras.layers.InputLayer(input_shape=(latent_dim,)),\n            tf.keras.layers.Dense(units=7*7*32, activation=tf.nn.relu),\n            tf.keras.layers.Reshape(target_shape=(7, 7, 32)),\n            tf.keras.layers.Conv2DTranspose(\n                filters=64, kernel_size=3, strides=2, padding=\'same\',\n                activation=\'relu\'),\n            tf.keras.layers.Conv2DTranspose(\n                filters=32, kernel_size=3, strides=2, padding=\'same\',\n                activation=\'relu\'),\n            tf.keras.layers.Conv2DTranspose(\n                filters=1, kernel_size=3, strides=1, padding=\'same\'),\n        ]\n    )\n\n  @tf.function\n  def sample(self, eps=None):\n    if eps is None:\n      eps = tf.random.normal(shape=(100, self.latent_dim))\n    return self.decode(eps, apply_sigmoid=True)\n\n  def encode(self, x):\n    mean, logvar = tf.split(self.encoder(x), num_or_size_splits=2, axis=1)\n    return mean, logvar\n\n  def reparameterize(self, mean, logvar):\n    eps = tf.random.normal(shape=mean.shape)\n    return eps * tf.exp(logvar * .5) + mean\n\n  def decode(self, z, apply_sigmoid=False):\n    logits = self.decoder(z)\n    if apply_sigmoid:\n      probs = tf.sigmoid(logits)\n      return probs\n    return logits')
    z = custom_method(
    model.reparameterize(mean, logvar), imports='import imageio;import time;import matplotlib.pyplot as plt;import tensorflow as tf;import numpy as np;import tensorflow_probability as tfp;import tensorflow_docs.vis.embed as embed;import glob;import PIL;from IPython import display', function_to_run='obj.reparameterize(*args)', method_object=eval('model'), object_signature=None, function_args=[eval('mean'), eval('logvar')], function_kwargs={}, custom_class='class CVAE(tf.keras.Model):\n  """Convolutional variational autoencoder."""\n\n  def __init__(self, latent_dim):\n    super(CVAE, self).__init__()\n    self.latent_dim = latent_dim\n    self.encoder = tf.keras.Sequential(\n        [\n            tf.keras.layers.InputLayer(input_shape=(28, 28, 1)),\n            tf.keras.layers.Conv2D(\n                filters=32, kernel_size=3, strides=(2, 2), activation=\'relu\'),\n            tf.keras.layers.Conv2D(\n                filters=64, kernel_size=3, strides=(2, 2), activation=\'relu\'),\n            tf.keras.layers.Flatten(),\n            tf.keras.layers.Dense(latent_dim + latent_dim),\n        ]\n    )\n\n    self.decoder = tf.keras.Sequential(\n        [\n            tf.keras.layers.InputLayer(input_shape=(latent_dim,)),\n            tf.keras.layers.Dense(units=7*7*32, activation=tf.nn.relu),\n            tf.keras.layers.Reshape(target_shape=(7, 7, 32)),\n            tf.keras.layers.Conv2DTranspose(\n                filters=64, kernel_size=3, strides=2, padding=\'same\',\n                activation=\'relu\'),\n            tf.keras.layers.Conv2DTranspose(\n                filters=32, kernel_size=3, strides=2, padding=\'same\',\n                activation=\'relu\'),\n            tf.keras.layers.Conv2DTranspose(\n                filters=1, kernel_size=3, strides=1, padding=\'same\'),\n        ]\n    )\n\n  @tf.function\n  def sample(self, eps=None):\n    if eps is None:\n      eps = tf.random.normal(shape=(100, self.latent_dim))\n    return self.decode(eps, apply_sigmoid=True)\n\n  def encode(self, x):\n    mean, logvar = tf.split(self.encoder(x), num_or_size_splits=2, axis=1)\n    return mean, logvar\n\n  def reparameterize(self, mean, logvar):\n    eps = tf.random.normal(shape=mean.shape)\n    return eps * tf.exp(logvar * .5) + mean\n\n  def decode(self, z, apply_sigmoid=False):\n    logits = self.decoder(z)\n    if apply_sigmoid:\n      probs = tf.sigmoid(logits)\n      return probs\n    return logits')
    predictions = custom_method(
    model.sample(z), imports='import imageio;import time;import matplotlib.pyplot as plt;import tensorflow as tf;import numpy as np;import tensorflow_probability as tfp;import tensorflow_docs.vis.embed as embed;import glob;import PIL;from IPython import display', function_to_run='obj.sample(*args)', method_object=eval('model'), object_signature=None, function_args=[eval('z')], function_kwargs={}, custom_class='class CVAE(tf.keras.Model):\n  """Convolutional variational autoencoder."""\n\n  def __init__(self, latent_dim):\n    super(CVAE, self).__init__()\n    self.latent_dim = latent_dim\n    self.encoder = tf.keras.Sequential(\n        [\n            tf.keras.layers.InputLayer(input_shape=(28, 28, 1)),\n            tf.keras.layers.Conv2D(\n                filters=32, kernel_size=3, strides=(2, 2), activation=\'relu\'),\n            tf.keras.layers.Conv2D(\n                filters=64, kernel_size=3, strides=(2, 2), activation=\'relu\'),\n            tf.keras.layers.Flatten(),\n            tf.keras.layers.Dense(latent_dim + latent_dim),\n        ]\n    )\n\n    self.decoder = tf.keras.Sequential(\n        [\n            tf.keras.layers.InputLayer(input_shape=(latent_dim,)),\n            tf.keras.layers.Dense(units=7*7*32, activation=tf.nn.relu),\n            tf.keras.layers.Reshape(target_shape=(7, 7, 32)),\n            tf.keras.layers.Conv2DTranspose(\n                filters=64, kernel_size=3, strides=2, padding=\'same\',\n                activation=\'relu\'),\n            tf.keras.layers.Conv2DTranspose(\n                filters=32, kernel_size=3, strides=2, padding=\'same\',\n                activation=\'relu\'),\n            tf.keras.layers.Conv2DTranspose(\n                filters=1, kernel_size=3, strides=1, padding=\'same\'),\n        ]\n    )\n\n  @tf.function\n  def sample(self, eps=None):\n    if eps is None:\n      eps = tf.random.normal(shape=(100, self.latent_dim))\n    return self.decode(eps, apply_sigmoid=True)\n\n  def encode(self, x):\n    mean, logvar = tf.split(self.encoder(x), num_or_size_splits=2, axis=1)\n    return mean, logvar\n\n  def reparameterize(self, mean, logvar):\n    eps = tf.random.normal(shape=mean.shape)\n    return eps * tf.exp(logvar * .5) + mean\n\n  def decode(self, z, apply_sigmoid=False):\n    logits = self.decoder(z)\n    if apply_sigmoid:\n      probs = tf.sigmoid(logits)\n      return probs\n    return logits')
    fig = plt.figure(figsize=(4, 4))
    for i in range(predictions.shape[0]):
        plt.subplot(4, 4, i + 1)
        plt.imshow(predictions[i, :, :, 0], cmap='gray')
        plt.axis('off')
    plt.savefig('image_at_epoch_{:04d}.png'.format(epoch))
    plt.show()
assert batch_size >= num_examples_to_generate
for test_batch in test_dataset.take(1):
    test_sample = test_batch[0:num_examples_to_generate, :, :, :]
generate_and_save_images(model, 0, test_sample)
for epoch in range(1, epochs + 1):
    start_time = time.time()
    for train_x in train_dataset:
        train_step(model, train_x, optimizer)
    end_time = time.time()
    loss = custom_method(
    tf.keras.metrics.Mean(), imports='import imageio;import time;import matplotlib.pyplot as plt;import tensorflow as tf;import numpy as np;import tensorflow_probability as tfp;import tensorflow_docs.vis.embed as embed;import glob;import PIL;from IPython import display', function_to_run='tf.keras.metrics.Mean()', method_object=None, object_signature=None, function_args=[], function_kwargs={})
    for test_x in test_dataset:
        loss(compute_loss(model, test_x))
    elbo = -loss.result()
    display.clear_output(wait=False)
    print('Epoch: {}, Test set ELBO: {}, time elapse for current epoch: {}'.format(epoch, elbo, end_time - start_time))
    generate_and_save_images(model, epoch, test_sample)

def display_image(epoch_no):
    return PIL.Image.open('image_at_epoch_{:04d}.png'.format(epoch_no))
plt.imshow(display_image(epoch))
plt.axis('off')
anim_file = 'cvae.gif'
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

def plot_latent_images(model, n, digit_size=28):
    """Plots n x n digit images decoded from the latent space."""
    norm = tfp.distributions.Normal(0, 1)
    grid_x = norm.quantile(np.linspace(0.05, 0.95, n))
    grid_y = norm.quantile(np.linspace(0.05, 0.95, n))
    image_width = digit_size * n
    image_height = image_width
    image = np.zeros((image_height, image_width))
    for (i, yi) in enumerate(grid_x):
        for (j, xi) in enumerate(grid_y):
            z = np.array([[xi, yi]])
            x_decoded = custom_method(
            model.sample(z), imports='import imageio;import time;import matplotlib.pyplot as plt;import tensorflow as tf;import numpy as np;import tensorflow_probability as tfp;import tensorflow_docs.vis.embed as embed;import glob;import PIL;from IPython import display', function_to_run='obj.sample(*args)', method_object=eval('model'), object_signature=None, function_args=[eval('z')], function_kwargs={}, custom_class='class CVAE(tf.keras.Model):\n  """Convolutional variational autoencoder."""\n\n  def __init__(self, latent_dim):\n    super(CVAE, self).__init__()\n    self.latent_dim = latent_dim\n    self.encoder = tf.keras.Sequential(\n        [\n            tf.keras.layers.InputLayer(input_shape=(28, 28, 1)),\n            tf.keras.layers.Conv2D(\n                filters=32, kernel_size=3, strides=(2, 2), activation=\'relu\'),\n            tf.keras.layers.Conv2D(\n                filters=64, kernel_size=3, strides=(2, 2), activation=\'relu\'),\n            tf.keras.layers.Flatten(),\n            tf.keras.layers.Dense(latent_dim + latent_dim),\n        ]\n    )\n\n    self.decoder = tf.keras.Sequential(\n        [\n            tf.keras.layers.InputLayer(input_shape=(latent_dim,)),\n            tf.keras.layers.Dense(units=7*7*32, activation=tf.nn.relu),\n            tf.keras.layers.Reshape(target_shape=(7, 7, 32)),\n            tf.keras.layers.Conv2DTranspose(\n                filters=64, kernel_size=3, strides=2, padding=\'same\',\n                activation=\'relu\'),\n            tf.keras.layers.Conv2DTranspose(\n                filters=32, kernel_size=3, strides=2, padding=\'same\',\n                activation=\'relu\'),\n            tf.keras.layers.Conv2DTranspose(\n                filters=1, kernel_size=3, strides=1, padding=\'same\'),\n        ]\n    )\n\n  @tf.function\n  def sample(self, eps=None):\n    if eps is None:\n      eps = tf.random.normal(shape=(100, self.latent_dim))\n    return self.decode(eps, apply_sigmoid=True)\n\n  def encode(self, x):\n    mean, logvar = tf.split(self.encoder(x), num_or_size_splits=2, axis=1)\n    return mean, logvar\n\n  def reparameterize(self, mean, logvar):\n    eps = tf.random.normal(shape=mean.shape)\n    return eps * tf.exp(logvar * .5) + mean\n\n  def decode(self, z, apply_sigmoid=False):\n    logits = self.decoder(z)\n    if apply_sigmoid:\n      probs = tf.sigmoid(logits)\n      return probs\n    return logits')
            digit = custom_method(
            tf.reshape(x_decoded[0], (digit_size, digit_size)), imports='import imageio;import time;import matplotlib.pyplot as plt;import tensorflow as tf;import numpy as np;import tensorflow_probability as tfp;import tensorflow_docs.vis.embed as embed;import glob;import PIL;from IPython import display', function_to_run='tf.reshape(*args)', method_object=None, object_signature=None, function_args=[eval('x_decoded[0]'), eval('(digit_size, digit_size)')], function_kwargs={})
            image[i * digit_size:(i + 1) * digit_size, j * digit_size:(j + 1) * digit_size] = digit.numpy()
    plt.figure(figsize=(10, 10))
    plt.imshow(image, cmap='Greys_r')
    plt.axis('Off')
    plt.show()
plot_latent_images(model, 20)
