import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_compression as tfc
import tensorflow_datasets as tfds
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

def make_analysis_transform(latent_dims):
    """Creates the analysis (encoder) transform."""
    return custom_method(
    tf.keras.Sequential([tf.keras.layers.Conv2D(20, 5, use_bias=True, strides=2, padding='same', activation='leaky_relu', name='conv_1'), tf.keras.layers.Conv2D(50, 5, use_bias=True, strides=2, padding='same', activation='leaky_relu', name='conv_2'), tf.keras.layers.Flatten(), tf.keras.layers.Dense(500, use_bias=True, activation='leaky_relu', name='fc_1'), tf.keras.layers.Dense(latent_dims, use_bias=True, activation=None, name='fc_2')], name='analysis_transform'), imports='import tensorflow as tf;import tensorflow_compression as tfc;import matplotlib.pyplot as plt;import os;import tensorflow_datasets as tfds', function_to_run='tf.keras.Sequential(*args, **kwargs)', method_object=None, object_signature=None, function_args=[eval('[\n      tf.keras.layers.Conv2D(\n          20, 5, use_bias=True, strides=2, padding="same",\n          activation="leaky_relu", name="conv_1"),\n      tf.keras.layers.Conv2D(\n          50, 5, use_bias=True, strides=2, padding="same",\n          activation="leaky_relu", name="conv_2"),\n      tf.keras.layers.Flatten(),\n      tf.keras.layers.Dense(\n          500, use_bias=True, activation="leaky_relu", name="fc_1"),\n      tf.keras.layers.Dense(\n          latent_dims, use_bias=True, activation=None, name="fc_2"),\n  ]')], function_kwargs={'name': eval('"analysis_transform"')})

def make_synthesis_transform():
    """Creates the synthesis (decoder) transform."""
    return custom_method(
    tf.keras.Sequential([tf.keras.layers.Dense(500, use_bias=True, activation='leaky_relu', name='fc_1'), tf.keras.layers.Dense(2450, use_bias=True, activation='leaky_relu', name='fc_2'), tf.keras.layers.Reshape((7, 7, 50)), tf.keras.layers.Conv2DTranspose(20, 5, use_bias=True, strides=2, padding='same', activation='leaky_relu', name='conv_1'), tf.keras.layers.Conv2DTranspose(1, 5, use_bias=True, strides=2, padding='same', activation='leaky_relu', name='conv_2')], name='synthesis_transform'), imports='import tensorflow as tf;import tensorflow_compression as tfc;import matplotlib.pyplot as plt;import os;import tensorflow_datasets as tfds', function_to_run='tf.keras.Sequential(*args, **kwargs)', method_object=None, object_signature=None, function_args=[eval('[\n      tf.keras.layers.Dense(\n          500, use_bias=True, activation="leaky_relu", name="fc_1"),\n      tf.keras.layers.Dense(\n          2450, use_bias=True, activation="leaky_relu", name="fc_2"),\n      tf.keras.layers.Reshape((7, 7, 50)),\n      tf.keras.layers.Conv2DTranspose(\n          20, 5, use_bias=True, strides=2, padding="same",\n          activation="leaky_relu", name="conv_1"),\n      tf.keras.layers.Conv2DTranspose(\n          1, 5, use_bias=True, strides=2, padding="same",\n          activation="leaky_relu", name="conv_2"),\n  ]')], function_kwargs={'name': eval('"synthesis_transform"')})

class MNISTCompressionTrainer(tf.keras.Model):
    """Model that trains a compressor/decompressor for MNIST."""

    def __init__(self, latent_dims):
        super().__init__()
        self.analysis_transform = make_analysis_transform(latent_dims)
        self.synthesis_transform = make_synthesis_transform()
        self.prior_log_scales = custom_method(
        tf.Variable(tf.zeros((latent_dims,))), imports='import tensorflow as tf;import tensorflow_compression as tfc;import matplotlib.pyplot as plt;import os;import tensorflow_datasets as tfds', function_to_run='tf.Variable(*args)', method_object=None, object_signature=None, function_args=[eval('tf.zeros((latent_dims,))')], function_kwargs={})

    @property
    def prior(self):
        return custom_method(
        tfc.NoisyLogistic(loc=0.0, scale=tf.exp(self.prior_log_scales)), imports='import tensorflow as tf;import tensorflow_compression as tfc;import matplotlib.pyplot as plt;import os;import tensorflow_datasets as tfds', function_to_run='tfc.NoisyLogistic(**kwargs)', method_object=None, object_signature=None, function_args=[], function_kwargs={'loc': eval('0.'), 'scale': eval('tf.exp(self.prior_log_scales)')})

    def call(self, x, training):
        """Computes rate and distortion losses."""
        x = custom_method(
        tf.cast(x, self.compute_dtype), imports='import tensorflow as tf;import tensorflow_compression as tfc;import matplotlib.pyplot as plt;import os;import tensorflow_datasets as tfds', function_to_run='tf.cast(*args)', method_object=None, object_signature=None, function_args=[eval('x'), eval('self.compute_dtype')], function_kwargs={}) / 255.0
        x = custom_method(
        tf.reshape(x, (-1, 28, 28, 1)), imports='import tensorflow as tf;import tensorflow_compression as tfc;import matplotlib.pyplot as plt;import os;import tensorflow_datasets as tfds', function_to_run='tf.reshape(*args)', method_object=None, object_signature=None, function_args=[eval('x'), eval('(-1, 28, 28, 1)')], function_kwargs={})
        y = self.analysis_transform(x)
        entropy_model = custom_method(
        tfc.ContinuousBatchedEntropyModel(self.prior, coding_rank=1, compression=False), imports='import tensorflow as tf;import tensorflow_compression as tfc;import matplotlib.pyplot as plt;import os;import tensorflow_datasets as tfds', function_to_run='tfc.ContinuousBatchedEntropyModel(*args, **kwargs)', method_object=None, object_signature=None, function_args=[eval('self.prior')], function_kwargs={'coding_rank': eval('1'), 'compression': eval('False')})
        (y_tilde, rate) = custom_method(
        entropy_model(y, training=training), imports='import tensorflow as tf;import tensorflow_compression as tfc;import matplotlib.pyplot as plt;import os;import tensorflow_datasets as tfds', function_to_run='obj(*args, **kwargs)', method_object=eval('entropy_model'), object_signature='tfc.ContinuousBatchedEntropyModel', function_args=[eval('y')], function_kwargs={'training': eval('training')}, custom_class=None)
        x_tilde = self.synthesis_transform(y_tilde)
        rate = custom_method(
        tf.reduce_mean(rate), imports='import tensorflow as tf;import tensorflow_compression as tfc;import matplotlib.pyplot as plt;import os;import tensorflow_datasets as tfds', function_to_run='tf.reduce_mean(*args)', method_object=None, object_signature=None, function_args=[eval('rate')], function_kwargs={})
        distortion = custom_method(
        tf.reduce_mean(abs(x - x_tilde)), imports='import tensorflow as tf;import tensorflow_compression as tfc;import matplotlib.pyplot as plt;import os;import tensorflow_datasets as tfds', function_to_run='tf.reduce_mean(*args)', method_object=None, object_signature=None, function_args=[eval('abs(x - x_tilde)')], function_kwargs={})
        return dict(rate=rate, distortion=distortion)
(training_dataset, validation_dataset) = custom_method(
tfds.load('mnist', split=['train', 'test'], shuffle_files=True, as_supervised=True, with_info=False), imports='import tensorflow as tf;import tensorflow_compression as tfc;import matplotlib.pyplot as plt;import os;import tensorflow_datasets as tfds', function_to_run='tfds.load(*args, **kwargs)', method_object=None, object_signature=None, function_args=[eval('"mnist"')], function_kwargs={'split': eval('["train", "test"]'), 'shuffle_files': eval('True'), 'as_supervised': eval('True'), 'with_info': eval('False')})
((x, _),) = validation_dataset.take(1)
plt.imshow(tf.squeeze(x))
print(f'Data type: {x.dtype}')
print(f'Shape: {x.shape}')
x = custom_method(
tf.cast(x, tf.float32), imports='import tensorflow as tf;import tensorflow_compression as tfc;import matplotlib.pyplot as plt;import os;import tensorflow_datasets as tfds', function_to_run='tf.cast(*args)', method_object=None, object_signature=None, function_args=[eval('x'), eval('tf.float32')], function_kwargs={}) / 255.0
x = custom_method(
tf.reshape(x, (-1, 28, 28, 1)), imports='import tensorflow as tf;import tensorflow_compression as tfc;import matplotlib.pyplot as plt;import os;import tensorflow_datasets as tfds', function_to_run='tf.reshape(*args)', method_object=None, object_signature=None, function_args=[eval('x'), eval('(-1, 28, 28, 1)')], function_kwargs={})
y = make_analysis_transform(10)(x)
print('y:', y)
y_tilde = y + custom_method(
tf.random.uniform(y.shape, -0.5, 0.5), imports='import tensorflow as tf;import tensorflow_compression as tfc;import matplotlib.pyplot as plt;import os;import tensorflow_datasets as tfds', function_to_run='tf.random.uniform(*args)', method_object=None, object_signature=None, function_args=[eval('y.shape'), eval('-.5'), eval('.5')], function_kwargs={})
print('y_tilde:', y_tilde)
prior = custom_method(
tfc.NoisyLogistic(loc=0.0, scale=tf.linspace(0.01, 2.0, 10)), imports='import tensorflow as tf;import tensorflow_compression as tfc;import matplotlib.pyplot as plt;import os;import tensorflow_datasets as tfds', function_to_run='tfc.NoisyLogistic(**kwargs)', method_object=None, object_signature=None, function_args=[], function_kwargs={'loc': eval('0.'), 'scale': eval('tf.linspace(.01, 2., 10)')})
_ = custom_method(
tf.linspace(-6.0, 6.0, 501), imports='import tensorflow as tf;import tensorflow_compression as tfc;import matplotlib.pyplot as plt;import os;import tensorflow_datasets as tfds', function_to_run='tf.linspace(*args)', method_object=None, object_signature=None, function_args=[eval('-6.'), eval('6.'), eval('501')], function_kwargs={})[:, None]
plt.plot(_, prior.prob(_))
entropy_model = custom_method(
tfc.ContinuousBatchedEntropyModel(prior, coding_rank=1, compression=False), imports='import tensorflow as tf;import tensorflow_compression as tfc;import matplotlib.pyplot as plt;import os;import tensorflow_datasets as tfds', function_to_run='tfc.ContinuousBatchedEntropyModel(*args, **kwargs)', method_object=None, object_signature=None, function_args=[eval('prior')], function_kwargs={'coding_rank': eval('1'), 'compression': eval('False')})
(y_tilde, rate) = custom_method(
entropy_model(y, training=True), imports='import tensorflow as tf;import tensorflow_compression as tfc;import matplotlib.pyplot as plt;import os;import tensorflow_datasets as tfds', function_to_run='obj(*args, **kwargs)', method_object=eval('entropy_model'), object_signature='tfc.ContinuousBatchedEntropyModel', function_args=[eval('y')], function_kwargs={'training': eval('True')}, custom_class=None)
print('rate:', rate)
print('y_tilde:', y_tilde)
x_tilde = make_synthesis_transform()(y_tilde)
distortion = custom_method(
tf.reduce_mean(abs(x - x_tilde)), imports='import tensorflow as tf;import tensorflow_compression as tfc;import matplotlib.pyplot as plt;import os;import tensorflow_datasets as tfds', function_to_run='tf.reduce_mean(*args)', method_object=None, object_signature=None, function_args=[eval('abs(x - x_tilde)')], function_kwargs={})
print('distortion:', distortion)
x_tilde = custom_method(
tf.saturate_cast(x_tilde[0] * 255, tf.uint8), imports='import tensorflow as tf;import tensorflow_compression as tfc;import matplotlib.pyplot as plt;import os;import tensorflow_datasets as tfds', function_to_run='tf.saturate_cast(*args)', method_object=None, object_signature=None, function_args=[eval('x_tilde[0] * 255'), eval('tf.uint8')], function_kwargs={})
plt.imshow(tf.squeeze(x_tilde))
print(f'Data type: {x_tilde.dtype}')
print(f'Shape: {x_tilde.shape}')
((example_batch, _),) = validation_dataset.batch(32).take(1)
trainer = MNISTCompressionTrainer(10)
example_output = custom_method(
trainer(example_batch), imports='import tensorflow as tf;import tensorflow_compression as tfc;import matplotlib.pyplot as plt;import os;import tensorflow_datasets as tfds', function_to_run='obj(*args)', method_object=eval('trainer'), object_signature='MNISTCompressionTrainer', function_args=[eval('example_batch')], function_kwargs={}, custom_class='class MNISTCompressionTrainer(tf.keras.Model):\n  """Model that trains a compressor/decompressor for MNIST."""\n\n  def __init__(self, latent_dims):\n    super().__init__()\n    self.analysis_transform = make_analysis_transform(latent_dims)\n    self.synthesis_transform = make_synthesis_transform()\n    self.prior_log_scales = tf.Variable(tf.zeros((latent_dims,)))\n\n  @property\n  def prior(self):\n    return tfc.NoisyLogistic(loc=0., scale=tf.exp(self.prior_log_scales))\n\n  def call(self, x, training):\n    """Computes rate and distortion losses."""\n    x = tf.cast(x, self.compute_dtype) / 255.\n    x = tf.reshape(x, (-1, 28, 28, 1))\n\n    y = self.analysis_transform(x)\n    entropy_model = tfc.ContinuousBatchedEntropyModel(\n        self.prior, coding_rank=1, compression=False)\n    y_tilde, rate = entropy_model(y, training=training)\n    x_tilde = self.synthesis_transform(y_tilde)\n\n    rate = tf.reduce_mean(rate)\n\n    distortion = tf.reduce_mean(abs(x - x_tilde))\n\n    return dict(rate=rate, distortion=distortion)')
print('rate: ', example_output['rate'])
print('distortion: ', example_output['distortion'])

def pass_through_loss(_, x):
    return x

def make_mnist_compression_trainer(lmbda, latent_dims=50):
    trainer = MNISTCompressionTrainer(latent_dims)
    custom_method(
    trainer.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss=dict(rate=pass_through_loss, distortion=pass_through_loss), metrics=dict(rate=pass_through_loss, distortion=pass_through_loss), loss_weights=dict(rate=1.0, distortion=lmbda)), imports='import tensorflow as tf;import tensorflow_compression as tfc;import matplotlib.pyplot as plt;import os;import tensorflow_datasets as tfds', function_to_run='obj.compile(**kwargs)', method_object=eval('trainer'), object_signature='MNISTCompressionTrainer', function_args=[], function_kwargs={'optimizer': eval('tf.keras.optimizers.Adam(learning_rate=1e-3)'), 'loss': eval('dict(rate=pass_through_loss, distortion=pass_through_loss)'), 'metrics': eval('dict(rate=pass_through_loss, distortion=pass_through_loss)'), 'loss_weights': eval('dict(rate=1., distortion=lmbda)')}, custom_class='class MNISTCompressionTrainer(tf.keras.Model):\n  """Model that trains a compressor/decompressor for MNIST."""\n\n  def __init__(self, latent_dims):\n    super().__init__()\n    self.analysis_transform = make_analysis_transform(latent_dims)\n    self.synthesis_transform = make_synthesis_transform()\n    self.prior_log_scales = tf.Variable(tf.zeros((latent_dims,)))\n\n  @property\n  def prior(self):\n    return tfc.NoisyLogistic(loc=0., scale=tf.exp(self.prior_log_scales))\n\n  def call(self, x, training):\n    """Computes rate and distortion losses."""\n    x = tf.cast(x, self.compute_dtype) / 255.\n    x = tf.reshape(x, (-1, 28, 28, 1))\n\n    y = self.analysis_transform(x)\n    entropy_model = tfc.ContinuousBatchedEntropyModel(\n        self.prior, coding_rank=1, compression=False)\n    y_tilde, rate = entropy_model(y, training=training)\n    x_tilde = self.synthesis_transform(y_tilde)\n\n    rate = tf.reduce_mean(rate)\n\n    distortion = tf.reduce_mean(abs(x - x_tilde))\n\n    return dict(rate=rate, distortion=distortion)')
    return trainer

def add_rd_targets(image, label):
    return (image, dict(rate=0.0, distortion=0.0))

def train_mnist_model(lmbda):
    trainer = make_mnist_compression_trainer(lmbda)
    custom_method(
    trainer.fit(training_dataset.map(add_rd_targets).batch(128).prefetch(8), epochs=15, validation_data=validation_dataset.map(add_rd_targets).batch(128).cache(), validation_freq=1, verbose=1), imports='import tensorflow as tf;import tensorflow_compression as tfc;import matplotlib.pyplot as plt;import os;import tensorflow_datasets as tfds', function_to_run='obj.fit(*args, **kwargs)', method_object=eval('trainer'), object_signature='MNISTCompressionTrainer', function_args=[eval('training_dataset.map(add_rd_targets).batch(128).prefetch(8)')], function_kwargs={'epochs': eval('15'), 'validation_data': eval('validation_dataset.map(add_rd_targets).batch(128).cache()'), 'validation_freq': eval('1'), 'verbose': eval('1')}, custom_class='class MNISTCompressionTrainer(tf.keras.Model):\n  """Model that trains a compressor/decompressor for MNIST."""\n\n  def __init__(self, latent_dims):\n    super().__init__()\n    self.analysis_transform = make_analysis_transform(latent_dims)\n    self.synthesis_transform = make_synthesis_transform()\n    self.prior_log_scales = tf.Variable(tf.zeros((latent_dims,)))\n\n  @property\n  def prior(self):\n    return tfc.NoisyLogistic(loc=0., scale=tf.exp(self.prior_log_scales))\n\n  def call(self, x, training):\n    """Computes rate and distortion losses."""\n    x = tf.cast(x, self.compute_dtype) / 255.\n    x = tf.reshape(x, (-1, 28, 28, 1))\n\n    y = self.analysis_transform(x)\n    entropy_model = tfc.ContinuousBatchedEntropyModel(\n        self.prior, coding_rank=1, compression=False)\n    y_tilde, rate = entropy_model(y, training=training)\n    x_tilde = self.synthesis_transform(y_tilde)\n\n    rate = tf.reduce_mean(rate)\n\n    distortion = tf.reduce_mean(abs(x - x_tilde))\n\n    return dict(rate=rate, distortion=distortion)')
    return trainer
trainer = train_mnist_model(lmbda=2000)

class MNISTCompressor(tf.keras.Model):
    """Compresses MNIST images to strings."""

    def __init__(self, analysis_transform, entropy_model):
        super().__init__()
        self.analysis_transform = analysis_transform
        self.entropy_model = entropy_model

    def call(self, x):
        x = custom_method(
        tf.cast(x, self.compute_dtype), imports='import tensorflow as tf;import tensorflow_compression as tfc;import matplotlib.pyplot as plt;import os;import tensorflow_datasets as tfds', function_to_run='tf.cast(*args)', method_object=None, object_signature=None, function_args=[eval('x'), eval('self.compute_dtype')], function_kwargs={}) / 255.0
        y = self.analysis_transform(x)
        (_, bits) = self.entropy_model(y, training=False)
        return (self.entropy_model.compress(y), bits)

class MNISTDecompressor(tf.keras.Model):
    """Decompresses MNIST images from strings."""

    def __init__(self, entropy_model, synthesis_transform):
        super().__init__()
        self.entropy_model = entropy_model
        self.synthesis_transform = synthesis_transform

    def call(self, string):
        y_hat = self.entropy_model.decompress(string, ())
        x_hat = self.synthesis_transform(y_hat)
        return custom_method(
        tf.saturate_cast(tf.round(x_hat * 255.0), tf.uint8), imports='import tensorflow as tf;import tensorflow_compression as tfc;import matplotlib.pyplot as plt;import os;import tensorflow_datasets as tfds', function_to_run='tf.saturate_cast(*args)', method_object=None, object_signature=None, function_args=[eval('tf.round(x_hat * 255.)'), eval('tf.uint8')], function_kwargs={})

def make_mnist_codec(trainer, **kwargs):
    entropy_model = custom_method(
    tfc.ContinuousBatchedEntropyModel(trainer.prior, coding_rank=1, compression=True, **kwargs), imports='import tensorflow as tf;import tensorflow_compression as tfc;import matplotlib.pyplot as plt;import os;import tensorflow_datasets as tfds', function_to_run='tfc.ContinuousBatchedEntropyModel(*args, **kwargs)', method_object=None, object_signature=None, function_args=[eval('trainer.prior')], function_kwargs={'coding_rank': eval('1'), 'compression': eval('True'), None: eval('kwargs')})
    compressor = MNISTCompressor(trainer.analysis_transform, entropy_model)
    decompressor = MNISTDecompressor(entropy_model, trainer.synthesis_transform)
    return (compressor, decompressor)
(compressor, decompressor) = make_mnist_codec(trainer)
((originals, _),) = validation_dataset.batch(16).skip(3).take(1)
(strings, entropies) = custom_method(
compressor(originals), imports='import tensorflow as tf;import tensorflow_compression as tfc;import matplotlib.pyplot as plt;import os;import tensorflow_datasets as tfds', function_to_run='obj(*args)', method_object=eval('compressor'), object_signature='MNISTCompressor', function_args=[eval('originals')], function_kwargs={}, custom_class='class MNISTCompressor(tf.keras.Model):\n  """Compresses MNIST images to strings."""\n\n  def __init__(self, analysis_transform, entropy_model):\n    super().__init__()\n    self.analysis_transform = analysis_transform\n    self.entropy_model = entropy_model\n\n  def call(self, x):\n    x = tf.cast(x, self.compute_dtype) / 255.\n    y = self.analysis_transform(x)\n    _, bits = self.entropy_model(y, training=False)\n    return self.entropy_model.compress(y), bits')
print(f'String representation of first digit in hexadecimal: 0x{strings[0].numpy().hex()}')
print(f'Number of bits actually needed to represent it: {entropies[0]:0.2f}')
reconstructions = custom_method(
decompressor(strings), imports='import tensorflow as tf;import tensorflow_compression as tfc;import matplotlib.pyplot as plt;import os;import tensorflow_datasets as tfds', function_to_run='obj(*args)', method_object=eval('decompressor'), object_signature='MNISTDecompressor', function_args=[eval('strings')], function_kwargs={}, custom_class='class MNISTDecompressor(tf.keras.Model):\n  """Decompresses MNIST images from strings."""\n\n  def __init__(self, entropy_model, synthesis_transform):\n    super().__init__()\n    self.entropy_model = entropy_model\n    self.synthesis_transform = synthesis_transform\n\n  def call(self, string):\n    y_hat = self.entropy_model.decompress(string, ())\n    x_hat = self.synthesis_transform(y_hat)\n    return tf.saturate_cast(tf.round(x_hat * 255.), tf.uint8)')

def display_digits(originals, strings, entropies, reconstructions):
    """Visualizes 16 digits together with their reconstructions."""
    (fig, axes) = plt.subplots(4, 4, sharex=True, sharey=True, figsize=(12.5, 5))
    axes = axes.ravel()
    for i in range(len(axes)):
        image = custom_method(
        tf.concat([tf.squeeze(originals[i]), tf.zeros((28, 14), tf.uint8), tf.squeeze(reconstructions[i])], 1), imports='import tensorflow as tf;import tensorflow_compression as tfc;import matplotlib.pyplot as plt;import os;import tensorflow_datasets as tfds', function_to_run='tf.concat(*args)', method_object=None, object_signature=None, function_args=[eval('[\n        tf.squeeze(originals[i]),\n        tf.zeros((28, 14), tf.uint8),\n        tf.squeeze(reconstructions[i]),\n    ]'), eval('1')], function_kwargs={})
        axes[i].imshow(image)
        axes[i].text(0.5, 0.5, f'→ 0x{strings[i].numpy().hex()} →\n{entropies[i]:0.2f} bits', ha='center', va='top', color='white', fontsize='small', transform=axes[i].transAxes)
        axes[i].axis('off')
    plt.subplots_adjust(wspace=0, hspace=0, left=0, right=1, bottom=0, top=1)
display_digits(originals, strings, entropies, reconstructions)

def train_and_visualize_model(lmbda):
    trainer = train_mnist_model(lmbda=lmbda)
    (compressor, decompressor) = make_mnist_codec(trainer)
    (strings, entropies) = custom_method(
    compressor(originals), imports='import tensorflow as tf;import tensorflow_compression as tfc;import matplotlib.pyplot as plt;import os;import tensorflow_datasets as tfds', function_to_run='obj(*args)', method_object=eval('compressor'), object_signature='MNISTCompressor', function_args=[eval('originals')], function_kwargs={}, custom_class='class MNISTCompressor(tf.keras.Model):\n  """Compresses MNIST images to strings."""\n\n  def __init__(self, analysis_transform, entropy_model):\n    super().__init__()\n    self.analysis_transform = analysis_transform\n    self.entropy_model = entropy_model\n\n  def call(self, x):\n    x = tf.cast(x, self.compute_dtype) / 255.\n    y = self.analysis_transform(x)\n    _, bits = self.entropy_model(y, training=False)\n    return self.entropy_model.compress(y), bits')
    reconstructions = custom_method(
    decompressor(strings), imports='import tensorflow as tf;import tensorflow_compression as tfc;import matplotlib.pyplot as plt;import os;import tensorflow_datasets as tfds', function_to_run='obj(*args)', method_object=eval('decompressor'), object_signature='MNISTDecompressor', function_args=[eval('strings')], function_kwargs={}, custom_class='class MNISTDecompressor(tf.keras.Model):\n  """Decompresses MNIST images from strings."""\n\n  def __init__(self, entropy_model, synthesis_transform):\n    super().__init__()\n    self.entropy_model = entropy_model\n    self.synthesis_transform = synthesis_transform\n\n  def call(self, string):\n    y_hat = self.entropy_model.decompress(string, ())\n    x_hat = self.synthesis_transform(y_hat)\n    return tf.saturate_cast(tf.round(x_hat * 255.), tf.uint8)')
    display_digits(originals, strings, entropies, reconstructions)
train_and_visualize_model(lmbda=500)
train_and_visualize_model(lmbda=300)
(compressor, decompressor) = make_mnist_codec(trainer, decode_sanity_check=False)
import os
strings = custom_method(
tf.constant([os.urandom(8) for _ in range(16)]), imports='import tensorflow as tf;import tensorflow_compression as tfc;import matplotlib.pyplot as plt;import os;import tensorflow_datasets as tfds', function_to_run='tf.constant(*args)', method_object=None, object_signature=None, function_args=[eval('[os.urandom(8) for _ in range(16)]')], function_kwargs={})
samples = custom_method(
decompressor(strings), imports='import tensorflow as tf;import tensorflow_compression as tfc;import matplotlib.pyplot as plt;import os;import tensorflow_datasets as tfds', function_to_run='obj(*args)', method_object=eval('decompressor'), object_signature='MNISTDecompressor', function_args=[eval('strings')], function_kwargs={}, custom_class='class MNISTDecompressor(tf.keras.Model):\n  """Decompresses MNIST images from strings."""\n\n  def __init__(self, entropy_model, synthesis_transform):\n    super().__init__()\n    self.entropy_model = entropy_model\n    self.synthesis_transform = synthesis_transform\n\n  def call(self, string):\n    y_hat = self.entropy_model.decompress(string, ())\n    x_hat = self.synthesis_transform(y_hat)\n    return tf.saturate_cast(tf.round(x_hat * 255.), tf.uint8)')
(fig, axes) = plt.subplots(4, 4, sharex=True, sharey=True, figsize=(5, 5))
axes = axes.ravel()
for i in range(len(axes)):
    axes[i].imshow(tf.squeeze(samples[i]))
    axes[i].axis('off')
plt.subplots_adjust(wspace=0, hspace=0, left=0, right=1, bottom=0, top=1)
