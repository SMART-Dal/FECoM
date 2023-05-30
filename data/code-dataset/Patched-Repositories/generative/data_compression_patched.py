import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_compression as tfc
import tensorflow_datasets as tfds
import os
from pathlib import Path
import dill as pickle
import sys
import numpy as np
from tool.client.client_config import EXPERIMENT_DIR, MAX_WAIT_S, WAIT_AFTER_RUN_S
from tool.server.send_request import send_request
from tool.server.function_details import FunctionDetails
import json
current_path = os.path.abspath(__file__)
experiment_number = sys.argv[1]
experiment_project = sys.argv[2]
EXPERIMENT_FILE_PATH = EXPERIMENT_DIR / 'method-level' / experiment_project / f'experiment-{experiment_number}.json'
skip_calls_file_path = EXPERIMENT_FILE_PATH.parent / 'skip_calls.json'
if skip_calls_file_path.exists():
    with open(skip_calls_file_path, 'r') as f:
        skip_calls = json.load(f)
else:
    skip_calls = []
    with open(skip_calls_file_path, 'w') as f:
        json.dump(skip_calls, f)

def custom_method(imports: str, function_to_run: str, method_object=None, object_signature=None, function_args: list=None, function_kwargs: dict=None, custom_class=None):
    if skip_calls is not None and any((call['function_to_run'] == function_to_run and np.array_equal(call['function_args'], function_args) and (call['function_kwargs'] == function_kwargs) for call in skip_calls)):
        print('skipping call: ', function_to_run)
        return
    result = send_request(imports=imports, function_to_run=function_to_run, function_args=function_args, function_kwargs=function_kwargs, max_wait_secs=MAX_WAIT_S, wait_after_run_secs=WAIT_AFTER_RUN_S, method_object=method_object, object_signature=object_signature, custom_class=custom_class, experiment_file_path=EXPERIMENT_FILE_PATH)
    if result is not None and isinstance(result, dict) and (len(result) == 1):
        energy_data = next(iter(result.values()))
        if skip_calls is not None and 'start_time_perf' in energy_data['times'] and ('end_time_perf' in energy_data['times']) and ('start_time_nvidia' in energy_data['times']) and ('end_time_nvidia' in energy_data['times']) and (energy_data['times']['start_time_perf'] == energy_data['times']['end_time_perf']) and (energy_data['times']['start_time_nvidia'] == energy_data['times']['end_time_nvidia']):
            call_to_skip = {'function_to_run': function_to_run, 'function_args': function_args, 'function_kwargs': function_kwargs}
            try:
                json.dumps(call_to_skip)
                if call_to_skip not in skip_calls:
                    skip_calls.append(call_to_skip)
                    with open(skip_calls_file_path, 'w') as f:
                        json.dump(skip_calls, f)
                    print('skipping call added, current list is: ', skip_calls)
                else:
                    print('Skipping call already exists.')
            except TypeError:
                print('Ignore: Skipping call is not JSON serializable, skipping append and dump.')
    else:
        print('Invalid dictionary object or does not have one key-value pair.')

def make_analysis_transform(latent_dims):
    """Creates the analysis (encoder) transform."""
    return tf.keras.Sequential([tf.keras.layers.Conv2D(20, 5, use_bias=True, strides=2, padding='same', activation='leaky_relu', name='conv_1'), tf.keras.layers.Conv2D(50, 5, use_bias=True, strides=2, padding='same', activation='leaky_relu', name='conv_2'), tf.keras.layers.Flatten(), tf.keras.layers.Dense(500, use_bias=True, activation='leaky_relu', name='fc_1'), tf.keras.layers.Dense(latent_dims, use_bias=True, activation=None, name='fc_2')], name='analysis_transform')

def make_synthesis_transform():
    """Creates the synthesis (decoder) transform."""
    return tf.keras.Sequential([tf.keras.layers.Dense(500, use_bias=True, activation='leaky_relu', name='fc_1'), tf.keras.layers.Dense(2450, use_bias=True, activation='leaky_relu', name='fc_2'), tf.keras.layers.Reshape((7, 7, 50)), tf.keras.layers.Conv2DTranspose(20, 5, use_bias=True, strides=2, padding='same', activation='leaky_relu', name='conv_1'), tf.keras.layers.Conv2DTranspose(1, 5, use_bias=True, strides=2, padding='same', activation='leaky_relu', name='conv_2')], name='synthesis_transform')

class MNISTCompressionTrainer(tf.keras.Model):
    """Model that trains a compressor/decompressor for MNIST."""

    def __init__(self, latent_dims):
        super().__init__()
        self.analysis_transform = make_analysis_transform(latent_dims)
        self.synthesis_transform = make_synthesis_transform()
        custom_method(imports='import matplotlib.pyplot as plt;import tensorflow_compression as tfc;import tensorflow as tf;import os;import tensorflow_datasets as tfds', function_to_run='tf.Variable(*args)', method_object=None, object_signature=None, function_args=[eval('tf.zeros((latent_dims,))')], function_kwargs={})
        self.prior_log_scales = tf.Variable(tf.zeros((latent_dims,)))

    @property
    def prior(self):
        return tfc.NoisyLogistic(loc=0.0, scale=tf.exp(self.prior_log_scales))

    def call(self, x, training):
        """Computes rate and distortion losses."""
        x = tf.cast(x, self.compute_dtype) / 255.0
        custom_method(imports='import matplotlib.pyplot as plt;import tensorflow_compression as tfc;import tensorflow as tf;import os;import tensorflow_datasets as tfds', function_to_run='tf.reshape(*args)', method_object=None, object_signature=None, function_args=[eval('x'), eval('(-1, 28, 28, 1)')], function_kwargs={})
        x = tf.reshape(x, (-1, 28, 28, 1))
        y = self.analysis_transform(x)
        entropy_model = tfc.ContinuousBatchedEntropyModel(self.prior, coding_rank=1, compression=False)
        (y_tilde, rate) = entropy_model(y, training=training)
        x_tilde = self.synthesis_transform(y_tilde)
        custom_method(imports='import matplotlib.pyplot as plt;import tensorflow_compression as tfc;import tensorflow as tf;import os;import tensorflow_datasets as tfds', function_to_run='tf.reduce_mean(*args)', method_object=None, object_signature=None, function_args=[eval('rate')], function_kwargs={})
        rate = tf.reduce_mean(rate)
        custom_method(imports='import matplotlib.pyplot as plt;import tensorflow_compression as tfc;import tensorflow as tf;import os;import tensorflow_datasets as tfds', function_to_run='tf.reduce_mean(*args)', method_object=None, object_signature=None, function_args=[eval('abs(x - x_tilde)')], function_kwargs={})
        distortion = tf.reduce_mean(abs(x - x_tilde))
        return dict(rate=rate, distortion=distortion)
(training_dataset, validation_dataset) = tfds.load('mnist', split=['train', 'test'], shuffle_files=True, as_supervised=True, with_info=False)
((x, _),) = validation_dataset.take(1)
plt.imshow(tf.squeeze(x))
print(f'Data type: {x.dtype}')
print(f'Shape: {x.shape}')
x = tf.cast(x, tf.float32) / 255.0
custom_method(imports='import matplotlib.pyplot as plt;import tensorflow_compression as tfc;import tensorflow as tf;import os;import tensorflow_datasets as tfds', function_to_run='tf.reshape(*args)', method_object=None, object_signature=None, function_args=[eval('x'), eval('(-1, 28, 28, 1)')], function_kwargs={})
x = tf.reshape(x, (-1, 28, 28, 1))
y = make_analysis_transform(10)(x)
print('y:', y)
y_tilde = y + tf.random.uniform(y.shape, -0.5, 0.5)
print('y_tilde:', y_tilde)
prior = tfc.NoisyLogistic(loc=0.0, scale=tf.linspace(0.01, 2.0, 10))
_ = tf.linspace(-6.0, 6.0, 501)[:, None]
plt.plot(_, prior.prob(_))
entropy_model = tfc.ContinuousBatchedEntropyModel(prior, coding_rank=1, compression=False)
(y_tilde, rate) = entropy_model(y, training=True)
print('rate:', rate)
print('y_tilde:', y_tilde)
x_tilde = make_synthesis_transform()(y_tilde)
custom_method(imports='import matplotlib.pyplot as plt;import tensorflow_compression as tfc;import tensorflow as tf;import os;import tensorflow_datasets as tfds', function_to_run='tf.reduce_mean(*args)', method_object=None, object_signature=None, function_args=[eval('abs(x - x_tilde)')], function_kwargs={})
distortion = tf.reduce_mean(abs(x - x_tilde))
print('distortion:', distortion)
custom_method(imports='import matplotlib.pyplot as plt;import tensorflow_compression as tfc;import tensorflow as tf;import os;import tensorflow_datasets as tfds', function_to_run='tf.saturate_cast(*args)', method_object=None, object_signature=None, function_args=[eval('x_tilde[0] * 255'), eval('tf.uint8')], function_kwargs={})
x_tilde = tf.saturate_cast(x_tilde[0] * 255, tf.uint8)
plt.imshow(tf.squeeze(x_tilde))
print(f'Data type: {x_tilde.dtype}')
print(f'Shape: {x_tilde.shape}')
((example_batch, _),) = validation_dataset.batch(32).take(1)
trainer = MNISTCompressionTrainer(10)
custom_method(imports='import matplotlib.pyplot as plt;import tensorflow_compression as tfc;import tensorflow as tf;import os;import tensorflow_datasets as tfds', function_to_run='obj(*args)', method_object=eval('trainer'), object_signature=None, function_args=[eval('example_batch')], function_kwargs={}, custom_class='class MNISTCompressionTrainer(tf.keras.Model):\n  """Model that trains a compressor/decompressor for MNIST."""\n\n  def __init__(self, latent_dims):\n    super().__init__()\n    self.analysis_transform = make_analysis_transform(latent_dims)\n    self.synthesis_transform = make_synthesis_transform()\n    self.prior_log_scales = tf.Variable(tf.zeros((latent_dims,)))\n\n  @property\n  def prior(self):\n    return tfc.NoisyLogistic(loc=0., scale=tf.exp(self.prior_log_scales))\n\n  def call(self, x, training):\n    """Computes rate and distortion losses."""\n    x = tf.cast(x, self.compute_dtype) / 255.\n    x = tf.reshape(x, (-1, 28, 28, 1))\n\n    y = self.analysis_transform(x)\n    entropy_model = tfc.ContinuousBatchedEntropyModel(\n        self.prior, coding_rank=1, compression=False)\n    y_tilde, rate = entropy_model(y, training=training)\n    x_tilde = self.synthesis_transform(y_tilde)\n\n    rate = tf.reduce_mean(rate)\n\n    distortion = tf.reduce_mean(abs(x - x_tilde))\n\n    return dict(rate=rate, distortion=distortion)')
example_output = trainer(example_batch)
print('rate: ', example_output['rate'])
print('distortion: ', example_output['distortion'])

def pass_through_loss(_, x):
    return x

def make_mnist_compression_trainer(lmbda, latent_dims=50):
    trainer = MNISTCompressionTrainer(latent_dims)
    custom_method(imports='import matplotlib.pyplot as plt;import tensorflow_compression as tfc;import tensorflow as tf;import os;import tensorflow_datasets as tfds', function_to_run='obj.compile(**kwargs)', method_object=eval('trainer'), object_signature='MNISTCompressionTrainer(latent_dims)', function_args=[], function_kwargs={'optimizer': eval('tf.keras.optimizers.Adam(learning_rate=1e-3)'), 'loss': eval('dict(rate=pass_through_loss, distortion=pass_through_loss)'), 'metrics': eval('dict(rate=pass_through_loss, distortion=pass_through_loss)'), 'loss_weights': eval('dict(rate=1., distortion=lmbda)')}, custom_class='class MNISTCompressionTrainer(tf.keras.Model):\n  """Model that trains a compressor/decompressor for MNIST."""\n\n  def __init__(self, latent_dims):\n    super().__init__()\n    self.analysis_transform = make_analysis_transform(latent_dims)\n    self.synthesis_transform = make_synthesis_transform()\n    self.prior_log_scales = tf.Variable(tf.zeros((latent_dims,)))\n\n  @property\n  def prior(self):\n    return tfc.NoisyLogistic(loc=0., scale=tf.exp(self.prior_log_scales))\n\n  def call(self, x, training):\n    """Computes rate and distortion losses."""\n    x = tf.cast(x, self.compute_dtype) / 255.\n    x = tf.reshape(x, (-1, 28, 28, 1))\n\n    y = self.analysis_transform(x)\n    entropy_model = tfc.ContinuousBatchedEntropyModel(\n        self.prior, coding_rank=1, compression=False)\n    y_tilde, rate = entropy_model(y, training=training)\n    x_tilde = self.synthesis_transform(y_tilde)\n\n    rate = tf.reduce_mean(rate)\n\n    distortion = tf.reduce_mean(abs(x - x_tilde))\n\n    return dict(rate=rate, distortion=distortion)')
    trainer.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss=dict(rate=pass_through_loss, distortion=pass_through_loss), metrics=dict(rate=pass_through_loss, distortion=pass_through_loss), loss_weights=dict(rate=1.0, distortion=lmbda))
    return trainer

def add_rd_targets(image, label):
    return (image, dict(rate=0.0, distortion=0.0))

def train_mnist_model(lmbda):
    trainer = make_mnist_compression_trainer(lmbda)
    custom_method(imports='import matplotlib.pyplot as plt;import tensorflow_compression as tfc;import tensorflow as tf;import os;import tensorflow_datasets as tfds', function_to_run='obj.fit(*args, **kwargs)', method_object=eval('trainer'), object_signature='MNISTCompressionTrainer(latent_dims)', function_args=[eval('training_dataset.map(add_rd_targets).batch(128).prefetch(8)')], function_kwargs={'epochs': eval('15'), 'validation_data': eval('validation_dataset.map(add_rd_targets).batch(128).cache()'), 'validation_freq': eval('1'), 'verbose': eval('1')}, custom_class='class MNISTCompressionTrainer(tf.keras.Model):\n  """Model that trains a compressor/decompressor for MNIST."""\n\n  def __init__(self, latent_dims):\n    super().__init__()\n    self.analysis_transform = make_analysis_transform(latent_dims)\n    self.synthesis_transform = make_synthesis_transform()\n    self.prior_log_scales = tf.Variable(tf.zeros((latent_dims,)))\n\n  @property\n  def prior(self):\n    return tfc.NoisyLogistic(loc=0., scale=tf.exp(self.prior_log_scales))\n\n  def call(self, x, training):\n    """Computes rate and distortion losses."""\n    x = tf.cast(x, self.compute_dtype) / 255.\n    x = tf.reshape(x, (-1, 28, 28, 1))\n\n    y = self.analysis_transform(x)\n    entropy_model = tfc.ContinuousBatchedEntropyModel(\n        self.prior, coding_rank=1, compression=False)\n    y_tilde, rate = entropy_model(y, training=training)\n    x_tilde = self.synthesis_transform(y_tilde)\n\n    rate = tf.reduce_mean(rate)\n\n    distortion = tf.reduce_mean(abs(x - x_tilde))\n\n    return dict(rate=rate, distortion=distortion)')
    trainer.fit(training_dataset.map(add_rd_targets).batch(128).prefetch(8), epochs=15, validation_data=validation_dataset.map(add_rd_targets).batch(128).cache(), validation_freq=1, verbose=1)
    return trainer
trainer = train_mnist_model(lmbda=2000)

class MNISTCompressor(tf.keras.Model):
    """Compresses MNIST images to strings."""

    def __init__(self, analysis_transform, entropy_model):
        super().__init__()
        self.analysis_transform = analysis_transform
        self.entropy_model = entropy_model

    def call(self, x):
        x = tf.cast(x, self.compute_dtype) / 255.0
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
        return tf.saturate_cast(tf.round(x_hat * 255.0), tf.uint8)

def make_mnist_codec(trainer, **kwargs):
    entropy_model = tfc.ContinuousBatchedEntropyModel(trainer.prior, coding_rank=1, compression=True, **kwargs)
    compressor = MNISTCompressor(trainer.analysis_transform, entropy_model)
    decompressor = MNISTDecompressor(entropy_model, trainer.synthesis_transform)
    return (compressor, decompressor)
(compressor, decompressor) = make_mnist_codec(trainer)
((originals, _),) = validation_dataset.batch(16).skip(3).take(1)
custom_method(imports='import matplotlib.pyplot as plt;import tensorflow_compression as tfc;import tensorflow as tf;import os;import tensorflow_datasets as tfds', function_to_run='obj(*args)', method_object=eval('compressor'), object_signature=None, function_args=[eval('originals')], function_kwargs={}, custom_class='class MNISTCompressor(tf.keras.Model):\n  """Compresses MNIST images to strings."""\n\n  def __init__(self, analysis_transform, entropy_model):\n    super().__init__()\n    self.analysis_transform = analysis_transform\n    self.entropy_model = entropy_model\n\n  def call(self, x):\n    x = tf.cast(x, self.compute_dtype) / 255.\n    y = self.analysis_transform(x)\n    _, bits = self.entropy_model(y, training=False)\n    return self.entropy_model.compress(y), bits')
(strings, entropies) = compressor(originals)
print(f'String representation of first digit in hexadecimal: 0x{strings[0].numpy().hex()}')
print(f'Number of bits actually needed to represent it: {entropies[0]:0.2f}')
custom_method(imports='import matplotlib.pyplot as plt;import tensorflow_compression as tfc;import tensorflow as tf;import os;import tensorflow_datasets as tfds', function_to_run='obj(*args)', method_object=eval('decompressor'), object_signature=None, function_args=[eval('strings')], function_kwargs={}, custom_class='class MNISTDecompressor(tf.keras.Model):\n  """Decompresses MNIST images from strings."""\n\n  def __init__(self, entropy_model, synthesis_transform):\n    super().__init__()\n    self.entropy_model = entropy_model\n    self.synthesis_transform = synthesis_transform\n\n  def call(self, string):\n    y_hat = self.entropy_model.decompress(string, ())\n    x_hat = self.synthesis_transform(y_hat)\n    return tf.saturate_cast(tf.round(x_hat * 255.), tf.uint8)')
reconstructions = decompressor(strings)

def display_digits(originals, strings, entropies, reconstructions):
    """Visualizes 16 digits together with their reconstructions."""
    (fig, axes) = plt.subplots(4, 4, sharex=True, sharey=True, figsize=(12.5, 5))
    axes = axes.ravel()
    for i in range(len(axes)):
        custom_method(imports='import matplotlib.pyplot as plt;import tensorflow_compression as tfc;import tensorflow as tf;import os;import tensorflow_datasets as tfds', function_to_run='tf.concat(*args)', method_object=None, object_signature=None, function_args=[eval('[\n        tf.squeeze(originals[i]),\n        tf.zeros((28, 14), tf.uint8),\n        tf.squeeze(reconstructions[i]),\n    ]'), eval('1')], function_kwargs={})
        image = tf.concat([tf.squeeze(originals[i]), tf.zeros((28, 14), tf.uint8), tf.squeeze(reconstructions[i])], 1)
        axes[i].imshow(image)
        axes[i].text(0.5, 0.5, f'→ 0x{strings[i].numpy().hex()} →\n{entropies[i]:0.2f} bits', ha='center', va='top', color='white', fontsize='small', transform=axes[i].transAxes)
        axes[i].axis('off')
    plt.subplots_adjust(wspace=0, hspace=0, left=0, right=1, bottom=0, top=1)
display_digits(originals, strings, entropies, reconstructions)

def train_and_visualize_model(lmbda):
    trainer = train_mnist_model(lmbda=lmbda)
    (compressor, decompressor) = make_mnist_codec(trainer)
    custom_method(imports='import matplotlib.pyplot as plt;import tensorflow_compression as tfc;import tensorflow as tf;import os;import tensorflow_datasets as tfds', function_to_run='obj(*args)', method_object=eval('compressor'), object_signature=None, function_args=[eval('originals')], function_kwargs={}, custom_class='class MNISTCompressor(tf.keras.Model):\n  """Compresses MNIST images to strings."""\n\n  def __init__(self, analysis_transform, entropy_model):\n    super().__init__()\n    self.analysis_transform = analysis_transform\n    self.entropy_model = entropy_model\n\n  def call(self, x):\n    x = tf.cast(x, self.compute_dtype) / 255.\n    y = self.analysis_transform(x)\n    _, bits = self.entropy_model(y, training=False)\n    return self.entropy_model.compress(y), bits')
    (strings, entropies) = compressor(originals)
    custom_method(imports='import matplotlib.pyplot as plt;import tensorflow_compression as tfc;import tensorflow as tf;import os;import tensorflow_datasets as tfds', function_to_run='obj(*args)', method_object=eval('decompressor'), object_signature=None, function_args=[eval('strings')], function_kwargs={}, custom_class='class MNISTDecompressor(tf.keras.Model):\n  """Decompresses MNIST images from strings."""\n\n  def __init__(self, entropy_model, synthesis_transform):\n    super().__init__()\n    self.entropy_model = entropy_model\n    self.synthesis_transform = synthesis_transform\n\n  def call(self, string):\n    y_hat = self.entropy_model.decompress(string, ())\n    x_hat = self.synthesis_transform(y_hat)\n    return tf.saturate_cast(tf.round(x_hat * 255.), tf.uint8)')
    reconstructions = decompressor(strings)
    display_digits(originals, strings, entropies, reconstructions)
train_and_visualize_model(lmbda=500)
train_and_visualize_model(lmbda=300)
(compressor, decompressor) = make_mnist_codec(trainer, decode_sanity_check=False)
import os
custom_method(imports='import matplotlib.pyplot as plt;import tensorflow_compression as tfc;import tensorflow as tf;import os;import tensorflow_datasets as tfds', function_to_run='tf.constant(*args)', method_object=None, object_signature=None, function_args=[eval('[os.urandom(8) for _ in range(16)]')], function_kwargs={})
strings = tf.constant([os.urandom(8) for _ in range(16)])
custom_method(imports='import matplotlib.pyplot as plt;import tensorflow_compression as tfc;import tensorflow as tf;import os;import tensorflow_datasets as tfds', function_to_run='obj(*args)', method_object=eval('decompressor'), object_signature=None, function_args=[eval('strings')], function_kwargs={}, custom_class='class MNISTDecompressor(tf.keras.Model):\n  """Decompresses MNIST images from strings."""\n\n  def __init__(self, entropy_model, synthesis_transform):\n    super().__init__()\n    self.entropy_model = entropy_model\n    self.synthesis_transform = synthesis_transform\n\n  def call(self, string):\n    y_hat = self.entropy_model.decompress(string, ())\n    x_hat = self.synthesis_transform(y_hat)\n    return tf.saturate_cast(tf.round(x_hat * 255.), tf.uint8)')
samples = decompressor(strings)
(fig, axes) = plt.subplots(4, 4, sharex=True, sharey=True, figsize=(5, 5))
axes = axes.ravel()
for i in range(len(axes)):
    axes[i].imshow(tf.squeeze(samples[i]))
    axes[i].axis('off')
plt.subplots_adjust(wspace=0, hspace=0, left=0, right=1, bottom=0, top=1)
