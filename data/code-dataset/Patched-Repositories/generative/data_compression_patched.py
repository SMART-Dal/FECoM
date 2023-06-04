import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_compression as tfc
import tensorflow_datasets as tfds
import sys
from tool.patching.patching_config import EXPERIMENT_DIR
from tool.measurement.execution import before_execution as before_execution_INSERTED_INTO_SCRIPT
from tool.measurement.execution import after_execution as after_execution_INSERTED_INTO_SCRIPT
experiment_number = sys.argv[1]
experiment_project = sys.argv[2]
EXPERIMENT_FILE_PATH = EXPERIMENT_DIR / 'local-execution' / experiment_project / f'experiment-{experiment_number}.json'

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
        start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT()
        self.prior_log_scales = tf.Variable(tf.zeros((latent_dims,)))
        after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tf.Variable()', method_object=None, function_args=[tf.zeros((latent_dims,))], function_kwargs=None)

    @property
    def prior(self):
        return tfc.NoisyLogistic(loc=0.0, scale=tf.exp(self.prior_log_scales))

    def call(self, x, training):
        """Computes rate and distortion losses."""
        x = tf.cast(x, self.compute_dtype) / 255.0
        start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT()
        x = tf.reshape(x, (-1, 28, 28, 1))
        after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tf.reshape()', method_object=None, function_args=[x, (-1, 28, 28, 1)], function_kwargs=None)
        y = self.analysis_transform(x)
        entropy_model = tfc.ContinuousBatchedEntropyModel(self.prior, coding_rank=1, compression=False)
        (y_tilde, rate) = entropy_model(y, training=training)
        x_tilde = self.synthesis_transform(y_tilde)
        start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT()
        rate = tf.reduce_mean(rate)
        after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tf.reduce_mean()', method_object=None, function_args=[rate], function_kwargs=None)
        start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT()
        distortion = tf.reduce_mean(abs(x - x_tilde))
        after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tf.reduce_mean()', method_object=None, function_args=[abs(x - x_tilde)], function_kwargs=None)
        return dict(rate=rate, distortion=distortion)
(training_dataset, validation_dataset) = tfds.load('mnist', split=['train', 'test'], shuffle_files=True, as_supervised=True, with_info=False)
((x, _),) = validation_dataset.take(1)
plt.imshow(tf.squeeze(x))
print(f'Data type: {x.dtype}')
print(f'Shape: {x.shape}')
x = tf.cast(x, tf.float32) / 255.0
start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT()
x = tf.reshape(x, (-1, 28, 28, 1))
after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tf.reshape()', method_object=None, function_args=[x, (-1, 28, 28, 1)], function_kwargs=None)
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
start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT()
distortion = tf.reduce_mean(abs(x - x_tilde))
after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tf.reduce_mean()', method_object=None, function_args=[abs(x - x_tilde)], function_kwargs=None)
print('distortion:', distortion)
start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT()
x_tilde = tf.saturate_cast(x_tilde[0] * 255, tf.uint8)
after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tf.saturate_cast()', method_object=None, function_args=[x_tilde[0] * 255, tf.uint8], function_kwargs=None)
plt.imshow(tf.squeeze(x_tilde))
print(f'Data type: {x_tilde.dtype}')
print(f'Shape: {x_tilde.shape}')
((example_batch, _),) = validation_dataset.batch(32).take(1)
trainer = MNISTCompressionTrainer(10)
start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT()
example_output = trainer(example_batch)
after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='MNISTCompressionTrainer()', method_object=trainer, function_args=[example_batch], function_kwargs=None)
print('rate: ', example_output['rate'])
print('distortion: ', example_output['distortion'])

def pass_through_loss(_, x):
    return x

def make_mnist_compression_trainer(lmbda, latent_dims=50):
    trainer = MNISTCompressionTrainer(latent_dims)
    start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT()
    trainer.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss=dict(rate=pass_through_loss, distortion=pass_through_loss), metrics=dict(rate=pass_through_loss, distortion=pass_through_loss), loss_weights=dict(rate=1.0, distortion=lmbda))
    after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='MNISTCompressionTrainer.compile()', method_object=trainer, function_args=None, function_kwargs={'optimizer': tf.keras.optimizers.Adam(learning_rate=0.001), 'loss': dict(rate=pass_through_loss, distortion=pass_through_loss), 'metrics': dict(rate=pass_through_loss, distortion=pass_through_loss), 'loss_weights': dict(rate=1.0, distortion=lmbda)})
    return trainer

def add_rd_targets(image, label):
    return (image, dict(rate=0.0, distortion=0.0))

def train_mnist_model(lmbda):
    trainer = make_mnist_compression_trainer(lmbda)
    start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT()
    trainer.fit(training_dataset.map(add_rd_targets).batch(128).prefetch(8), epochs=15, validation_data=validation_dataset.map(add_rd_targets).batch(128).cache(), validation_freq=1, verbose=1)
    after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='make_mnist_compression_trainer.fit()', method_object=trainer, function_args=[training_dataset.map(add_rd_targets).batch(128).prefetch(8)], function_kwargs={'epochs': 15, 'validation_data': validation_dataset.map(add_rd_targets).batch(128).cache(), 'validation_freq': 1, 'verbose': 1})
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
start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT()
(strings, entropies) = compressor(originals)
after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='MNISTCompressor()', method_object=compressor, function_args=[originals], function_kwargs=None)
print(f'String representation of first digit in hexadecimal: 0x{strings[0].numpy().hex()}')
print(f'Number of bits actually needed to represent it: {entropies[0]:0.2f}')
start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT()
reconstructions = decompressor(strings)
after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='MNISTDecompressor()', method_object=decompressor, function_args=[strings], function_kwargs=None)

def display_digits(originals, strings, entropies, reconstructions):
    """Visualizes 16 digits together with their reconstructions."""
    (fig, axes) = plt.subplots(4, 4, sharex=True, sharey=True, figsize=(12.5, 5))
    axes = axes.ravel()
    for i in range(len(axes)):
        start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT()
        image = tf.concat([tf.squeeze(originals[i]), tf.zeros((28, 14), tf.uint8), tf.squeeze(reconstructions[i])], 1)
        after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tf.concat()', method_object=None, function_args=[[tf.squeeze(originals[i]), tf.zeros((28, 14), tf.uint8), tf.squeeze(reconstructions[i])], 1], function_kwargs=None)
        axes[i].imshow(image)
        axes[i].text(0.5, 0.5, f'→ 0x{strings[i].numpy().hex()} →\n{entropies[i]:0.2f} bits', ha='center', va='top', color='white', fontsize='small', transform=axes[i].transAxes)
        axes[i].axis('off')
    plt.subplots_adjust(wspace=0, hspace=0, left=0, right=1, bottom=0, top=1)
display_digits(originals, strings, entropies, reconstructions)

def train_and_visualize_model(lmbda):
    trainer = train_mnist_model(lmbda=lmbda)
    (compressor, decompressor) = make_mnist_codec(trainer)
    start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT()
    (strings, entropies) = compressor(originals)
    after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='MNISTCompressor()', method_object=compressor, function_args=[originals], function_kwargs=None)
    start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT()
    reconstructions = decompressor(strings)
    after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='MNISTDecompressor()', method_object=decompressor, function_args=[strings], function_kwargs=None)
    display_digits(originals, strings, entropies, reconstructions)
train_and_visualize_model(lmbda=500)
train_and_visualize_model(lmbda=300)
(compressor, decompressor) = make_mnist_codec(trainer, decode_sanity_check=False)
import os
start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT()
strings = tf.constant([os.urandom(8) for _ in range(16)])
after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tf.constant()', method_object=None, function_args=[[os.urandom(8) for _ in range(16)]], function_kwargs=None)
start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT()
samples = decompressor(strings)
after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='MNISTDecompressor()', method_object=decompressor, function_args=[strings], function_kwargs=None)
(fig, axes) = plt.subplots(4, 4, sharex=True, sharey=True, figsize=(5, 5))
axes = axes.ravel()
for i in range(len(axes)):
    axes[i].imshow(tf.squeeze(samples[i]))
    axes[i].axis('off')
plt.subplots_adjust(wspace=0, hspace=0, left=0, right=1, bottom=0, top=1)
