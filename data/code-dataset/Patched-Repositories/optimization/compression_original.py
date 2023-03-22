import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_compression as tfc
import tensorflow_datasets as tfds

class CustomDense(tf.keras.layers.Layer):

  def __init__(self, filters, name="dense"):
    super().__init__(name=name)
    self.filters = filters

  @classmethod
  def copy(cls, other, **kwargs):
    """Returns an instantiated and built layer, initialized from `other`."""
    self = cls(filters=other.filters, name=other.name, **kwargs)
    self.build(None, other=other)
    return self

  def build(self, input_shape, other=None):
    """Instantiates weights, optionally initializing them from `other`."""
    if other is None:
      kernel_shape = (input_shape[-1], self.filters)
      kernel = tf.keras.initializers.GlorotUniform()(shape=kernel_shape)
      bias = tf.keras.initializers.Zeros()(shape=(self.filters,))
    else:
      kernel, bias = other.kernel, other.bias
    self.kernel = tf.Variable(
        tf.cast(kernel, self.variable_dtype), name="kernel")
    self.bias = tf.Variable(
        tf.cast(bias, self.variable_dtype), name="bias")
    self.built = True

  def call(self, inputs):
    outputs = tf.linalg.matvec(self.kernel, inputs, transpose_a=True)
    outputs = tf.nn.bias_add(outputs, self.bias)
    return tf.nn.leaky_relu(outputs)

class CustomConv2D(tf.keras.layers.Layer):

  def __init__(self, filters, kernel_size,
               strides=1, padding="SAME", name="conv2d"):
    super().__init__(name=name)
    self.filters = filters
    self.kernel_size = kernel_size
    self.strides = strides
    self.padding = padding

  @classmethod
  def copy(cls, other, **kwargs):
    """Returns an instantiated and built layer, initialized from `other`."""
    self = cls(filters=other.filters, kernel_size=other.kernel_size,
               strides=other.strides, padding=other.padding, name=other.name,
               **kwargs)
    self.build(None, other=other)
    return self

  def build(self, input_shape, other=None):
    """Instantiates weights, optionally initializing them from `other`."""
    if other is None:
      kernel_shape = 2 * (self.kernel_size,) + (input_shape[-1], self.filters)
      kernel = tf.keras.initializers.GlorotUniform()(shape=kernel_shape)
      bias = tf.keras.initializers.Zeros()(shape=(self.filters,))
    else:
      kernel, bias = other.kernel, other.bias
    self.kernel = tf.Variable(
        tf.cast(kernel, self.variable_dtype), name="kernel")
    self.bias = tf.Variable(
        tf.cast(bias, self.variable_dtype), name="bias")
    self.built = True

  def call(self, inputs):
    outputs = tf.nn.convolution(
        inputs, self.kernel, strides=self.strides, padding=self.padding)
    outputs = tf.nn.bias_add(outputs, self.bias)
    return tf.nn.leaky_relu(outputs)

classifier = tf.keras.Sequential([
    CustomConv2D(20, 5, strides=2, name="conv_1"),
    CustomConv2D(50, 5, strides=2, name="conv_2"),
    tf.keras.layers.Flatten(),
    CustomDense(500, name="fc_1"),
    CustomDense(10, name="fc_2"),
], name="classifier")

def normalize_img(image, label):
  """Normalizes images: `uint8` -> `float32`."""
  return tf.cast(image, tf.float32) / 255., label

training_dataset, validation_dataset = tfds.load(
    "mnist",
    split=["train", "test"],
    shuffle_files=True,
    as_supervised=True,
    with_info=False,
)
training_dataset = training_dataset.map(normalize_img)
validation_dataset = validation_dataset.map(normalize_img)

def train_model(model, training_data, validation_data, **kwargs):
  model.compile(
      optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
      loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
      metrics=[tf.keras.metrics.SparseCategoricalAccuracy()],
  )
  kwargs.setdefault("epochs", 5)
  kwargs.setdefault("verbose", 1)
  log = model.fit(
      training_data.batch(128).prefetch(8),
      validation_data=validation_data.batch(128).cache(),
      validation_freq=1,
      **kwargs,
  )
  return log.history["val_sparse_categorical_accuracy"][-1]

classifier_accuracy = train_model(
    classifier, training_dataset, validation_dataset)

print(f"Accuracy: {classifier_accuracy:0.4f}")

_ = tf.linspace(-5., 5., 501)
plt.plot(_, tfc.PowerLawEntropyModel(0).penalty(_));

class PowerLawRegularizer(tf.keras.regularizers.Regularizer):

  def __init__(self, lmbda):
    super().__init__()
    self.lmbda = lmbda

  def __call__(self, variable):
    em = tfc.PowerLawEntropyModel(coding_rank=variable.shape.rank)
    return self.lmbda * em.penalty(variable)

regularizer = PowerLawRegularizer(lmbda=2./classifier.count_params())

def quantize(latent, log_step):
  step = tf.exp(log_step)
  return tfc.round_st(latent / step) * step

class CompressibleDense(CustomDense):

  def __init__(self, regularizer, *args, **kwargs):
    super().__init__(*args, **kwargs)
    self.regularizer = regularizer

  def build(self, input_shape, other=None):
    """Instantiates weights, optionally initializing them from `other`."""
    super().build(input_shape, other=other)
    if other is not None and hasattr(other, "kernel_log_step"):
      kernel_log_step = other.kernel_log_step
      bias_log_step = other.bias_log_step
    else:
      kernel_log_step = bias_log_step = -4.
    self.kernel_log_step = tf.Variable(
        tf.cast(kernel_log_step, self.variable_dtype), name="kernel_log_step")
    self.bias_log_step = tf.Variable(
        tf.cast(bias_log_step, self.variable_dtype), name="bias_log_step")
    self.add_loss(lambda: self.regularizer(
        self.kernel_latent / tf.exp(self.kernel_log_step)))
    self.add_loss(lambda: self.regularizer(
        self.bias_latent / tf.exp(self.bias_log_step)))

  @property
  def kernel(self):
    return quantize(self.kernel_latent, self.kernel_log_step)

  @kernel.setter
  def kernel(self, kernel):
    self.kernel_latent = tf.Variable(kernel, name="kernel_latent")

  @property
  def bias(self):
    return quantize(self.bias_latent, self.bias_log_step)

  @bias.setter
  def bias(self, bias):
    self.bias_latent = tf.Variable(bias, name="bias_latent")

def to_rdft(kernel, kernel_size):
  kernel = tf.transpose(kernel, (2, 3, 0, 1))
  kernel_rdft = tf.signal.rfft2d(kernel)
  kernel_rdft = tf.stack(
      [tf.math.real(kernel_rdft), tf.math.imag(kernel_rdft)], axis=-1)
  return kernel_rdft / kernel_size

def from_rdft(kernel_rdft, kernel_size):
  kernel_rdft *= kernel_size
  kernel_rdft = tf.dtypes.complex(*tf.unstack(kernel_rdft, axis=-1))
  kernel = tf.signal.irfft2d(kernel_rdft, fft_length=2 * (kernel_size,))
  return tf.transpose(kernel, (2, 3, 0, 1))

class CompressibleConv2D(CustomConv2D):

  def __init__(self, regularizer, *args, **kwargs):
    super().__init__(*args, **kwargs)
    self.regularizer = regularizer

  def build(self, input_shape, other=None):
    """Instantiates weights, optionally initializing them from `other`."""
    super().build(input_shape, other=other)
    if other is not None and hasattr(other, "kernel_log_step"):
      kernel_log_step = other.kernel_log_step
      bias_log_step = other.bias_log_step
    else:
      kernel_log_step = tf.fill(self.kernel_latent.shape[2:], -4.)
      bias_log_step = -4.
    self.kernel_log_step = tf.Variable(
        tf.cast(kernel_log_step, self.variable_dtype), name="kernel_log_step")
    self.bias_log_step = tf.Variable(
        tf.cast(bias_log_step, self.variable_dtype), name="bias_log_step")
    self.add_loss(lambda: self.regularizer(
        self.kernel_latent / tf.exp(self.kernel_log_step)))
    self.add_loss(lambda: self.regularizer(
        self.bias_latent / tf.exp(self.bias_log_step)))

  @property
  def kernel(self):
    kernel_rdft = quantize(self.kernel_latent, self.kernel_log_step)
    return from_rdft(kernel_rdft, self.kernel_size)

  @kernel.setter
  def kernel(self, kernel):
    kernel_rdft = to_rdft(kernel, self.kernel_size)
    self.kernel_latent = tf.Variable(kernel_rdft, name="kernel_latent")

  @property
  def bias(self):
    return quantize(self.bias_latent, self.bias_log_step)

  @bias.setter
  def bias(self, bias):
    self.bias_latent = tf.Variable(bias, name="bias_latent")

def make_mnist_classifier(regularizer):
  return tf.keras.Sequential([
      CompressibleConv2D(regularizer, 20, 5, strides=2, name="conv_1"),
      CompressibleConv2D(regularizer, 50, 5, strides=2, name="conv_2"),
      tf.keras.layers.Flatten(),
      CompressibleDense(regularizer, 500, name="fc_1"),
      CompressibleDense(regularizer, 10, name="fc_2"),
  ], name="classifier")

compressible_classifier = make_mnist_classifier(regularizer)

penalized_accuracy = train_model(
    compressible_classifier, training_dataset, validation_dataset)

print(f"Accuracy: {penalized_accuracy:0.4f}")

def compress_latent(latent, log_step, name):
  em = tfc.PowerLawEntropyModel(latent.shape.rank)
  compressed = em.compress(latent / tf.exp(log_step))
  compressed = tf.Variable(compressed, name=f"{name}_compressed")
  log_step = tf.cast(log_step, tf.float16)
  log_step = tf.Variable(log_step, name=f"{name}_log_step")
  return compressed, log_step

def decompress_latent(compressed, shape, log_step):
  latent = tfc.PowerLawEntropyModel(len(shape)).decompress(compressed, shape)
  step = tf.exp(tf.cast(log_step, latent.dtype))
  return latent * step

class CompressedDense(CustomDense):

  def build(self, input_shape, other=None):
    assert isinstance(other, CompressibleDense)
    self.input_channels = other.kernel.shape[0]
    self.kernel_compressed, self.kernel_log_step = compress_latent(
        other.kernel_latent, other.kernel_log_step, "kernel")
    self.bias_compressed, self.bias_log_step = compress_latent(
        other.bias_latent, other.bias_log_step, "bias")
    self.built = True

  @property
  def kernel(self):
    kernel_shape = (self.input_channels, self.filters)
    return decompress_latent(
        self.kernel_compressed, kernel_shape, self.kernel_log_step)

  @property
  def bias(self):
    bias_shape = (self.filters,)
    return decompress_latent(
        self.bias_compressed, bias_shape, self.bias_log_step)

class CompressedConv2D(CustomConv2D):

  def build(self, input_shape, other=None):
    assert isinstance(other, CompressibleConv2D)
    self.input_channels = other.kernel.shape[2]
    self.kernel_compressed, self.kernel_log_step = compress_latent(
        other.kernel_latent, other.kernel_log_step, "kernel")
    self.bias_compressed, self.bias_log_step = compress_latent(
        other.bias_latent, other.bias_log_step, "bias")
    self.built = True

  @property
  def kernel(self):
    rdft_shape = (self.input_channels, self.filters,
                  self.kernel_size, self.kernel_size // 2 + 1, 2)
    kernel_rdft = decompress_latent(
        self.kernel_compressed, rdft_shape, self.kernel_log_step)
    return from_rdft(kernel_rdft, self.kernel_size)

  @property
  def bias(self):
    bias_shape = (self.filters,)
    return decompress_latent(
        self.bias_compressed, bias_shape, self.bias_log_step)

def compress_layer(layer):
  if isinstance(layer, CompressibleDense):
    return CompressedDense.copy(layer)
  if isinstance(layer, CompressibleConv2D):
    return CompressedConv2D.copy(layer)
  return type(layer).from_config(layer.get_config())

compressed_classifier = tf.keras.models.clone_model(
    compressible_classifier, clone_function=compress_layer)

compressed_classifier.compile(metrics=[tf.keras.metrics.SparseCategoricalAccuracy()])
_, compressed_accuracy = compressed_classifier.evaluate(validation_dataset.batch(128))

print(f"Accuracy of the compressible classifier: {penalized_accuracy:0.4f}")
print(f"Accuracy of the compressed classifier: {compressed_accuracy:0.4f}")

def get_weight_size_in_bytes(weight):
  if weight.dtype == tf.string:
    return tf.reduce_sum(tf.strings.length(weight, unit="BYTE"))
  else:
    return tf.size(weight) * weight.dtype.size

original_size = sum(map(get_weight_size_in_bytes, classifier.weights))
compressed_size = sum(map(get_weight_size_in_bytes, compressed_classifier.weights))

print(f"Size of original model weights: {original_size} bytes")
print(f"Size of compressed model weights: {compressed_size} bytes")
print(f"Compression ratio: {(original_size/compressed_size):0.0f}x")

import os
import shutil

def get_disk_size(model, path):
  model.save(path)
  zip_path = shutil.make_archive(path, "zip", path)
  return os.path.getsize(zip_path)

original_zip_size = get_disk_size(classifier, "/tmp/classifier")
compressed_zip_size = get_disk_size(
    compressed_classifier, "/tmp/compressed_classifier")

print(f"Original on-disk size (ZIP compressed): {original_zip_size} bytes")
print(f"Compressed on-disk size (ZIP compressed): {compressed_zip_size} bytes")
print(f"Compression ratio: {(original_zip_size/compressed_zip_size):0.0f}x")


print(f"Accuracy of the vanilla classifier: {classifier_accuracy:0.4f}")
print(f"Accuracy of the penalized classifier: {penalized_accuracy:0.4f}")

def compress_and_evaluate_model(lmbda):
  print(f"lambda={lmbda:0.0f}: training...", flush=True)
  regularizer = PowerLawRegularizer(lmbda=lmbda/classifier.count_params())
  compressible_classifier = make_mnist_classifier(regularizer)
  train_model(
      compressible_classifier, training_dataset, validation_dataset, verbose=0)
  print("compressing...", flush=True)
  compressed_classifier = tf.keras.models.clone_model(
      compressible_classifier, clone_function=compress_layer)
  compressed_size = sum(map(
      get_weight_size_in_bytes, compressed_classifier.weights))
  compressed_zip_size = float(get_disk_size(
      compressed_classifier, "/tmp/compressed_classifier"))
  print("evaluating...", flush=True)
  compressed_classifier = tf.keras.models.load_model(
      "/tmp/compressed_classifier")
  compressed_classifier.compile(
      metrics=[tf.keras.metrics.SparseCategoricalAccuracy()])
  _, compressed_accuracy = compressed_classifier.evaluate(
      validation_dataset.batch(128), verbose=0)
  print()
  return compressed_size, compressed_zip_size, compressed_accuracy

lambdas = (2., 5., 10., 20., 50.)
metrics = [compress_and_evaluate_model(l) for l in lambdas]
metrics = tf.convert_to_tensor(metrics, tf.float32)


def plot_broken_xaxis(ax, compressed_sizes, original_size, original_accuracy):
  xticks = list(range(
      int(tf.math.floor(min(compressed_sizes) / 5) * 5),
      int(tf.math.ceil(max(compressed_sizes) / 5) * 5) + 1,
      5))
  xticks.append(xticks[-1] + 10)
  ax.set_xlim(xticks[0], xticks[-1] + 2)
  ax.set_xticks(xticks[1:])
  ax.set_xticklabels(xticks[1:-1] + [f"{original_size:0.2f}"])
  ax.plot(xticks[-1], original_accuracy, "o", label="float32")

sizes, zip_sizes, accuracies = tf.transpose(metrics)
sizes /= 1024
zip_sizes /= 1024

fig, (axl, axr) = plt.subplots(1, 2, sharey=True, figsize=(10, 4))
axl.plot(sizes, accuracies, "o-", label="EPR compressed")
axr.plot(zip_sizes, accuracies, "o-", label="EPR compressed")
plot_broken_xaxis(axl, sizes, original_size/1024, classifier_accuracy)
plot_broken_xaxis(axr, zip_sizes, original_zip_size/1024, classifier_accuracy)

axl.set_xlabel("size of model weights [kbytes]")
axr.set_xlabel("ZIP compressed on-disk model size [kbytes]")
axl.set_ylabel("accuracy")
axl.legend(loc="lower right")
axr.legend(loc="lower right")
axl.grid()
axr.grid()
for i in range(len(lambdas)):
  axl.annotate(f"$\lambda = {lambdas[i]:0.0f}$", (sizes[i], accuracies[i]),
               xytext=(10, -5), xycoords="data", textcoords="offset points")
  axr.annotate(f"$\lambda = {lambdas[i]:0.0f}$", (zip_sizes[i], accuracies[i]),
               xytext=(10, -5), xycoords="data", textcoords="offset points")
plt.tight_layout()

def decompress_layer(layer):
  if isinstance(layer, CompressedDense):
    return CustomDense.copy(layer)
  if isinstance(layer, CompressedConv2D):
    return CustomConv2D.copy(layer)
  return type(layer).from_config(layer.get_config())

decompressed_classifier = tf.keras.models.clone_model(
    compressed_classifier, clone_function=decompress_layer)

decompressed_accuracy = train_model(
    decompressed_classifier, training_dataset, validation_dataset, epochs=1)

print(f"Accuracy of the compressed classifier: {compressed_accuracy:0.4f}")
print(f"Accuracy of the decompressed classifier after one more epoch of training: {decompressed_accuracy:0.4f}")

def decompress_layer_with_penalty(layer):
  if isinstance(layer, CompressedDense):
    return CompressibleDense.copy(layer, regularizer=regularizer)
  if isinstance(layer, CompressedConv2D):
    return CompressibleConv2D.copy(layer, regularizer=regularizer)
  return type(layer).from_config(layer.get_config())

decompressed_classifier = tf.keras.models.clone_model(
    compressed_classifier, clone_function=decompress_layer_with_penalty)

decompressed_accuracy = train_model(
    decompressed_classifier, training_dataset, validation_dataset, epochs=1)

print(f"Accuracy of the compressed classifier: {compressed_accuracy:0.4f}")
print(f"Accuracy of the decompressed classifier after one more epoch of training: {decompressed_accuracy:0.4f}")

