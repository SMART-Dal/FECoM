import pkg_resources
import importlib
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

def custom_method(func, imports: str, function_to_run: str, method_object=None, function_args: list=None, function_kwargs: dict=None, max_wait_secs=MAX_WAIT_S, custom_class=None, wait_after_run_secs=WAIT_AFTER_RUN_S):
    result = send_request(imports=imports, function_to_run=function_to_run, function_args=function_args, function_kwargs=function_kwargs, max_wait_secs=max_wait_secs, wait_after_run_secs=wait_after_run_secs, method_object=method_object, custom_class=custom_class, experiment_file_path=EXPERIMENT_FILE_PATH)
    return func
if __name__ == '__main__':
    print(EXPERIMENT_FILE_PATH)
importlib.reload(pkg_resources)
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import sklearn.datasets
import numpy as np
import tensorflow as tf
import official.nlp.modeling.layers as nlp_layers
plt.rcParams['figure.dpi'] = 140
DEFAULT_X_RANGE = (-3.5, 3.5)
DEFAULT_Y_RANGE = (-2.5, 2.5)
DEFAULT_CMAP = colors.ListedColormap(['#377eb8', '#ff7f00'])
DEFAULT_NORM = colors.Normalize(vmin=0, vmax=1)
DEFAULT_N_GRID = 100

def make_training_data(sample_size=500):
    """Create two moon training dataset."""
    (train_examples, train_labels) = sklearn.datasets.make_moons(n_samples=2 * sample_size, noise=0.1)
    train_examples[train_labels == 0] += [-0.1, 0.2]
    train_examples[train_labels == 1] += [0.1, -0.2]
    return (train_examples, train_labels)

def make_testing_data(x_range=DEFAULT_X_RANGE, y_range=DEFAULT_Y_RANGE, n_grid=DEFAULT_N_GRID):
    """Create a mesh grid in 2D space."""
    x = np.linspace(x_range[0], x_range[1], n_grid)
    y = np.linspace(y_range[0], y_range[1], n_grid)
    (xv, yv) = np.meshgrid(x, y)
    return np.stack([xv.flatten(), yv.flatten()], axis=-1)

def make_ood_data(sample_size=500, means=(2.5, -1.75), vars=(0.01, 0.01)):
    return np.random.multivariate_normal(means, cov=np.diag(vars), size=sample_size)
(train_examples, train_labels) = make_training_data(sample_size=500)
test_examples = make_testing_data()
ood_examples = make_ood_data(sample_size=500)
pos_examples = train_examples[train_labels == 0]
neg_examples = train_examples[train_labels == 1]
plt.figure(figsize=(7, 5.5))
plt.scatter(pos_examples[:, 0], pos_examples[:, 1], c='#377eb8', alpha=0.5)
plt.scatter(neg_examples[:, 0], neg_examples[:, 1], c='#ff7f00', alpha=0.5)
plt.scatter(ood_examples[:, 0], ood_examples[:, 1], c='red', alpha=0.1)
plt.legend(['Positive', 'Negative', 'Out-of-Domain'])
plt.ylim(DEFAULT_Y_RANGE)
plt.xlim(DEFAULT_X_RANGE)
plt.show()

class DeepResNet(tf.keras.Model):
    """Defines a multi-layer residual network."""

    def __init__(self, num_classes, num_layers=3, num_hidden=128, dropout_rate=0.1, **classifier_kwargs):
        super().__init__()
        self.num_hidden = num_hidden
        self.num_layers = num_layers
        self.dropout_rate = dropout_rate
        self.classifier_kwargs = classifier_kwargs
        self.input_layer = custom_method(
        tf.keras.layers.Dense(self.num_hidden, trainable=False), imports='import pkg_resources;import tensorflow as tf;import numpy as np;import importlib;import matplotlib.pyplot as plt;import official.nlp.modeling.layers as nlp_layers;import matplotlib.colors as colors;import sklearn.datasets', function_to_run='tf.keras.layers.Dense(*args, **kwargs)', method_object=None, function_args=[eval('self.num_hidden')], function_kwargs={'trainable': eval('False')}, max_wait_secs=0)
        self.dense_layers = [self.make_dense_layer() for _ in range(num_layers)]
        self.classifier = self.make_output_layer(num_classes)

    def call(self, inputs):
        hidden = self.input_layer(inputs)
        for i in range(self.num_layers):
            resid = self.dense_layers[i](hidden)
            resid = custom_method(
            tf.keras.layers.Dropout(self.dropout_rate)(resid), imports='import pkg_resources;import tensorflow as tf;import numpy as np;import importlib;import matplotlib.pyplot as plt;import official.nlp.modeling.layers as nlp_layers;import matplotlib.colors as colors;import sklearn.datasets', function_to_run='tf.keras.layers.Dropout(self.dropout_rate)(*args)', method_object=None, function_args=[eval('resid')], function_kwargs={}, max_wait_secs=0)
            hidden += resid
        return self.classifier(hidden)

    def make_dense_layer(self):
        """Uses the Dense layer as the hidden layer."""
        return custom_method(
        tf.keras.layers.Dense(self.num_hidden, activation='relu'), imports='import pkg_resources;import tensorflow as tf;import numpy as np;import importlib;import matplotlib.pyplot as plt;import official.nlp.modeling.layers as nlp_layers;import matplotlib.colors as colors;import sklearn.datasets', function_to_run='tf.keras.layers.Dense(*args, **kwargs)', method_object=None, function_args=[eval('self.num_hidden')], function_kwargs={'activation': eval('"relu"')}, max_wait_secs=0)

    def make_output_layer(self, num_classes):
        """Uses the Dense layer as the output layer."""
        return custom_method(
        tf.keras.layers.Dense(num_classes, **self.classifier_kwargs), imports='import pkg_resources;import tensorflow as tf;import numpy as np;import importlib;import matplotlib.pyplot as plt;import official.nlp.modeling.layers as nlp_layers;import matplotlib.colors as colors;import sklearn.datasets', function_to_run='tf.keras.layers.Dense(*args, **kwargs)', method_object=None, function_args=[eval('num_classes')], function_kwargs={None: eval('self.classifier_kwargs')}, max_wait_secs=0)
resnet_config = dict(num_classes=2, num_layers=6, num_hidden=128)
resnet_model = DeepResNet(**resnet_config)
custom_method(
resnet_model.build((None, 2)), imports='import pkg_resources;import tensorflow as tf;import numpy as np;import importlib;import matplotlib.pyplot as plt;import official.nlp.modeling.layers as nlp_layers;import matplotlib.colors as colors;import sklearn.datasets', function_to_run='obj.build(*args)', method_object=eval('resnet_model'), function_args=[eval('(None, 2)')], function_kwargs={}, max_wait_secs=0, custom_class='class DeepResNet(tf.keras.Model):\n  """Defines a multi-layer residual network."""\n  def __init__(self, num_classes, num_layers=3, num_hidden=128,\n               dropout_rate=0.1, **classifier_kwargs):\n    super().__init__()\n    self.num_hidden = num_hidden\n    self.num_layers = num_layers\n    self.dropout_rate = dropout_rate\n    self.classifier_kwargs = classifier_kwargs\n\n    self.input_layer = tf.keras.layers.Dense(self.num_hidden, trainable=False)\n    self.dense_layers = [self.make_dense_layer() for _ in range(num_layers)]\n\n    self.classifier = self.make_output_layer(num_classes)\n\n  def call(self, inputs):\n    hidden = self.input_layer(inputs)\n\n    for i in range(self.num_layers):\n      resid = self.dense_layers[i](hidden)\n      resid = tf.keras.layers.Dropout(self.dropout_rate)(resid)\n      hidden += resid\n\n    return self.classifier(hidden)\n\n  def make_dense_layer(self):\n    """Uses the Dense layer as the hidden layer."""\n    return tf.keras.layers.Dense(self.num_hidden, activation="relu")\n\n  def make_output_layer(self, num_classes):\n    """Uses the Dense layer as the output layer."""\n    return tf.keras.layers.Dense(\n        num_classes, **self.classifier_kwargs)')
custom_method(
resnet_model.summary(), imports='import pkg_resources;import tensorflow as tf;import numpy as np;import importlib;import matplotlib.pyplot as plt;import official.nlp.modeling.layers as nlp_layers;import matplotlib.colors as colors;import sklearn.datasets', function_to_run='obj.summary()', method_object=eval('resnet_model'), function_args=[], function_kwargs={}, max_wait_secs=0, custom_class='class DeepResNet(tf.keras.Model):\n  """Defines a multi-layer residual network."""\n  def __init__(self, num_classes, num_layers=3, num_hidden=128,\n               dropout_rate=0.1, **classifier_kwargs):\n    super().__init__()\n    self.num_hidden = num_hidden\n    self.num_layers = num_layers\n    self.dropout_rate = dropout_rate\n    self.classifier_kwargs = classifier_kwargs\n\n    self.input_layer = tf.keras.layers.Dense(self.num_hidden, trainable=False)\n    self.dense_layers = [self.make_dense_layer() for _ in range(num_layers)]\n\n    self.classifier = self.make_output_layer(num_classes)\n\n  def call(self, inputs):\n    hidden = self.input_layer(inputs)\n\n    for i in range(self.num_layers):\n      resid = self.dense_layers[i](hidden)\n      resid = tf.keras.layers.Dropout(self.dropout_rate)(resid)\n      hidden += resid\n\n    return self.classifier(hidden)\n\n  def make_dense_layer(self):\n    """Uses the Dense layer as the hidden layer."""\n    return tf.keras.layers.Dense(self.num_hidden, activation="relu")\n\n  def make_output_layer(self, num_classes):\n    """Uses the Dense layer as the output layer."""\n    return tf.keras.layers.Dense(\n        num_classes, **self.classifier_kwargs)')
loss = custom_method(
tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), imports='import pkg_resources;import tensorflow as tf;import numpy as np;import importlib;import matplotlib.pyplot as plt;import official.nlp.modeling.layers as nlp_layers;import matplotlib.colors as colors;import sklearn.datasets', function_to_run='tf.keras.losses.SparseCategoricalCrossentropy(**kwargs)', method_object=None, function_args=[], function_kwargs={'from_logits': eval('True')}, max_wait_secs=0)
metrics = (custom_method(
tf.keras.metrics.SparseCategoricalAccuracy(), imports='import pkg_resources;import tensorflow as tf;import numpy as np;import importlib;import matplotlib.pyplot as plt;import official.nlp.modeling.layers as nlp_layers;import matplotlib.colors as colors;import sklearn.datasets', function_to_run='tf.keras.metrics.SparseCategoricalAccuracy()', method_object=None, function_args=[], function_kwargs={}, max_wait_secs=0),)
optimizer = custom_method(
tf.keras.optimizers.legacy.Adam(learning_rate=0.0001), imports='import pkg_resources;import tensorflow as tf;import numpy as np;import importlib;import matplotlib.pyplot as plt;import official.nlp.modeling.layers as nlp_layers;import matplotlib.colors as colors;import sklearn.datasets', function_to_run='tf.keras.optimizers.legacy.Adam(**kwargs)', method_object=None, function_args=[], function_kwargs={'learning_rate': eval('1e-4')}, max_wait_secs=0)
train_config = dict(loss=loss, metrics=metrics, optimizer=optimizer)
fit_config = dict(batch_size=128, epochs=100)
custom_method(
resnet_model.compile(**train_config), imports='import pkg_resources;import tensorflow as tf;import numpy as np;import importlib;import matplotlib.pyplot as plt;import official.nlp.modeling.layers as nlp_layers;import matplotlib.colors as colors;import sklearn.datasets', function_to_run='obj.compile(**kwargs)', method_object=eval('resnet_model'), function_args=[], function_kwargs={None: eval('train_config')}, max_wait_secs=0, custom_class='class DeepResNet(tf.keras.Model):\n  """Defines a multi-layer residual network."""\n  def __init__(self, num_classes, num_layers=3, num_hidden=128,\n               dropout_rate=0.1, **classifier_kwargs):\n    super().__init__()\n    self.num_hidden = num_hidden\n    self.num_layers = num_layers\n    self.dropout_rate = dropout_rate\n    self.classifier_kwargs = classifier_kwargs\n\n    self.input_layer = tf.keras.layers.Dense(self.num_hidden, trainable=False)\n    self.dense_layers = [self.make_dense_layer() for _ in range(num_layers)]\n\n    self.classifier = self.make_output_layer(num_classes)\n\n  def call(self, inputs):\n    hidden = self.input_layer(inputs)\n\n    for i in range(self.num_layers):\n      resid = self.dense_layers[i](hidden)\n      resid = tf.keras.layers.Dropout(self.dropout_rate)(resid)\n      hidden += resid\n\n    return self.classifier(hidden)\n\n  def make_dense_layer(self):\n    """Uses the Dense layer as the hidden layer."""\n    return tf.keras.layers.Dense(self.num_hidden, activation="relu")\n\n  def make_output_layer(self, num_classes):\n    """Uses the Dense layer as the output layer."""\n    return tf.keras.layers.Dense(\n        num_classes, **self.classifier_kwargs)')
custom_method(
resnet_model.fit(train_examples, train_labels, **fit_config), imports='import pkg_resources;import tensorflow as tf;import numpy as np;import importlib;import matplotlib.pyplot as plt;import official.nlp.modeling.layers as nlp_layers;import matplotlib.colors as colors;import sklearn.datasets', function_to_run='obj.fit(*args, **kwargs)', method_object=eval('resnet_model'), function_args=[eval('train_examples'), eval('train_labels')], function_kwargs={None: eval('fit_config')}, max_wait_secs=0, custom_class='class DeepResNet(tf.keras.Model):\n  """Defines a multi-layer residual network."""\n  def __init__(self, num_classes, num_layers=3, num_hidden=128,\n               dropout_rate=0.1, **classifier_kwargs):\n    super().__init__()\n    self.num_hidden = num_hidden\n    self.num_layers = num_layers\n    self.dropout_rate = dropout_rate\n    self.classifier_kwargs = classifier_kwargs\n\n    self.input_layer = tf.keras.layers.Dense(self.num_hidden, trainable=False)\n    self.dense_layers = [self.make_dense_layer() for _ in range(num_layers)]\n\n    self.classifier = self.make_output_layer(num_classes)\n\n  def call(self, inputs):\n    hidden = self.input_layer(inputs)\n\n    for i in range(self.num_layers):\n      resid = self.dense_layers[i](hidden)\n      resid = tf.keras.layers.Dropout(self.dropout_rate)(resid)\n      hidden += resid\n\n    return self.classifier(hidden)\n\n  def make_dense_layer(self):\n    """Uses the Dense layer as the hidden layer."""\n    return tf.keras.layers.Dense(self.num_hidden, activation="relu")\n\n  def make_output_layer(self, num_classes):\n    """Uses the Dense layer as the output layer."""\n    return tf.keras.layers.Dense(\n        num_classes, **self.classifier_kwargs)')

def plot_uncertainty_surface(test_uncertainty, ax, cmap=None):
    """Visualizes the 2D uncertainty surface.
  
  For simplicity, assume these objects already exist in the memory:

    test_examples: Array of test examples, shape (num_test, 2).
    train_labels: Array of train labels, shape (num_train, ).
    train_examples: Array of train examples, shape (num_train, 2).
  
  Arguments:
    test_uncertainty: Array of uncertainty scores, shape (num_test,).
    ax: A matplotlib Axes object that specifies a matplotlib figure.
    cmap: A matplotlib colormap object specifying the palette of the
      predictive surface.

  Returns:
    pcm: A matplotlib PathCollection object that contains the palette
      information of the uncertainty plot.
  """
    test_uncertainty = test_uncertainty / np.max(test_uncertainty)
    ax.set_ylim(DEFAULT_Y_RANGE)
    ax.set_xlim(DEFAULT_X_RANGE)
    pcm = ax.imshow(np.reshape(test_uncertainty, [DEFAULT_N_GRID, DEFAULT_N_GRID]), cmap=cmap, origin='lower', extent=DEFAULT_X_RANGE + DEFAULT_Y_RANGE, vmin=DEFAULT_NORM.vmin, vmax=DEFAULT_NORM.vmax, interpolation='bicubic', aspect='auto')
    ax.scatter(train_examples[:, 0], train_examples[:, 1], c=train_labels, cmap=DEFAULT_CMAP, alpha=0.5)
    ax.scatter(ood_examples[:, 0], ood_examples[:, 1], c='red', alpha=0.1)
    return pcm
resnet_logits = custom_method(
resnet_model(test_examples), imports='import pkg_resources;import tensorflow as tf;import numpy as np;import importlib;import matplotlib.pyplot as plt;import official.nlp.modeling.layers as nlp_layers;import matplotlib.colors as colors;import sklearn.datasets', function_to_run='obj(*args)', method_object=eval('resnet_model'), function_args=[eval('test_examples')], function_kwargs={}, max_wait_secs=0, custom_class='class DeepResNet(tf.keras.Model):\n  """Defines a multi-layer residual network."""\n  def __init__(self, num_classes, num_layers=3, num_hidden=128,\n               dropout_rate=0.1, **classifier_kwargs):\n    super().__init__()\n    self.num_hidden = num_hidden\n    self.num_layers = num_layers\n    self.dropout_rate = dropout_rate\n    self.classifier_kwargs = classifier_kwargs\n\n    self.input_layer = tf.keras.layers.Dense(self.num_hidden, trainable=False)\n    self.dense_layers = [self.make_dense_layer() for _ in range(num_layers)]\n\n    self.classifier = self.make_output_layer(num_classes)\n\n  def call(self, inputs):\n    hidden = self.input_layer(inputs)\n\n    for i in range(self.num_layers):\n      resid = self.dense_layers[i](hidden)\n      resid = tf.keras.layers.Dropout(self.dropout_rate)(resid)\n      hidden += resid\n\n    return self.classifier(hidden)\n\n  def make_dense_layer(self):\n    """Uses the Dense layer as the hidden layer."""\n    return tf.keras.layers.Dense(self.num_hidden, activation="relu")\n\n  def make_output_layer(self, num_classes):\n    """Uses the Dense layer as the output layer."""\n    return tf.keras.layers.Dense(\n        num_classes, **self.classifier_kwargs)')
resnet_probs = custom_method(
tf.nn.softmax(resnet_logits, axis=-1), imports='import pkg_resources;import tensorflow as tf;import numpy as np;import importlib;import matplotlib.pyplot as plt;import official.nlp.modeling.layers as nlp_layers;import matplotlib.colors as colors;import sklearn.datasets', function_to_run='tf.nn.softmax(*args, **kwargs)', method_object=None, function_args=[eval('resnet_logits')], function_kwargs={'axis': eval('-1')}, max_wait_secs=0)[:, 0]
(_, ax) = plt.subplots(figsize=(7, 5.5))
pcm = plot_uncertainty_surface(resnet_probs, ax=ax)
plt.colorbar(pcm, ax=ax)
plt.title('Class Probability, Deterministic Model')
plt.show()
resnet_uncertainty = resnet_probs * (1 - resnet_probs)
(_, ax) = plt.subplots(figsize=(7, 5.5))
pcm = plot_uncertainty_surface(resnet_uncertainty, ax=ax)
plt.colorbar(pcm, ax=ax)
plt.title('Predictive Uncertainty, Deterministic Model')
plt.show()
dense = custom_method(
tf.keras.layers.Dense(units=10), imports='import pkg_resources;import tensorflow as tf;import numpy as np;import importlib;import matplotlib.pyplot as plt;import official.nlp.modeling.layers as nlp_layers;import matplotlib.colors as colors;import sklearn.datasets', function_to_run='tf.keras.layers.Dense(**kwargs)', method_object=None, function_args=[], function_kwargs={'units': eval('10')}, max_wait_secs=0)
dense = nlp_layers.SpectralNormalization(dense, norm_multiplier=0.9)
batch_size = 32
input_dim = 1024
num_classes = 10
gp_layer = nlp_layers.RandomFeatureGaussianProcess(units=num_classes, num_inducing=1024, normalize_input=False, scale_random_features=True, gp_cov_momentum=-1)
embedding = custom_method(
tf.random.normal(shape=(batch_size, input_dim)), imports='import pkg_resources;import tensorflow as tf;import numpy as np;import importlib;import matplotlib.pyplot as plt;import official.nlp.modeling.layers as nlp_layers;import matplotlib.colors as colors;import sklearn.datasets', function_to_run='tf.random.normal(**kwargs)', method_object=None, function_args=[], function_kwargs={'shape': eval('(batch_size, input_dim)')}, max_wait_secs=0)
(logits, covmat) = gp_layer(embedding)

class DeepResNetSNGP(DeepResNet):

    def __init__(self, spec_norm_bound=0.9, **kwargs):
        self.spec_norm_bound = spec_norm_bound
        super().__init__(**kwargs)

    def make_dense_layer(self):
        """Applies spectral normalization to the hidden layer."""
        dense_layer = super().make_dense_layer()
        return nlp_layers.SpectralNormalization(dense_layer, norm_multiplier=self.spec_norm_bound)

    def make_output_layer(self, num_classes):
        """Uses Gaussian process as the output layer."""
        return nlp_layers.RandomFeatureGaussianProcess(num_classes, gp_cov_momentum=-1, **self.classifier_kwargs)

    def call(self, inputs, training=False, return_covmat=False):
        (logits, covmat) = super().call(inputs)
        if not training and return_covmat:
            return (logits, covmat)
        return logits
resnet_config
sngp_model = DeepResNetSNGP(**resnet_config)
custom_method(
sngp_model.build((None, 2)), imports='import pkg_resources;import tensorflow as tf;import numpy as np;import importlib;import matplotlib.pyplot as plt;import official.nlp.modeling.layers as nlp_layers;import matplotlib.colors as colors;import sklearn.datasets', function_to_run='obj.build(*args)', method_object=eval('sngp_model'), function_args=[eval('(None, 2)')], function_kwargs={}, max_wait_secs=0, custom_class=None)
custom_method(
sngp_model.summary(), imports='import pkg_resources;import tensorflow as tf;import numpy as np;import importlib;import matplotlib.pyplot as plt;import official.nlp.modeling.layers as nlp_layers;import matplotlib.colors as colors;import sklearn.datasets', function_to_run='obj.summary()', method_object=eval('sngp_model'), function_args=[], function_kwargs={}, max_wait_secs=0, custom_class=None)

class ResetCovarianceCallback(tf.keras.callbacks.Callback):

    def on_epoch_begin(self, epoch, logs=None):
        """Resets covariance matrix at the beginning of the epoch."""
        if epoch > 0:
            self.model.classifier.reset_covariance_matrix()

class DeepResNetSNGPWithCovReset(DeepResNetSNGP):

    def fit(self, *args, **kwargs):
        """Adds ResetCovarianceCallback to model callbacks."""
        kwargs['callbacks'] = list(kwargs.get('callbacks', []))
        kwargs['callbacks'].append(ResetCovarianceCallback())
        return super().fit(*args, **kwargs)
sngp_model = DeepResNetSNGPWithCovReset(**resnet_config)
custom_method(
sngp_model.compile(**train_config), imports='import pkg_resources;import tensorflow as tf;import numpy as np;import importlib;import matplotlib.pyplot as plt;import official.nlp.modeling.layers as nlp_layers;import matplotlib.colors as colors;import sklearn.datasets', function_to_run='obj.compile(**kwargs)', method_object=eval('sngp_model'), function_args=[], function_kwargs={None: eval('train_config')}, max_wait_secs=0, custom_class=None)
custom_method(
sngp_model.fit(train_examples, train_labels, **fit_config), imports='import pkg_resources;import tensorflow as tf;import numpy as np;import importlib;import matplotlib.pyplot as plt;import official.nlp.modeling.layers as nlp_layers;import matplotlib.colors as colors;import sklearn.datasets', function_to_run='obj.fit(*args, **kwargs)', method_object=eval('sngp_model'), function_args=[eval('train_examples'), eval('train_labels')], function_kwargs={None: eval('fit_config')}, max_wait_secs=0, custom_class=None)
(sngp_logits, sngp_covmat) = custom_method(
sngp_model(test_examples, return_covmat=True), imports='import pkg_resources;import tensorflow as tf;import numpy as np;import importlib;import matplotlib.pyplot as plt;import official.nlp.modeling.layers as nlp_layers;import matplotlib.colors as colors;import sklearn.datasets', function_to_run='obj(*args, **kwargs)', method_object=eval('sngp_model'), function_args=[eval('test_examples')], function_kwargs={'return_covmat': eval('True')}, max_wait_secs=0, custom_class=None)
sngp_variance = custom_method(
tf.linalg.diag_part(sngp_covmat), imports='import pkg_resources;import tensorflow as tf;import numpy as np;import importlib;import matplotlib.pyplot as plt;import official.nlp.modeling.layers as nlp_layers;import matplotlib.colors as colors;import sklearn.datasets', function_to_run='tf.linalg.diag_part(*args)', method_object=None, function_args=[eval('sngp_covmat')], function_kwargs={}, max_wait_secs=0)[:, None]
sngp_logits_adjusted = sngp_logits / custom_method(
tf.sqrt(1.0 + np.pi / 8.0 * sngp_variance), imports='import pkg_resources;import tensorflow as tf;import numpy as np;import importlib;import matplotlib.pyplot as plt;import official.nlp.modeling.layers as nlp_layers;import matplotlib.colors as colors;import sklearn.datasets', function_to_run='tf.sqrt(*args)', method_object=None, function_args=[eval('1. + (np.pi / 8.) * sngp_variance')], function_kwargs={}, max_wait_secs=0)
sngp_probs = custom_method(
tf.nn.softmax(sngp_logits_adjusted, axis=-1), imports='import pkg_resources;import tensorflow as tf;import numpy as np;import importlib;import matplotlib.pyplot as plt;import official.nlp.modeling.layers as nlp_layers;import matplotlib.colors as colors;import sklearn.datasets', function_to_run='tf.nn.softmax(*args, **kwargs)', method_object=None, function_args=[eval('sngp_logits_adjusted')], function_kwargs={'axis': eval('-1')}, max_wait_secs=0)[:, 0]

def compute_posterior_mean_probability(logits, covmat, lambda_param=np.pi / 8.0):
    logits_adjusted = nlp_layers.gaussian_process.mean_field_logits(logits, covmat, mean_field_factor=lambda_param)
    return custom_method(
    tf.nn.softmax(logits_adjusted, axis=-1), imports='import pkg_resources;import tensorflow as tf;import numpy as np;import importlib;import matplotlib.pyplot as plt;import official.nlp.modeling.layers as nlp_layers;import matplotlib.colors as colors;import sklearn.datasets', function_to_run='tf.nn.softmax(*args, **kwargs)', method_object=None, function_args=[eval('logits_adjusted')], function_kwargs={'axis': eval('-1')}, max_wait_secs=0)[:, 0]
(sngp_logits, sngp_covmat) = custom_method(
sngp_model(test_examples, return_covmat=True), imports='import pkg_resources;import tensorflow as tf;import numpy as np;import importlib;import matplotlib.pyplot as plt;import official.nlp.modeling.layers as nlp_layers;import matplotlib.colors as colors;import sklearn.datasets', function_to_run='obj(*args, **kwargs)', method_object=eval('sngp_model'), function_args=[eval('test_examples')], function_kwargs={'return_covmat': eval('True')}, max_wait_secs=0, custom_class=None)
sngp_probs = compute_posterior_mean_probability(sngp_logits, sngp_covmat)

def plot_predictions(pred_probs, model_name=''):
    """Plot normalized class probabilities and predictive uncertainties."""
    uncertainty = pred_probs * (1.0 - pred_probs)
    (fig, axs) = plt.subplots(1, 2, figsize=(14, 5))
    pcm_0 = plot_uncertainty_surface(pred_probs, ax=axs[0])
    pcm_1 = plot_uncertainty_surface(uncertainty, ax=axs[1])
    fig.colorbar(pcm_0, ax=axs[0])
    fig.colorbar(pcm_1, ax=axs[1])
    axs[0].set_title(f'Class Probability, {model_name}')
    axs[1].set_title(f'(Normalized) Predictive Uncertainty, {model_name}')
    plt.show()

def train_and_test_sngp(train_examples, test_examples):
    sngp_model = DeepResNetSNGPWithCovReset(**resnet_config)
    custom_method(
    sngp_model.compile(**train_config), imports='import pkg_resources;import tensorflow as tf;import numpy as np;import importlib;import matplotlib.pyplot as plt;import official.nlp.modeling.layers as nlp_layers;import matplotlib.colors as colors;import sklearn.datasets', function_to_run='obj.compile(**kwargs)', method_object=eval('sngp_model'), function_args=[], function_kwargs={None: eval('train_config')}, max_wait_secs=0, custom_class=None)
    custom_method(
    sngp_model.fit(train_examples, train_labels, verbose=0, **fit_config), imports='import pkg_resources;import tensorflow as tf;import numpy as np;import importlib;import matplotlib.pyplot as plt;import official.nlp.modeling.layers as nlp_layers;import matplotlib.colors as colors;import sklearn.datasets', function_to_run='obj.fit(*args, **kwargs)', method_object=eval('sngp_model'), function_args=[eval('train_examples'), eval('train_labels')], function_kwargs={'verbose': eval('0'), None: eval('fit_config')}, max_wait_secs=0, custom_class=None)
    (sngp_logits, sngp_covmat) = custom_method(
    sngp_model(test_examples, return_covmat=True), imports='import pkg_resources;import tensorflow as tf;import numpy as np;import importlib;import matplotlib.pyplot as plt;import official.nlp.modeling.layers as nlp_layers;import matplotlib.colors as colors;import sklearn.datasets', function_to_run='obj(*args, **kwargs)', method_object=eval('sngp_model'), function_args=[eval('test_examples')], function_kwargs={'return_covmat': eval('True')}, max_wait_secs=0, custom_class=None)
    sngp_probs = compute_posterior_mean_probability(sngp_logits, sngp_covmat)
    return sngp_probs
sngp_probs = train_and_test_sngp(train_examples, test_examples)
plot_predictions(sngp_probs, model_name='SNGP')
plot_predictions(resnet_probs, model_name='Deterministic')
num_ensemble = 10

def mc_dropout_sampling(test_examples):
    return custom_method(
    resnet_model(test_examples, training=True), imports='import pkg_resources;import tensorflow as tf;import numpy as np;import importlib;import matplotlib.pyplot as plt;import official.nlp.modeling.layers as nlp_layers;import matplotlib.colors as colors;import sklearn.datasets', function_to_run='obj(*args, **kwargs)', method_object=eval('resnet_model'), function_args=[eval('test_examples')], function_kwargs={'training': eval('True')}, max_wait_secs=0, custom_class='class DeepResNet(tf.keras.Model):\n  """Defines a multi-layer residual network."""\n  def __init__(self, num_classes, num_layers=3, num_hidden=128,\n               dropout_rate=0.1, **classifier_kwargs):\n    super().__init__()\n    self.num_hidden = num_hidden\n    self.num_layers = num_layers\n    self.dropout_rate = dropout_rate\n    self.classifier_kwargs = classifier_kwargs\n\n    self.input_layer = tf.keras.layers.Dense(self.num_hidden, trainable=False)\n    self.dense_layers = [self.make_dense_layer() for _ in range(num_layers)]\n\n    self.classifier = self.make_output_layer(num_classes)\n\n  def call(self, inputs):\n    hidden = self.input_layer(inputs)\n\n    for i in range(self.num_layers):\n      resid = self.dense_layers[i](hidden)\n      resid = tf.keras.layers.Dropout(self.dropout_rate)(resid)\n      hidden += resid\n\n    return self.classifier(hidden)\n\n  def make_dense_layer(self):\n    """Uses the Dense layer as the hidden layer."""\n    return tf.keras.layers.Dense(self.num_hidden, activation="relu")\n\n  def make_output_layer(self, num_classes):\n    """Uses the Dense layer as the output layer."""\n    return tf.keras.layers.Dense(\n        num_classes, **self.classifier_kwargs)')
dropout_logit_samples = [mc_dropout_sampling(test_examples) for _ in range(num_ensemble)]
dropout_prob_samples = [custom_method(
tf.nn.softmax(dropout_logits, axis=-1), imports='import pkg_resources;import tensorflow as tf;import numpy as np;import importlib;import matplotlib.pyplot as plt;import official.nlp.modeling.layers as nlp_layers;import matplotlib.colors as colors;import sklearn.datasets', function_to_run='tf.nn.softmax(*args, **kwargs)', method_object=None, function_args=[eval('dropout_logits')], function_kwargs={'axis': eval('-1')}, max_wait_secs=0)[:, 0] for dropout_logits in dropout_logit_samples]
dropout_probs = custom_method(
tf.reduce_mean(dropout_prob_samples, axis=0), imports='import pkg_resources;import tensorflow as tf;import numpy as np;import importlib;import matplotlib.pyplot as plt;import official.nlp.modeling.layers as nlp_layers;import matplotlib.colors as colors;import sklearn.datasets', function_to_run='tf.reduce_mean(*args, **kwargs)', method_object=None, function_args=[eval('dropout_prob_samples')], function_kwargs={'axis': eval('0')}, max_wait_secs=0)
dropout_probs = custom_method(
tf.reduce_mean(dropout_prob_samples, axis=0), imports='import pkg_resources;import tensorflow as tf;import numpy as np;import importlib;import matplotlib.pyplot as plt;import official.nlp.modeling.layers as nlp_layers;import matplotlib.colors as colors;import sklearn.datasets', function_to_run='tf.reduce_mean(*args, **kwargs)', method_object=None, function_args=[eval('dropout_prob_samples')], function_kwargs={'axis': eval('0')}, max_wait_secs=0)
plot_predictions(dropout_probs, model_name='MC Dropout')
resnet_ensemble = []
for _ in range(num_ensemble):
    resnet_model = DeepResNet(**resnet_config)
    custom_method(
    resnet_model.compile(optimizer=optimizer, loss=loss, metrics=metrics), imports='import pkg_resources;import tensorflow as tf;import numpy as np;import importlib;import matplotlib.pyplot as plt;import official.nlp.modeling.layers as nlp_layers;import matplotlib.colors as colors;import sklearn.datasets', function_to_run='obj.compile(**kwargs)', method_object=eval('resnet_model'), function_args=[], function_kwargs={'optimizer': eval('optimizer'), 'loss': eval('loss'), 'metrics': eval('metrics')}, max_wait_secs=0, custom_class='class DeepResNet(tf.keras.Model):\n  """Defines a multi-layer residual network."""\n  def __init__(self, num_classes, num_layers=3, num_hidden=128,\n               dropout_rate=0.1, **classifier_kwargs):\n    super().__init__()\n    self.num_hidden = num_hidden\n    self.num_layers = num_layers\n    self.dropout_rate = dropout_rate\n    self.classifier_kwargs = classifier_kwargs\n\n    self.input_layer = tf.keras.layers.Dense(self.num_hidden, trainable=False)\n    self.dense_layers = [self.make_dense_layer() for _ in range(num_layers)]\n\n    self.classifier = self.make_output_layer(num_classes)\n\n  def call(self, inputs):\n    hidden = self.input_layer(inputs)\n\n    for i in range(self.num_layers):\n      resid = self.dense_layers[i](hidden)\n      resid = tf.keras.layers.Dropout(self.dropout_rate)(resid)\n      hidden += resid\n\n    return self.classifier(hidden)\n\n  def make_dense_layer(self):\n    """Uses the Dense layer as the hidden layer."""\n    return tf.keras.layers.Dense(self.num_hidden, activation="relu")\n\n  def make_output_layer(self, num_classes):\n    """Uses the Dense layer as the output layer."""\n    return tf.keras.layers.Dense(\n        num_classes, **self.classifier_kwargs)')
    custom_method(
    resnet_model.fit(train_examples, train_labels, verbose=0, **fit_config), imports='import pkg_resources;import tensorflow as tf;import numpy as np;import importlib;import matplotlib.pyplot as plt;import official.nlp.modeling.layers as nlp_layers;import matplotlib.colors as colors;import sklearn.datasets', function_to_run='obj.fit(*args, **kwargs)', method_object=eval('resnet_model'), function_args=[eval('train_examples'), eval('train_labels')], function_kwargs={'verbose': eval('0'), None: eval('fit_config')}, max_wait_secs=0, custom_class='class DeepResNet(tf.keras.Model):\n  """Defines a multi-layer residual network."""\n  def __init__(self, num_classes, num_layers=3, num_hidden=128,\n               dropout_rate=0.1, **classifier_kwargs):\n    super().__init__()\n    self.num_hidden = num_hidden\n    self.num_layers = num_layers\n    self.dropout_rate = dropout_rate\n    self.classifier_kwargs = classifier_kwargs\n\n    self.input_layer = tf.keras.layers.Dense(self.num_hidden, trainable=False)\n    self.dense_layers = [self.make_dense_layer() for _ in range(num_layers)]\n\n    self.classifier = self.make_output_layer(num_classes)\n\n  def call(self, inputs):\n    hidden = self.input_layer(inputs)\n\n    for i in range(self.num_layers):\n      resid = self.dense_layers[i](hidden)\n      resid = tf.keras.layers.Dropout(self.dropout_rate)(resid)\n      hidden += resid\n\n    return self.classifier(hidden)\n\n  def make_dense_layer(self):\n    """Uses the Dense layer as the hidden layer."""\n    return tf.keras.layers.Dense(self.num_hidden, activation="relu")\n\n  def make_output_layer(self, num_classes):\n    """Uses the Dense layer as the output layer."""\n    return tf.keras.layers.Dense(\n        num_classes, **self.classifier_kwargs)')
    resnet_ensemble.append(resnet_model)
ensemble_logit_samples = [model(test_examples) for model in resnet_ensemble]
ensemble_prob_samples = [custom_method(
tf.nn.softmax(logits, axis=-1), imports='import pkg_resources;import tensorflow as tf;import numpy as np;import importlib;import matplotlib.pyplot as plt;import official.nlp.modeling.layers as nlp_layers;import matplotlib.colors as colors;import sklearn.datasets', function_to_run='tf.nn.softmax(*args, **kwargs)', method_object=None, function_args=[eval('logits')], function_kwargs={'axis': eval('-1')}, max_wait_secs=0)[:, 0] for logits in ensemble_logit_samples]
ensemble_probs = custom_method(
tf.reduce_mean(ensemble_prob_samples, axis=0), imports='import pkg_resources;import tensorflow as tf;import numpy as np;import importlib;import matplotlib.pyplot as plt;import official.nlp.modeling.layers as nlp_layers;import matplotlib.colors as colors;import sklearn.datasets', function_to_run='tf.reduce_mean(*args, **kwargs)', method_object=None, function_args=[eval('ensemble_prob_samples')], function_kwargs={'axis': eval('0')}, max_wait_secs=0)
plot_predictions(ensemble_probs, model_name='Deep ensemble')
