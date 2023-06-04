import pkg_resources
import importlib
import sys
from tool.client.client_config import EXPERIMENT_DIR
from tool.measurement.execution import before_execution as before_execution_INSERTED_INTO_SCRIPT
from tool.measurement.execution import after_execution as after_execution_INSERTED_INTO_SCRIPT
experiment_number = sys.argv[1]
experiment_project = sys.argv[2]
EXPERIMENT_FILE_PATH = EXPERIMENT_DIR / 'local-execution' / experiment_project / f'experiment-{experiment_number}.json'
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
        start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT()
        self.input_layer = tf.keras.layers.Dense(self.num_hidden, trainable=False)
        after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tf.keras.layers.Dense()', method_object=None, function_args=[self.num_hidden], function_kwargs={'trainable': False})
        self.dense_layers = [self.make_dense_layer() for _ in range(num_layers)]
        self.classifier = self.make_output_layer(num_classes)

    def call(self, inputs):
        hidden = self.input_layer(inputs)
        for i in range(self.num_layers):
            resid = self.dense_layers[i](hidden)
            start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT()
            resid = tf.keras.layers.Dropout(self.dropout_rate)(resid)
            after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tf.keras.layers.Dropout(self.dropout_rate)()', method_object=None, function_args=[resid], function_kwargs=None)
            hidden += resid
        return self.classifier(hidden)

    def make_dense_layer(self):
        """Uses the Dense layer as the hidden layer."""
        return tf.keras.layers.Dense(self.num_hidden, activation='relu')

    def make_output_layer(self, num_classes):
        """Uses the Dense layer as the output layer."""
        return tf.keras.layers.Dense(num_classes, **self.classifier_kwargs)
resnet_config = dict(num_classes=2, num_layers=6, num_hidden=128)
resnet_model = DeepResNet(**resnet_config)
start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT()
resnet_model.build((None, 2))
after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='DeepResNet.build()', method_object=resnet_model, function_args=[(None, 2)], function_kwargs=None)
start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT()
resnet_model.summary()
after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='DeepResNet.summary()', method_object=resnet_model, function_args=None, function_kwargs=None)
start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT()
loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tf.keras.losses.SparseCategoricalCrossentropy()', method_object=None, function_args=None, function_kwargs={'from_logits': True})
metrics = (tf.keras.metrics.SparseCategoricalAccuracy(),)
start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT()
optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=0.0001)
after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tf.keras.optimizers.legacy.Adam()', method_object=None, function_args=None, function_kwargs={'learning_rate': 0.0001})
train_config = dict(loss=loss, metrics=metrics, optimizer=optimizer)
fit_config = dict(batch_size=128, epochs=100)
start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT()
resnet_model.compile(**train_config)
after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='DeepResNet.compile()', method_object=resnet_model, function_args=None, function_kwargs={None: train_config})
start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT()
resnet_model.fit(train_examples, train_labels, **fit_config)
after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='DeepResNet.fit()', method_object=resnet_model, function_args=[train_examples, train_labels], function_kwargs={None: fit_config})

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
start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT()
resnet_logits = resnet_model(test_examples)
after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='DeepResNet()', method_object=resnet_model, function_args=[test_examples], function_kwargs=None)
resnet_probs = tf.nn.softmax(resnet_logits, axis=-1)[:, 0]
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
start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT()
dense = tf.keras.layers.Dense(units=10)
after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tf.keras.layers.Dense()', method_object=None, function_args=None, function_kwargs={'units': 10})
dense = nlp_layers.SpectralNormalization(dense, norm_multiplier=0.9)
batch_size = 32
input_dim = 1024
num_classes = 10
gp_layer = nlp_layers.RandomFeatureGaussianProcess(units=num_classes, num_inducing=1024, normalize_input=False, scale_random_features=True, gp_cov_momentum=-1)
start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT()
embedding = tf.random.normal(shape=(batch_size, input_dim))
after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tf.random.normal()', method_object=None, function_args=None, function_kwargs={'shape': (batch_size, input_dim)})
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
start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT()
sngp_model.build((None, 2))
after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='DeepResNetSNGP.build()', method_object=sngp_model, function_args=[(None, 2)], function_kwargs=None)
start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT()
sngp_model.summary()
after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='DeepResNetSNGP.summary()', method_object=sngp_model, function_args=None, function_kwargs=None)

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
start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT()
sngp_model.compile(**train_config)
after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='DeepResNetSNGPWithCovReset.compile()', method_object=sngp_model, function_args=None, function_kwargs={None: train_config})
start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT()
sngp_model.fit(train_examples, train_labels, **fit_config)
after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='DeepResNetSNGPWithCovReset.fit()', method_object=sngp_model, function_args=[train_examples, train_labels], function_kwargs={None: fit_config})
start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT()
(sngp_logits, sngp_covmat) = sngp_model(test_examples, return_covmat=True)
after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='DeepResNetSNGPWithCovReset()', method_object=sngp_model, function_args=[test_examples], function_kwargs={'return_covmat': True})
sngp_variance = tf.linalg.diag_part(sngp_covmat)[:, None]
sngp_logits_adjusted = sngp_logits / tf.sqrt(1.0 + np.pi / 8.0 * sngp_variance)
sngp_probs = tf.nn.softmax(sngp_logits_adjusted, axis=-1)[:, 0]

def compute_posterior_mean_probability(logits, covmat, lambda_param=np.pi / 8.0):
    logits_adjusted = nlp_layers.gaussian_process.mean_field_logits(logits, covmat, mean_field_factor=lambda_param)
    return tf.nn.softmax(logits_adjusted, axis=-1)[:, 0]
start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT()
(sngp_logits, sngp_covmat) = sngp_model(test_examples, return_covmat=True)
after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='DeepResNetSNGPWithCovReset()', method_object=sngp_model, function_args=[test_examples], function_kwargs={'return_covmat': True})
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
    start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT()
    sngp_model.compile(**train_config)
    after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='DeepResNetSNGPWithCovReset.compile()', method_object=sngp_model, function_args=None, function_kwargs={None: train_config})
    start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT()
    sngp_model.fit(train_examples, train_labels, verbose=0, **fit_config)
    after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='DeepResNetSNGPWithCovReset.fit()', method_object=sngp_model, function_args=[train_examples, train_labels], function_kwargs={'verbose': 0, None: fit_config})
    start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT()
    (sngp_logits, sngp_covmat) = sngp_model(test_examples, return_covmat=True)
    after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='DeepResNetSNGPWithCovReset()', method_object=sngp_model, function_args=[test_examples], function_kwargs={'return_covmat': True})
    sngp_probs = compute_posterior_mean_probability(sngp_logits, sngp_covmat)
    return sngp_probs
sngp_probs = train_and_test_sngp(train_examples, test_examples)
plot_predictions(sngp_probs, model_name='SNGP')
plot_predictions(resnet_probs, model_name='Deterministic')
num_ensemble = 10

def mc_dropout_sampling(test_examples):
    return resnet_model(test_examples, training=True)
dropout_logit_samples = [mc_dropout_sampling(test_examples) for _ in range(num_ensemble)]
dropout_prob_samples = [tf.nn.softmax(dropout_logits, axis=-1)[:, 0] for dropout_logits in dropout_logit_samples]
start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT()
dropout_probs = tf.reduce_mean(dropout_prob_samples, axis=0)
after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tf.reduce_mean()', method_object=None, function_args=[dropout_prob_samples], function_kwargs={'axis': 0})
start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT()
dropout_probs = tf.reduce_mean(dropout_prob_samples, axis=0)
after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tf.reduce_mean()', method_object=None, function_args=[dropout_prob_samples], function_kwargs={'axis': 0})
plot_predictions(dropout_probs, model_name='MC Dropout')
resnet_ensemble = []
for _ in range(num_ensemble):
    resnet_model = DeepResNet(**resnet_config)
    start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT()
    resnet_model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
    after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='DeepResNet.compile()', method_object=resnet_model, function_args=None, function_kwargs={'optimizer': optimizer, 'loss': loss, 'metrics': metrics})
    start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT()
    resnet_model.fit(train_examples, train_labels, verbose=0, **fit_config)
    after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='DeepResNet.fit()', method_object=resnet_model, function_args=[train_examples, train_labels], function_kwargs={'verbose': 0, None: fit_config})
    resnet_ensemble.append(resnet_model)
ensemble_logit_samples = [model(test_examples) for model in resnet_ensemble]
ensemble_prob_samples = [tf.nn.softmax(logits, axis=-1)[:, 0] for logits in ensemble_logit_samples]
start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT()
ensemble_probs = tf.reduce_mean(ensemble_prob_samples, axis=0)
after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tf.reduce_mean()', method_object=None, function_args=[ensemble_prob_samples], function_kwargs={'axis': 0})
plot_predictions(ensemble_probs, model_name='Deep ensemble')
