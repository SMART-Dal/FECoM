import sys
from fecom.patching.patching_config import EXPERIMENT_DIR
from fecom.measurement.execution import before_execution as before_execution_INSERTED_INTO_SCRIPT
from fecom.measurement.execution import after_execution as after_execution_INSERTED_INTO_SCRIPT
from fecom.experiment.experiment_kinds import ExperimentKinds
experiment_number = sys.argv[1]
experiment_project = sys.argv[2]
EXPERIMENT_FILE_PATH = EXPERIMENT_DIR / ExperimentKinds.PROJECT_LEVEL.value / experiment_project / f'experiment-{experiment_number}.json'
start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT(experiment_file_path=EXPERIMENT_FILE_PATH, enable_skip_calls=False)
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import regularizers
print(tf.__version__)
import tensorflow_docs as tfdocs
import tensorflow_docs.modeling
import tensorflow_docs.plots
from IPython import display
from matplotlib import pyplot as plt
import numpy as np
import pathlib
import shutil
import tempfile
logdir = pathlib.Path(tempfile.mkdtemp()) / 'tensorboard_logs'
shutil.rmtree(logdir, ignore_errors=True)
gz = tf.keras.utils.get_file('HIGGS.csv.gz', 'http://mlphysics.ics.uci.edu/data/higgs/HIGGS.csv.gz')
FEATURES = 28
ds = tf.data.experimental.CsvDataset(gz, [float()] * (FEATURES + 1), compression_type='GZIP')

def pack_row(*row):
    label = row[0]
    features = tf.stack(row[1:], 1)
    return (features, label)
packed_ds = ds.batch(10000).map(pack_row).unbatch()
for (features, label) in packed_ds.batch(1000).take(1):
    print(features[0])
    plt.hist(features.numpy().flatten(), bins=101)
N_VALIDATION = int(1000.0)
N_TRAIN = int(10000.0)
BUFFER_SIZE = int(10000.0)
BATCH_SIZE = 500
STEPS_PER_EPOCH = N_TRAIN // BATCH_SIZE
validate_ds = packed_ds.take(N_VALIDATION).cache()
train_ds = packed_ds.skip(N_VALIDATION).take(N_TRAIN).cache()
train_ds
validate_ds = validate_ds.batch(BATCH_SIZE)
train_ds = train_ds.shuffle(BUFFER_SIZE).repeat().batch(BATCH_SIZE)
lr_schedule = tf.keras.optimizers.schedules.InverseTimeDecay(0.001, decay_steps=STEPS_PER_EPOCH * 1000, decay_rate=1, staircase=False)

def get_optimizer():
    return tf.keras.optimizers.Adam(lr_schedule)
step = np.linspace(0, 100000)
lr = lr_schedule(step)
plt.figure(figsize=(8, 6))
plt.plot(step / STEPS_PER_EPOCH, lr)
plt.ylim([0, max(plt.ylim())])
plt.xlabel('Epoch')
_ = plt.ylabel('Learning Rate')

def get_callbacks(name):
    return [tfdocs.modeling.EpochDots(), tf.keras.callbacks.EarlyStopping(monitor='val_binary_crossentropy', patience=200), tf.keras.callbacks.TensorBoard(logdir / name)]

def compile_and_fit(model, name, optimizer=None, max_epochs=10000):
    if optimizer is None:
        optimizer = get_optimizer()
    model.compile(optimizer=optimizer, loss=tf.keras.losses.BinaryCrossentropy(from_logits=True), metrics=[tf.keras.metrics.BinaryCrossentropy(from_logits=True, name='binary_crossentropy'), 'accuracy'])
    model.summary()
    history = model.fit(train_ds, steps_per_epoch=STEPS_PER_EPOCH, epochs=max_epochs, validation_data=validate_ds, callbacks=get_callbacks(name), verbose=0)
    return history
tiny_model = tf.keras.Sequential([layers.Dense(16, activation='elu', input_shape=(FEATURES,)), layers.Dense(1)])
size_histories = {}
size_histories['Tiny'] = compile_and_fit(tiny_model, 'sizes/Tiny')
plotter = tfdocs.plots.HistoryPlotter(metric='binary_crossentropy', smoothing_std=10)
plotter.plot(size_histories)
plt.ylim([0.5, 0.7])
small_model = tf.keras.Sequential([layers.Dense(16, activation='elu', input_shape=(FEATURES,)), layers.Dense(16, activation='elu'), layers.Dense(1)])
size_histories['Small'] = compile_and_fit(small_model, 'sizes/Small')
medium_model = tf.keras.Sequential([layers.Dense(64, activation='elu', input_shape=(FEATURES,)), layers.Dense(64, activation='elu'), layers.Dense(64, activation='elu'), layers.Dense(1)])
size_histories['Medium'] = compile_and_fit(medium_model, 'sizes/Medium')
large_model = tf.keras.Sequential([layers.Dense(512, activation='elu', input_shape=(FEATURES,)), layers.Dense(512, activation='elu'), layers.Dense(512, activation='elu'), layers.Dense(512, activation='elu'), layers.Dense(1)])
size_histories['large'] = compile_and_fit(large_model, 'sizes/large')
plotter.plot(size_histories)
a = plt.xscale('log')
plt.xlim([5, max(plt.xlim())])
plt.ylim([0.5, 0.7])
plt.xlabel('Epochs [Log Scale]')
shutil.rmtree(logdir / 'regularizers/Tiny', ignore_errors=True)
shutil.copytree(logdir / 'sizes/Tiny', logdir / 'regularizers/Tiny')
regularizer_histories = {}
regularizer_histories['Tiny'] = size_histories['Tiny']
l2_model = tf.keras.Sequential([layers.Dense(512, activation='elu', kernel_regularizer=regularizers.l2(0.001), input_shape=(FEATURES,)), layers.Dense(512, activation='elu', kernel_regularizer=regularizers.l2(0.001)), layers.Dense(512, activation='elu', kernel_regularizer=regularizers.l2(0.001)), layers.Dense(512, activation='elu', kernel_regularizer=regularizers.l2(0.001)), layers.Dense(1)])
regularizer_histories['l2'] = compile_and_fit(l2_model, 'regularizers/l2')
plotter.plot(regularizer_histories)
plt.ylim([0.5, 0.7])
result = l2_model(features)
regularization_loss = tf.add_n(l2_model.losses)
dropout_model = tf.keras.Sequential([layers.Dense(512, activation='elu', input_shape=(FEATURES,)), layers.Dropout(0.5), layers.Dense(512, activation='elu'), layers.Dropout(0.5), layers.Dense(512, activation='elu'), layers.Dropout(0.5), layers.Dense(512, activation='elu'), layers.Dropout(0.5), layers.Dense(1)])
regularizer_histories['dropout'] = compile_and_fit(dropout_model, 'regularizers/dropout')
plotter.plot(regularizer_histories)
plt.ylim([0.5, 0.7])
combined_model = tf.keras.Sequential([layers.Dense(512, kernel_regularizer=regularizers.l2(0.0001), activation='elu', input_shape=(FEATURES,)), layers.Dropout(0.5), layers.Dense(512, kernel_regularizer=regularizers.l2(0.0001), activation='elu'), layers.Dropout(0.5), layers.Dense(512, kernel_regularizer=regularizers.l2(0.0001), activation='elu'), layers.Dropout(0.5), layers.Dense(512, kernel_regularizer=regularizers.l2(0.0001), activation='elu'), layers.Dropout(0.5), layers.Dense(1)])
regularizer_histories['combined'] = compile_and_fit(combined_model, 'regularizers/combined')
plotter.plot(regularizer_histories)
plt.ylim([0.5, 0.7])
after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, enable_skip_calls=False)