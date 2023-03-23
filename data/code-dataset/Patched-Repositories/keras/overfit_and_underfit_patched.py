import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import regularizers
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
    result = send_request(imports=imports, function_to_run=function_to_run, function_args=function_args, function_kwargs=function_kwargs, max_wait_secs=max_wait_secs, wait_after_run_secs=wait_after_run_secs, method_object=method_object, object_signature=object_signature, custom_class=custom_class, experiment_file_path=EXPERIMENT_FILE_PATH)
    return func
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
gz = custom_method(
tf.keras.utils.get_file('HIGGS.csv.gz', 'http://mlphysics.ics.uci.edu/data/higgs/HIGGS.csv.gz'), imports='import tensorflow_docs.plots;from tensorflow.keras import layers;from  IPython import display;import pathlib;from tensorflow.keras import regularizers;import shutil;import tempfile;import tensorflow_docs as tfdocs;from matplotlib import pyplot as plt;import numpy as np;import tensorflow_docs.modeling;import tensorflow as tf', function_to_run='tf.keras.utils.get_file(*args)', method_object=None, object_signature=None, function_args=[eval("'HIGGS.csv.gz'"), eval("'http://mlphysics.ics.uci.edu/data/higgs/HIGGS.csv.gz'")], function_kwargs={})
FEATURES = 28
ds = custom_method(
tf.data.experimental.CsvDataset(gz, [float()] * (FEATURES + 1), compression_type='GZIP'), imports='import tensorflow_docs.plots;from tensorflow.keras import layers;from  IPython import display;import pathlib;from tensorflow.keras import regularizers;import shutil;import tempfile;import tensorflow_docs as tfdocs;from matplotlib import pyplot as plt;import numpy as np;import tensorflow_docs.modeling;import tensorflow as tf', function_to_run='tf.data.experimental.CsvDataset(*args, **kwargs)', method_object=None, object_signature=None, function_args=[eval('gz'), eval('[float(),]*(FEATURES+1)')], function_kwargs={'compression_type': eval('"GZIP"')})

def pack_row(*row):
    label = row[0]
    features = custom_method(
    tf.stack(row[1:], 1), imports='import tensorflow_docs.plots;from tensorflow.keras import layers;from  IPython import display;import pathlib;from tensorflow.keras import regularizers;import shutil;import tempfile;import tensorflow_docs as tfdocs;from matplotlib import pyplot as plt;import numpy as np;import tensorflow_docs.modeling;import tensorflow as tf', function_to_run='tf.stack(*args)', method_object=None, object_signature=None, function_args=[eval('row[1:]'), eval('1')], function_kwargs={})
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
lr_schedule = custom_method(
tf.keras.optimizers.schedules.InverseTimeDecay(0.001, decay_steps=STEPS_PER_EPOCH * 1000, decay_rate=1, staircase=False), imports='import tensorflow_docs.plots;from tensorflow.keras import layers;from  IPython import display;import pathlib;from tensorflow.keras import regularizers;import shutil;import tempfile;import tensorflow_docs as tfdocs;from matplotlib import pyplot as plt;import numpy as np;import tensorflow_docs.modeling;import tensorflow as tf', function_to_run='tf.keras.optimizers.schedules.InverseTimeDecay(*args, **kwargs)', method_object=None, object_signature=None, function_args=[eval('0.001')], function_kwargs={'decay_steps': eval('STEPS_PER_EPOCH*1000'), 'decay_rate': eval('1'), 'staircase': eval('False')})

def get_optimizer():
    return custom_method(
    tf.keras.optimizers.Adam(lr_schedule), imports='import tensorflow_docs.plots;from tensorflow.keras import layers;from  IPython import display;import pathlib;from tensorflow.keras import regularizers;import shutil;import tempfile;import tensorflow_docs as tfdocs;from matplotlib import pyplot as plt;import numpy as np;import tensorflow_docs.modeling;import tensorflow as tf', function_to_run='tf.keras.optimizers.Adam(*args)', method_object=None, object_signature=None, function_args=[eval('lr_schedule')], function_kwargs={})
step = np.linspace(0, 100000)
lr = custom_method(
lr_schedule(step), imports='import tensorflow_docs.plots;from tensorflow.keras import layers;from  IPython import display;import pathlib;from tensorflow.keras import regularizers;import shutil;import tempfile;import tensorflow_docs as tfdocs;from matplotlib import pyplot as plt;import numpy as np;import tensorflow_docs.modeling;import tensorflow as tf', function_to_run='obj(*args)', method_object=eval('lr_schedule'), object_signature='tf.keras.optimizers.schedules.InverseTimeDecay', function_args=[eval('step')], function_kwargs={}, custom_class=None)
plt.figure(figsize=(8, 6))
plt.plot(step / STEPS_PER_EPOCH, lr)
plt.ylim([0, max(plt.ylim())])
plt.xlabel('Epoch')
_ = plt.ylabel('Learning Rate')

def get_callbacks(name):
    return [custom_method(
    tfdocs.modeling.EpochDots(), imports='import tensorflow_docs.plots;from tensorflow.keras import layers;from  IPython import display;import pathlib;from tensorflow.keras import regularizers;import shutil;import tempfile;import tensorflow_docs as tfdocs;from matplotlib import pyplot as plt;import numpy as np;import tensorflow_docs.modeling;import tensorflow as tf', function_to_run='tfdocs.modeling.EpochDots()', method_object=None, object_signature=None, function_args=[], function_kwargs={}), custom_method(
    tf.keras.callbacks.EarlyStopping(monitor='val_binary_crossentropy', patience=200), imports='import tensorflow_docs.plots;from tensorflow.keras import layers;from  IPython import display;import pathlib;from tensorflow.keras import regularizers;import shutil;import tempfile;import tensorflow_docs as tfdocs;from matplotlib import pyplot as plt;import numpy as np;import tensorflow_docs.modeling;import tensorflow as tf', function_to_run='tf.keras.callbacks.EarlyStopping(**kwargs)', method_object=None, object_signature=None, function_args=[], function_kwargs={'monitor': eval("'val_binary_crossentropy'"), 'patience': eval('200')}), custom_method(
    tf.keras.callbacks.TensorBoard(logdir / name), imports='import tensorflow_docs.plots;from tensorflow.keras import layers;from  IPython import display;import pathlib;from tensorflow.keras import regularizers;import shutil;import tempfile;import tensorflow_docs as tfdocs;from matplotlib import pyplot as plt;import numpy as np;import tensorflow_docs.modeling;import tensorflow as tf', function_to_run='tf.keras.callbacks.TensorBoard(*args)', method_object=None, object_signature=None, function_args=[eval('logdir/name')], function_kwargs={})]

def compile_and_fit(model, name, optimizer=None, max_epochs=10000):
    if optimizer is None:
        optimizer = get_optimizer()
    model.compile(optimizer=optimizer, loss=tf.keras.losses.BinaryCrossentropy(from_logits=True), metrics=[tf.keras.metrics.BinaryCrossentropy(from_logits=True, name='binary_crossentropy'), 'accuracy'])
    model.summary()
    history = model.fit(train_ds, steps_per_epoch=STEPS_PER_EPOCH, epochs=max_epochs, validation_data=validate_ds, callbacks=get_callbacks(name), verbose=0)
    return history
tiny_model = custom_method(
tf.keras.Sequential([layers.Dense(16, activation='elu', input_shape=(FEATURES,)), layers.Dense(1)]), imports='import tensorflow_docs.plots;from tensorflow.keras import layers;from  IPython import display;import pathlib;from tensorflow.keras import regularizers;import shutil;import tempfile;import tensorflow_docs as tfdocs;from matplotlib import pyplot as plt;import numpy as np;import tensorflow_docs.modeling;import tensorflow as tf', function_to_run='tf.keras.Sequential(*args)', method_object=None, object_signature=None, function_args=[eval("[\n    layers.Dense(16, activation='elu', input_shape=(FEATURES,)),\n    layers.Dense(1)\n]")], function_kwargs={})
size_histories = {}
size_histories['Tiny'] = compile_and_fit(tiny_model, 'sizes/Tiny')
plotter = custom_method(
tfdocs.plots.HistoryPlotter(metric='binary_crossentropy', smoothing_std=10), imports='import tensorflow_docs.plots;from tensorflow.keras import layers;from  IPython import display;import pathlib;from tensorflow.keras import regularizers;import shutil;import tempfile;import tensorflow_docs as tfdocs;from matplotlib import pyplot as plt;import numpy as np;import tensorflow_docs.modeling;import tensorflow as tf', function_to_run='tfdocs.plots.HistoryPlotter(**kwargs)', method_object=None, object_signature=None, function_args=[], function_kwargs={'metric': eval("'binary_crossentropy'"), 'smoothing_std': eval('10')})
custom_method(
plotter.plot(size_histories), imports='import tensorflow_docs.plots;from tensorflow.keras import layers;from  IPython import display;import pathlib;from tensorflow.keras import regularizers;import shutil;import tempfile;import tensorflow_docs as tfdocs;from matplotlib import pyplot as plt;import numpy as np;import tensorflow_docs.modeling;import tensorflow as tf', function_to_run='obj.plot(*args)', method_object=eval('plotter'), object_signature='tfdocs.plots.HistoryPlotter', function_args=[eval('size_histories')], function_kwargs={}, custom_class=None)
plt.ylim([0.5, 0.7])
small_model = custom_method(
tf.keras.Sequential([layers.Dense(16, activation='elu', input_shape=(FEATURES,)), layers.Dense(16, activation='elu'), layers.Dense(1)]), imports='import tensorflow_docs.plots;from tensorflow.keras import layers;from  IPython import display;import pathlib;from tensorflow.keras import regularizers;import shutil;import tempfile;import tensorflow_docs as tfdocs;from matplotlib import pyplot as plt;import numpy as np;import tensorflow_docs.modeling;import tensorflow as tf', function_to_run='tf.keras.Sequential(*args)', method_object=None, object_signature=None, function_args=[eval("[\n    layers.Dense(16, activation='elu', input_shape=(FEATURES,)),\n    layers.Dense(16, activation='elu'),\n    layers.Dense(1)\n]")], function_kwargs={})
size_histories['Small'] = compile_and_fit(small_model, 'sizes/Small')
medium_model = custom_method(
tf.keras.Sequential([layers.Dense(64, activation='elu', input_shape=(FEATURES,)), layers.Dense(64, activation='elu'), layers.Dense(64, activation='elu'), layers.Dense(1)]), imports='import tensorflow_docs.plots;from tensorflow.keras import layers;from  IPython import display;import pathlib;from tensorflow.keras import regularizers;import shutil;import tempfile;import tensorflow_docs as tfdocs;from matplotlib import pyplot as plt;import numpy as np;import tensorflow_docs.modeling;import tensorflow as tf', function_to_run='tf.keras.Sequential(*args)', method_object=None, object_signature=None, function_args=[eval("[\n    layers.Dense(64, activation='elu', input_shape=(FEATURES,)),\n    layers.Dense(64, activation='elu'),\n    layers.Dense(64, activation='elu'),\n    layers.Dense(1)\n]")], function_kwargs={})
size_histories['Medium'] = compile_and_fit(medium_model, 'sizes/Medium')
large_model = custom_method(
tf.keras.Sequential([layers.Dense(512, activation='elu', input_shape=(FEATURES,)), layers.Dense(512, activation='elu'), layers.Dense(512, activation='elu'), layers.Dense(512, activation='elu'), layers.Dense(1)]), imports='import tensorflow_docs.plots;from tensorflow.keras import layers;from  IPython import display;import pathlib;from tensorflow.keras import regularizers;import shutil;import tempfile;import tensorflow_docs as tfdocs;from matplotlib import pyplot as plt;import numpy as np;import tensorflow_docs.modeling;import tensorflow as tf', function_to_run='tf.keras.Sequential(*args)', method_object=None, object_signature=None, function_args=[eval("[\n    layers.Dense(512, activation='elu', input_shape=(FEATURES,)),\n    layers.Dense(512, activation='elu'),\n    layers.Dense(512, activation='elu'),\n    layers.Dense(512, activation='elu'),\n    layers.Dense(1)\n]")], function_kwargs={})
size_histories['large'] = compile_and_fit(large_model, 'sizes/large')
custom_method(
plotter.plot(size_histories), imports='import tensorflow_docs.plots;from tensorflow.keras import layers;from  IPython import display;import pathlib;from tensorflow.keras import regularizers;import shutil;import tempfile;import tensorflow_docs as tfdocs;from matplotlib import pyplot as plt;import numpy as np;import tensorflow_docs.modeling;import tensorflow as tf', function_to_run='obj.plot(*args)', method_object=eval('plotter'), object_signature='tfdocs.plots.HistoryPlotter', function_args=[eval('size_histories')], function_kwargs={}, custom_class=None)
a = plt.xscale('log')
plt.xlim([5, max(plt.xlim())])
plt.ylim([0.5, 0.7])
plt.xlabel('Epochs [Log Scale]')
shutil.rmtree(logdir / 'regularizers/Tiny', ignore_errors=True)
shutil.copytree(logdir / 'sizes/Tiny', logdir / 'regularizers/Tiny')
regularizer_histories = {}
regularizer_histories['Tiny'] = size_histories['Tiny']
l2_model = custom_method(
tf.keras.Sequential([layers.Dense(512, activation='elu', kernel_regularizer=regularizers.l2(0.001), input_shape=(FEATURES,)), layers.Dense(512, activation='elu', kernel_regularizer=regularizers.l2(0.001)), layers.Dense(512, activation='elu', kernel_regularizer=regularizers.l2(0.001)), layers.Dense(512, activation='elu', kernel_regularizer=regularizers.l2(0.001)), layers.Dense(1)]), imports='import tensorflow_docs.plots;from tensorflow.keras import layers;from  IPython import display;import pathlib;from tensorflow.keras import regularizers;import shutil;import tempfile;import tensorflow_docs as tfdocs;from matplotlib import pyplot as plt;import numpy as np;import tensorflow_docs.modeling;import tensorflow as tf', function_to_run='tf.keras.Sequential(*args)', method_object=None, object_signature=None, function_args=[eval("[\n    layers.Dense(512, activation='elu',\n                 kernel_regularizer=regularizers.l2(0.001),\n                 input_shape=(FEATURES,)),\n    layers.Dense(512, activation='elu',\n                 kernel_regularizer=regularizers.l2(0.001)),\n    layers.Dense(512, activation='elu',\n                 kernel_regularizer=regularizers.l2(0.001)),\n    layers.Dense(512, activation='elu',\n                 kernel_regularizer=regularizers.l2(0.001)),\n    layers.Dense(1)\n]")], function_kwargs={})
regularizer_histories['l2'] = compile_and_fit(l2_model, 'regularizers/l2')
custom_method(
plotter.plot(regularizer_histories), imports='import tensorflow_docs.plots;from tensorflow.keras import layers;from  IPython import display;import pathlib;from tensorflow.keras import regularizers;import shutil;import tempfile;import tensorflow_docs as tfdocs;from matplotlib import pyplot as plt;import numpy as np;import tensorflow_docs.modeling;import tensorflow as tf', function_to_run='obj.plot(*args)', method_object=eval('plotter'), object_signature='tfdocs.plots.HistoryPlotter', function_args=[eval('regularizer_histories')], function_kwargs={}, custom_class=None)
plt.ylim([0.5, 0.7])
result = custom_method(
l2_model(features), imports='import tensorflow_docs.plots;from tensorflow.keras import layers;from  IPython import display;import pathlib;from tensorflow.keras import regularizers;import shutil;import tempfile;import tensorflow_docs as tfdocs;from matplotlib import pyplot as plt;import numpy as np;import tensorflow_docs.modeling;import tensorflow as tf', function_to_run='obj(*args)', method_object=eval('l2_model'), object_signature='tf.keras.Sequential', function_args=[eval('features')], function_kwargs={}, custom_class=None)
regularization_loss = custom_method(
tf.add_n(l2_model.losses), imports='import tensorflow_docs.plots;from tensorflow.keras import layers;from  IPython import display;import pathlib;from tensorflow.keras import regularizers;import shutil;import tempfile;import tensorflow_docs as tfdocs;from matplotlib import pyplot as plt;import numpy as np;import tensorflow_docs.modeling;import tensorflow as tf', function_to_run='tf.add_n(*args)', method_object=None, object_signature=None, function_args=[eval('l2_model.losses')], function_kwargs={})
dropout_model = custom_method(
tf.keras.Sequential([layers.Dense(512, activation='elu', input_shape=(FEATURES,)), layers.Dropout(0.5), layers.Dense(512, activation='elu'), layers.Dropout(0.5), layers.Dense(512, activation='elu'), layers.Dropout(0.5), layers.Dense(512, activation='elu'), layers.Dropout(0.5), layers.Dense(1)]), imports='import tensorflow_docs.plots;from tensorflow.keras import layers;from  IPython import display;import pathlib;from tensorflow.keras import regularizers;import shutil;import tempfile;import tensorflow_docs as tfdocs;from matplotlib import pyplot as plt;import numpy as np;import tensorflow_docs.modeling;import tensorflow as tf', function_to_run='tf.keras.Sequential(*args)', method_object=None, object_signature=None, function_args=[eval("[\n    layers.Dense(512, activation='elu', input_shape=(FEATURES,)),\n    layers.Dropout(0.5),\n    layers.Dense(512, activation='elu'),\n    layers.Dropout(0.5),\n    layers.Dense(512, activation='elu'),\n    layers.Dropout(0.5),\n    layers.Dense(512, activation='elu'),\n    layers.Dropout(0.5),\n    layers.Dense(1)\n]")], function_kwargs={})
regularizer_histories['dropout'] = compile_and_fit(dropout_model, 'regularizers/dropout')
custom_method(
plotter.plot(regularizer_histories), imports='import tensorflow_docs.plots;from tensorflow.keras import layers;from  IPython import display;import pathlib;from tensorflow.keras import regularizers;import shutil;import tempfile;import tensorflow_docs as tfdocs;from matplotlib import pyplot as plt;import numpy as np;import tensorflow_docs.modeling;import tensorflow as tf', function_to_run='obj.plot(*args)', method_object=eval('plotter'), object_signature='tfdocs.plots.HistoryPlotter', function_args=[eval('regularizer_histories')], function_kwargs={}, custom_class=None)
plt.ylim([0.5, 0.7])
combined_model = custom_method(
tf.keras.Sequential([layers.Dense(512, kernel_regularizer=regularizers.l2(0.0001), activation='elu', input_shape=(FEATURES,)), layers.Dropout(0.5), layers.Dense(512, kernel_regularizer=regularizers.l2(0.0001), activation='elu'), layers.Dropout(0.5), layers.Dense(512, kernel_regularizer=regularizers.l2(0.0001), activation='elu'), layers.Dropout(0.5), layers.Dense(512, kernel_regularizer=regularizers.l2(0.0001), activation='elu'), layers.Dropout(0.5), layers.Dense(1)]), imports='import tensorflow_docs.plots;from tensorflow.keras import layers;from  IPython import display;import pathlib;from tensorflow.keras import regularizers;import shutil;import tempfile;import tensorflow_docs as tfdocs;from matplotlib import pyplot as plt;import numpy as np;import tensorflow_docs.modeling;import tensorflow as tf', function_to_run='tf.keras.Sequential(*args)', method_object=None, object_signature=None, function_args=[eval("[\n    layers.Dense(512, kernel_regularizer=regularizers.l2(0.0001),\n                 activation='elu', input_shape=(FEATURES,)),\n    layers.Dropout(0.5),\n    layers.Dense(512, kernel_regularizer=regularizers.l2(0.0001),\n                 activation='elu'),\n    layers.Dropout(0.5),\n    layers.Dense(512, kernel_regularizer=regularizers.l2(0.0001),\n                 activation='elu'),\n    layers.Dropout(0.5),\n    layers.Dense(512, kernel_regularizer=regularizers.l2(0.0001),\n                 activation='elu'),\n    layers.Dropout(0.5),\n    layers.Dense(1)\n]")], function_kwargs={})
regularizer_histories['combined'] = compile_and_fit(combined_model, 'regularizers/combined')
custom_method(
plotter.plot(regularizer_histories), imports='import tensorflow_docs.plots;from tensorflow.keras import layers;from  IPython import display;import pathlib;from tensorflow.keras import regularizers;import shutil;import tempfile;import tensorflow_docs as tfdocs;from matplotlib import pyplot as plt;import numpy as np;import tensorflow_docs.modeling;import tensorflow as tf', function_to_run='obj.plot(*args)', method_object=eval('plotter'), object_signature='tfdocs.plots.HistoryPlotter', function_args=[eval('regularizer_histories')], function_kwargs={}, custom_class=None)
plt.ylim([0.5, 0.7])
