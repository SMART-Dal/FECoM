import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import regularizers
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
custom_method(imports='from  IPython import display;import tensorflow_docs.plots;import tempfile;from tensorflow.keras import layers;from tensorflow.keras import regularizers;import tensorflow as tf;from matplotlib import pyplot as plt;import shutil;import numpy as np;import pathlib;import tensorflow_docs as tfdocs;import tensorflow_docs.modeling', function_to_run='tf.keras.utils.get_file(*args)', method_object=None, object_signature=None, function_args=[eval("'HIGGS.csv.gz'"), eval("'http://mlphysics.ics.uci.edu/data/higgs/HIGGS.csv.gz'")], function_kwargs={})
gz = tf.keras.utils.get_file('HIGGS.csv.gz', 'http://mlphysics.ics.uci.edu/data/higgs/HIGGS.csv.gz')
FEATURES = 28
custom_method(imports='from  IPython import display;import tensorflow_docs.plots;import tempfile;from tensorflow.keras import layers;from tensorflow.keras import regularizers;import tensorflow as tf;from matplotlib import pyplot as plt;import shutil;import numpy as np;import pathlib;import tensorflow_docs as tfdocs;import tensorflow_docs.modeling', function_to_run='tf.data.experimental.CsvDataset(*args, **kwargs)', method_object=None, object_signature=None, function_args=[eval('gz'), eval('[float(),]*(FEATURES+1)')], function_kwargs={'compression_type': eval('"GZIP"')})
ds = tf.data.experimental.CsvDataset(gz, [float()] * (FEATURES + 1), compression_type='GZIP')

def pack_row(*row):
    label = row[0]
    custom_method(imports='from  IPython import display;import tensorflow_docs.plots;import tempfile;from tensorflow.keras import layers;from tensorflow.keras import regularizers;import tensorflow as tf;from matplotlib import pyplot as plt;import shutil;import numpy as np;import pathlib;import tensorflow_docs as tfdocs;import tensorflow_docs.modeling', function_to_run='tf.stack(*args)', method_object=None, object_signature=None, function_args=[eval('row[1:]'), eval('1')], function_kwargs={})
    features = tf.stack(row[1:], 1)
    return (features, label)
custom_method(imports='from  IPython import display;import tensorflow_docs.plots;import tempfile;from tensorflow.keras import layers;from tensorflow.keras import regularizers;import tensorflow as tf;from matplotlib import pyplot as plt;import shutil;import numpy as np;import pathlib;import tensorflow_docs as tfdocs;import tensorflow_docs.modeling', function_to_run='obj.batch(10000).map(pack_row).unbatch()', method_object=eval('ds'), object_signature=None, function_args=[], function_kwargs={}, custom_class=None)
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
custom_method(imports='from  IPython import display;import tensorflow_docs.plots;import tempfile;from tensorflow.keras import layers;from tensorflow.keras import regularizers;import tensorflow as tf;from matplotlib import pyplot as plt;import shutil;import numpy as np;import pathlib;import tensorflow_docs as tfdocs;import tensorflow_docs.modeling', function_to_run='tf.keras.optimizers.schedules.InverseTimeDecay(*args, **kwargs)', method_object=None, object_signature=None, function_args=[eval('0.001')], function_kwargs={'decay_steps': eval('STEPS_PER_EPOCH*1000'), 'decay_rate': eval('1'), 'staircase': eval('False')})
lr_schedule = tf.keras.optimizers.schedules.InverseTimeDecay(0.001, decay_steps=STEPS_PER_EPOCH * 1000, decay_rate=1, staircase=False)

def get_optimizer():
    return tf.keras.optimizers.Adam(lr_schedule)
step = np.linspace(0, 100000)
custom_method(imports='from  IPython import display;import tensorflow_docs.plots;import tempfile;from tensorflow.keras import layers;from tensorflow.keras import regularizers;import tensorflow as tf;from matplotlib import pyplot as plt;import shutil;import numpy as np;import pathlib;import tensorflow_docs as tfdocs;import tensorflow_docs.modeling', function_to_run='obj(*args)', method_object=eval('lr_schedule'), object_signature=None, function_args=[eval('step')], function_kwargs={}, custom_class=None)
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
custom_method(imports='from  IPython import display;import tensorflow_docs.plots;import tempfile;from tensorflow.keras import layers;from tensorflow.keras import regularizers;import tensorflow as tf;from matplotlib import pyplot as plt;import shutil;import numpy as np;import pathlib;import tensorflow_docs as tfdocs;import tensorflow_docs.modeling', function_to_run='tf.keras.Sequential(*args)', method_object=None, object_signature=None, function_args=[eval("[\n    layers.Dense(16, activation='elu', input_shape=(FEATURES,)),\n    layers.Dense(1)\n]")], function_kwargs={})
tiny_model = tf.keras.Sequential([layers.Dense(16, activation='elu', input_shape=(FEATURES,)), layers.Dense(1)])
size_histories = {}
size_histories['Tiny'] = compile_and_fit(tiny_model, 'sizes/Tiny')
plotter = tfdocs.plots.HistoryPlotter(metric='binary_crossentropy', smoothing_std=10)
plotter.plot(size_histories)
plt.ylim([0.5, 0.7])
custom_method(imports='from  IPython import display;import tensorflow_docs.plots;import tempfile;from tensorflow.keras import layers;from tensorflow.keras import regularizers;import tensorflow as tf;from matplotlib import pyplot as plt;import shutil;import numpy as np;import pathlib;import tensorflow_docs as tfdocs;import tensorflow_docs.modeling', function_to_run='tf.keras.Sequential(*args)', method_object=None, object_signature=None, function_args=[eval("[\n    layers.Dense(16, activation='elu', input_shape=(FEATURES,)),\n    layers.Dense(16, activation='elu'),\n    layers.Dense(1)\n]")], function_kwargs={})
small_model = tf.keras.Sequential([layers.Dense(16, activation='elu', input_shape=(FEATURES,)), layers.Dense(16, activation='elu'), layers.Dense(1)])
size_histories['Small'] = compile_and_fit(small_model, 'sizes/Small')
custom_method(imports='from  IPython import display;import tensorflow_docs.plots;import tempfile;from tensorflow.keras import layers;from tensorflow.keras import regularizers;import tensorflow as tf;from matplotlib import pyplot as plt;import shutil;import numpy as np;import pathlib;import tensorflow_docs as tfdocs;import tensorflow_docs.modeling', function_to_run='tf.keras.Sequential(*args)', method_object=None, object_signature=None, function_args=[eval("[\n    layers.Dense(64, activation='elu', input_shape=(FEATURES,)),\n    layers.Dense(64, activation='elu'),\n    layers.Dense(64, activation='elu'),\n    layers.Dense(1)\n]")], function_kwargs={})
medium_model = tf.keras.Sequential([layers.Dense(64, activation='elu', input_shape=(FEATURES,)), layers.Dense(64, activation='elu'), layers.Dense(64, activation='elu'), layers.Dense(1)])
size_histories['Medium'] = compile_and_fit(medium_model, 'sizes/Medium')
custom_method(imports='from  IPython import display;import tensorflow_docs.plots;import tempfile;from tensorflow.keras import layers;from tensorflow.keras import regularizers;import tensorflow as tf;from matplotlib import pyplot as plt;import shutil;import numpy as np;import pathlib;import tensorflow_docs as tfdocs;import tensorflow_docs.modeling', function_to_run='tf.keras.Sequential(*args)', method_object=None, object_signature=None, function_args=[eval("[\n    layers.Dense(512, activation='elu', input_shape=(FEATURES,)),\n    layers.Dense(512, activation='elu'),\n    layers.Dense(512, activation='elu'),\n    layers.Dense(512, activation='elu'),\n    layers.Dense(1)\n]")], function_kwargs={})
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
custom_method(imports='from  IPython import display;import tensorflow_docs.plots;import tempfile;from tensorflow.keras import layers;from tensorflow.keras import regularizers;import tensorflow as tf;from matplotlib import pyplot as plt;import shutil;import numpy as np;import pathlib;import tensorflow_docs as tfdocs;import tensorflow_docs.modeling', function_to_run='tf.keras.Sequential(*args)', method_object=None, object_signature=None, function_args=[eval("[\n    layers.Dense(512, activation='elu',\n                 kernel_regularizer=regularizers.l2(0.001),\n                 input_shape=(FEATURES,)),\n    layers.Dense(512, activation='elu',\n                 kernel_regularizer=regularizers.l2(0.001)),\n    layers.Dense(512, activation='elu',\n                 kernel_regularizer=regularizers.l2(0.001)),\n    layers.Dense(512, activation='elu',\n                 kernel_regularizer=regularizers.l2(0.001)),\n    layers.Dense(1)\n]")], function_kwargs={})
l2_model = tf.keras.Sequential([layers.Dense(512, activation='elu', kernel_regularizer=regularizers.l2(0.001), input_shape=(FEATURES,)), layers.Dense(512, activation='elu', kernel_regularizer=regularizers.l2(0.001)), layers.Dense(512, activation='elu', kernel_regularizer=regularizers.l2(0.001)), layers.Dense(512, activation='elu', kernel_regularizer=regularizers.l2(0.001)), layers.Dense(1)])
regularizer_histories['l2'] = compile_and_fit(l2_model, 'regularizers/l2')
plotter.plot(regularizer_histories)
plt.ylim([0.5, 0.7])
custom_method(imports='from  IPython import display;import tensorflow_docs.plots;import tempfile;from tensorflow.keras import layers;from tensorflow.keras import regularizers;import tensorflow as tf;from matplotlib import pyplot as plt;import shutil;import numpy as np;import pathlib;import tensorflow_docs as tfdocs;import tensorflow_docs.modeling', function_to_run='obj(*args)', method_object=eval('l2_model'), object_signature=None, function_args=[eval('features')], function_kwargs={}, custom_class=None)
result = l2_model(features)
custom_method(imports='from  IPython import display;import tensorflow_docs.plots;import tempfile;from tensorflow.keras import layers;from tensorflow.keras import regularizers;import tensorflow as tf;from matplotlib import pyplot as plt;import shutil;import numpy as np;import pathlib;import tensorflow_docs as tfdocs;import tensorflow_docs.modeling', function_to_run='tf.add_n(*args)', method_object=None, object_signature=None, function_args=[eval('l2_model.losses')], function_kwargs={})
regularization_loss = tf.add_n(l2_model.losses)
custom_method(imports='from  IPython import display;import tensorflow_docs.plots;import tempfile;from tensorflow.keras import layers;from tensorflow.keras import regularizers;import tensorflow as tf;from matplotlib import pyplot as plt;import shutil;import numpy as np;import pathlib;import tensorflow_docs as tfdocs;import tensorflow_docs.modeling', function_to_run='tf.keras.Sequential(*args)', method_object=None, object_signature=None, function_args=[eval("[\n    layers.Dense(512, activation='elu', input_shape=(FEATURES,)),\n    layers.Dropout(0.5),\n    layers.Dense(512, activation='elu'),\n    layers.Dropout(0.5),\n    layers.Dense(512, activation='elu'),\n    layers.Dropout(0.5),\n    layers.Dense(512, activation='elu'),\n    layers.Dropout(0.5),\n    layers.Dense(1)\n]")], function_kwargs={})
dropout_model = tf.keras.Sequential([layers.Dense(512, activation='elu', input_shape=(FEATURES,)), layers.Dropout(0.5), layers.Dense(512, activation='elu'), layers.Dropout(0.5), layers.Dense(512, activation='elu'), layers.Dropout(0.5), layers.Dense(512, activation='elu'), layers.Dropout(0.5), layers.Dense(1)])
regularizer_histories['dropout'] = compile_and_fit(dropout_model, 'regularizers/dropout')
plotter.plot(regularizer_histories)
plt.ylim([0.5, 0.7])
custom_method(imports='from  IPython import display;import tensorflow_docs.plots;import tempfile;from tensorflow.keras import layers;from tensorflow.keras import regularizers;import tensorflow as tf;from matplotlib import pyplot as plt;import shutil;import numpy as np;import pathlib;import tensorflow_docs as tfdocs;import tensorflow_docs.modeling', function_to_run='tf.keras.Sequential(*args)', method_object=None, object_signature=None, function_args=[eval("[\n    layers.Dense(512, kernel_regularizer=regularizers.l2(0.0001),\n                 activation='elu', input_shape=(FEATURES,)),\n    layers.Dropout(0.5),\n    layers.Dense(512, kernel_regularizer=regularizers.l2(0.0001),\n                 activation='elu'),\n    layers.Dropout(0.5),\n    layers.Dense(512, kernel_regularizer=regularizers.l2(0.0001),\n                 activation='elu'),\n    layers.Dropout(0.5),\n    layers.Dense(512, kernel_regularizer=regularizers.l2(0.0001),\n                 activation='elu'),\n    layers.Dropout(0.5),\n    layers.Dense(1)\n]")], function_kwargs={})
combined_model = tf.keras.Sequential([layers.Dense(512, kernel_regularizer=regularizers.l2(0.0001), activation='elu', input_shape=(FEATURES,)), layers.Dropout(0.5), layers.Dense(512, kernel_regularizer=regularizers.l2(0.0001), activation='elu'), layers.Dropout(0.5), layers.Dense(512, kernel_regularizer=regularizers.l2(0.0001), activation='elu'), layers.Dropout(0.5), layers.Dense(512, kernel_regularizer=regularizers.l2(0.0001), activation='elu'), layers.Dropout(0.5), layers.Dense(1)])
regularizer_histories['combined'] = compile_and_fit(combined_model, 'regularizers/combined')
plotter.plot(regularizer_histories)
plt.ylim([0.5, 0.7])
