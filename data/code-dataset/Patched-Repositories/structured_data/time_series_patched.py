import os
import datetime
import IPython
import IPython.display
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
import os
from pathlib import Path
import dill as pickle
from tool.client.client_config import EXPERIMENT_DIR, EXPERIMENT_TAG, MAX_WAIT_S, WAIT_AFTER_RUN_S
from tool.server.send_request import send_request
from tool.server.function_details import FunctionDetails
current_path = os.path.abspath(__file__)
(immediate_folder, file_name) = os.path.split(current_path)
immediate_folder = os.path.basename(immediate_folder)
experiment_file_name = os.path.splitext(file_name)[0]
EXPERIMENT_FILE_PATH = EXPERIMENT_DIR / immediate_folder / EXPERIMENT_TAG / (experiment_file_name + '-energy.json')

def custom_method(func, imports: str, function_to_run: str, method_object=None, function_args: list=None, function_kwargs: dict=None, max_wait_secs=MAX_WAIT_S, custom_class=None, wait_after_run_secs=WAIT_AFTER_RUN_S):
    result = send_request(imports=imports, function_to_run=function_to_run, function_args=function_args, function_kwargs=function_kwargs, max_wait_secs=max_wait_secs, method_object=method_object, custom_class=custom_class, experiment_file_path=EXPERIMENT_FILE_PATH)
    return func
if __name__ == '__main__':
    print(EXPERIMENT_FILE_PATH)
mpl.rcParams['figure.figsize'] = (8, 6)
mpl.rcParams['axes.grid'] = False
zip_path = custom_method(
tf.keras.utils.get_file(origin='https://storage.googleapis.com/tensorflow/tf-keras-datasets/jena_climate_2009_2016.csv.zip', fname='jena_climate_2009_2016.csv.zip', extract=True), imports='import os;import IPython;import pandas as pd;import IPython.display;import matplotlib.pyplot as plt;import datetime;import matplotlib as mpl;import numpy as np;import tensorflow as tf;import seaborn as sns', function_to_run='tf.keras.utils.get_file(**kwargs)', method_object=None, function_args=[], function_kwargs={'origin': eval("'https://storage.googleapis.com/tensorflow/tf-keras-datasets/jena_climate_2009_2016.csv.zip'"), 'fname': eval("'jena_climate_2009_2016.csv.zip'"), 'extract': eval('True')}, max_wait_secs=0)
(csv_path, _) = os.path.splitext(zip_path)
df = pd.read_csv(csv_path)
df = df[5::6]
date_time = pd.to_datetime(df.pop('Date Time'), format='%d.%m.%Y %H:%M:%S')
df.head()
plot_cols = ['T (degC)', 'p (mbar)', 'rho (g/m**3)']
plot_features = df[plot_cols]
plot_features.index = date_time
_ = plot_features.plot(subplots=True)
plot_features = df[plot_cols][:480]
plot_features.index = date_time[:480]
_ = plot_features.plot(subplots=True)
df.describe().transpose()
wv = df['wv (m/s)']
bad_wv = wv == -9999.0
wv[bad_wv] = 0.0
max_wv = df['max. wv (m/s)']
bad_max_wv = max_wv == -9999.0
max_wv[bad_max_wv] = 0.0
df['wv (m/s)'].min()
plt.hist2d(df['wd (deg)'], df['wv (m/s)'], bins=(50, 50), vmax=400)
plt.colorbar()
plt.xlabel('Wind Direction [deg]')
plt.ylabel('Wind Velocity [m/s]')
wv = df.pop('wv (m/s)')
max_wv = df.pop('max. wv (m/s)')
wd_rad = df.pop('wd (deg)') * np.pi / 180
df['Wx'] = wv * np.cos(wd_rad)
df['Wy'] = wv * np.sin(wd_rad)
df['max Wx'] = max_wv * np.cos(wd_rad)
df['max Wy'] = max_wv * np.sin(wd_rad)
plt.hist2d(df['Wx'], df['Wy'], bins=(50, 50), vmax=400)
plt.colorbar()
plt.xlabel('Wind X [m/s]')
plt.ylabel('Wind Y [m/s]')
ax = plt.gca()
ax.axis('tight')
timestamp_s = date_time.map(pd.Timestamp.timestamp)
day = 24 * 60 * 60
year = 365.2425 * day
df['Day sin'] = np.sin(timestamp_s * (2 * np.pi / day))
df['Day cos'] = np.cos(timestamp_s * (2 * np.pi / day))
df['Year sin'] = np.sin(timestamp_s * (2 * np.pi / year))
df['Year cos'] = np.cos(timestamp_s * (2 * np.pi / year))
plt.plot(np.array(df['Day sin'])[:25])
plt.plot(np.array(df['Day cos'])[:25])
plt.xlabel('Time [h]')
plt.title('Time of day signal')
fft = custom_method(
tf.signal.rfft(df['T (degC)']), imports='import os;import IPython;import pandas as pd;import IPython.display;import matplotlib.pyplot as plt;import datetime;import matplotlib as mpl;import numpy as np;import tensorflow as tf;import seaborn as sns', function_to_run='tf.signal.rfft(*args)', method_object=None, function_args=[eval("df['T (degC)']")], function_kwargs={}, max_wait_secs=0)
f_per_dataset = np.arange(0, len(fft))
n_samples_h = len(df['T (degC)'])
hours_per_year = 24 * 365.2524
years_per_dataset = n_samples_h / hours_per_year
f_per_year = f_per_dataset / years_per_dataset
plt.step(f_per_year, np.abs(fft))
plt.xscale('log')
plt.ylim(0, 400000)
plt.xlim([0.1, max(plt.xlim())])
plt.xticks([1, 365.2524], labels=['1/Year', '1/day'])
_ = plt.xlabel('Frequency (log scale)')
column_indices = {name: i for (i, name) in enumerate(df.columns)}
n = len(df)
train_df = df[0:int(n * 0.7)]
val_df = df[int(n * 0.7):int(n * 0.9)]
test_df = df[int(n * 0.9):]
num_features = df.shape[1]
train_mean = train_df.mean()
train_std = train_df.std()
train_df = (train_df - train_mean) / train_std
val_df = (val_df - train_mean) / train_std
test_df = (test_df - train_mean) / train_std
df_std = (df - train_mean) / train_std
df_std = df_std.melt(var_name='Column', value_name='Normalized')
plt.figure(figsize=(12, 6))
ax = sns.violinplot(x='Column', y='Normalized', data=df_std)
_ = ax.set_xticklabels(df.keys(), rotation=90)

class WindowGenerator:

    def __init__(self, input_width, label_width, shift, train_df=train_df, val_df=val_df, test_df=test_df, label_columns=None):
        self.train_df = train_df
        self.val_df = val_df
        self.test_df = test_df
        self.label_columns = label_columns
        if label_columns is not None:
            self.label_columns_indices = {name: i for (i, name) in enumerate(label_columns)}
        self.column_indices = {name: i for (i, name) in enumerate(train_df.columns)}
        self.input_width = input_width
        self.label_width = label_width
        self.shift = shift
        self.total_window_size = input_width + shift
        self.input_slice = slice(0, input_width)
        self.input_indices = np.arange(self.total_window_size)[self.input_slice]
        self.label_start = self.total_window_size - self.label_width
        self.labels_slice = slice(self.label_start, None)
        self.label_indices = np.arange(self.total_window_size)[self.labels_slice]

    def __repr__(self):
        return '\n'.join([f'Total window size: {self.total_window_size}', f'Input indices: {self.input_indices}', f'Label indices: {self.label_indices}', f'Label column name(s): {self.label_columns}'])
w1 = WindowGenerator(input_width=24, label_width=1, shift=24, label_columns=['T (degC)'])
w1
w2 = WindowGenerator(input_width=6, label_width=1, shift=1, label_columns=['T (degC)'])
w2

def split_window(self, features):
    inputs = features[:, self.input_slice, :]
    labels = features[:, self.labels_slice, :]
    if self.label_columns is not None:
        labels = custom_method(
        tf.stack([labels[:, :, self.column_indices[name]] for name in self.label_columns], axis=-1), imports='import os;import IPython;import pandas as pd;import IPython.display;import matplotlib.pyplot as plt;import datetime;import matplotlib as mpl;import numpy as np;import tensorflow as tf;import seaborn as sns', function_to_run='tf.stack(*args, **kwargs)', method_object=None, function_args=[eval('[labels[:, :, self.column_indices[name]] for name in self.label_columns]')], function_kwargs={'axis': eval('-1')}, max_wait_secs=0)
    inputs.set_shape([None, self.input_width, None])
    custom_method(
    labels.set_shape([None, self.label_width, None]), imports='import os;import IPython;import pandas as pd;import IPython.display;import matplotlib.pyplot as plt;import datetime;import matplotlib as mpl;import numpy as np;import tensorflow as tf;import seaborn as sns', function_to_run='obj.set_shape(*args)', method_object=eval('labels'), function_args=[eval('[None, self.label_width, None]')], function_kwargs={}, max_wait_secs=0, custom_class=None)
    return (inputs, labels)
WindowGenerator.split_window = split_window
example_window = custom_method(
tf.stack([np.array(train_df[:w2.total_window_size]), np.array(train_df[100:100 + w2.total_window_size]), np.array(train_df[200:200 + w2.total_window_size])]), imports='import os;import IPython;import pandas as pd;import IPython.display;import matplotlib.pyplot as plt;import datetime;import matplotlib as mpl;import numpy as np;import tensorflow as tf;import seaborn as sns', function_to_run='tf.stack(*args)', method_object=None, function_args=[eval('[np.array(train_df[:w2.total_window_size]),\n                           np.array(train_df[100:100+w2.total_window_size]),\n                           np.array(train_df[200:200+w2.total_window_size])]')], function_kwargs={}, max_wait_secs=0)
(example_inputs, example_labels) = w2.split_window(example_window)
print('All shapes are: (batch, time, features)')
print(f'Window shape: {example_window.shape}')
print(f'Inputs shape: {example_inputs.shape}')
print(f'Labels shape: {example_labels.shape}')
w2.example = (example_inputs, example_labels)

def plot(self, model=None, plot_col='T (degC)', max_subplots=3):
    (inputs, labels) = self.example
    plt.figure(figsize=(12, 8))
    plot_col_index = self.column_indices[plot_col]
    max_n = min(max_subplots, len(inputs))
    for n in range(max_n):
        plt.subplot(max_n, 1, n + 1)
        plt.ylabel(f'{plot_col} [normed]')
        plt.plot(self.input_indices, inputs[n, :, plot_col_index], label='Inputs', marker='.', zorder=-10)
        if self.label_columns:
            label_col_index = self.label_columns_indices.get(plot_col, None)
        else:
            label_col_index = plot_col_index
        if label_col_index is None:
            continue
        plt.scatter(self.label_indices, labels[n, :, label_col_index], edgecolors='k', label='Labels', c='#2ca02c', s=64)
        if model is not None:
            predictions = model(inputs)
            plt.scatter(self.label_indices, predictions[n, :, label_col_index], marker='X', edgecolors='k', label='Predictions', c='#ff7f0e', s=64)
        if n == 0:
            plt.legend()
    plt.xlabel('Time [h]')
WindowGenerator.plot = plot
w2.plot()
w2.plot(plot_col='p (mbar)')

def make_dataset(self, data):
    data = np.array(data, dtype=np.float32)
    ds = custom_method(
    tf.keras.utils.timeseries_dataset_from_array(data=data, targets=None, sequence_length=self.total_window_size, sequence_stride=1, shuffle=True, batch_size=32), imports='import os;import IPython;import pandas as pd;import IPython.display;import matplotlib.pyplot as plt;import datetime;import matplotlib as mpl;import numpy as np;import tensorflow as tf;import seaborn as sns', function_to_run='tf.keras.utils.timeseries_dataset_from_array(**kwargs)', method_object=None, function_args=[], function_kwargs={'data': eval('data'), 'targets': eval('None'), 'sequence_length': eval('self.total_window_size'), 'sequence_stride': eval('1'), 'shuffle': eval('True'), 'batch_size': eval('32')}, max_wait_secs=0)
    ds = custom_method(
    ds.map(self.split_window), imports='import os;import IPython;import pandas as pd;import IPython.display;import matplotlib.pyplot as plt;import datetime;import matplotlib as mpl;import numpy as np;import tensorflow as tf;import seaborn as sns', function_to_run='obj.map(*args)', method_object=eval('ds'), function_args=[eval('self.split_window')], function_kwargs={}, max_wait_secs=0, custom_class=None)
    return ds
WindowGenerator.make_dataset = make_dataset

@property
def train(self):
    return self.make_dataset(self.train_df)

@property
def val(self):
    return self.make_dataset(self.val_df)

@property
def test(self):
    return self.make_dataset(self.test_df)

@property
def example(self):
    """Get and cache an example batch of `inputs, labels` for plotting."""
    result = getattr(self, '_example', None)
    if result is None:
        result = next(iter(self.train))
        self._example = result
    return result
WindowGenerator.train = train
WindowGenerator.val = val
WindowGenerator.test = test
WindowGenerator.example = example
w2.train.element_spec
for (example_inputs, example_labels) in w2.train.take(1):
    print(f'Inputs shape (batch, time, features): {example_inputs.shape}')
    print(f'Labels shape (batch, time, features): {example_labels.shape}')
single_step_window = WindowGenerator(input_width=1, label_width=1, shift=1, label_columns=['T (degC)'])
single_step_window
for (example_inputs, example_labels) in single_step_window.train.take(1):
    print(f'Inputs shape (batch, time, features): {example_inputs.shape}')
    print(f'Labels shape (batch, time, features): {example_labels.shape}')

class Baseline(tf.keras.Model):

    def __init__(self, label_index=None):
        super().__init__()
        self.label_index = label_index

    def call(self, inputs):
        if self.label_index is None:
            return inputs
        result = inputs[:, :, self.label_index]
        return result[:, :, tf.newaxis]
baseline = Baseline(label_index=column_indices['T (degC)'])
custom_method(
baseline.compile(loss=tf.keras.losses.MeanSquaredError(), metrics=[tf.keras.metrics.MeanAbsoluteError()]), imports='import os;import IPython;import pandas as pd;import IPython.display;import matplotlib.pyplot as plt;import datetime;import matplotlib as mpl;import numpy as np;import tensorflow as tf;import seaborn as sns', function_to_run='obj.compile(**kwargs)', method_object=eval('baseline'), function_args=[], function_kwargs={'loss': eval('tf.keras.losses.MeanSquaredError()'), 'metrics': eval('[tf.keras.metrics.MeanAbsoluteError()]')}, max_wait_secs=0, custom_class='class Baseline(tf.keras.Model):\n  def __init__(self, label_index=None):\n    super().__init__()\n    self.label_index = label_index\n\n  def call(self, inputs):\n    if self.label_index is None:\n      return inputs\n    result = inputs[:, :, self.label_index]\n    return result[:, :, tf.newaxis]')
val_performance = {}
performance = {}
val_performance['Baseline'] = custom_method(
baseline.evaluate(single_step_window.val), imports='import os;import IPython;import pandas as pd;import IPython.display;import matplotlib.pyplot as plt;import datetime;import matplotlib as mpl;import numpy as np;import tensorflow as tf;import seaborn as sns', function_to_run='obj.evaluate(*args)', method_object=eval('baseline'), function_args=[eval('single_step_window.val')], function_kwargs={}, max_wait_secs=0, custom_class='class Baseline(tf.keras.Model):\n  def __init__(self, label_index=None):\n    super().__init__()\n    self.label_index = label_index\n\n  def call(self, inputs):\n    if self.label_index is None:\n      return inputs\n    result = inputs[:, :, self.label_index]\n    return result[:, :, tf.newaxis]')
performance['Baseline'] = custom_method(
baseline.evaluate(single_step_window.test, verbose=0), imports='import os;import IPython;import pandas as pd;import IPython.display;import matplotlib.pyplot as plt;import datetime;import matplotlib as mpl;import numpy as np;import tensorflow as tf;import seaborn as sns', function_to_run='obj.evaluate(*args, **kwargs)', method_object=eval('baseline'), function_args=[eval('single_step_window.test')], function_kwargs={'verbose': eval('0')}, max_wait_secs=0, custom_class='class Baseline(tf.keras.Model):\n  def __init__(self, label_index=None):\n    super().__init__()\n    self.label_index = label_index\n\n  def call(self, inputs):\n    if self.label_index is None:\n      return inputs\n    result = inputs[:, :, self.label_index]\n    return result[:, :, tf.newaxis]')
wide_window = WindowGenerator(input_width=24, label_width=24, shift=1, label_columns=['T (degC)'])
wide_window
print('Input shape:', wide_window.example[0].shape)
print('Output shape:', baseline(wide_window.example[0]).shape)
wide_window.plot(baseline)
linear = custom_method(
tf.keras.Sequential([tf.keras.layers.Dense(units=1)]), imports='import os;import IPython;import pandas as pd;import IPython.display;import matplotlib.pyplot as plt;import datetime;import matplotlib as mpl;import numpy as np;import tensorflow as tf;import seaborn as sns', function_to_run='tf.keras.Sequential(*args)', method_object=None, function_args=[eval('[\n    tf.keras.layers.Dense(units=1)\n]')], function_kwargs={}, max_wait_secs=0)
print('Input shape:', single_step_window.example[0].shape)
print('Output shape:', linear(single_step_window.example[0]).shape)
MAX_EPOCHS = 20

def compile_and_fit(model, window, patience=2):
    early_stopping = custom_method(
    tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=patience, mode='min'), imports='import os;import IPython;import pandas as pd;import IPython.display;import matplotlib.pyplot as plt;import datetime;import matplotlib as mpl;import numpy as np;import tensorflow as tf;import seaborn as sns', function_to_run='tf.keras.callbacks.EarlyStopping(**kwargs)', method_object=None, function_args=[], function_kwargs={'monitor': eval("'val_loss'"), 'patience': eval('patience'), 'mode': eval("'min'")}, max_wait_secs=0)
    model.compile(loss=tf.keras.losses.MeanSquaredError(), optimizer=tf.keras.optimizers.Adam(), metrics=[tf.keras.metrics.MeanAbsoluteError()])
    history = model.fit(window.train, epochs=MAX_EPOCHS, validation_data=window.val, callbacks=[early_stopping])
    return history
history = compile_and_fit(linear, single_step_window)
val_performance['Linear'] = custom_method(
linear.evaluate(single_step_window.val), imports='import os;import IPython;import pandas as pd;import IPython.display;import matplotlib.pyplot as plt;import datetime;import matplotlib as mpl;import numpy as np;import tensorflow as tf;import seaborn as sns', function_to_run='obj.evaluate(*args)', method_object=eval('linear'), function_args=[eval('single_step_window.val')], function_kwargs={}, max_wait_secs=0, custom_class=None)
performance['Linear'] = custom_method(
linear.evaluate(single_step_window.test, verbose=0), imports='import os;import IPython;import pandas as pd;import IPython.display;import matplotlib.pyplot as plt;import datetime;import matplotlib as mpl;import numpy as np;import tensorflow as tf;import seaborn as sns', function_to_run='obj.evaluate(*args, **kwargs)', method_object=eval('linear'), function_args=[eval('single_step_window.test')], function_kwargs={'verbose': eval('0')}, max_wait_secs=0, custom_class=None)
print('Input shape:', wide_window.example[0].shape)
print('Output shape:', baseline(wide_window.example[0]).shape)
wide_window.plot(linear)
plt.bar(x=range(len(train_df.columns)), height=linear.layers[0].kernel[:, 0].numpy())
axis = plt.gca()
axis.set_xticks(range(len(train_df.columns)))
_ = axis.set_xticklabels(train_df.columns, rotation=90)
dense = custom_method(
tf.keras.Sequential([tf.keras.layers.Dense(units=64, activation='relu'), tf.keras.layers.Dense(units=64, activation='relu'), tf.keras.layers.Dense(units=1)]), imports='import os;import IPython;import pandas as pd;import IPython.display;import matplotlib.pyplot as plt;import datetime;import matplotlib as mpl;import numpy as np;import tensorflow as tf;import seaborn as sns', function_to_run='tf.keras.Sequential(*args)', method_object=None, function_args=[eval("[\n    tf.keras.layers.Dense(units=64, activation='relu'),\n    tf.keras.layers.Dense(units=64, activation='relu'),\n    tf.keras.layers.Dense(units=1)\n]")], function_kwargs={}, max_wait_secs=0)
history = compile_and_fit(dense, single_step_window)
val_performance['Dense'] = custom_method(
dense.evaluate(single_step_window.val), imports='import os;import IPython;import pandas as pd;import IPython.display;import matplotlib.pyplot as plt;import datetime;import matplotlib as mpl;import numpy as np;import tensorflow as tf;import seaborn as sns', function_to_run='obj.evaluate(*args)', method_object=eval('dense'), function_args=[eval('single_step_window.val')], function_kwargs={}, max_wait_secs=0, custom_class=None)
performance['Dense'] = custom_method(
dense.evaluate(single_step_window.test, verbose=0), imports='import os;import IPython;import pandas as pd;import IPython.display;import matplotlib.pyplot as plt;import datetime;import matplotlib as mpl;import numpy as np;import tensorflow as tf;import seaborn as sns', function_to_run='obj.evaluate(*args, **kwargs)', method_object=eval('dense'), function_args=[eval('single_step_window.test')], function_kwargs={'verbose': eval('0')}, max_wait_secs=0, custom_class=None)
CONV_WIDTH = 3
conv_window = WindowGenerator(input_width=CONV_WIDTH, label_width=1, shift=1, label_columns=['T (degC)'])
conv_window
conv_window.plot()
plt.title('Given 3 hours of inputs, predict 1 hour into the future.')
multi_step_dense = custom_method(
tf.keras.Sequential([tf.keras.layers.Flatten(), tf.keras.layers.Dense(units=32, activation='relu'), tf.keras.layers.Dense(units=32, activation='relu'), tf.keras.layers.Dense(units=1), tf.keras.layers.Reshape([1, -1])]), imports='import os;import IPython;import pandas as pd;import IPython.display;import matplotlib.pyplot as plt;import datetime;import matplotlib as mpl;import numpy as np;import tensorflow as tf;import seaborn as sns', function_to_run='tf.keras.Sequential(*args)', method_object=None, function_args=[eval("[\n    tf.keras.layers.Flatten(),\n    tf.keras.layers.Dense(units=32, activation='relu'),\n    tf.keras.layers.Dense(units=32, activation='relu'),\n    tf.keras.layers.Dense(units=1),\n    tf.keras.layers.Reshape([1, -1]),\n]")], function_kwargs={}, max_wait_secs=0)
print('Input shape:', conv_window.example[0].shape)
print('Output shape:', multi_step_dense(conv_window.example[0]).shape)
history = compile_and_fit(multi_step_dense, conv_window)
IPython.display.clear_output()
val_performance['Multi step dense'] = custom_method(
multi_step_dense.evaluate(conv_window.val), imports='import os;import IPython;import pandas as pd;import IPython.display;import matplotlib.pyplot as plt;import datetime;import matplotlib as mpl;import numpy as np;import tensorflow as tf;import seaborn as sns', function_to_run='obj.evaluate(*args)', method_object=eval('multi_step_dense'), function_args=[eval('conv_window.val')], function_kwargs={}, max_wait_secs=0, custom_class=None)
performance['Multi step dense'] = custom_method(
multi_step_dense.evaluate(conv_window.test, verbose=0), imports='import os;import IPython;import pandas as pd;import IPython.display;import matplotlib.pyplot as plt;import datetime;import matplotlib as mpl;import numpy as np;import tensorflow as tf;import seaborn as sns', function_to_run='obj.evaluate(*args, **kwargs)', method_object=eval('multi_step_dense'), function_args=[eval('conv_window.test')], function_kwargs={'verbose': eval('0')}, max_wait_secs=0, custom_class=None)
conv_window.plot(multi_step_dense)
print('Input shape:', wide_window.example[0].shape)
try:
    print('Output shape:', multi_step_dense(wide_window.example[0]).shape)
except Exception as e:
    print(f'\n{type(e).__name__}:{e}')
conv_model = custom_method(
tf.keras.Sequential([tf.keras.layers.Conv1D(filters=32, kernel_size=(CONV_WIDTH,), activation='relu'), tf.keras.layers.Dense(units=32, activation='relu'), tf.keras.layers.Dense(units=1)]), imports='import os;import IPython;import pandas as pd;import IPython.display;import matplotlib.pyplot as plt;import datetime;import matplotlib as mpl;import numpy as np;import tensorflow as tf;import seaborn as sns', function_to_run='tf.keras.Sequential(*args)', method_object=None, function_args=[eval("[\n    tf.keras.layers.Conv1D(filters=32,\n                           kernel_size=(CONV_WIDTH,),\n                           activation='relu'),\n    tf.keras.layers.Dense(units=32, activation='relu'),\n    tf.keras.layers.Dense(units=1),\n]")], function_kwargs={}, max_wait_secs=0)
print('Conv model on `conv_window`')
print('Input shape:', conv_window.example[0].shape)
print('Output shape:', conv_model(conv_window.example[0]).shape)
history = compile_and_fit(conv_model, conv_window)
IPython.display.clear_output()
val_performance['Conv'] = custom_method(
conv_model.evaluate(conv_window.val), imports='import os;import IPython;import pandas as pd;import IPython.display;import matplotlib.pyplot as plt;import datetime;import matplotlib as mpl;import numpy as np;import tensorflow as tf;import seaborn as sns', function_to_run='obj.evaluate(*args)', method_object=eval('conv_model'), function_args=[eval('conv_window.val')], function_kwargs={}, max_wait_secs=0, custom_class=None)
performance['Conv'] = custom_method(
conv_model.evaluate(conv_window.test, verbose=0), imports='import os;import IPython;import pandas as pd;import IPython.display;import matplotlib.pyplot as plt;import datetime;import matplotlib as mpl;import numpy as np;import tensorflow as tf;import seaborn as sns', function_to_run='obj.evaluate(*args, **kwargs)', method_object=eval('conv_model'), function_args=[eval('conv_window.test')], function_kwargs={'verbose': eval('0')}, max_wait_secs=0, custom_class=None)
print('Wide window')
print('Input shape:', wide_window.example[0].shape)
print('Labels shape:', wide_window.example[1].shape)
print('Output shape:', conv_model(wide_window.example[0]).shape)
LABEL_WIDTH = 24
INPUT_WIDTH = LABEL_WIDTH + (CONV_WIDTH - 1)
wide_conv_window = WindowGenerator(input_width=INPUT_WIDTH, label_width=LABEL_WIDTH, shift=1, label_columns=['T (degC)'])
wide_conv_window
print('Wide conv window')
print('Input shape:', wide_conv_window.example[0].shape)
print('Labels shape:', wide_conv_window.example[1].shape)
print('Output shape:', conv_model(wide_conv_window.example[0]).shape)
wide_conv_window.plot(conv_model)
lstm_model = custom_method(
tf.keras.models.Sequential([tf.keras.layers.LSTM(32, return_sequences=True), tf.keras.layers.Dense(units=1)]), imports='import os;import IPython;import pandas as pd;import IPython.display;import matplotlib.pyplot as plt;import datetime;import matplotlib as mpl;import numpy as np;import tensorflow as tf;import seaborn as sns', function_to_run='tf.keras.models.Sequential(*args)', method_object=None, function_args=[eval('[\n    tf.keras.layers.LSTM(32, return_sequences=True),\n    tf.keras.layers.Dense(units=1)\n]')], function_kwargs={}, max_wait_secs=0)
print('Input shape:', wide_window.example[0].shape)
print('Output shape:', lstm_model(wide_window.example[0]).shape)
history = compile_and_fit(lstm_model, wide_window)
IPython.display.clear_output()
val_performance['LSTM'] = custom_method(
lstm_model.evaluate(wide_window.val), imports='import os;import IPython;import pandas as pd;import IPython.display;import matplotlib.pyplot as plt;import datetime;import matplotlib as mpl;import numpy as np;import tensorflow as tf;import seaborn as sns', function_to_run='obj.evaluate(*args)', method_object=eval('lstm_model'), function_args=[eval('wide_window.val')], function_kwargs={}, max_wait_secs=0, custom_class=None)
performance['LSTM'] = custom_method(
lstm_model.evaluate(wide_window.test, verbose=0), imports='import os;import IPython;import pandas as pd;import IPython.display;import matplotlib.pyplot as plt;import datetime;import matplotlib as mpl;import numpy as np;import tensorflow as tf;import seaborn as sns', function_to_run='obj.evaluate(*args, **kwargs)', method_object=eval('lstm_model'), function_args=[eval('wide_window.test')], function_kwargs={'verbose': eval('0')}, max_wait_secs=0, custom_class=None)
wide_window.plot(lstm_model)
x = np.arange(len(performance))
width = 0.3
metric_name = 'mean_absolute_error'
metric_index = custom_method(
lstm_model.metrics_names.index('mean_absolute_error'), imports='import os;import IPython;import pandas as pd;import IPython.display;import matplotlib.pyplot as plt;import datetime;import matplotlib as mpl;import numpy as np;import tensorflow as tf;import seaborn as sns', function_to_run='obj.metrics_names.index(*args)', method_object=eval('lstm_model'), function_args=[eval("'mean_absolute_error'")], function_kwargs={}, max_wait_secs=0, custom_class=None)
val_mae = [v[metric_index] for v in val_performance.values()]
test_mae = [v[metric_index] for v in performance.values()]
plt.ylabel('mean_absolute_error [T (degC), normalized]')
plt.bar(x - 0.17, val_mae, width, label='Validation')
plt.bar(x + 0.17, test_mae, width, label='Test')
plt.xticks(ticks=x, labels=performance.keys(), rotation=45)
_ = plt.legend()
for (name, value) in performance.items():
    print(f'{name:12s}: {value[1]:0.4f}')
single_step_window = WindowGenerator(input_width=1, label_width=1, shift=1)
wide_window = WindowGenerator(input_width=24, label_width=24, shift=1)
for (example_inputs, example_labels) in wide_window.train.take(1):
    print(f'Inputs shape (batch, time, features): {example_inputs.shape}')
    print(f'Labels shape (batch, time, features): {example_labels.shape}')
baseline = Baseline()
custom_method(
baseline.compile(loss=tf.keras.losses.MeanSquaredError(), metrics=[tf.keras.metrics.MeanAbsoluteError()]), imports='import os;import IPython;import pandas as pd;import IPython.display;import matplotlib.pyplot as plt;import datetime;import matplotlib as mpl;import numpy as np;import tensorflow as tf;import seaborn as sns', function_to_run='obj.compile(**kwargs)', method_object=eval('baseline'), function_args=[], function_kwargs={'loss': eval('tf.keras.losses.MeanSquaredError()'), 'metrics': eval('[tf.keras.metrics.MeanAbsoluteError()]')}, max_wait_secs=0, custom_class='class Baseline(tf.keras.Model):\n  def __init__(self, label_index=None):\n    super().__init__()\n    self.label_index = label_index\n\n  def call(self, inputs):\n    if self.label_index is None:\n      return inputs\n    result = inputs[:, :, self.label_index]\n    return result[:, :, tf.newaxis]')
val_performance = {}
performance = {}
val_performance['Baseline'] = custom_method(
baseline.evaluate(wide_window.val), imports='import os;import IPython;import pandas as pd;import IPython.display;import matplotlib.pyplot as plt;import datetime;import matplotlib as mpl;import numpy as np;import tensorflow as tf;import seaborn as sns', function_to_run='obj.evaluate(*args)', method_object=eval('baseline'), function_args=[eval('wide_window.val')], function_kwargs={}, max_wait_secs=0, custom_class='class Baseline(tf.keras.Model):\n  def __init__(self, label_index=None):\n    super().__init__()\n    self.label_index = label_index\n\n  def call(self, inputs):\n    if self.label_index is None:\n      return inputs\n    result = inputs[:, :, self.label_index]\n    return result[:, :, tf.newaxis]')
performance['Baseline'] = custom_method(
baseline.evaluate(wide_window.test, verbose=0), imports='import os;import IPython;import pandas as pd;import IPython.display;import matplotlib.pyplot as plt;import datetime;import matplotlib as mpl;import numpy as np;import tensorflow as tf;import seaborn as sns', function_to_run='obj.evaluate(*args, **kwargs)', method_object=eval('baseline'), function_args=[eval('wide_window.test')], function_kwargs={'verbose': eval('0')}, max_wait_secs=0, custom_class='class Baseline(tf.keras.Model):\n  def __init__(self, label_index=None):\n    super().__init__()\n    self.label_index = label_index\n\n  def call(self, inputs):\n    if self.label_index is None:\n      return inputs\n    result = inputs[:, :, self.label_index]\n    return result[:, :, tf.newaxis]')
dense = custom_method(
tf.keras.Sequential([tf.keras.layers.Dense(units=64, activation='relu'), tf.keras.layers.Dense(units=64, activation='relu'), tf.keras.layers.Dense(units=num_features)]), imports='import os;import IPython;import pandas as pd;import IPython.display;import matplotlib.pyplot as plt;import datetime;import matplotlib as mpl;import numpy as np;import tensorflow as tf;import seaborn as sns', function_to_run='tf.keras.Sequential(*args)', method_object=None, function_args=[eval("[\n    tf.keras.layers.Dense(units=64, activation='relu'),\n    tf.keras.layers.Dense(units=64, activation='relu'),\n    tf.keras.layers.Dense(units=num_features)\n]")], function_kwargs={}, max_wait_secs=0)
history = compile_and_fit(dense, single_step_window)
IPython.display.clear_output()
val_performance['Dense'] = custom_method(
dense.evaluate(single_step_window.val), imports='import os;import IPython;import pandas as pd;import IPython.display;import matplotlib.pyplot as plt;import datetime;import matplotlib as mpl;import numpy as np;import tensorflow as tf;import seaborn as sns', function_to_run='obj.evaluate(*args)', method_object=eval('dense'), function_args=[eval('single_step_window.val')], function_kwargs={}, max_wait_secs=0, custom_class=None)
performance['Dense'] = custom_method(
dense.evaluate(single_step_window.test, verbose=0), imports='import os;import IPython;import pandas as pd;import IPython.display;import matplotlib.pyplot as plt;import datetime;import matplotlib as mpl;import numpy as np;import tensorflow as tf;import seaborn as sns', function_to_run='obj.evaluate(*args, **kwargs)', method_object=eval('dense'), function_args=[eval('single_step_window.test')], function_kwargs={'verbose': eval('0')}, max_wait_secs=0, custom_class=None)
wide_window = WindowGenerator(input_width=24, label_width=24, shift=1)
lstm_model = custom_method(
tf.keras.models.Sequential([tf.keras.layers.LSTM(32, return_sequences=True), tf.keras.layers.Dense(units=num_features)]), imports='import os;import IPython;import pandas as pd;import IPython.display;import matplotlib.pyplot as plt;import datetime;import matplotlib as mpl;import numpy as np;import tensorflow as tf;import seaborn as sns', function_to_run='tf.keras.models.Sequential(*args)', method_object=None, function_args=[eval('[\n    tf.keras.layers.LSTM(32, return_sequences=True),\n    tf.keras.layers.Dense(units=num_features)\n]')], function_kwargs={}, max_wait_secs=0)
history = compile_and_fit(lstm_model, wide_window)
IPython.display.clear_output()
val_performance['LSTM'] = custom_method(
lstm_model.evaluate(wide_window.val), imports='import os;import IPython;import pandas as pd;import IPython.display;import matplotlib.pyplot as plt;import datetime;import matplotlib as mpl;import numpy as np;import tensorflow as tf;import seaborn as sns', function_to_run='obj.evaluate(*args)', method_object=eval('lstm_model'), function_args=[eval('wide_window.val')], function_kwargs={}, max_wait_secs=0, custom_class=None)
performance['LSTM'] = custom_method(
lstm_model.evaluate(wide_window.test, verbose=0), imports='import os;import IPython;import pandas as pd;import IPython.display;import matplotlib.pyplot as plt;import datetime;import matplotlib as mpl;import numpy as np;import tensorflow as tf;import seaborn as sns', function_to_run='obj.evaluate(*args, **kwargs)', method_object=eval('lstm_model'), function_args=[eval('wide_window.test')], function_kwargs={'verbose': eval('0')}, max_wait_secs=0, custom_class=None)
print()

class ResidualWrapper(tf.keras.Model):

    def __init__(self, model):
        super().__init__()
        self.model = model

    def call(self, inputs, *args, **kwargs):
        delta = self.model(inputs, *args, **kwargs)
        return inputs + delta
residual_lstm = ResidualWrapper(tf.keras.Sequential([tf.keras.layers.LSTM(32, return_sequences=True), tf.keras.layers.Dense(num_features, kernel_initializer=tf.initializers.zeros())]))
history = compile_and_fit(residual_lstm, wide_window)
IPython.display.clear_output()
val_performance['Residual LSTM'] = custom_method(
residual_lstm.evaluate(wide_window.val), imports='import os;import IPython;import pandas as pd;import IPython.display;import matplotlib.pyplot as plt;import datetime;import matplotlib as mpl;import numpy as np;import tensorflow as tf;import seaborn as sns', function_to_run='obj.evaluate(*args)', method_object=eval('residual_lstm'), function_args=[eval('wide_window.val')], function_kwargs={}, max_wait_secs=0, custom_class='class ResidualWrapper(tf.keras.Model):\n  def __init__(self, model):\n    super().__init__()\n    self.model = model\n\n  def call(self, inputs, *args, **kwargs):\n    delta = self.model(inputs, *args, **kwargs)\n\n    return inputs + delta')
performance['Residual LSTM'] = custom_method(
residual_lstm.evaluate(wide_window.test, verbose=0), imports='import os;import IPython;import pandas as pd;import IPython.display;import matplotlib.pyplot as plt;import datetime;import matplotlib as mpl;import numpy as np;import tensorflow as tf;import seaborn as sns', function_to_run='obj.evaluate(*args, **kwargs)', method_object=eval('residual_lstm'), function_args=[eval('wide_window.test')], function_kwargs={'verbose': eval('0')}, max_wait_secs=0, custom_class='class ResidualWrapper(tf.keras.Model):\n  def __init__(self, model):\n    super().__init__()\n    self.model = model\n\n  def call(self, inputs, *args, **kwargs):\n    delta = self.model(inputs, *args, **kwargs)\n\n    return inputs + delta')
print()
x = np.arange(len(performance))
width = 0.3
metric_name = 'mean_absolute_error'
metric_index = custom_method(
lstm_model.metrics_names.index('mean_absolute_error'), imports='import os;import IPython;import pandas as pd;import IPython.display;import matplotlib.pyplot as plt;import datetime;import matplotlib as mpl;import numpy as np;import tensorflow as tf;import seaborn as sns', function_to_run='obj.metrics_names.index(*args)', method_object=eval('lstm_model'), function_args=[eval("'mean_absolute_error'")], function_kwargs={}, max_wait_secs=0, custom_class=None)
val_mae = [v[metric_index] for v in val_performance.values()]
test_mae = [v[metric_index] for v in performance.values()]
plt.bar(x - 0.17, val_mae, width, label='Validation')
plt.bar(x + 0.17, test_mae, width, label='Test')
plt.xticks(ticks=x, labels=performance.keys(), rotation=45)
plt.ylabel('MAE (average over all outputs)')
_ = plt.legend()
for (name, value) in performance.items():
    print(f'{name:15s}: {value[1]:0.4f}')
OUT_STEPS = 24
multi_window = WindowGenerator(input_width=24, label_width=OUT_STEPS, shift=OUT_STEPS)
multi_window.plot()
multi_window

class MultiStepLastBaseline(tf.keras.Model):

    def call(self, inputs):
        return custom_method(
        tf.tile(inputs[:, -1:, :], [1, OUT_STEPS, 1]), imports='import os;import IPython;import pandas as pd;import IPython.display;import matplotlib.pyplot as plt;import datetime;import matplotlib as mpl;import numpy as np;import tensorflow as tf;import seaborn as sns', function_to_run='tf.tile(*args)', method_object=None, function_args=[eval('inputs[:, -1:, :]'), eval('[1, OUT_STEPS, 1]')], function_kwargs={}, max_wait_secs=0)
last_baseline = MultiStepLastBaseline()
custom_method(
last_baseline.compile(loss=tf.keras.losses.MeanSquaredError(), metrics=[tf.keras.metrics.MeanAbsoluteError()]), imports='import os;import IPython;import pandas as pd;import IPython.display;import matplotlib.pyplot as plt;import datetime;import matplotlib as mpl;import numpy as np;import tensorflow as tf;import seaborn as sns', function_to_run='obj.compile(**kwargs)', method_object=eval('last_baseline'), function_args=[], function_kwargs={'loss': eval('tf.keras.losses.MeanSquaredError()'), 'metrics': eval('[tf.keras.metrics.MeanAbsoluteError()]')}, max_wait_secs=0, custom_class='class MultiStepLastBaseline(tf.keras.Model):\n  def call(self, inputs):\n    return tf.tile(inputs[:, -1:, :], [1, OUT_STEPS, 1])')
multi_val_performance = {}
multi_performance = {}
multi_val_performance['Last'] = custom_method(
last_baseline.evaluate(multi_window.val), imports='import os;import IPython;import pandas as pd;import IPython.display;import matplotlib.pyplot as plt;import datetime;import matplotlib as mpl;import numpy as np;import tensorflow as tf;import seaborn as sns', function_to_run='obj.evaluate(*args)', method_object=eval('last_baseline'), function_args=[eval('multi_window.val')], function_kwargs={}, max_wait_secs=0, custom_class='class MultiStepLastBaseline(tf.keras.Model):\n  def call(self, inputs):\n    return tf.tile(inputs[:, -1:, :], [1, OUT_STEPS, 1])')
multi_performance['Last'] = custom_method(
last_baseline.evaluate(multi_window.test, verbose=0), imports='import os;import IPython;import pandas as pd;import IPython.display;import matplotlib.pyplot as plt;import datetime;import matplotlib as mpl;import numpy as np;import tensorflow as tf;import seaborn as sns', function_to_run='obj.evaluate(*args, **kwargs)', method_object=eval('last_baseline'), function_args=[eval('multi_window.test')], function_kwargs={'verbose': eval('0')}, max_wait_secs=0, custom_class='class MultiStepLastBaseline(tf.keras.Model):\n  def call(self, inputs):\n    return tf.tile(inputs[:, -1:, :], [1, OUT_STEPS, 1])')
multi_window.plot(last_baseline)

class RepeatBaseline(tf.keras.Model):

    def call(self, inputs):
        return inputs
repeat_baseline = RepeatBaseline()
custom_method(
repeat_baseline.compile(loss=tf.keras.losses.MeanSquaredError(), metrics=[tf.keras.metrics.MeanAbsoluteError()]), imports='import os;import IPython;import pandas as pd;import IPython.display;import matplotlib.pyplot as plt;import datetime;import matplotlib as mpl;import numpy as np;import tensorflow as tf;import seaborn as sns', function_to_run='obj.compile(**kwargs)', method_object=eval('repeat_baseline'), function_args=[], function_kwargs={'loss': eval('tf.keras.losses.MeanSquaredError()'), 'metrics': eval('[tf.keras.metrics.MeanAbsoluteError()]')}, max_wait_secs=0, custom_class='class RepeatBaseline(tf.keras.Model):\n  def call(self, inputs):\n    return inputs')
multi_val_performance['Repeat'] = custom_method(
repeat_baseline.evaluate(multi_window.val), imports='import os;import IPython;import pandas as pd;import IPython.display;import matplotlib.pyplot as plt;import datetime;import matplotlib as mpl;import numpy as np;import tensorflow as tf;import seaborn as sns', function_to_run='obj.evaluate(*args)', method_object=eval('repeat_baseline'), function_args=[eval('multi_window.val')], function_kwargs={}, max_wait_secs=0, custom_class='class RepeatBaseline(tf.keras.Model):\n  def call(self, inputs):\n    return inputs')
multi_performance['Repeat'] = custom_method(
repeat_baseline.evaluate(multi_window.test, verbose=0), imports='import os;import IPython;import pandas as pd;import IPython.display;import matplotlib.pyplot as plt;import datetime;import matplotlib as mpl;import numpy as np;import tensorflow as tf;import seaborn as sns', function_to_run='obj.evaluate(*args, **kwargs)', method_object=eval('repeat_baseline'), function_args=[eval('multi_window.test')], function_kwargs={'verbose': eval('0')}, max_wait_secs=0, custom_class='class RepeatBaseline(tf.keras.Model):\n  def call(self, inputs):\n    return inputs')
multi_window.plot(repeat_baseline)
multi_linear_model = custom_method(
tf.keras.Sequential([tf.keras.layers.Lambda(lambda x: x[:, -1:, :]), tf.keras.layers.Dense(OUT_STEPS * num_features, kernel_initializer=tf.initializers.zeros()), tf.keras.layers.Reshape([OUT_STEPS, num_features])]), imports='import os;import IPython;import pandas as pd;import IPython.display;import matplotlib.pyplot as plt;import datetime;import matplotlib as mpl;import numpy as np;import tensorflow as tf;import seaborn as sns', function_to_run='tf.keras.Sequential(*args)', method_object=None, function_args=[eval('[\n    tf.keras.layers.Lambda(lambda x: x[:, -1:, :]),\n    tf.keras.layers.Dense(OUT_STEPS*num_features,\n                          kernel_initializer=tf.initializers.zeros()),\n    tf.keras.layers.Reshape([OUT_STEPS, num_features])\n]')], function_kwargs={}, max_wait_secs=0)
history = compile_and_fit(multi_linear_model, multi_window)
IPython.display.clear_output()
multi_val_performance['Linear'] = custom_method(
multi_linear_model.evaluate(multi_window.val), imports='import os;import IPython;import pandas as pd;import IPython.display;import matplotlib.pyplot as plt;import datetime;import matplotlib as mpl;import numpy as np;import tensorflow as tf;import seaborn as sns', function_to_run='obj.evaluate(*args)', method_object=eval('multi_linear_model'), function_args=[eval('multi_window.val')], function_kwargs={}, max_wait_secs=0, custom_class=None)
multi_performance['Linear'] = custom_method(
multi_linear_model.evaluate(multi_window.test, verbose=0), imports='import os;import IPython;import pandas as pd;import IPython.display;import matplotlib.pyplot as plt;import datetime;import matplotlib as mpl;import numpy as np;import tensorflow as tf;import seaborn as sns', function_to_run='obj.evaluate(*args, **kwargs)', method_object=eval('multi_linear_model'), function_args=[eval('multi_window.test')], function_kwargs={'verbose': eval('0')}, max_wait_secs=0, custom_class=None)
multi_window.plot(multi_linear_model)
multi_dense_model = custom_method(
tf.keras.Sequential([tf.keras.layers.Lambda(lambda x: x[:, -1:, :]), tf.keras.layers.Dense(512, activation='relu'), tf.keras.layers.Dense(OUT_STEPS * num_features, kernel_initializer=tf.initializers.zeros()), tf.keras.layers.Reshape([OUT_STEPS, num_features])]), imports='import os;import IPython;import pandas as pd;import IPython.display;import matplotlib.pyplot as plt;import datetime;import matplotlib as mpl;import numpy as np;import tensorflow as tf;import seaborn as sns', function_to_run='tf.keras.Sequential(*args)', method_object=None, function_args=[eval("[\n    tf.keras.layers.Lambda(lambda x: x[:, -1:, :]),\n    tf.keras.layers.Dense(512, activation='relu'),\n    tf.keras.layers.Dense(OUT_STEPS*num_features,\n                          kernel_initializer=tf.initializers.zeros()),\n    tf.keras.layers.Reshape([OUT_STEPS, num_features])\n]")], function_kwargs={}, max_wait_secs=0)
history = compile_and_fit(multi_dense_model, multi_window)
IPython.display.clear_output()
multi_val_performance['Dense'] = custom_method(
multi_dense_model.evaluate(multi_window.val), imports='import os;import IPython;import pandas as pd;import IPython.display;import matplotlib.pyplot as plt;import datetime;import matplotlib as mpl;import numpy as np;import tensorflow as tf;import seaborn as sns', function_to_run='obj.evaluate(*args)', method_object=eval('multi_dense_model'), function_args=[eval('multi_window.val')], function_kwargs={}, max_wait_secs=0, custom_class=None)
multi_performance['Dense'] = custom_method(
multi_dense_model.evaluate(multi_window.test, verbose=0), imports='import os;import IPython;import pandas as pd;import IPython.display;import matplotlib.pyplot as plt;import datetime;import matplotlib as mpl;import numpy as np;import tensorflow as tf;import seaborn as sns', function_to_run='obj.evaluate(*args, **kwargs)', method_object=eval('multi_dense_model'), function_args=[eval('multi_window.test')], function_kwargs={'verbose': eval('0')}, max_wait_secs=0, custom_class=None)
multi_window.plot(multi_dense_model)
CONV_WIDTH = 3
multi_conv_model = custom_method(
tf.keras.Sequential([tf.keras.layers.Lambda(lambda x: x[:, -CONV_WIDTH:, :]), tf.keras.layers.Conv1D(256, activation='relu', kernel_size=CONV_WIDTH), tf.keras.layers.Dense(OUT_STEPS * num_features, kernel_initializer=tf.initializers.zeros()), tf.keras.layers.Reshape([OUT_STEPS, num_features])]), imports='import os;import IPython;import pandas as pd;import IPython.display;import matplotlib.pyplot as plt;import datetime;import matplotlib as mpl;import numpy as np;import tensorflow as tf;import seaborn as sns', function_to_run='tf.keras.Sequential(*args)', method_object=None, function_args=[eval("[\n    tf.keras.layers.Lambda(lambda x: x[:, -CONV_WIDTH:, :]),\n    tf.keras.layers.Conv1D(256, activation='relu', kernel_size=(CONV_WIDTH)),\n    tf.keras.layers.Dense(OUT_STEPS*num_features,\n                          kernel_initializer=tf.initializers.zeros()),\n    tf.keras.layers.Reshape([OUT_STEPS, num_features])\n]")], function_kwargs={}, max_wait_secs=0)
history = compile_and_fit(multi_conv_model, multi_window)
IPython.display.clear_output()
multi_val_performance['Conv'] = custom_method(
multi_conv_model.evaluate(multi_window.val), imports='import os;import IPython;import pandas as pd;import IPython.display;import matplotlib.pyplot as plt;import datetime;import matplotlib as mpl;import numpy as np;import tensorflow as tf;import seaborn as sns', function_to_run='obj.evaluate(*args)', method_object=eval('multi_conv_model'), function_args=[eval('multi_window.val')], function_kwargs={}, max_wait_secs=0, custom_class=None)
multi_performance['Conv'] = custom_method(
multi_conv_model.evaluate(multi_window.test, verbose=0), imports='import os;import IPython;import pandas as pd;import IPython.display;import matplotlib.pyplot as plt;import datetime;import matplotlib as mpl;import numpy as np;import tensorflow as tf;import seaborn as sns', function_to_run='obj.evaluate(*args, **kwargs)', method_object=eval('multi_conv_model'), function_args=[eval('multi_window.test')], function_kwargs={'verbose': eval('0')}, max_wait_secs=0, custom_class=None)
multi_window.plot(multi_conv_model)
multi_lstm_model = custom_method(
tf.keras.Sequential([tf.keras.layers.LSTM(32, return_sequences=False), tf.keras.layers.Dense(OUT_STEPS * num_features, kernel_initializer=tf.initializers.zeros()), tf.keras.layers.Reshape([OUT_STEPS, num_features])]), imports='import os;import IPython;import pandas as pd;import IPython.display;import matplotlib.pyplot as plt;import datetime;import matplotlib as mpl;import numpy as np;import tensorflow as tf;import seaborn as sns', function_to_run='tf.keras.Sequential(*args)', method_object=None, function_args=[eval('[\n    tf.keras.layers.LSTM(32, return_sequences=False),\n    tf.keras.layers.Dense(OUT_STEPS*num_features,\n                          kernel_initializer=tf.initializers.zeros()),\n    tf.keras.layers.Reshape([OUT_STEPS, num_features])\n]')], function_kwargs={}, max_wait_secs=0)
history = compile_and_fit(multi_lstm_model, multi_window)
IPython.display.clear_output()
multi_val_performance['LSTM'] = custom_method(
multi_lstm_model.evaluate(multi_window.val), imports='import os;import IPython;import pandas as pd;import IPython.display;import matplotlib.pyplot as plt;import datetime;import matplotlib as mpl;import numpy as np;import tensorflow as tf;import seaborn as sns', function_to_run='obj.evaluate(*args)', method_object=eval('multi_lstm_model'), function_args=[eval('multi_window.val')], function_kwargs={}, max_wait_secs=0, custom_class=None)
multi_performance['LSTM'] = custom_method(
multi_lstm_model.evaluate(multi_window.test, verbose=0), imports='import os;import IPython;import pandas as pd;import IPython.display;import matplotlib.pyplot as plt;import datetime;import matplotlib as mpl;import numpy as np;import tensorflow as tf;import seaborn as sns', function_to_run='obj.evaluate(*args, **kwargs)', method_object=eval('multi_lstm_model'), function_args=[eval('multi_window.test')], function_kwargs={'verbose': eval('0')}, max_wait_secs=0, custom_class=None)
multi_window.plot(multi_lstm_model)

class FeedBack(tf.keras.Model):

    def __init__(self, units, out_steps):
        super().__init__()
        self.out_steps = out_steps
        self.units = units
        self.lstm_cell = custom_method(
        tf.keras.layers.LSTMCell(units), imports='import os;import IPython;import pandas as pd;import IPython.display;import matplotlib.pyplot as plt;import datetime;import matplotlib as mpl;import numpy as np;import tensorflow as tf;import seaborn as sns', function_to_run='tf.keras.layers.LSTMCell(*args)', method_object=None, function_args=[eval('units')], function_kwargs={}, max_wait_secs=0)
        self.lstm_rnn = custom_method(
        tf.keras.layers.RNN(self.lstm_cell, return_state=True), imports='import os;import IPython;import pandas as pd;import IPython.display;import matplotlib.pyplot as plt;import datetime;import matplotlib as mpl;import numpy as np;import tensorflow as tf;import seaborn as sns', function_to_run='tf.keras.layers.RNN(*args, **kwargs)', method_object=None, function_args=[eval('self.lstm_cell')], function_kwargs={'return_state': eval('True')}, max_wait_secs=0)
        self.dense = custom_method(
        tf.keras.layers.Dense(num_features), imports='import os;import IPython;import pandas as pd;import IPython.display;import matplotlib.pyplot as plt;import datetime;import matplotlib as mpl;import numpy as np;import tensorflow as tf;import seaborn as sns', function_to_run='tf.keras.layers.Dense(*args)', method_object=None, function_args=[eval('num_features')], function_kwargs={}, max_wait_secs=0)
feedback_model = FeedBack(units=32, out_steps=OUT_STEPS)

def warmup(self, inputs):
    (x, *state) = self.lstm_rnn(inputs)
    prediction = self.dense(x)
    return (prediction, state)
FeedBack.warmup = warmup
(prediction, state) = custom_method(
feedback_model.warmup(multi_window.example[0]), imports='import os;import IPython;import pandas as pd;import IPython.display;import matplotlib.pyplot as plt;import datetime;import matplotlib as mpl;import numpy as np;import tensorflow as tf;import seaborn as sns', function_to_run='obj.warmup(*args)', method_object=eval('feedback_model'), function_args=[eval('multi_window.example[0]')], function_kwargs={}, max_wait_secs=0, custom_class='class FeedBack(tf.keras.Model):\n  def __init__(self, units, out_steps):\n    super().__init__()\n    self.out_steps = out_steps\n    self.units = units\n    self.lstm_cell = tf.keras.layers.LSTMCell(units)\n    self.lstm_rnn = tf.keras.layers.RNN(self.lstm_cell, return_state=True)\n    self.dense = tf.keras.layers.Dense(num_features)')
prediction.shape

def call(self, inputs, training=None):
    predictions = []
    (prediction, state) = self.warmup(inputs)
    custom_method(
    predictions.append(prediction), imports='import os;import IPython;import pandas as pd;import IPython.display;import matplotlib.pyplot as plt;import datetime;import matplotlib as mpl;import numpy as np;import tensorflow as tf;import seaborn as sns', function_to_run='obj.append(*args)', method_object=eval('predictions'), function_args=[eval('prediction')], function_kwargs={}, max_wait_secs=0, custom_class=None)
    for n in range(1, self.out_steps):
        x = prediction
        (x, state) = self.lstm_cell(x, states=state, training=training)
        prediction = self.dense(x)
        custom_method(
        predictions.append(prediction), imports='import os;import IPython;import pandas as pd;import IPython.display;import matplotlib.pyplot as plt;import datetime;import matplotlib as mpl;import numpy as np;import tensorflow as tf;import seaborn as sns', function_to_run='obj.append(*args)', method_object=eval('predictions'), function_args=[eval('prediction')], function_kwargs={}, max_wait_secs=0, custom_class=None)
    predictions = custom_method(
    tf.stack(predictions), imports='import os;import IPython;import pandas as pd;import IPython.display;import matplotlib.pyplot as plt;import datetime;import matplotlib as mpl;import numpy as np;import tensorflow as tf;import seaborn as sns', function_to_run='tf.stack(*args)', method_object=None, function_args=[eval('predictions')], function_kwargs={}, max_wait_secs=0)
    predictions = custom_method(
    tf.transpose(predictions, [1, 0, 2]), imports='import os;import IPython;import pandas as pd;import IPython.display;import matplotlib.pyplot as plt;import datetime;import matplotlib as mpl;import numpy as np;import tensorflow as tf;import seaborn as sns', function_to_run='tf.transpose(*args)', method_object=None, function_args=[eval('predictions'), eval('[1, 0, 2]')], function_kwargs={}, max_wait_secs=0)
    return predictions
FeedBack.call = call
print('Output shape (batch, time, features): ', feedback_model(multi_window.example[0]).shape)
history = compile_and_fit(feedback_model, multi_window)
IPython.display.clear_output()
multi_val_performance['AR LSTM'] = custom_method(
feedback_model.evaluate(multi_window.val), imports='import os;import IPython;import pandas as pd;import IPython.display;import matplotlib.pyplot as plt;import datetime;import matplotlib as mpl;import numpy as np;import tensorflow as tf;import seaborn as sns', function_to_run='obj.evaluate(*args)', method_object=eval('feedback_model'), function_args=[eval('multi_window.val')], function_kwargs={}, max_wait_secs=0, custom_class='class FeedBack(tf.keras.Model):\n  def __init__(self, units, out_steps):\n    super().__init__()\n    self.out_steps = out_steps\n    self.units = units\n    self.lstm_cell = tf.keras.layers.LSTMCell(units)\n    self.lstm_rnn = tf.keras.layers.RNN(self.lstm_cell, return_state=True)\n    self.dense = tf.keras.layers.Dense(num_features)')
multi_performance['AR LSTM'] = custom_method(
feedback_model.evaluate(multi_window.test, verbose=0), imports='import os;import IPython;import pandas as pd;import IPython.display;import matplotlib.pyplot as plt;import datetime;import matplotlib as mpl;import numpy as np;import tensorflow as tf;import seaborn as sns', function_to_run='obj.evaluate(*args, **kwargs)', method_object=eval('feedback_model'), function_args=[eval('multi_window.test')], function_kwargs={'verbose': eval('0')}, max_wait_secs=0, custom_class='class FeedBack(tf.keras.Model):\n  def __init__(self, units, out_steps):\n    super().__init__()\n    self.out_steps = out_steps\n    self.units = units\n    self.lstm_cell = tf.keras.layers.LSTMCell(units)\n    self.lstm_rnn = tf.keras.layers.RNN(self.lstm_cell, return_state=True)\n    self.dense = tf.keras.layers.Dense(num_features)')
multi_window.plot(feedback_model)
x = np.arange(len(multi_performance))
width = 0.3
metric_name = 'mean_absolute_error'
metric_index = custom_method(
lstm_model.metrics_names.index('mean_absolute_error'), imports='import os;import IPython;import pandas as pd;import IPython.display;import matplotlib.pyplot as plt;import datetime;import matplotlib as mpl;import numpy as np;import tensorflow as tf;import seaborn as sns', function_to_run='obj.metrics_names.index(*args)', method_object=eval('lstm_model'), function_args=[eval("'mean_absolute_error'")], function_kwargs={}, max_wait_secs=0, custom_class=None)
val_mae = [v[metric_index] for v in multi_val_performance.values()]
test_mae = [v[metric_index] for v in multi_performance.values()]
plt.bar(x - 0.17, val_mae, width, label='Validation')
plt.bar(x + 0.17, test_mae, width, label='Test')
plt.xticks(ticks=x, labels=multi_performance.keys(), rotation=45)
plt.ylabel(f'MAE (average over all times and outputs)')
_ = plt.legend()
for (name, value) in multi_performance.items():
    print(f'{name:8s}: {value[1]:0.4f}')
