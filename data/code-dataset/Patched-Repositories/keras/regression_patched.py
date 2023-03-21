import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
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
np.set_printoptions(precision=3, suppress=True)
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
print(tf.__version__)
url = 'http://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data'
column_names = ['MPG', 'Cylinders', 'Displacement', 'Horsepower', 'Weight', 'Acceleration', 'Model Year', 'Origin']
raw_dataset = pd.read_csv(url, names=column_names, na_values='?', comment='\t', sep=' ', skipinitialspace=True)
dataset = raw_dataset.copy()
dataset.tail()
dataset.isna().sum()
dataset = dataset.dropna()
dataset['Origin'] = dataset['Origin'].map({1: 'USA', 2: 'Europe', 3: 'Japan'})
dataset = pd.get_dummies(dataset, columns=['Origin'], prefix='', prefix_sep='')
dataset.tail()
train_dataset = dataset.sample(frac=0.8, random_state=0)
test_dataset = dataset.drop(train_dataset.index)
sns.pairplot(train_dataset[['MPG', 'Cylinders', 'Displacement', 'Weight']], diag_kind='kde')
train_dataset.describe().transpose()
train_features = train_dataset.copy()
test_features = test_dataset.copy()
train_labels = train_features.pop('MPG')
test_labels = test_features.pop('MPG')
train_dataset.describe().transpose()[['mean', 'std']]
normalizer = custom_method(
tf.keras.layers.Normalization(axis=-1), imports='import tensorflow as tf;import numpy as np;import matplotlib.pyplot as plt;import pandas as pd;from tensorflow.keras import layers;from tensorflow import keras;import seaborn as sns', function_to_run='tf.keras.layers.Normalization(**kwargs)', method_object=None, function_args=[], function_kwargs={'axis': eval('-1')}, max_wait_secs=0)
custom_method(
normalizer.adapt(np.array(train_features)), imports='import tensorflow as tf;import numpy as np;import matplotlib.pyplot as plt;import pandas as pd;from tensorflow.keras import layers;from tensorflow import keras;import seaborn as sns', function_to_run='obj.adapt(*args)', method_object=eval('normalizer'), function_args=[eval('np.array(train_features)')], function_kwargs={}, max_wait_secs=0, custom_class=None)
print(normalizer.mean.numpy())
first = np.array(train_features[:1])
with np.printoptions(precision=2, suppress=True):
    print('First example:', first)
    print()
    print('Normalized:', normalizer(first).numpy())
horsepower = np.array(train_features['Horsepower'])
horsepower_normalizer = custom_method(
layers.Normalization(input_shape=[1], axis=None), imports='import tensorflow as tf;import numpy as np;import matplotlib.pyplot as plt;import pandas as pd;from tensorflow.keras import layers;from tensorflow import keras;import seaborn as sns', function_to_run='layers.Normalization(**kwargs)', method_object=None, function_args=[], function_kwargs={'input_shape': eval('[1,]'), 'axis': eval('None')}, max_wait_secs=0)
custom_method(
horsepower_normalizer.adapt(horsepower), imports='import tensorflow as tf;import numpy as np;import matplotlib.pyplot as plt;import pandas as pd;from tensorflow.keras import layers;from tensorflow import keras;import seaborn as sns', function_to_run='obj.adapt(*args)', method_object=eval('horsepower_normalizer'), function_args=[eval('horsepower')], function_kwargs={}, max_wait_secs=0, custom_class=None)
horsepower_model = custom_method(
tf.keras.Sequential([horsepower_normalizer, layers.Dense(units=1)]), imports='import tensorflow as tf;import numpy as np;import matplotlib.pyplot as plt;import pandas as pd;from tensorflow.keras import layers;from tensorflow import keras;import seaborn as sns', function_to_run='tf.keras.Sequential(*args)', method_object=None, function_args=[eval('[\n    horsepower_normalizer,\n    layers.Dense(units=1)\n]')], function_kwargs={}, max_wait_secs=0)
custom_method(
horsepower_model.summary(), imports='import tensorflow as tf;import numpy as np;import matplotlib.pyplot as plt;import pandas as pd;from tensorflow.keras import layers;from tensorflow import keras;import seaborn as sns', function_to_run='obj.summary()', method_object=eval('horsepower_model'), function_args=[], function_kwargs={}, max_wait_secs=0, custom_class=None)
custom_method(
horsepower_model.predict(horsepower[:10]), imports='import tensorflow as tf;import numpy as np;import matplotlib.pyplot as plt;import pandas as pd;from tensorflow.keras import layers;from tensorflow import keras;import seaborn as sns', function_to_run='obj.predict(*args)', method_object=eval('horsepower_model'), function_args=[eval('horsepower[:10]')], function_kwargs={}, max_wait_secs=0, custom_class=None)
custom_method(
horsepower_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.1), loss='mean_absolute_error'), imports='import tensorflow as tf;import numpy as np;import matplotlib.pyplot as plt;import pandas as pd;from tensorflow.keras import layers;from tensorflow import keras;import seaborn as sns', function_to_run='obj.compile(**kwargs)', method_object=eval('horsepower_model'), function_args=[], function_kwargs={'optimizer': eval('tf.keras.optimizers.Adam(learning_rate=0.1)'), 'loss': eval("'mean_absolute_error'")}, max_wait_secs=0, custom_class=None)
history = custom_method(
horsepower_model.fit(train_features['Horsepower'], train_labels, epochs=100, verbose=0, validation_split=0.2), imports='import tensorflow as tf;import numpy as np;import matplotlib.pyplot as plt;import pandas as pd;from tensorflow.keras import layers;from tensorflow import keras;import seaborn as sns', function_to_run='obj.fit(*args, **kwargs)', method_object=eval('horsepower_model'), function_args=[eval("train_features['Horsepower']"), eval('train_labels')], function_kwargs={'epochs': eval('100'), 'verbose': eval('0'), 'validation_split': eval('0.2')}, max_wait_secs=0, custom_class=None)
hist = pd.DataFrame(history.history)
hist['epoch'] = history.epoch
hist.tail()

def plot_loss(history):
    plt.plot(history.history['loss'], label='loss')
    plt.plot(history.history['val_loss'], label='val_loss')
    plt.ylim([0, 10])
    plt.xlabel('Epoch')
    plt.ylabel('Error [MPG]')
    plt.legend()
    plt.grid(True)
plot_loss(history)
test_results = {}
test_results['horsepower_model'] = custom_method(
horsepower_model.evaluate(test_features['Horsepower'], test_labels, verbose=0), imports='import tensorflow as tf;import numpy as np;import matplotlib.pyplot as plt;import pandas as pd;from tensorflow.keras import layers;from tensorflow import keras;import seaborn as sns', function_to_run='obj.evaluate(*args, **kwargs)', method_object=eval('horsepower_model'), function_args=[eval("test_features['Horsepower']"), eval('test_labels')], function_kwargs={'verbose': eval('0')}, max_wait_secs=0, custom_class=None)
x = custom_method(
tf.linspace(0.0, 250, 251), imports='import tensorflow as tf;import numpy as np;import matplotlib.pyplot as plt;import pandas as pd;from tensorflow.keras import layers;from tensorflow import keras;import seaborn as sns', function_to_run='tf.linspace(*args)', method_object=None, function_args=[eval('0.0'), eval('250'), eval('251')], function_kwargs={}, max_wait_secs=0)
y = custom_method(
horsepower_model.predict(x), imports='import tensorflow as tf;import numpy as np;import matplotlib.pyplot as plt;import pandas as pd;from tensorflow.keras import layers;from tensorflow import keras;import seaborn as sns', function_to_run='obj.predict(*args)', method_object=eval('horsepower_model'), function_args=[eval('x')], function_kwargs={}, max_wait_secs=0, custom_class=None)

def plot_horsepower(x, y):
    plt.scatter(train_features['Horsepower'], train_labels, label='Data')
    plt.plot(x, y, color='k', label='Predictions')
    plt.xlabel('Horsepower')
    plt.ylabel('MPG')
    plt.legend()
plot_horsepower(x, y)
linear_model = custom_method(
tf.keras.Sequential([normalizer, layers.Dense(units=1)]), imports='import tensorflow as tf;import numpy as np;import matplotlib.pyplot as plt;import pandas as pd;from tensorflow.keras import layers;from tensorflow import keras;import seaborn as sns', function_to_run='tf.keras.Sequential(*args)', method_object=None, function_args=[eval('[\n    normalizer,\n    layers.Dense(units=1)\n]')], function_kwargs={}, max_wait_secs=0)
custom_method(
linear_model.predict(train_features[:10]), imports='import tensorflow as tf;import numpy as np;import matplotlib.pyplot as plt;import pandas as pd;from tensorflow.keras import layers;from tensorflow import keras;import seaborn as sns', function_to_run='obj.predict(*args)', method_object=eval('linear_model'), function_args=[eval('train_features[:10]')], function_kwargs={}, max_wait_secs=0, custom_class=None)
linear_model.layers[1].kernel
custom_method(
linear_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.1), loss='mean_absolute_error'), imports='import tensorflow as tf;import numpy as np;import matplotlib.pyplot as plt;import pandas as pd;from tensorflow.keras import layers;from tensorflow import keras;import seaborn as sns', function_to_run='obj.compile(**kwargs)', method_object=eval('linear_model'), function_args=[], function_kwargs={'optimizer': eval('tf.keras.optimizers.Adam(learning_rate=0.1)'), 'loss': eval("'mean_absolute_error'")}, max_wait_secs=0, custom_class=None)
history = custom_method(
linear_model.fit(train_features, train_labels, epochs=100, verbose=0, validation_split=0.2), imports='import tensorflow as tf;import numpy as np;import matplotlib.pyplot as plt;import pandas as pd;from tensorflow.keras import layers;from tensorflow import keras;import seaborn as sns', function_to_run='obj.fit(*args, **kwargs)', method_object=eval('linear_model'), function_args=[eval('train_features'), eval('train_labels')], function_kwargs={'epochs': eval('100'), 'verbose': eval('0'), 'validation_split': eval('0.2')}, max_wait_secs=0, custom_class=None)
plot_loss(history)
test_results['linear_model'] = custom_method(
linear_model.evaluate(test_features, test_labels, verbose=0), imports='import tensorflow as tf;import numpy as np;import matplotlib.pyplot as plt;import pandas as pd;from tensorflow.keras import layers;from tensorflow import keras;import seaborn as sns', function_to_run='obj.evaluate(*args, **kwargs)', method_object=eval('linear_model'), function_args=[eval('test_features'), eval('test_labels')], function_kwargs={'verbose': eval('0')}, max_wait_secs=0, custom_class=None)

def build_and_compile_model(norm):
    model = custom_method(
    keras.Sequential([norm, layers.Dense(64, activation='relu'), layers.Dense(64, activation='relu'), layers.Dense(1)]), imports='import tensorflow as tf;import numpy as np;import matplotlib.pyplot as plt;import pandas as pd;from tensorflow.keras import layers;from tensorflow import keras;import seaborn as sns', function_to_run='keras.Sequential(*args)', method_object=None, function_args=[eval("[\n      norm,\n      layers.Dense(64, activation='relu'),\n      layers.Dense(64, activation='relu'),\n      layers.Dense(1)\n  ]")], function_kwargs={}, max_wait_secs=0)
    custom_method(
    model.compile(loss='mean_absolute_error', optimizer=tf.keras.optimizers.Adam(0.001)), imports='import tensorflow as tf;import numpy as np;import matplotlib.pyplot as plt;import pandas as pd;from tensorflow.keras import layers;from tensorflow import keras;import seaborn as sns', function_to_run='obj.compile(**kwargs)', method_object=eval('model'), function_args=[], function_kwargs={'loss': eval("'mean_absolute_error'"), 'optimizer': eval('tf.keras.optimizers.Adam(0.001)')}, max_wait_secs=0, custom_class=None)
    return model
dnn_horsepower_model = custom_method(
build_and_compile_model(horsepower_normalizer), imports='import tensorflow as tf;import numpy as np;import matplotlib.pyplot as plt;import pandas as pd;from tensorflow.keras import layers;from tensorflow import keras;import seaborn as sns', function_to_run='obj(*args)', method_object=eval('build_and_compile_model'), function_args=[eval('horsepower_normalizer')], function_kwargs={}, max_wait_secs=0, custom_class=None)
custom_method(
dnn_horsepower_model.summary(), imports='import tensorflow as tf;import numpy as np;import matplotlib.pyplot as plt;import pandas as pd;from tensorflow.keras import layers;from tensorflow import keras;import seaborn as sns', function_to_run='obj.summary()', method_object=eval('dnn_horsepower_model'), function_args=[], function_kwargs={}, max_wait_secs=0, custom_class=None)
history = custom_method(
dnn_horsepower_model.fit(train_features['Horsepower'], train_labels, validation_split=0.2, verbose=0, epochs=100), imports='import tensorflow as tf;import numpy as np;import matplotlib.pyplot as plt;import pandas as pd;from tensorflow.keras import layers;from tensorflow import keras;import seaborn as sns', function_to_run='obj.fit(*args, **kwargs)', method_object=eval('dnn_horsepower_model'), function_args=[eval("train_features['Horsepower']"), eval('train_labels')], function_kwargs={'validation_split': eval('0.2'), 'verbose': eval('0'), 'epochs': eval('100')}, max_wait_secs=0, custom_class=None)
plot_loss(history)
x = custom_method(
tf.linspace(0.0, 250, 251), imports='import tensorflow as tf;import numpy as np;import matplotlib.pyplot as plt;import pandas as pd;from tensorflow.keras import layers;from tensorflow import keras;import seaborn as sns', function_to_run='tf.linspace(*args)', method_object=None, function_args=[eval('0.0'), eval('250'), eval('251')], function_kwargs={}, max_wait_secs=0)
y = custom_method(
dnn_horsepower_model.predict(x), imports='import tensorflow as tf;import numpy as np;import matplotlib.pyplot as plt;import pandas as pd;from tensorflow.keras import layers;from tensorflow import keras;import seaborn as sns', function_to_run='obj.predict(*args)', method_object=eval('dnn_horsepower_model'), function_args=[eval('x')], function_kwargs={}, max_wait_secs=0, custom_class=None)
plot_horsepower(x, y)
test_results['dnn_horsepower_model'] = custom_method(
dnn_horsepower_model.evaluate(test_features['Horsepower'], test_labels, verbose=0), imports='import tensorflow as tf;import numpy as np;import matplotlib.pyplot as plt;import pandas as pd;from tensorflow.keras import layers;from tensorflow import keras;import seaborn as sns', function_to_run='obj.evaluate(*args, **kwargs)', method_object=eval('dnn_horsepower_model'), function_args=[eval("test_features['Horsepower']"), eval('test_labels')], function_kwargs={'verbose': eval('0')}, max_wait_secs=0, custom_class=None)
dnn_model = custom_method(
build_and_compile_model(normalizer), imports='import tensorflow as tf;import numpy as np;import matplotlib.pyplot as plt;import pandas as pd;from tensorflow.keras import layers;from tensorflow import keras;import seaborn as sns', function_to_run='obj(*args)', method_object=eval('build_and_compile_model'), function_args=[eval('normalizer')], function_kwargs={}, max_wait_secs=0, custom_class=None)
custom_method(
dnn_model.summary(), imports='import tensorflow as tf;import numpy as np;import matplotlib.pyplot as plt;import pandas as pd;from tensorflow.keras import layers;from tensorflow import keras;import seaborn as sns', function_to_run='obj.summary()', method_object=eval('dnn_model'), function_args=[], function_kwargs={}, max_wait_secs=0, custom_class=None)
history = custom_method(
dnn_model.fit(train_features, train_labels, validation_split=0.2, verbose=0, epochs=100), imports='import tensorflow as tf;import numpy as np;import matplotlib.pyplot as plt;import pandas as pd;from tensorflow.keras import layers;from tensorflow import keras;import seaborn as sns', function_to_run='obj.fit(*args, **kwargs)', method_object=eval('dnn_model'), function_args=[eval('train_features'), eval('train_labels')], function_kwargs={'validation_split': eval('0.2'), 'verbose': eval('0'), 'epochs': eval('100')}, max_wait_secs=0, custom_class=None)
plot_loss(history)
test_results['dnn_model'] = custom_method(
dnn_model.evaluate(test_features, test_labels, verbose=0), imports='import tensorflow as tf;import numpy as np;import matplotlib.pyplot as plt;import pandas as pd;from tensorflow.keras import layers;from tensorflow import keras;import seaborn as sns', function_to_run='obj.evaluate(*args, **kwargs)', method_object=eval('dnn_model'), function_args=[eval('test_features'), eval('test_labels')], function_kwargs={'verbose': eval('0')}, max_wait_secs=0, custom_class=None)
pd.DataFrame(test_results, index=['Mean absolute error [MPG]']).T
test_predictions = dnn_model.predict(test_features).flatten()
a = plt.axes(aspect='equal')
plt.scatter(test_labels, test_predictions)
plt.xlabel('True Values [MPG]')
plt.ylabel('Predictions [MPG]')
lims = [0, 50]
plt.xlim(lims)
plt.ylim(lims)
_ = plt.plot(lims, lims)
error = test_predictions - test_labels
plt.hist(error, bins=25)
plt.xlabel('Prediction Error [MPG]')
_ = plt.ylabel('Count')
custom_method(
dnn_model.save('dnn_model'), imports='import tensorflow as tf;import numpy as np;import matplotlib.pyplot as plt;import pandas as pd;from tensorflow.keras import layers;from tensorflow import keras;import seaborn as sns', function_to_run='obj.save(*args)', method_object=eval('dnn_model'), function_args=[eval("'dnn_model'")], function_kwargs={}, max_wait_secs=0, custom_class=None)
reloaded = custom_method(
tf.keras.models.load_model('dnn_model'), imports='import tensorflow as tf;import numpy as np;import matplotlib.pyplot as plt;import pandas as pd;from tensorflow.keras import layers;from tensorflow import keras;import seaborn as sns', function_to_run='tf.keras.models.load_model(*args)', method_object=None, function_args=[eval("'dnn_model'")], function_kwargs={}, max_wait_secs=0)
test_results['reloaded'] = custom_method(
reloaded.evaluate(test_features, test_labels, verbose=0), imports='import tensorflow as tf;import numpy as np;import matplotlib.pyplot as plt;import pandas as pd;from tensorflow.keras import layers;from tensorflow import keras;import seaborn as sns', function_to_run='obj.evaluate(*args, **kwargs)', method_object=eval('reloaded'), function_args=[eval('test_features'), eval('test_labels')], function_kwargs={'verbose': eval('0')}, max_wait_secs=0, custom_class=None)
pd.DataFrame(test_results, index=['Mean absolute error [MPG]']).T
