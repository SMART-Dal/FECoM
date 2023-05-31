import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
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
custom_method(imports='import numpy as np;import seaborn as sns;from tensorflow.keras import layers;from tensorflow import keras;import tensorflow as tf;import matplotlib.pyplot as plt;import pandas as pd', function_to_run='tf.keras.layers.Normalization(**kwargs)', method_object=None, object_signature=None, function_args=[], function_kwargs={'axis': eval('-1')})
normalizer = tf.keras.layers.Normalization(axis=-1)
custom_method(imports='import numpy as np;import seaborn as sns;from tensorflow.keras import layers;from tensorflow import keras;import tensorflow as tf;import matplotlib.pyplot as plt;import pandas as pd', function_to_run='obj.adapt(*args)', method_object=eval('normalizer'), object_signature=None, function_args=[eval('np.array(train_features)')], function_kwargs={}, custom_class=None)
normalizer.adapt(np.array(train_features))
print(normalizer.mean.numpy())
first = np.array(train_features[:1])
with np.printoptions(precision=2, suppress=True):
    print('First example:', first)
    print()
    print('Normalized:', normalizer(first).numpy())
horsepower = np.array(train_features['Horsepower'])
custom_method(imports='import numpy as np;import seaborn as sns;from tensorflow.keras import layers;from tensorflow import keras;import tensorflow as tf;import matplotlib.pyplot as plt;import pandas as pd', function_to_run='layers.Normalization(**kwargs)', method_object=None, object_signature=None, function_args=[], function_kwargs={'input_shape': eval('[1,]'), 'axis': eval('None')})
horsepower_normalizer = layers.Normalization(input_shape=[1], axis=None)
custom_method(imports='import numpy as np;import seaborn as sns;from tensorflow.keras import layers;from tensorflow import keras;import tensorflow as tf;import matplotlib.pyplot as plt;import pandas as pd', function_to_run='obj.adapt(*args)', method_object=eval('horsepower_normalizer'), object_signature=None, function_args=[eval('horsepower')], function_kwargs={}, custom_class=None)
horsepower_normalizer.adapt(horsepower)
custom_method(imports='import numpy as np;import seaborn as sns;from tensorflow.keras import layers;from tensorflow import keras;import tensorflow as tf;import matplotlib.pyplot as plt;import pandas as pd', function_to_run='tf.keras.Sequential(*args)', method_object=None, object_signature=None, function_args=[eval('[\n    horsepower_normalizer,\n    layers.Dense(units=1)\n]')], function_kwargs={})
horsepower_model = tf.keras.Sequential([horsepower_normalizer, layers.Dense(units=1)])
custom_method(imports='import numpy as np;import seaborn as sns;from tensorflow.keras import layers;from tensorflow import keras;import tensorflow as tf;import matplotlib.pyplot as plt;import pandas as pd', function_to_run='obj.summary()', method_object=eval('horsepower_model'), object_signature=None, function_args=[], function_kwargs={}, custom_class=None)
horsepower_model.summary()
custom_method(imports='import numpy as np;import seaborn as sns;from tensorflow.keras import layers;from tensorflow import keras;import tensorflow as tf;import matplotlib.pyplot as plt;import pandas as pd', function_to_run='obj.predict(*args)', method_object=eval('horsepower_model'), object_signature=None, function_args=[eval('horsepower[:10]')], function_kwargs={}, custom_class=None)
horsepower_model.predict(horsepower[:10])
custom_method(imports='import numpy as np;import seaborn as sns;from tensorflow.keras import layers;from tensorflow import keras;import tensorflow as tf;import matplotlib.pyplot as plt;import pandas as pd', function_to_run='obj.compile(**kwargs)', method_object=eval('horsepower_model'), object_signature=None, function_args=[], function_kwargs={'optimizer': eval('tf.keras.optimizers.Adam(learning_rate=0.1)'), 'loss': eval("'mean_absolute_error'")}, custom_class=None)
horsepower_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.1), loss='mean_absolute_error')
custom_method(imports='import numpy as np;import seaborn as sns;from tensorflow.keras import layers;from tensorflow import keras;import tensorflow as tf;import matplotlib.pyplot as plt;import pandas as pd', function_to_run='obj.fit(*args, **kwargs)', method_object=eval('horsepower_model'), object_signature=None, function_args=[eval("train_features['Horsepower']"), eval('train_labels')], function_kwargs={'epochs': eval('100'), 'verbose': eval('0'), 'validation_split': eval('0.2')}, custom_class=None)
history = horsepower_model.fit(train_features['Horsepower'], train_labels, epochs=100, verbose=0, validation_split=0.2)
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
custom_method(imports='import numpy as np;import seaborn as sns;from tensorflow.keras import layers;from tensorflow import keras;import tensorflow as tf;import matplotlib.pyplot as plt;import pandas as pd', function_to_run='obj.evaluate(*args, **kwargs)', method_object=eval('horsepower_model'), object_signature=None, function_args=[eval("test_features['Horsepower']"), eval('test_labels')], function_kwargs={'verbose': eval('0')}, custom_class=None)
test_results['horsepower_model'] = horsepower_model.evaluate(test_features['Horsepower'], test_labels, verbose=0)
custom_method(imports='import numpy as np;import seaborn as sns;from tensorflow.keras import layers;from tensorflow import keras;import tensorflow as tf;import matplotlib.pyplot as plt;import pandas as pd', function_to_run='tf.linspace(*args)', method_object=None, object_signature=None, function_args=[eval('0.0'), eval('250'), eval('251')], function_kwargs={})
x = tf.linspace(0.0, 250, 251)
custom_method(imports='import numpy as np;import seaborn as sns;from tensorflow.keras import layers;from tensorflow import keras;import tensorflow as tf;import matplotlib.pyplot as plt;import pandas as pd', function_to_run='obj.predict(*args)', method_object=eval('horsepower_model'), object_signature=None, function_args=[eval('x')], function_kwargs={}, custom_class=None)
y = horsepower_model.predict(x)

def plot_horsepower(x, y):
    plt.scatter(train_features['Horsepower'], train_labels, label='Data')
    plt.plot(x, y, color='k', label='Predictions')
    plt.xlabel('Horsepower')
    plt.ylabel('MPG')
    plt.legend()
plot_horsepower(x, y)
custom_method(imports='import numpy as np;import seaborn as sns;from tensorflow.keras import layers;from tensorflow import keras;import tensorflow as tf;import matplotlib.pyplot as plt;import pandas as pd', function_to_run='tf.keras.Sequential(*args)', method_object=None, object_signature=None, function_args=[eval('[\n    normalizer,\n    layers.Dense(units=1)\n]')], function_kwargs={})
linear_model = tf.keras.Sequential([normalizer, layers.Dense(units=1)])
custom_method(imports='import numpy as np;import seaborn as sns;from tensorflow.keras import layers;from tensorflow import keras;import tensorflow as tf;import matplotlib.pyplot as plt;import pandas as pd', function_to_run='obj.predict(*args)', method_object=eval('linear_model'), object_signature=None, function_args=[eval('train_features[:10]')], function_kwargs={}, custom_class=None)
linear_model.predict(train_features[:10])
linear_model.layers[1].kernel
custom_method(imports='import numpy as np;import seaborn as sns;from tensorflow.keras import layers;from tensorflow import keras;import tensorflow as tf;import matplotlib.pyplot as plt;import pandas as pd', function_to_run='obj.compile(**kwargs)', method_object=eval('linear_model'), object_signature=None, function_args=[], function_kwargs={'optimizer': eval('tf.keras.optimizers.Adam(learning_rate=0.1)'), 'loss': eval("'mean_absolute_error'")}, custom_class=None)
linear_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.1), loss='mean_absolute_error')
custom_method(imports='import numpy as np;import seaborn as sns;from tensorflow.keras import layers;from tensorflow import keras;import tensorflow as tf;import matplotlib.pyplot as plt;import pandas as pd', function_to_run='obj.fit(*args, **kwargs)', method_object=eval('linear_model'), object_signature=None, function_args=[eval('train_features'), eval('train_labels')], function_kwargs={'epochs': eval('100'), 'verbose': eval('0'), 'validation_split': eval('0.2')}, custom_class=None)
history = linear_model.fit(train_features, train_labels, epochs=100, verbose=0, validation_split=0.2)
plot_loss(history)
custom_method(imports='import numpy as np;import seaborn as sns;from tensorflow.keras import layers;from tensorflow import keras;import tensorflow as tf;import matplotlib.pyplot as plt;import pandas as pd', function_to_run='obj.evaluate(*args, **kwargs)', method_object=eval('linear_model'), object_signature=None, function_args=[eval('test_features'), eval('test_labels')], function_kwargs={'verbose': eval('0')}, custom_class=None)
test_results['linear_model'] = linear_model.evaluate(test_features, test_labels, verbose=0)

def build_and_compile_model(norm):
    custom_method(imports='import numpy as np;import seaborn as sns;from tensorflow.keras import layers;from tensorflow import keras;import tensorflow as tf;import matplotlib.pyplot as plt;import pandas as pd', function_to_run='keras.Sequential(*args)', method_object=None, object_signature=None, function_args=[eval("[\n      norm,\n      layers.Dense(64, activation='relu'),\n      layers.Dense(64, activation='relu'),\n      layers.Dense(1)\n  ]")], function_kwargs={})
    model = keras.Sequential([norm, layers.Dense(64, activation='relu'), layers.Dense(64, activation='relu'), layers.Dense(1)])
    custom_method(imports='import numpy as np;import seaborn as sns;from tensorflow.keras import layers;from tensorflow import keras;import tensorflow as tf;import matplotlib.pyplot as plt;import pandas as pd', function_to_run='obj.compile(**kwargs)', method_object=eval('model'), object_signature=None, function_args=[], function_kwargs={'loss': eval("'mean_absolute_error'"), 'optimizer': eval('tf.keras.optimizers.Adam(0.001)')}, custom_class=None)
    model.compile(loss='mean_absolute_error', optimizer=tf.keras.optimizers.Adam(0.001))
    return model
dnn_horsepower_model = build_and_compile_model(horsepower_normalizer)
dnn_horsepower_model.summary()
history = dnn_horsepower_model.fit(train_features['Horsepower'], train_labels, validation_split=0.2, verbose=0, epochs=100)
plot_loss(history)
custom_method(imports='import numpy as np;import seaborn as sns;from tensorflow.keras import layers;from tensorflow import keras;import tensorflow as tf;import matplotlib.pyplot as plt;import pandas as pd', function_to_run='tf.linspace(*args)', method_object=None, object_signature=None, function_args=[eval('0.0'), eval('250'), eval('251')], function_kwargs={})
x = tf.linspace(0.0, 250, 251)
y = dnn_horsepower_model.predict(x)
plot_horsepower(x, y)
test_results['dnn_horsepower_model'] = dnn_horsepower_model.evaluate(test_features['Horsepower'], test_labels, verbose=0)
dnn_model = build_and_compile_model(normalizer)
dnn_model.summary()
history = dnn_model.fit(train_features, train_labels, validation_split=0.2, verbose=0, epochs=100)
plot_loss(history)
test_results['dnn_model'] = dnn_model.evaluate(test_features, test_labels, verbose=0)
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
dnn_model.save('dnn_model')
custom_method(imports='import numpy as np;import seaborn as sns;from tensorflow.keras import layers;from tensorflow import keras;import tensorflow as tf;import matplotlib.pyplot as plt;import pandas as pd', function_to_run='tf.keras.models.load_model(*args)', method_object=None, object_signature=None, function_args=[eval("'dnn_model'")], function_kwargs={})
reloaded = tf.keras.models.load_model('dnn_model')
custom_method(imports='import numpy as np;import seaborn as sns;from tensorflow.keras import layers;from tensorflow import keras;import tensorflow as tf;import matplotlib.pyplot as plt;import pandas as pd', function_to_run='obj.evaluate(*args, **kwargs)', method_object=eval('reloaded'), object_signature=None, function_args=[eval('test_features'), eval('test_labels')], function_kwargs={'verbose': eval('0')}, custom_class=None)
test_results['reloaded'] = reloaded.evaluate(test_features, test_labels, verbose=0)
pd.DataFrame(test_results, index=['Mean absolute error [MPG]']).T
