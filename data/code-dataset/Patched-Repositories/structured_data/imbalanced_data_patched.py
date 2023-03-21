import tensorflow as tf
from tensorflow import keras
import os
import tempfile
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import sklearn
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
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

def custom_method(func, imports: str, function_to_run: str, method_object=None, object_signature=None, function_args: list=None, function_kwargs: dict=None, max_wait_secs=MAX_WAIT_S, custom_class=None, wait_after_run_secs=WAIT_AFTER_RUN_S):
    result = send_request(imports=imports, function_to_run=function_to_run, function_args=function_args, function_kwargs=function_kwargs, max_wait_secs=max_wait_secs, wait_after_run_secs=wait_after_run_secs, method_object=method_object, object_signature=object_signature, custom_class=custom_class, experiment_file_path=EXPERIMENT_FILE_PATH)
    return func
mpl.rcParams['figure.figsize'] = (12, 10)
colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
file = tf.keras.utils
raw_df = pd.read_csv('https://storage.googleapis.com/download.tensorflow.org/data/creditcard.csv')
raw_df.head()
raw_df[['Time', 'V1', 'V2', 'V3', 'V4', 'V5', 'V26', 'V27', 'V28', 'Amount', 'Class']].describe()
(neg, pos) = np.bincount(raw_df['Class'])
total = neg + pos
print('Examples:\n    Total: {}\n    Positive: {} ({:.2f}% of total)\n'.format(total, pos, 100 * pos / total))
cleaned_df = raw_df.copy()
cleaned_df.pop('Time')
eps = 0.001
cleaned_df['Log Amount'] = np.log(cleaned_df.pop('Amount') + eps)
(train_df, test_df) = train_test_split(cleaned_df, test_size=0.2)
(train_df, val_df) = train_test_split(train_df, test_size=0.2)
train_labels = np.array(train_df.pop('Class'))
bool_train_labels = train_labels != 0
val_labels = np.array(val_df.pop('Class'))
test_labels = np.array(test_df.pop('Class'))
train_features = np.array(train_df)
val_features = np.array(val_df)
test_features = np.array(test_df)
print(f'Average class probability in training set:   {train_labels.mean():.4f}')
print(f'Average class probability in validation set: {val_labels.mean():.4f}')
print(f'Average class probability in test set:       {test_labels.mean():.4f}')
scaler = StandardScaler()
train_features = scaler.fit_transform(train_features)
val_features = scaler.transform(val_features)
test_features = scaler.transform(test_features)
train_features = np.clip(train_features, -5, 5)
val_features = np.clip(val_features, -5, 5)
test_features = np.clip(test_features, -5, 5)
print('Training labels shape:', train_labels.shape)
print('Validation labels shape:', val_labels.shape)
print('Test labels shape:', test_labels.shape)
print('Training features shape:', train_features.shape)
print('Validation features shape:', val_features.shape)
print('Test features shape:', test_features.shape)
pos_df = pd.DataFrame(train_features[bool_train_labels], columns=train_df.columns)
neg_df = pd.DataFrame(train_features[~bool_train_labels], columns=train_df.columns)
sns.jointplot(x=pos_df['V5'], y=pos_df['V6'], kind='hex', xlim=(-5, 5), ylim=(-5, 5))
plt.suptitle('Positive distribution')
sns.jointplot(x=neg_df['V5'], y=neg_df['V6'], kind='hex', xlim=(-5, 5), ylim=(-5, 5))
_ = plt.suptitle('Negative distribution')
METRICS = [custom_method(
keras.metrics.BinaryCrossentropy(name='cross entropy'), imports='import seaborn as sns;import matplotlib.pyplot as plt;from sklearn.model_selection import train_test_split;import tempfile;import pandas as pd;from tensorflow import keras;import sklearn;from sklearn.preprocessing import StandardScaler;import os;from sklearn.metrics import confusion_matrix;import matplotlib as mpl;import tensorflow as tf;import numpy as np', function_to_run='keras.metrics.BinaryCrossentropy(**kwargs)', method_object=None, object_signature=None, function_args=[], function_kwargs={'name': eval("'cross entropy'")}, max_wait_secs=0), custom_method(
keras.metrics.MeanSquaredError(name='Brier score'), imports='import seaborn as sns;import matplotlib.pyplot as plt;from sklearn.model_selection import train_test_split;import tempfile;import pandas as pd;from tensorflow import keras;import sklearn;from sklearn.preprocessing import StandardScaler;import os;from sklearn.metrics import confusion_matrix;import matplotlib as mpl;import tensorflow as tf;import numpy as np', function_to_run='keras.metrics.MeanSquaredError(**kwargs)', method_object=None, object_signature=None, function_args=[], function_kwargs={'name': eval("'Brier score'")}, max_wait_secs=0), custom_method(
keras.metrics.TruePositives(name='tp'), imports='import seaborn as sns;import matplotlib.pyplot as plt;from sklearn.model_selection import train_test_split;import tempfile;import pandas as pd;from tensorflow import keras;import sklearn;from sklearn.preprocessing import StandardScaler;import os;from sklearn.metrics import confusion_matrix;import matplotlib as mpl;import tensorflow as tf;import numpy as np', function_to_run='keras.metrics.TruePositives(**kwargs)', method_object=None, object_signature=None, function_args=[], function_kwargs={'name': eval("'tp'")}, max_wait_secs=0), custom_method(
keras.metrics.FalsePositives(name='fp'), imports='import seaborn as sns;import matplotlib.pyplot as plt;from sklearn.model_selection import train_test_split;import tempfile;import pandas as pd;from tensorflow import keras;import sklearn;from sklearn.preprocessing import StandardScaler;import os;from sklearn.metrics import confusion_matrix;import matplotlib as mpl;import tensorflow as tf;import numpy as np', function_to_run='keras.metrics.FalsePositives(**kwargs)', method_object=None, object_signature=None, function_args=[], function_kwargs={'name': eval("'fp'")}, max_wait_secs=0), custom_method(
keras.metrics.TrueNegatives(name='tn'), imports='import seaborn as sns;import matplotlib.pyplot as plt;from sklearn.model_selection import train_test_split;import tempfile;import pandas as pd;from tensorflow import keras;import sklearn;from sklearn.preprocessing import StandardScaler;import os;from sklearn.metrics import confusion_matrix;import matplotlib as mpl;import tensorflow as tf;import numpy as np', function_to_run='keras.metrics.TrueNegatives(**kwargs)', method_object=None, object_signature=None, function_args=[], function_kwargs={'name': eval("'tn'")}, max_wait_secs=0), custom_method(
keras.metrics.FalseNegatives(name='fn'), imports='import seaborn as sns;import matplotlib.pyplot as plt;from sklearn.model_selection import train_test_split;import tempfile;import pandas as pd;from tensorflow import keras;import sklearn;from sklearn.preprocessing import StandardScaler;import os;from sklearn.metrics import confusion_matrix;import matplotlib as mpl;import tensorflow as tf;import numpy as np', function_to_run='keras.metrics.FalseNegatives(**kwargs)', method_object=None, object_signature=None, function_args=[], function_kwargs={'name': eval("'fn'")}, max_wait_secs=0), custom_method(
keras.metrics.BinaryAccuracy(name='accuracy'), imports='import seaborn as sns;import matplotlib.pyplot as plt;from sklearn.model_selection import train_test_split;import tempfile;import pandas as pd;from tensorflow import keras;import sklearn;from sklearn.preprocessing import StandardScaler;import os;from sklearn.metrics import confusion_matrix;import matplotlib as mpl;import tensorflow as tf;import numpy as np', function_to_run='keras.metrics.BinaryAccuracy(**kwargs)', method_object=None, object_signature=None, function_args=[], function_kwargs={'name': eval("'accuracy'")}, max_wait_secs=0), custom_method(
keras.metrics.Precision(name='precision'), imports='import seaborn as sns;import matplotlib.pyplot as plt;from sklearn.model_selection import train_test_split;import tempfile;import pandas as pd;from tensorflow import keras;import sklearn;from sklearn.preprocessing import StandardScaler;import os;from sklearn.metrics import confusion_matrix;import matplotlib as mpl;import tensorflow as tf;import numpy as np', function_to_run='keras.metrics.Precision(**kwargs)', method_object=None, object_signature=None, function_args=[], function_kwargs={'name': eval("'precision'")}, max_wait_secs=0), custom_method(
keras.metrics.Recall(name='recall'), imports='import seaborn as sns;import matplotlib.pyplot as plt;from sklearn.model_selection import train_test_split;import tempfile;import pandas as pd;from tensorflow import keras;import sklearn;from sklearn.preprocessing import StandardScaler;import os;from sklearn.metrics import confusion_matrix;import matplotlib as mpl;import tensorflow as tf;import numpy as np', function_to_run='keras.metrics.Recall(**kwargs)', method_object=None, object_signature=None, function_args=[], function_kwargs={'name': eval("'recall'")}, max_wait_secs=0), custom_method(
keras.metrics.AUC(name='auc'), imports='import seaborn as sns;import matplotlib.pyplot as plt;from sklearn.model_selection import train_test_split;import tempfile;import pandas as pd;from tensorflow import keras;import sklearn;from sklearn.preprocessing import StandardScaler;import os;from sklearn.metrics import confusion_matrix;import matplotlib as mpl;import tensorflow as tf;import numpy as np', function_to_run='keras.metrics.AUC(**kwargs)', method_object=None, object_signature=None, function_args=[], function_kwargs={'name': eval("'auc'")}, max_wait_secs=0), custom_method(
keras.metrics.AUC(name='prc', curve='PR'), imports='import seaborn as sns;import matplotlib.pyplot as plt;from sklearn.model_selection import train_test_split;import tempfile;import pandas as pd;from tensorflow import keras;import sklearn;from sklearn.preprocessing import StandardScaler;import os;from sklearn.metrics import confusion_matrix;import matplotlib as mpl;import tensorflow as tf;import numpy as np', function_to_run='keras.metrics.AUC(**kwargs)', method_object=None, object_signature=None, function_args=[], function_kwargs={'name': eval("'prc'"), 'curve': eval("'PR'")}, max_wait_secs=0)]

def make_model(metrics=METRICS, output_bias=None):
    if output_bias is not None:
        output_bias = custom_method(
        tf.keras.initializers.Constant(output_bias), imports='import seaborn as sns;import matplotlib.pyplot as plt;from sklearn.model_selection import train_test_split;import tempfile;import pandas as pd;from tensorflow import keras;import sklearn;from sklearn.preprocessing import StandardScaler;import os;from sklearn.metrics import confusion_matrix;import matplotlib as mpl;import tensorflow as tf;import numpy as np', function_to_run='tf.keras.initializers.Constant(*args)', method_object=None, object_signature=None, function_args=[eval('output_bias')], function_kwargs={}, max_wait_secs=0)
    model = custom_method(
    keras.Sequential([keras.layers.Dense(16, activation='relu', input_shape=(train_features.shape[-1],)), keras.layers.Dropout(0.5), keras.layers.Dense(1, activation='sigmoid', bias_initializer=output_bias)]), imports='import seaborn as sns;import matplotlib.pyplot as plt;from sklearn.model_selection import train_test_split;import tempfile;import pandas as pd;from tensorflow import keras;import sklearn;from sklearn.preprocessing import StandardScaler;import os;from sklearn.metrics import confusion_matrix;import matplotlib as mpl;import tensorflow as tf;import numpy as np', function_to_run='keras.Sequential(*args)', method_object=None, object_signature=None, function_args=[eval("[\n      keras.layers.Dense(\n          16, activation='relu',\n          input_shape=(train_features.shape[-1],)),\n      keras.layers.Dropout(0.5),\n      keras.layers.Dense(1, activation='sigmoid',\n                         bias_initializer=output_bias),\n  ]")], function_kwargs={}, max_wait_secs=0)
    custom_method(
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001), loss=keras.losses.BinaryCrossentropy(), metrics=metrics), imports='import seaborn as sns;import matplotlib.pyplot as plt;from sklearn.model_selection import train_test_split;import tempfile;import pandas as pd;from tensorflow import keras;import sklearn;from sklearn.preprocessing import StandardScaler;import os;from sklearn.metrics import confusion_matrix;import matplotlib as mpl;import tensorflow as tf;import numpy as np', function_to_run='obj.compile(**kwargs)', method_object=eval('model'), object_signature='keras.Sequential', function_args=[], function_kwargs={'optimizer': eval('keras.optimizers.Adam(learning_rate=1e-3)'), 'loss': eval('keras.losses.BinaryCrossentropy()'), 'metrics': eval('metrics')}, max_wait_secs=0, custom_class=None)
    return model
EPOCHS = 100
BATCH_SIZE = 2048
early_stopping = custom_method(
tf.keras.callbacks.EarlyStopping(monitor='val_prc', verbose=1, patience=10, mode='max', restore_best_weights=True), imports='import seaborn as sns;import matplotlib.pyplot as plt;from sklearn.model_selection import train_test_split;import tempfile;import pandas as pd;from tensorflow import keras;import sklearn;from sklearn.preprocessing import StandardScaler;import os;from sklearn.metrics import confusion_matrix;import matplotlib as mpl;import tensorflow as tf;import numpy as np', function_to_run='tf.keras.callbacks.EarlyStopping(**kwargs)', method_object=None, object_signature=None, function_args=[], function_kwargs={'monitor': eval("'val_prc'"), 'verbose': eval('1'), 'patience': eval('10'), 'mode': eval("'max'"), 'restore_best_weights': eval('True')}, max_wait_secs=0)
model = custom_method(
make_model(), imports='import seaborn as sns;import matplotlib.pyplot as plt;from sklearn.model_selection import train_test_split;import tempfile;import pandas as pd;from tensorflow import keras;import sklearn;from sklearn.preprocessing import StandardScaler;import os;from sklearn.metrics import confusion_matrix;import matplotlib as mpl;import tensorflow as tf;import numpy as np', function_to_run='obj()', method_object=eval('make_model'), object_signature='keras.Sequential', function_args=[], function_kwargs={}, max_wait_secs=0, custom_class=None)
custom_method(
model.summary(), imports='import seaborn as sns;import matplotlib.pyplot as plt;from sklearn.model_selection import train_test_split;import tempfile;import pandas as pd;from tensorflow import keras;import sklearn;from sklearn.preprocessing import StandardScaler;import os;from sklearn.metrics import confusion_matrix;import matplotlib as mpl;import tensorflow as tf;import numpy as np', function_to_run='obj.summary()', method_object=eval('model'), object_signature='keras.Sequential', function_args=[], function_kwargs={}, max_wait_secs=0, custom_class=None)
custom_method(
model.predict(train_features[:10]), imports='import seaborn as sns;import matplotlib.pyplot as plt;from sklearn.model_selection import train_test_split;import tempfile;import pandas as pd;from tensorflow import keras;import sklearn;from sklearn.preprocessing import StandardScaler;import os;from sklearn.metrics import confusion_matrix;import matplotlib as mpl;import tensorflow as tf;import numpy as np', function_to_run='obj.predict(*args)', method_object=eval('model'), object_signature='keras.Sequential', function_args=[eval('train_features[:10]')], function_kwargs={}, max_wait_secs=0, custom_class=None)
results = custom_method(
model.evaluate(train_features, train_labels, batch_size=BATCH_SIZE, verbose=0), imports='import seaborn as sns;import matplotlib.pyplot as plt;from sklearn.model_selection import train_test_split;import tempfile;import pandas as pd;from tensorflow import keras;import sklearn;from sklearn.preprocessing import StandardScaler;import os;from sklearn.metrics import confusion_matrix;import matplotlib as mpl;import tensorflow as tf;import numpy as np', function_to_run='obj.evaluate(*args, **kwargs)', method_object=eval('model'), object_signature='keras.Sequential', function_args=[eval('train_features'), eval('train_labels')], function_kwargs={'batch_size': eval('BATCH_SIZE'), 'verbose': eval('0')}, max_wait_secs=0, custom_class=None)
print('Loss: {:0.4f}'.format(results[0]))
initial_bias = np.log([pos / neg])
initial_bias
model = custom_method(
make_model(output_bias=initial_bias), imports='import seaborn as sns;import matplotlib.pyplot as plt;from sklearn.model_selection import train_test_split;import tempfile;import pandas as pd;from tensorflow import keras;import sklearn;from sklearn.preprocessing import StandardScaler;import os;from sklearn.metrics import confusion_matrix;import matplotlib as mpl;import tensorflow as tf;import numpy as np', function_to_run='obj(**kwargs)', method_object=eval('make_model'), object_signature='keras.Sequential', function_args=[], function_kwargs={'output_bias': eval('initial_bias')}, max_wait_secs=0, custom_class=None)
custom_method(
model.predict(train_features[:10]), imports='import seaborn as sns;import matplotlib.pyplot as plt;from sklearn.model_selection import train_test_split;import tempfile;import pandas as pd;from tensorflow import keras;import sklearn;from sklearn.preprocessing import StandardScaler;import os;from sklearn.metrics import confusion_matrix;import matplotlib as mpl;import tensorflow as tf;import numpy as np', function_to_run='obj.predict(*args)', method_object=eval('model'), object_signature='keras.Sequential', function_args=[eval('train_features[:10]')], function_kwargs={}, max_wait_secs=0, custom_class=None)
results = custom_method(
model.evaluate(train_features, train_labels, batch_size=BATCH_SIZE, verbose=0), imports='import seaborn as sns;import matplotlib.pyplot as plt;from sklearn.model_selection import train_test_split;import tempfile;import pandas as pd;from tensorflow import keras;import sklearn;from sklearn.preprocessing import StandardScaler;import os;from sklearn.metrics import confusion_matrix;import matplotlib as mpl;import tensorflow as tf;import numpy as np', function_to_run='obj.evaluate(*args, **kwargs)', method_object=eval('model'), object_signature='keras.Sequential', function_args=[eval('train_features'), eval('train_labels')], function_kwargs={'batch_size': eval('BATCH_SIZE'), 'verbose': eval('0')}, max_wait_secs=0, custom_class=None)
print('Loss: {:0.4f}'.format(results[0]))
initial_weights = os.path.join(tempfile.mkdtemp(), 'initial_weights')
custom_method(
model.save_weights(initial_weights), imports='import seaborn as sns;import matplotlib.pyplot as plt;from sklearn.model_selection import train_test_split;import tempfile;import pandas as pd;from tensorflow import keras;import sklearn;from sklearn.preprocessing import StandardScaler;import os;from sklearn.metrics import confusion_matrix;import matplotlib as mpl;import tensorflow as tf;import numpy as np', function_to_run='obj.save_weights(*args)', method_object=eval('model'), object_signature='keras.Sequential', function_args=[eval('initial_weights')], function_kwargs={}, max_wait_secs=0, custom_class=None)
model = custom_method(
make_model(), imports='import seaborn as sns;import matplotlib.pyplot as plt;from sklearn.model_selection import train_test_split;import tempfile;import pandas as pd;from tensorflow import keras;import sklearn;from sklearn.preprocessing import StandardScaler;import os;from sklearn.metrics import confusion_matrix;import matplotlib as mpl;import tensorflow as tf;import numpy as np', function_to_run='obj()', method_object=eval('make_model'), object_signature='keras.Sequential', function_args=[], function_kwargs={}, max_wait_secs=0, custom_class=None)
custom_method(
model.load_weights(initial_weights), imports='import seaborn as sns;import matplotlib.pyplot as plt;from sklearn.model_selection import train_test_split;import tempfile;import pandas as pd;from tensorflow import keras;import sklearn;from sklearn.preprocessing import StandardScaler;import os;from sklearn.metrics import confusion_matrix;import matplotlib as mpl;import tensorflow as tf;import numpy as np', function_to_run='obj.load_weights(*args)', method_object=eval('model'), object_signature='keras.Sequential', function_args=[eval('initial_weights')], function_kwargs={}, max_wait_secs=0, custom_class=None)
custom_method(
model.layers[-1].bias.assign([0.0]), imports='import seaborn as sns;import matplotlib.pyplot as plt;from sklearn.model_selection import train_test_split;import tempfile;import pandas as pd;from tensorflow import keras;import sklearn;from sklearn.preprocessing import StandardScaler;import os;from sklearn.metrics import confusion_matrix;import matplotlib as mpl;import tensorflow as tf;import numpy as np', function_to_run='obj.layers[-1].bias.assign(*args)', method_object=eval('model'), object_signature='keras.Sequential', function_args=[eval('[0.0]')], function_kwargs={}, max_wait_secs=0, custom_class=None)
zero_bias_history = custom_method(
model.fit(train_features, train_labels, batch_size=BATCH_SIZE, epochs=20, validation_data=(val_features, val_labels), verbose=0), imports='import seaborn as sns;import matplotlib.pyplot as plt;from sklearn.model_selection import train_test_split;import tempfile;import pandas as pd;from tensorflow import keras;import sklearn;from sklearn.preprocessing import StandardScaler;import os;from sklearn.metrics import confusion_matrix;import matplotlib as mpl;import tensorflow as tf;import numpy as np', function_to_run='obj.fit(*args, **kwargs)', method_object=eval('model'), object_signature='keras.Sequential', function_args=[eval('train_features'), eval('train_labels')], function_kwargs={'batch_size': eval('BATCH_SIZE'), 'epochs': eval('20'), 'validation_data': eval('(val_features, val_labels)'), 'verbose': eval('0')}, max_wait_secs=0, custom_class=None)
model = custom_method(
make_model(), imports='import seaborn as sns;import matplotlib.pyplot as plt;from sklearn.model_selection import train_test_split;import tempfile;import pandas as pd;from tensorflow import keras;import sklearn;from sklearn.preprocessing import StandardScaler;import os;from sklearn.metrics import confusion_matrix;import matplotlib as mpl;import tensorflow as tf;import numpy as np', function_to_run='obj()', method_object=eval('make_model'), object_signature='keras.Sequential', function_args=[], function_kwargs={}, max_wait_secs=0, custom_class=None)
custom_method(
model.load_weights(initial_weights), imports='import seaborn as sns;import matplotlib.pyplot as plt;from sklearn.model_selection import train_test_split;import tempfile;import pandas as pd;from tensorflow import keras;import sklearn;from sklearn.preprocessing import StandardScaler;import os;from sklearn.metrics import confusion_matrix;import matplotlib as mpl;import tensorflow as tf;import numpy as np', function_to_run='obj.load_weights(*args)', method_object=eval('model'), object_signature='keras.Sequential', function_args=[eval('initial_weights')], function_kwargs={}, max_wait_secs=0, custom_class=None)
careful_bias_history = custom_method(
model.fit(train_features, train_labels, batch_size=BATCH_SIZE, epochs=20, validation_data=(val_features, val_labels), verbose=0), imports='import seaborn as sns;import matplotlib.pyplot as plt;from sklearn.model_selection import train_test_split;import tempfile;import pandas as pd;from tensorflow import keras;import sklearn;from sklearn.preprocessing import StandardScaler;import os;from sklearn.metrics import confusion_matrix;import matplotlib as mpl;import tensorflow as tf;import numpy as np', function_to_run='obj.fit(*args, **kwargs)', method_object=eval('model'), object_signature='keras.Sequential', function_args=[eval('train_features'), eval('train_labels')], function_kwargs={'batch_size': eval('BATCH_SIZE'), 'epochs': eval('20'), 'validation_data': eval('(val_features, val_labels)'), 'verbose': eval('0')}, max_wait_secs=0, custom_class=None)

def plot_loss(history, label, n):
    plt.semilogy(history.epoch, history.history['loss'], color=colors[n], label='Train ' + label)
    plt.semilogy(history.epoch, history.history['val_loss'], color=colors[n], label='Val ' + label, linestyle='--')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
plot_loss(zero_bias_history, 'Zero Bias', 0)
plot_loss(careful_bias_history, 'Careful Bias', 1)
model = custom_method(
make_model(), imports='import seaborn as sns;import matplotlib.pyplot as plt;from sklearn.model_selection import train_test_split;import tempfile;import pandas as pd;from tensorflow import keras;import sklearn;from sklearn.preprocessing import StandardScaler;import os;from sklearn.metrics import confusion_matrix;import matplotlib as mpl;import tensorflow as tf;import numpy as np', function_to_run='obj()', method_object=eval('make_model'), object_signature='keras.Sequential', function_args=[], function_kwargs={}, max_wait_secs=0, custom_class=None)
custom_method(
model.load_weights(initial_weights), imports='import seaborn as sns;import matplotlib.pyplot as plt;from sklearn.model_selection import train_test_split;import tempfile;import pandas as pd;from tensorflow import keras;import sklearn;from sklearn.preprocessing import StandardScaler;import os;from sklearn.metrics import confusion_matrix;import matplotlib as mpl;import tensorflow as tf;import numpy as np', function_to_run='obj.load_weights(*args)', method_object=eval('model'), object_signature='keras.Sequential', function_args=[eval('initial_weights')], function_kwargs={}, max_wait_secs=0, custom_class=None)
baseline_history = custom_method(
model.fit(train_features, train_labels, batch_size=BATCH_SIZE, epochs=EPOCHS, callbacks=[early_stopping], validation_data=(val_features, val_labels)), imports='import seaborn as sns;import matplotlib.pyplot as plt;from sklearn.model_selection import train_test_split;import tempfile;import pandas as pd;from tensorflow import keras;import sklearn;from sklearn.preprocessing import StandardScaler;import os;from sklearn.metrics import confusion_matrix;import matplotlib as mpl;import tensorflow as tf;import numpy as np', function_to_run='obj.fit(*args, **kwargs)', method_object=eval('model'), object_signature='keras.Sequential', function_args=[eval('train_features'), eval('train_labels')], function_kwargs={'batch_size': eval('BATCH_SIZE'), 'epochs': eval('EPOCHS'), 'callbacks': eval('[early_stopping]'), 'validation_data': eval('(val_features, val_labels)')}, max_wait_secs=0, custom_class=None)

def plot_metrics(history):
    metrics = ['loss', 'prc', 'precision', 'recall']
    for (n, metric) in enumerate(metrics):
        name = metric.replace('_', ' ').capitalize()
        plt.subplot(2, 2, n + 1)
        plt.plot(history.epoch, history.history[metric], color=colors[0], label='Train')
        plt.plot(history.epoch, history.history['val_' + metric], color=colors[0], linestyle='--', label='Val')
        plt.xlabel('Epoch')
        plt.ylabel(name)
        if metric == 'loss':
            plt.ylim([0, plt.ylim()[1]])
        elif metric == 'auc':
            plt.ylim([0.8, 1])
        else:
            plt.ylim([0, 1])
        plt.legend()
plot_metrics(baseline_history)
train_predictions_baseline = custom_method(
model.predict(train_features, batch_size=BATCH_SIZE), imports='import seaborn as sns;import matplotlib.pyplot as plt;from sklearn.model_selection import train_test_split;import tempfile;import pandas as pd;from tensorflow import keras;import sklearn;from sklearn.preprocessing import StandardScaler;import os;from sklearn.metrics import confusion_matrix;import matplotlib as mpl;import tensorflow as tf;import numpy as np', function_to_run='obj.predict(*args, **kwargs)', method_object=eval('model'), object_signature='keras.Sequential', function_args=[eval('train_features')], function_kwargs={'batch_size': eval('BATCH_SIZE')}, max_wait_secs=0, custom_class=None)
test_predictions_baseline = custom_method(
model.predict(test_features, batch_size=BATCH_SIZE), imports='import seaborn as sns;import matplotlib.pyplot as plt;from sklearn.model_selection import train_test_split;import tempfile;import pandas as pd;from tensorflow import keras;import sklearn;from sklearn.preprocessing import StandardScaler;import os;from sklearn.metrics import confusion_matrix;import matplotlib as mpl;import tensorflow as tf;import numpy as np', function_to_run='obj.predict(*args, **kwargs)', method_object=eval('model'), object_signature='keras.Sequential', function_args=[eval('test_features')], function_kwargs={'batch_size': eval('BATCH_SIZE')}, max_wait_secs=0, custom_class=None)

def plot_cm(labels, predictions, threshold=0.5):
    cm = confusion_matrix(labels, predictions > threshold)
    plt.figure(figsize=(5, 5))
    sns.heatmap(cm, annot=True, fmt='d')
    plt.title('Confusion matrix @{:.2f}'.format(threshold))
    plt.ylabel('Actual label')
    plt.xlabel('Predicted label')
    print('Legitimate Transactions Detected (True Negatives): ', cm[0][0])
    print('Legitimate Transactions Incorrectly Detected (False Positives): ', cm[0][1])
    print('Fraudulent Transactions Missed (False Negatives): ', cm[1][0])
    print('Fraudulent Transactions Detected (True Positives): ', cm[1][1])
    print('Total Fraudulent Transactions: ', np.sum(cm[1]))
baseline_results = custom_method(
model.evaluate(test_features, test_labels, batch_size=BATCH_SIZE, verbose=0), imports='import seaborn as sns;import matplotlib.pyplot as plt;from sklearn.model_selection import train_test_split;import tempfile;import pandas as pd;from tensorflow import keras;import sklearn;from sklearn.preprocessing import StandardScaler;import os;from sklearn.metrics import confusion_matrix;import matplotlib as mpl;import tensorflow as tf;import numpy as np', function_to_run='obj.evaluate(*args, **kwargs)', method_object=eval('model'), object_signature='keras.Sequential', function_args=[eval('test_features'), eval('test_labels')], function_kwargs={'batch_size': eval('BATCH_SIZE'), 'verbose': eval('0')}, max_wait_secs=0, custom_class=None)
for (name, value) in zip(model.metrics_names, baseline_results):
    print(name, ': ', value)
print()
plot_cm(test_labels, test_predictions_baseline)
plot_cm(test_labels, test_predictions_baseline, threshold=0.1)
plot_cm(test_labels, test_predictions_baseline, threshold=0.01)

def plot_roc(name, labels, predictions, **kwargs):
    (fp, tp, _) = sklearn.metrics.roc_curve(labels, predictions)
    plt.plot(100 * fp, 100 * tp, label=name, linewidth=2, **kwargs)
    plt.xlabel('False positives [%]')
    plt.ylabel('True positives [%]')
    plt.xlim([-0.5, 20])
    plt.ylim([80, 100.5])
    plt.grid(True)
    ax = plt.gca()
    ax.set_aspect('equal')
plot_roc('Train Baseline', train_labels, train_predictions_baseline, color=colors[0])
plot_roc('Test Baseline', test_labels, test_predictions_baseline, color=colors[0], linestyle='--')
plt.legend(loc='lower right')

def plot_prc(name, labels, predictions, **kwargs):
    (precision, recall, _) = sklearn.metrics.precision_recall_curve(labels, predictions)
    plt.plot(precision, recall, label=name, linewidth=2, **kwargs)
    plt.xlabel('Precision')
    plt.ylabel('Recall')
    plt.grid(True)
    ax = plt.gca()
    ax.set_aspect('equal')
plot_prc('Train Baseline', train_labels, train_predictions_baseline, color=colors[0])
plot_prc('Test Baseline', test_labels, test_predictions_baseline, color=colors[0], linestyle='--')
plt.legend(loc='lower right')
weight_for_0 = 1 / neg * (total / 2.0)
weight_for_1 = 1 / pos * (total / 2.0)
class_weight = {0: weight_for_0, 1: weight_for_1}
print('Weight for class 0: {:.2f}'.format(weight_for_0))
print('Weight for class 1: {:.2f}'.format(weight_for_1))
weighted_model = custom_method(
make_model(), imports='import seaborn as sns;import matplotlib.pyplot as plt;from sklearn.model_selection import train_test_split;import tempfile;import pandas as pd;from tensorflow import keras;import sklearn;from sklearn.preprocessing import StandardScaler;import os;from sklearn.metrics import confusion_matrix;import matplotlib as mpl;import tensorflow as tf;import numpy as np', function_to_run='obj()', method_object=eval('make_model'), object_signature='keras.Sequential', function_args=[], function_kwargs={}, max_wait_secs=0, custom_class=None)
custom_method(
weighted_model.load_weights(initial_weights), imports='import seaborn as sns;import matplotlib.pyplot as plt;from sklearn.model_selection import train_test_split;import tempfile;import pandas as pd;from tensorflow import keras;import sklearn;from sklearn.preprocessing import StandardScaler;import os;from sklearn.metrics import confusion_matrix;import matplotlib as mpl;import tensorflow as tf;import numpy as np', function_to_run='obj.load_weights(*args)', method_object=eval('weighted_model'), object_signature='keras.Sequential', function_args=[eval('initial_weights')], function_kwargs={}, max_wait_secs=0, custom_class=None)
weighted_history = custom_method(
weighted_model.fit(train_features, train_labels, batch_size=BATCH_SIZE, epochs=EPOCHS, callbacks=[early_stopping], validation_data=(val_features, val_labels), class_weight=class_weight), imports='import seaborn as sns;import matplotlib.pyplot as plt;from sklearn.model_selection import train_test_split;import tempfile;import pandas as pd;from tensorflow import keras;import sklearn;from sklearn.preprocessing import StandardScaler;import os;from sklearn.metrics import confusion_matrix;import matplotlib as mpl;import tensorflow as tf;import numpy as np', function_to_run='obj.fit(*args, **kwargs)', method_object=eval('weighted_model'), object_signature='keras.Sequential', function_args=[eval('train_features'), eval('train_labels')], function_kwargs={'batch_size': eval('BATCH_SIZE'), 'epochs': eval('EPOCHS'), 'callbacks': eval('[early_stopping]'), 'validation_data': eval('(val_features, val_labels)'), 'class_weight': eval('class_weight')}, max_wait_secs=0, custom_class=None)
plot_metrics(weighted_history)
train_predictions_weighted = custom_method(
weighted_model.predict(train_features, batch_size=BATCH_SIZE), imports='import seaborn as sns;import matplotlib.pyplot as plt;from sklearn.model_selection import train_test_split;import tempfile;import pandas as pd;from tensorflow import keras;import sklearn;from sklearn.preprocessing import StandardScaler;import os;from sklearn.metrics import confusion_matrix;import matplotlib as mpl;import tensorflow as tf;import numpy as np', function_to_run='obj.predict(*args, **kwargs)', method_object=eval('weighted_model'), object_signature='keras.Sequential', function_args=[eval('train_features')], function_kwargs={'batch_size': eval('BATCH_SIZE')}, max_wait_secs=0, custom_class=None)
test_predictions_weighted = custom_method(
weighted_model.predict(test_features, batch_size=BATCH_SIZE), imports='import seaborn as sns;import matplotlib.pyplot as plt;from sklearn.model_selection import train_test_split;import tempfile;import pandas as pd;from tensorflow import keras;import sklearn;from sklearn.preprocessing import StandardScaler;import os;from sklearn.metrics import confusion_matrix;import matplotlib as mpl;import tensorflow as tf;import numpy as np', function_to_run='obj.predict(*args, **kwargs)', method_object=eval('weighted_model'), object_signature='keras.Sequential', function_args=[eval('test_features')], function_kwargs={'batch_size': eval('BATCH_SIZE')}, max_wait_secs=0, custom_class=None)
weighted_results = custom_method(
weighted_model.evaluate(test_features, test_labels, batch_size=BATCH_SIZE, verbose=0), imports='import seaborn as sns;import matplotlib.pyplot as plt;from sklearn.model_selection import train_test_split;import tempfile;import pandas as pd;from tensorflow import keras;import sklearn;from sklearn.preprocessing import StandardScaler;import os;from sklearn.metrics import confusion_matrix;import matplotlib as mpl;import tensorflow as tf;import numpy as np', function_to_run='obj.evaluate(*args, **kwargs)', method_object=eval('weighted_model'), object_signature='keras.Sequential', function_args=[eval('test_features'), eval('test_labels')], function_kwargs={'batch_size': eval('BATCH_SIZE'), 'verbose': eval('0')}, max_wait_secs=0, custom_class=None)
for (name, value) in zip(weighted_model.metrics_names, weighted_results):
    print(name, ': ', value)
print()
plot_cm(test_labels, test_predictions_weighted)
plot_roc('Train Baseline', train_labels, train_predictions_baseline, color=colors[0])
plot_roc('Test Baseline', test_labels, test_predictions_baseline, color=colors[0], linestyle='--')
plot_roc('Train Weighted', train_labels, train_predictions_weighted, color=colors[1])
plot_roc('Test Weighted', test_labels, test_predictions_weighted, color=colors[1], linestyle='--')
plt.legend(loc='lower right')
plot_prc('Train Baseline', train_labels, train_predictions_baseline, color=colors[0])
plot_prc('Test Baseline', test_labels, test_predictions_baseline, color=colors[0], linestyle='--')
plot_prc('Train Weighted', train_labels, train_predictions_weighted, color=colors[1])
plot_prc('Test Weighted', test_labels, test_predictions_weighted, color=colors[1], linestyle='--')
plt.legend(loc='lower right')
pos_features = train_features[bool_train_labels]
neg_features = train_features[~bool_train_labels]
pos_labels = train_labels[bool_train_labels]
neg_labels = train_labels[~bool_train_labels]
ids = np.arange(len(pos_features))
choices = np.random.choice(ids, len(neg_features))
res_pos_features = pos_features[choices]
res_pos_labels = pos_labels[choices]
res_pos_features.shape
resampled_features = np.concatenate([res_pos_features, neg_features], axis=0)
resampled_labels = np.concatenate([res_pos_labels, neg_labels], axis=0)
order = np.arange(len(resampled_labels))
np.random.shuffle(order)
resampled_features = resampled_features[order]
resampled_labels = resampled_labels[order]
resampled_features.shape
BUFFER_SIZE = 100000

def make_ds(features, labels):
    ds = custom_method(
    tf.data.Dataset.from_tensor_slices((features, labels)), imports='import seaborn as sns;import matplotlib.pyplot as plt;from sklearn.model_selection import train_test_split;import tempfile;import pandas as pd;from tensorflow import keras;import sklearn;from sklearn.preprocessing import StandardScaler;import os;from sklearn.metrics import confusion_matrix;import matplotlib as mpl;import tensorflow as tf;import numpy as np', function_to_run='tf.data.Dataset.from_tensor_slices(*args)', method_object=None, object_signature=None, function_args=[eval('(features, labels)')], function_kwargs={}, max_wait_secs=0)
    ds = ds.shuffle(BUFFER_SIZE).repeat()
    return ds
pos_ds = custom_method(
make_ds(pos_features, pos_labels), imports='import seaborn as sns;import matplotlib.pyplot as plt;from sklearn.model_selection import train_test_split;import tempfile;import pandas as pd;from tensorflow import keras;import sklearn;from sklearn.preprocessing import StandardScaler;import os;from sklearn.metrics import confusion_matrix;import matplotlib as mpl;import tensorflow as tf;import numpy as np', function_to_run='obj(*args)', method_object=eval('make_ds'), object_signature='tf.data.Dataset.from_tensor_slices', function_args=[eval('pos_features'), eval('pos_labels')], function_kwargs={}, max_wait_secs=0, custom_class=None)
neg_ds = custom_method(
make_ds(neg_features, neg_labels), imports='import seaborn as sns;import matplotlib.pyplot as plt;from sklearn.model_selection import train_test_split;import tempfile;import pandas as pd;from tensorflow import keras;import sklearn;from sklearn.preprocessing import StandardScaler;import os;from sklearn.metrics import confusion_matrix;import matplotlib as mpl;import tensorflow as tf;import numpy as np', function_to_run='obj(*args)', method_object=eval('make_ds'), object_signature='tf.data.Dataset.from_tensor_slices', function_args=[eval('neg_features'), eval('neg_labels')], function_kwargs={}, max_wait_secs=0, custom_class=None)
for (features, label) in custom_method(
pos_ds.take(1), imports='import seaborn as sns;import matplotlib.pyplot as plt;from sklearn.model_selection import train_test_split;import tempfile;import pandas as pd;from tensorflow import keras;import sklearn;from sklearn.preprocessing import StandardScaler;import os;from sklearn.metrics import confusion_matrix;import matplotlib as mpl;import tensorflow as tf;import numpy as np', function_to_run='obj.take(*args)', method_object=eval('pos_ds'), object_signature='tf.data.Dataset.from_tensor_slices', function_args=[eval('1')], function_kwargs={}, max_wait_secs=0, custom_class=None):
    print('Features:\n', features.numpy())
    print()
    print('Label: ', label.numpy())
resampled_ds = custom_method(
tf.data.Dataset.sample_from_datasets([pos_ds, neg_ds], weights=[0.5, 0.5]), imports='import seaborn as sns;import matplotlib.pyplot as plt;from sklearn.model_selection import train_test_split;import tempfile;import pandas as pd;from tensorflow import keras;import sklearn;from sklearn.preprocessing import StandardScaler;import os;from sklearn.metrics import confusion_matrix;import matplotlib as mpl;import tensorflow as tf;import numpy as np', function_to_run='tf.data.Dataset.sample_from_datasets(*args, **kwargs)', method_object=None, object_signature=None, function_args=[eval('[pos_ds, neg_ds]')], function_kwargs={'weights': eval('[0.5, 0.5]')}, max_wait_secs=0)
resampled_ds = resampled_ds.batch(BATCH_SIZE).prefetch(2)
for (features, label) in custom_method(
resampled_ds.take(1), imports='import seaborn as sns;import matplotlib.pyplot as plt;from sklearn.model_selection import train_test_split;import tempfile;import pandas as pd;from tensorflow import keras;import sklearn;from sklearn.preprocessing import StandardScaler;import os;from sklearn.metrics import confusion_matrix;import matplotlib as mpl;import tensorflow as tf;import numpy as np', function_to_run='obj.take(*args)', method_object=eval('resampled_ds'), object_signature='tf.data.Dataset.from_tensor_slices', function_args=[eval('1')], function_kwargs={}, max_wait_secs=0, custom_class=None):
    print(label.numpy().mean())
resampled_steps_per_epoch = np.ceil(2.0 * neg / BATCH_SIZE)
resampled_steps_per_epoch
resampled_model = custom_method(
make_model(), imports='import seaborn as sns;import matplotlib.pyplot as plt;from sklearn.model_selection import train_test_split;import tempfile;import pandas as pd;from tensorflow import keras;import sklearn;from sklearn.preprocessing import StandardScaler;import os;from sklearn.metrics import confusion_matrix;import matplotlib as mpl;import tensorflow as tf;import numpy as np', function_to_run='obj()', method_object=eval('make_model'), object_signature='keras.Sequential', function_args=[], function_kwargs={}, max_wait_secs=0, custom_class=None)
custom_method(
resampled_model.load_weights(initial_weights), imports='import seaborn as sns;import matplotlib.pyplot as plt;from sklearn.model_selection import train_test_split;import tempfile;import pandas as pd;from tensorflow import keras;import sklearn;from sklearn.preprocessing import StandardScaler;import os;from sklearn.metrics import confusion_matrix;import matplotlib as mpl;import tensorflow as tf;import numpy as np', function_to_run='obj.load_weights(*args)', method_object=eval('resampled_model'), object_signature='keras.Sequential', function_args=[eval('initial_weights')], function_kwargs={}, max_wait_secs=0, custom_class=None)
output_layer = resampled_model.layers[-1]
output_layer.bias.assign([0])
val_ds = custom_method(
tf.data.Dataset.from_tensor_slices((val_features, val_labels)).cache(), imports='import seaborn as sns;import matplotlib.pyplot as plt;from sklearn.model_selection import train_test_split;import tempfile;import pandas as pd;from tensorflow import keras;import sklearn;from sklearn.preprocessing import StandardScaler;import os;from sklearn.metrics import confusion_matrix;import matplotlib as mpl;import tensorflow as tf;import numpy as np', function_to_run='tf.data.Dataset.from_tensor_slices((val_features, val_labels)).cache()', method_object=None, object_signature=None, function_args=[], function_kwargs={}, max_wait_secs=0)
val_ds = val_ds.batch(BATCH_SIZE).prefetch(2)
resampled_history = custom_method(
resampled_model.fit(resampled_ds, epochs=EPOCHS, steps_per_epoch=resampled_steps_per_epoch, callbacks=[early_stopping], validation_data=val_ds), imports='import seaborn as sns;import matplotlib.pyplot as plt;from sklearn.model_selection import train_test_split;import tempfile;import pandas as pd;from tensorflow import keras;import sklearn;from sklearn.preprocessing import StandardScaler;import os;from sklearn.metrics import confusion_matrix;import matplotlib as mpl;import tensorflow as tf;import numpy as np', function_to_run='obj.fit(*args, **kwargs)', method_object=eval('resampled_model'), object_signature='keras.Sequential', function_args=[eval('resampled_ds')], function_kwargs={'epochs': eval('EPOCHS'), 'steps_per_epoch': eval('resampled_steps_per_epoch'), 'callbacks': eval('[early_stopping]'), 'validation_data': eval('val_ds')}, max_wait_secs=0, custom_class=None)
plot_metrics(resampled_history)
resampled_model = custom_method(
make_model(), imports='import seaborn as sns;import matplotlib.pyplot as plt;from sklearn.model_selection import train_test_split;import tempfile;import pandas as pd;from tensorflow import keras;import sklearn;from sklearn.preprocessing import StandardScaler;import os;from sklearn.metrics import confusion_matrix;import matplotlib as mpl;import tensorflow as tf;import numpy as np', function_to_run='obj()', method_object=eval('make_model'), object_signature='keras.Sequential', function_args=[], function_kwargs={}, max_wait_secs=0, custom_class=None)
custom_method(
resampled_model.load_weights(initial_weights), imports='import seaborn as sns;import matplotlib.pyplot as plt;from sklearn.model_selection import train_test_split;import tempfile;import pandas as pd;from tensorflow import keras;import sklearn;from sklearn.preprocessing import StandardScaler;import os;from sklearn.metrics import confusion_matrix;import matplotlib as mpl;import tensorflow as tf;import numpy as np', function_to_run='obj.load_weights(*args)', method_object=eval('resampled_model'), object_signature='keras.Sequential', function_args=[eval('initial_weights')], function_kwargs={}, max_wait_secs=0, custom_class=None)
output_layer = resampled_model.layers[-1]
output_layer.bias.assign([0])
resampled_history = custom_method(
resampled_model.fit(resampled_ds, steps_per_epoch=20, epochs=10 * EPOCHS, callbacks=[early_stopping], validation_data=val_ds), imports='import seaborn as sns;import matplotlib.pyplot as plt;from sklearn.model_selection import train_test_split;import tempfile;import pandas as pd;from tensorflow import keras;import sklearn;from sklearn.preprocessing import StandardScaler;import os;from sklearn.metrics import confusion_matrix;import matplotlib as mpl;import tensorflow as tf;import numpy as np', function_to_run='obj.fit(*args, **kwargs)', method_object=eval('resampled_model'), object_signature='keras.Sequential', function_args=[eval('resampled_ds')], function_kwargs={'steps_per_epoch': eval('20'), 'epochs': eval('10*EPOCHS'), 'callbacks': eval('[early_stopping]'), 'validation_data': eval('val_ds')}, max_wait_secs=0, custom_class=None)
plot_metrics(resampled_history)
train_predictions_resampled = custom_method(
resampled_model.predict(train_features, batch_size=BATCH_SIZE), imports='import seaborn as sns;import matplotlib.pyplot as plt;from sklearn.model_selection import train_test_split;import tempfile;import pandas as pd;from tensorflow import keras;import sklearn;from sklearn.preprocessing import StandardScaler;import os;from sklearn.metrics import confusion_matrix;import matplotlib as mpl;import tensorflow as tf;import numpy as np', function_to_run='obj.predict(*args, **kwargs)', method_object=eval('resampled_model'), object_signature='keras.Sequential', function_args=[eval('train_features')], function_kwargs={'batch_size': eval('BATCH_SIZE')}, max_wait_secs=0, custom_class=None)
test_predictions_resampled = custom_method(
resampled_model.predict(test_features, batch_size=BATCH_SIZE), imports='import seaborn as sns;import matplotlib.pyplot as plt;from sklearn.model_selection import train_test_split;import tempfile;import pandas as pd;from tensorflow import keras;import sklearn;from sklearn.preprocessing import StandardScaler;import os;from sklearn.metrics import confusion_matrix;import matplotlib as mpl;import tensorflow as tf;import numpy as np', function_to_run='obj.predict(*args, **kwargs)', method_object=eval('resampled_model'), object_signature='keras.Sequential', function_args=[eval('test_features')], function_kwargs={'batch_size': eval('BATCH_SIZE')}, max_wait_secs=0, custom_class=None)
resampled_results = custom_method(
resampled_model.evaluate(test_features, test_labels, batch_size=BATCH_SIZE, verbose=0), imports='import seaborn as sns;import matplotlib.pyplot as plt;from sklearn.model_selection import train_test_split;import tempfile;import pandas as pd;from tensorflow import keras;import sklearn;from sklearn.preprocessing import StandardScaler;import os;from sklearn.metrics import confusion_matrix;import matplotlib as mpl;import tensorflow as tf;import numpy as np', function_to_run='obj.evaluate(*args, **kwargs)', method_object=eval('resampled_model'), object_signature='keras.Sequential', function_args=[eval('test_features'), eval('test_labels')], function_kwargs={'batch_size': eval('BATCH_SIZE'), 'verbose': eval('0')}, max_wait_secs=0, custom_class=None)
for (name, value) in zip(resampled_model.metrics_names, resampled_results):
    print(name, ': ', value)
print()
plot_cm(test_labels, test_predictions_resampled)
plot_roc('Train Baseline', train_labels, train_predictions_baseline, color=colors[0])
plot_roc('Test Baseline', test_labels, test_predictions_baseline, color=colors[0], linestyle='--')
plot_roc('Train Weighted', train_labels, train_predictions_weighted, color=colors[1])
plot_roc('Test Weighted', test_labels, test_predictions_weighted, color=colors[1], linestyle='--')
plot_roc('Train Resampled', train_labels, train_predictions_resampled, color=colors[2])
plot_roc('Test Resampled', test_labels, test_predictions_resampled, color=colors[2], linestyle='--')
plt.legend(loc='lower right')
plot_prc('Train Baseline', train_labels, train_predictions_baseline, color=colors[0])
plot_prc('Test Baseline', test_labels, test_predictions_baseline, color=colors[0], linestyle='--')
plot_prc('Train Weighted', train_labels, train_predictions_weighted, color=colors[1])
plot_prc('Test Weighted', test_labels, test_predictions_weighted, color=colors[1], linestyle='--')
plot_prc('Train Resampled', train_labels, train_predictions_resampled, color=colors[2])
plot_prc('Test Resampled', test_labels, test_predictions_resampled, color=colors[2], linestyle='--')
plt.legend(loc='lower right')
