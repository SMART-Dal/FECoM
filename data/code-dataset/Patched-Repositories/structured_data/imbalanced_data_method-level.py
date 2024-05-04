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
import sys
from fecom.patching.patching_config import EXPERIMENT_DIR
from fecom.measurement.execution import before_execution as before_execution_INSERTED_INTO_SCRIPT
from fecom.measurement.execution import after_execution as after_execution_INSERTED_INTO_SCRIPT
from fecom.experiment.experiment_kinds import ExperimentKinds
experiment_number = sys.argv[1]
experiment_project = sys.argv[2]
EXPERIMENT_FILE_PATH = EXPERIMENT_DIR / ExperimentKinds.METHOD_LEVEL.value / experiment_project / f'experiment-{experiment_number}.json'
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
METRICS = [keras.metrics.BinaryCrossentropy(name='cross entropy'), keras.metrics.MeanSquaredError(name='Brier score'), keras.metrics.TruePositives(name='tp'), keras.metrics.FalsePositives(name='fp'), keras.metrics.TrueNegatives(name='tn'), keras.metrics.FalseNegatives(name='fn'), keras.metrics.BinaryAccuracy(name='accuracy'), keras.metrics.Precision(name='precision'), keras.metrics.Recall(name='recall'), keras.metrics.AUC(name='auc'), keras.metrics.AUC(name='prc', curve='PR')]

def make_model(metrics=METRICS, output_bias=None):
    if output_bias is not None:
        start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT(experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tensorflow.keras.initializers.Constant()')
        output_bias = tf.keras.initializers.Constant(output_bias)
        after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tensorflow.keras.initializers.Constant()', method_object=None, function_args=[output_bias], function_kwargs=None)
    start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT(experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tensorflow.keras.Sequential()')
    model = keras.Sequential([keras.layers.Dense(16, activation='relu', input_shape=(train_features.shape[-1],)), keras.layers.Dropout(0.5), keras.layers.Dense(1, activation='sigmoid', bias_initializer=output_bias)])
    after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tensorflow.keras.Sequential()', method_object=None, function_args=[[keras.layers.Dense(16, activation='relu', input_shape=(train_features.shape[-1],)), keras.layers.Dropout(0.5), keras.layers.Dense(1, activation='sigmoid', bias_initializer=output_bias)]], function_kwargs=None)
    start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT(experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tensorflow.keras.Sequential.compile()')
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001), loss=keras.losses.BinaryCrossentropy(), metrics=metrics)
    after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tensorflow.keras.Sequential.compile()', method_object=model, function_args=None, function_kwargs={'optimizer': keras.optimizers.Adam(learning_rate=0.001), 'loss': keras.losses.BinaryCrossentropy(), 'metrics': metrics})
    return model
EPOCHS = 100
BATCH_SIZE = 2048
start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT(experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tensorflow.keras.callbacks.EarlyStopping()')
early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_prc', verbose=1, patience=10, mode='max', restore_best_weights=True)
after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tensorflow.keras.callbacks.EarlyStopping()', method_object=None, function_args=None, function_kwargs={'monitor': 'val_prc', 'verbose': 1, 'patience': 10, 'mode': 'max', 'restore_best_weights': True})
model = make_model()
model.summary()
model.predict(train_features[:10])
results = model.evaluate(train_features, train_labels, batch_size=BATCH_SIZE, verbose=0)
print('Loss: {:0.4f}'.format(results[0]))
initial_bias = np.log([pos / neg])
initial_bias
model = make_model(output_bias=initial_bias)
model.predict(train_features[:10])
results = model.evaluate(train_features, train_labels, batch_size=BATCH_SIZE, verbose=0)
print('Loss: {:0.4f}'.format(results[0]))
initial_weights = os.path.join(tempfile.mkdtemp(), 'initial_weights')
model.save_weights(initial_weights)
model = make_model()
model.load_weights(initial_weights)
model.layers[-1].bias.assign([0.0])
zero_bias_history = model.fit(train_features, train_labels, batch_size=BATCH_SIZE, epochs=20, validation_data=(val_features, val_labels), verbose=0)
model = make_model()
model.load_weights(initial_weights)
careful_bias_history = model.fit(train_features, train_labels, batch_size=BATCH_SIZE, epochs=20, validation_data=(val_features, val_labels), verbose=0)

def plot_loss(history, label, n):
    plt.semilogy(history.epoch, history.history['loss'], color=colors[n], label='Train ' + label)
    plt.semilogy(history.epoch, history.history['val_loss'], color=colors[n], label='Val ' + label, linestyle='--')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
plot_loss(zero_bias_history, 'Zero Bias', 0)
plot_loss(careful_bias_history, 'Careful Bias', 1)
model = make_model()
model.load_weights(initial_weights)
baseline_history = model.fit(train_features, train_labels, batch_size=BATCH_SIZE, epochs=EPOCHS, callbacks=[early_stopping], validation_data=(val_features, val_labels))

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
train_predictions_baseline = model.predict(train_features, batch_size=BATCH_SIZE)
test_predictions_baseline = model.predict(test_features, batch_size=BATCH_SIZE)

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
baseline_results = model.evaluate(test_features, test_labels, batch_size=BATCH_SIZE, verbose=0)
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
weighted_model = make_model()
weighted_model.load_weights(initial_weights)
weighted_history = weighted_model.fit(train_features, train_labels, batch_size=BATCH_SIZE, epochs=EPOCHS, callbacks=[early_stopping], validation_data=(val_features, val_labels), class_weight=class_weight)
plot_metrics(weighted_history)
train_predictions_weighted = weighted_model.predict(train_features, batch_size=BATCH_SIZE)
test_predictions_weighted = weighted_model.predict(test_features, batch_size=BATCH_SIZE)
weighted_results = weighted_model.evaluate(test_features, test_labels, batch_size=BATCH_SIZE, verbose=0)
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
    start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT(experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tensorflow.data.Dataset.from_tensor_slices()')
    ds = tf.data.Dataset.from_tensor_slices((features, labels))
    after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tensorflow.data.Dataset.from_tensor_slices()', method_object=None, function_args=[(features, labels)], function_kwargs=None)
    start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT(experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tensorflow.data.Dataset.from_tensor_slices.shuffle(BUFFER_SIZE).repeat()')
    ds = ds.shuffle(BUFFER_SIZE).repeat()
    after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tensorflow.data.Dataset.from_tensor_slices.shuffle(BUFFER_SIZE).repeat()', method_object=ds, function_args=None, function_kwargs=None)
    return ds
pos_ds = make_ds(pos_features, pos_labels)
neg_ds = make_ds(neg_features, neg_labels)
for (features, label) in pos_ds.take(1):
    print('Features:\n', features.numpy())
    print()
    print('Label: ', label.numpy())
start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT(experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tensorflow.data.Dataset.sample_from_datasets()')
resampled_ds = tf.data.Dataset.sample_from_datasets([pos_ds, neg_ds], weights=[0.5, 0.5])
after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tensorflow.data.Dataset.sample_from_datasets()', method_object=None, function_args=[[pos_ds, neg_ds]], function_kwargs={'weights': [0.5, 0.5]})
start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT(experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tensorflow.data.Dataset.sample_from_datasets.batch(BATCH_SIZE).prefetch()')
resampled_ds = resampled_ds.batch(BATCH_SIZE).prefetch(2)
after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tensorflow.data.Dataset.sample_from_datasets.batch(BATCH_SIZE).prefetch()', method_object=resampled_ds, function_args=[2], function_kwargs=None)
for (features, label) in resampled_ds.take(1):
    print(label.numpy().mean())
resampled_steps_per_epoch = np.ceil(2.0 * neg / BATCH_SIZE)
resampled_steps_per_epoch
resampled_model = make_model()
resampled_model.load_weights(initial_weights)
output_layer = resampled_model.layers[-1]
output_layer.bias.assign([0])
start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT(experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tensorflow.data.Dataset.from_tensor_slices((val_features, val_labels)).cache()')
val_ds = tf.data.Dataset.from_tensor_slices((val_features, val_labels)).cache()
after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tensorflow.data.Dataset.from_tensor_slices((val_features, val_labels)).cache()', method_object=None, function_args=None, function_kwargs=None)
start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT(experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tensorflow.data.Dataset.from_tensor_slices((val_features, val_labels)).cache.batch(BATCH_SIZE).prefetch()')
val_ds = val_ds.batch(BATCH_SIZE).prefetch(2)
after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tensorflow.data.Dataset.from_tensor_slices((val_features, val_labels)).cache.batch(BATCH_SIZE).prefetch()', method_object=val_ds, function_args=[2], function_kwargs=None)
resampled_history = resampled_model.fit(resampled_ds, epochs=EPOCHS, steps_per_epoch=resampled_steps_per_epoch, callbacks=[early_stopping], validation_data=val_ds)
plot_metrics(resampled_history)
resampled_model = make_model()
resampled_model.load_weights(initial_weights)
output_layer = resampled_model.layers[-1]
output_layer.bias.assign([0])
resampled_history = resampled_model.fit(resampled_ds, steps_per_epoch=20, epochs=10 * EPOCHS, callbacks=[early_stopping], validation_data=val_ds)
plot_metrics(resampled_history)
train_predictions_resampled = resampled_model.predict(train_features, batch_size=BATCH_SIZE)
test_predictions_resampled = resampled_model.predict(test_features, batch_size=BATCH_SIZE)
resampled_results = resampled_model.evaluate(test_features, test_labels, batch_size=BATCH_SIZE, verbose=0)
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
