import os
import tensorflow as tf
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt
import sys
from tool.patching.patching_config import EXPERIMENT_DIR
from tool.measurement.execution import before_execution as before_execution_INSERTED_INTO_SCRIPT
from tool.measurement.execution import after_execution as after_execution_INSERTED_INTO_SCRIPT
experiment_number = sys.argv[1]
experiment_project = sys.argv[2]
EXPERIMENT_FILE_PATH = EXPERIMENT_DIR / 'local-execution' / experiment_project / f'experiment-{experiment_number}.json'
print('TensorFlow version: {}'.format(tf.__version__))
print('TensorFlow Datasets version: ', tfds.__version__)
(ds_preview, info) = tfds.load('penguins/simple', split='train', with_info=True)
df = tfds.as_dataframe(ds_preview.take(5), info)
print(df)
print(info.features)
class_names = ['Adélie', 'Chinstrap', 'Gentoo']
(ds_split, info) = tfds.load('penguins/processed', split=['train[:20%]', 'train[20%:]'], as_supervised=True, with_info=True)
ds_test = ds_split[0]
ds_train = ds_split[1]
assert isinstance(ds_test, tf.data.Dataset)
print(info.features)
df_test = tfds.as_dataframe(ds_test.take(5), info)
print('Test dataset sample: ')
print(df_test)
df_train = tfds.as_dataframe(ds_train.take(5), info)
print('Train dataset sample: ')
print(df_train)
ds_train_batch = ds_train.batch(32)
(features, labels) = next(iter(ds_train_batch))
print(features)
print(labels)
plt.scatter(features[:, 0], features[:, 2], c=labels, cmap='viridis')
plt.xlabel('Body Mass')
plt.ylabel('Culmen Length')
plt.show()
start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT()
model = tf.keras.Sequential([tf.keras.layers.Dense(10, activation=tf.nn.relu, input_shape=(4,)), tf.keras.layers.Dense(10, activation=tf.nn.relu), tf.keras.layers.Dense(3)])
after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tf.keras.Sequential()', method_object=None, function_args=[[tf.keras.layers.Dense(10, activation=tf.nn.relu, input_shape=(4,)), tf.keras.layers.Dense(10, activation=tf.nn.relu), tf.keras.layers.Dense(3)]], function_kwargs=None)
start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT()
predictions = model(features)
after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tf.keras.Sequential()', method_object=model, function_args=[features], function_kwargs=None)
predictions[:5]
start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT()
tf.nn.softmax(predictions[:5])
after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tf.nn.softmax()', method_object=None, function_args=[predictions[:5]], function_kwargs=None)
print('Prediction: {}'.format(tf.math.argmax(predictions, axis=1)))
print('    Labels: {}'.format(labels))
start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT()
loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tf.keras.losses.SparseCategoricalCrossentropy()', method_object=None, function_args=None, function_kwargs={'from_logits': True})

def loss(model, x, y, training):
    start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT()
    y_ = model(x, training=training)
    after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tf.keras.Sequential()', method_object=model, function_args=[x], function_kwargs={'training': training})
    return loss_object(y_true=y, y_pred=y_)
l = loss(model, features, labels, training=False)
print('Loss test: {}'.format(l))

def grad(model, inputs, targets):
    with tf.GradientTape() as tape:
        loss_value = loss(model, inputs, targets, training=True)
    return (loss_value, tape.gradient(loss_value, model.trainable_variables))
start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT()
optimizer = tf.keras.optimizers.SGD(learning_rate=0.01)
after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tf.keras.optimizers.SGD()', method_object=None, function_args=None, function_kwargs={'learning_rate': 0.01})
(loss_value, grads) = grad(model, features, labels)
print('Step: {}, Initial Loss: {}'.format(optimizer.iterations.numpy(), loss_value.numpy()))
start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT()
optimizer.apply_gradients(zip(grads, model.trainable_variables))
after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tf.keras.optimizers.SGD.apply_gradients()', method_object=optimizer, function_args=[zip(grads, model.trainable_variables)], function_kwargs=None)
print('Step: {},         Loss: {}'.format(optimizer.iterations.numpy(), loss(model, features, labels, training=True).numpy()))
train_loss_results = []
train_accuracy_results = []
num_epochs = 201
for epoch in range(num_epochs):
    start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT()
    epoch_loss_avg = tf.keras.metrics.Mean()
    after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tf.keras.metrics.Mean()', method_object=None, function_args=None, function_kwargs=None)
    start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT()
    epoch_accuracy = tf.keras.metrics.SparseCategoricalAccuracy()
    after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tf.keras.metrics.SparseCategoricalAccuracy()', method_object=None, function_args=None, function_kwargs=None)
    for (x, y) in ds_train_batch:
        (loss_value, grads) = grad(model, x, y)
        start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT()
        optimizer.apply_gradients(zip(grads, model.trainable_variables))
        after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tf.keras.optimizers.SGD.apply_gradients()', method_object=optimizer, function_args=[zip(grads, model.trainable_variables)], function_kwargs=None)
        start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT()
        epoch_loss_avg.update_state(loss_value)
        after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tf.keras.metrics.Mean.update_state()', method_object=epoch_loss_avg, function_args=[loss_value], function_kwargs=None)
        start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT()
        epoch_accuracy.update_state(y, model(x, training=True))
        after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tf.keras.metrics.SparseCategoricalAccuracy.update_state()', method_object=epoch_accuracy, function_args=[y, model(x, training=True)], function_kwargs=None)
    train_loss_results.append(epoch_loss_avg.result())
    train_accuracy_results.append(epoch_accuracy.result())
    if epoch % 50 == 0:
        print('Epoch {:03d}: Loss: {:.3f}, Accuracy: {:.3%}'.format(epoch, epoch_loss_avg.result(), epoch_accuracy.result()))
(fig, axes) = plt.subplots(2, sharex=True, figsize=(12, 8))
fig.suptitle('Training Metrics')
axes[0].set_ylabel('Loss', fontsize=14)
axes[0].plot(train_loss_results)
axes[1].set_ylabel('Accuracy', fontsize=14)
axes[1].set_xlabel('Epoch', fontsize=14)
axes[1].plot(train_accuracy_results)
plt.show()
start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT()
test_accuracy = tf.keras.metrics.Accuracy()
after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tf.keras.metrics.Accuracy()', method_object=None, function_args=None, function_kwargs=None)
ds_test_batch = ds_test.batch(10)
for (x, y) in ds_test_batch:
    start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT()
    logits = model(x, training=False)
    after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tf.keras.Sequential()', method_object=model, function_args=[x], function_kwargs={'training': False})
    start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT()
    prediction = tf.math.argmax(logits, axis=1, output_type=tf.int64)
    after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tf.math.argmax()', method_object=None, function_args=[logits], function_kwargs={'axis': 1, 'output_type': tf.int64})
    start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT()
    test_accuracy(prediction, y)
    after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tf.keras.metrics.Accuracy()', method_object=test_accuracy, function_args=[prediction, y], function_kwargs=None)
print('Test set accuracy: {:.3%}'.format(test_accuracy.result()))
start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT()
tf.stack([y, prediction], axis=1)
after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tf.stack()', method_object=None, function_args=[[y, prediction]], function_kwargs={'axis': 1})
start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT()
predict_dataset = tf.convert_to_tensor([[0.3, 0.8, 0.4, 0.5], [0.4, 0.1, 0.8, 0.5], [0.7, 0.9, 0.8, 0.4]])
after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tf.convert_to_tensor()', method_object=None, function_args=[[[0.3, 0.8, 0.4, 0.5], [0.4, 0.1, 0.8, 0.5], [0.7, 0.9, 0.8, 0.4]]], function_kwargs=None)
start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT()
predictions = model(predict_dataset, training=False)
after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tf.keras.Sequential()', method_object=model, function_args=[predict_dataset], function_kwargs={'training': False})
for (i, logits) in enumerate(predictions):
    start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT()
    class_idx = tf.math.argmax(logits).numpy()
    after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tf.math.argmax(logits).numpy()', method_object=None, function_args=None, function_kwargs=None)
    p = tf.nn.softmax(logits)[class_idx]
    name = class_names[class_idx]
    print('Example {} prediction: {} ({:4.1f}%)'.format(i, name, 100 * p))
