import os
import tensorflow as tf
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt
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
    result = send_request(imports=imports, function_to_run=function_to_run, function_args=function_args, function_kwargs=function_kwargs, max_wait_secs=MAX_WAIT_S, wait_after_run_secs=WAIT_AFTER_RUN_S, method_object=method_object, object_signature=object_signature, custom_class=custom_class, experiment_file_path=EXPERIMENT_FILE_PATH)
    return func
print('TensorFlow version: {}'.format(tf.__version__))
print('TensorFlow Datasets version: ', tfds.__version__)
(ds_preview, info) = custom_method(
tfds.load('penguins/simple', split='train', with_info=True), imports='import tensorflow as tf;import matplotlib.pyplot as plt;import os;import tensorflow_datasets as tfds', function_to_run='tfds.load(*args, **kwargs)', method_object=None, object_signature=None, function_args=[eval("'penguins/simple'")], function_kwargs={'split': eval("'train'"), 'with_info': eval('True')})
df = custom_method(
tfds.as_dataframe(ds_preview.take(5), info), imports='import tensorflow as tf;import matplotlib.pyplot as plt;import os;import tensorflow_datasets as tfds', function_to_run='tfds.as_dataframe(*args)', method_object=None, object_signature=None, function_args=[eval('ds_preview.take(5)'), eval('info')], function_kwargs={})
print(df)
print(info.features)
class_names = ['Adélie', 'Chinstrap', 'Gentoo']
(ds_split, info) = custom_method(
tfds.load('penguins/processed', split=['train[:20%]', 'train[20%:]'], as_supervised=True, with_info=True), imports='import tensorflow as tf;import matplotlib.pyplot as plt;import os;import tensorflow_datasets as tfds', function_to_run='tfds.load(*args, **kwargs)', method_object=None, object_signature=None, function_args=[eval('"penguins/processed"')], function_kwargs={'split': eval("['train[:20%]', 'train[20%:]']"), 'as_supervised': eval('True'), 'with_info': eval('True')})
ds_test = ds_split[0]
ds_train = ds_split[1]
assert isinstance(ds_test, tf.data.Dataset)
print(info.features)
df_test = custom_method(
tfds.as_dataframe(ds_test.take(5), info), imports='import tensorflow as tf;import matplotlib.pyplot as plt;import os;import tensorflow_datasets as tfds', function_to_run='tfds.as_dataframe(*args)', method_object=None, object_signature=None, function_args=[eval('ds_test.take(5)'), eval('info')], function_kwargs={})
print('Test dataset sample: ')
print(df_test)
df_train = custom_method(
tfds.as_dataframe(ds_train.take(5), info), imports='import tensorflow as tf;import matplotlib.pyplot as plt;import os;import tensorflow_datasets as tfds', function_to_run='tfds.as_dataframe(*args)', method_object=None, object_signature=None, function_args=[eval('ds_train.take(5)'), eval('info')], function_kwargs={})
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
model = custom_method(
tf.keras.Sequential([tf.keras.layers.Dense(10, activation=tf.nn.relu, input_shape=(4,)), tf.keras.layers.Dense(10, activation=tf.nn.relu), tf.keras.layers.Dense(3)]), imports='import tensorflow as tf;import matplotlib.pyplot as plt;import os;import tensorflow_datasets as tfds', function_to_run='tf.keras.Sequential(*args)', method_object=None, object_signature=None, function_args=[eval('[\n  tf.keras.layers.Dense(10, activation=tf.nn.relu, input_shape=(4,)),  # input shape required\n  tf.keras.layers.Dense(10, activation=tf.nn.relu),\n  tf.keras.layers.Dense(3)\n]')], function_kwargs={})
predictions = custom_method(
model(features), imports='import tensorflow as tf;import matplotlib.pyplot as plt;import os;import tensorflow_datasets as tfds', function_to_run='obj(*args)', method_object=eval('model'), object_signature='tf.keras.Sequential', function_args=[eval('features')], function_kwargs={}, custom_class=None)
predictions[:5]
custom_method(
tf.nn.softmax(predictions[:5]), imports='import tensorflow as tf;import matplotlib.pyplot as plt;import os;import tensorflow_datasets as tfds', function_to_run='tf.nn.softmax(*args)', method_object=None, object_signature=None, function_args=[eval('predictions[:5]')], function_kwargs={})
print('Prediction: {}'.format(tf.math.argmax(predictions, axis=1)))
print('    Labels: {}'.format(labels))
loss_object = custom_method(
tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), imports='import tensorflow as tf;import matplotlib.pyplot as plt;import os;import tensorflow_datasets as tfds', function_to_run='tf.keras.losses.SparseCategoricalCrossentropy(**kwargs)', method_object=None, object_signature=None, function_args=[], function_kwargs={'from_logits': eval('True')})

def loss(model, x, y, training):
    y_ = custom_method(
    model(x, training=training), imports='import tensorflow as tf;import matplotlib.pyplot as plt;import os;import tensorflow_datasets as tfds', function_to_run='obj(*args, **kwargs)', method_object=eval('model'), object_signature='tf.keras.Sequential', function_args=[eval('x')], function_kwargs={'training': eval('training')}, custom_class=None)
    return custom_method(
    loss_object(y_true=y, y_pred=y_), imports='import tensorflow as tf;import matplotlib.pyplot as plt;import os;import tensorflow_datasets as tfds', function_to_run='obj(**kwargs)', method_object=eval('loss_object'), object_signature='tf.keras.losses.SparseCategoricalCrossentropy', function_args=[], function_kwargs={'y_true': eval('y'), 'y_pred': eval('y_')}, custom_class=None)
l = loss(model, features, labels, training=False)
print('Loss test: {}'.format(l))

def grad(model, inputs, targets):
    with custom_method(
    tf.GradientTape(), imports='import tensorflow as tf;import matplotlib.pyplot as plt;import os;import tensorflow_datasets as tfds', function_to_run='tf.GradientTape()', method_object=None, object_signature=None, function_args=[], function_kwargs={}) as tape:
        loss_value = loss(model, inputs, targets, training=True)
    return (loss_value, tape.gradient(loss_value, model.trainable_variables))
optimizer = custom_method(
tf.keras.optimizers.SGD(learning_rate=0.01), imports='import tensorflow as tf;import matplotlib.pyplot as plt;import os;import tensorflow_datasets as tfds', function_to_run='tf.keras.optimizers.SGD(**kwargs)', method_object=None, object_signature=None, function_args=[], function_kwargs={'learning_rate': eval('0.01')})
(loss_value, grads) = grad(model, features, labels)
print('Step: {}, Initial Loss: {}'.format(optimizer.iterations.numpy(), loss_value.numpy()))
custom_method(
optimizer.apply_gradients(zip(grads, model.trainable_variables)), imports='import tensorflow as tf;import matplotlib.pyplot as plt;import os;import tensorflow_datasets as tfds', function_to_run='obj.apply_gradients(*args)', method_object=eval('optimizer'), object_signature='tf.keras.optimizers.SGD', function_args=[eval('zip(grads, model.trainable_variables)')], function_kwargs={}, custom_class=None)
print('Step: {},         Loss: {}'.format(optimizer.iterations.numpy(), loss(model, features, labels, training=True).numpy()))
train_loss_results = []
train_accuracy_results = []
num_epochs = 201
for epoch in range(num_epochs):
    epoch_loss_avg = custom_method(
    tf.keras.metrics.Mean(), imports='import tensorflow as tf;import matplotlib.pyplot as plt;import os;import tensorflow_datasets as tfds', function_to_run='tf.keras.metrics.Mean()', method_object=None, object_signature=None, function_args=[], function_kwargs={})
    epoch_accuracy = custom_method(
    tf.keras.metrics.SparseCategoricalAccuracy(), imports='import tensorflow as tf;import matplotlib.pyplot as plt;import os;import tensorflow_datasets as tfds', function_to_run='tf.keras.metrics.SparseCategoricalAccuracy()', method_object=None, object_signature=None, function_args=[], function_kwargs={})
    for (x, y) in ds_train_batch:
        (loss_value, grads) = grad(model, x, y)
        custom_method(
        optimizer.apply_gradients(zip(grads, model.trainable_variables)), imports='import tensorflow as tf;import matplotlib.pyplot as plt;import os;import tensorflow_datasets as tfds', function_to_run='obj.apply_gradients(*args)', method_object=eval('optimizer'), object_signature='tf.keras.optimizers.SGD', function_args=[eval('zip(grads, model.trainable_variables)')], function_kwargs={}, custom_class=None)
        custom_method(
        epoch_loss_avg.update_state(loss_value), imports='import tensorflow as tf;import matplotlib.pyplot as plt;import os;import tensorflow_datasets as tfds', function_to_run='obj.update_state(*args)', method_object=eval('epoch_loss_avg'), object_signature='tf.keras.metrics.Mean', function_args=[eval('loss_value')], function_kwargs={}, custom_class=None)
        custom_method(
        epoch_accuracy.update_state(y, model(x, training=True)), imports='import tensorflow as tf;import matplotlib.pyplot as plt;import os;import tensorflow_datasets as tfds', function_to_run='obj.update_state(*args)', method_object=eval('epoch_accuracy'), object_signature='tf.keras.metrics.SparseCategoricalAccuracy', function_args=[eval('y'), eval('model(x, training=True)')], function_kwargs={}, custom_class=None)
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
test_accuracy = custom_method(
tf.keras.metrics.Accuracy(), imports='import tensorflow as tf;import matplotlib.pyplot as plt;import os;import tensorflow_datasets as tfds', function_to_run='tf.keras.metrics.Accuracy()', method_object=None, object_signature=None, function_args=[], function_kwargs={})
ds_test_batch = ds_test.batch(10)
for (x, y) in ds_test_batch:
    logits = custom_method(
    model(x, training=False), imports='import tensorflow as tf;import matplotlib.pyplot as plt;import os;import tensorflow_datasets as tfds', function_to_run='obj(*args, **kwargs)', method_object=eval('model'), object_signature='tf.keras.Sequential', function_args=[eval('x')], function_kwargs={'training': eval('False')}, custom_class=None)
    prediction = custom_method(
    tf.math.argmax(logits, axis=1, output_type=tf.int64), imports='import tensorflow as tf;import matplotlib.pyplot as plt;import os;import tensorflow_datasets as tfds', function_to_run='tf.math.argmax(*args, **kwargs)', method_object=None, object_signature=None, function_args=[eval('logits')], function_kwargs={'axis': eval('1'), 'output_type': eval('tf.int64')})
    custom_method(
    test_accuracy(prediction, y), imports='import tensorflow as tf;import matplotlib.pyplot as plt;import os;import tensorflow_datasets as tfds', function_to_run='obj(*args)', method_object=eval('test_accuracy'), object_signature='tf.keras.metrics.Accuracy', function_args=[eval('prediction'), eval('y')], function_kwargs={}, custom_class=None)
print('Test set accuracy: {:.3%}'.format(test_accuracy.result()))
custom_method(
tf.stack([y, prediction], axis=1), imports='import tensorflow as tf;import matplotlib.pyplot as plt;import os;import tensorflow_datasets as tfds', function_to_run='tf.stack(*args, **kwargs)', method_object=None, object_signature=None, function_args=[eval('[y,prediction]')], function_kwargs={'axis': eval('1')})
predict_dataset = custom_method(
tf.convert_to_tensor([[0.3, 0.8, 0.4, 0.5], [0.4, 0.1, 0.8, 0.5], [0.7, 0.9, 0.8, 0.4]]), imports='import tensorflow as tf;import matplotlib.pyplot as plt;import os;import tensorflow_datasets as tfds', function_to_run='tf.convert_to_tensor(*args)', method_object=None, object_signature=None, function_args=[eval('[\n    [0.3, 0.8, 0.4, 0.5,],\n    [0.4, 0.1, 0.8, 0.5,],\n    [0.7, 0.9, 0.8, 0.4]\n]')], function_kwargs={})
predictions = custom_method(
model(predict_dataset, training=False), imports='import tensorflow as tf;import matplotlib.pyplot as plt;import os;import tensorflow_datasets as tfds', function_to_run='obj(*args, **kwargs)', method_object=eval('model'), object_signature='tf.keras.Sequential', function_args=[eval('predict_dataset')], function_kwargs={'training': eval('False')}, custom_class=None)
for (i, logits) in enumerate(predictions):
    class_idx = custom_method(
    tf.math.argmax(logits).numpy(), imports='import tensorflow as tf;import matplotlib.pyplot as plt;import os;import tensorflow_datasets as tfds', function_to_run='tf.math.argmax(logits).numpy()', method_object=None, object_signature=None, function_args=[], function_kwargs={})
    p = custom_method(
    tf.nn.softmax(logits), imports='import tensorflow as tf;import matplotlib.pyplot as plt;import os;import tensorflow_datasets as tfds', function_to_run='tf.nn.softmax(*args)', method_object=None, object_signature=None, function_args=[eval('logits')], function_kwargs={})[class_idx]
    name = class_names[class_idx]
    print('Example {} prediction: {} ({:4.1f}%)'.format(i, name, 100 * p))
