import pickle
import requests
import sys
import os

# Define the path to the subdirectory, relative to the home directory
subdir_path = os.path.join(os.path.expanduser("~"), "/home/saurabh/code-energy-consumption/server")

# Get the current working directory
cwd = os.path.dirname(os.path.realpath(__file__))


# Get the relative path from the current working directory to the subdirectory
rel_path = os.path.relpath(subdir_path, cwd)

sys.path.append(rel_path)
print(rel_path)
from send_request import send_request, send_single_thread_request

def custom_method(func, imports: str, function_to_run: str, method_object=None, function_args: list=None, function_kwargs: dict=None, max_wait_secs=0, custom_class=None):
    result = send_single_thread_request(imports=imports, function_to_run=function_to_run, function_args=function_args, function_kwargs=function_kwargs, max_wait_secs=max_wait_secs, method_object=method_object, custom_class=custom_class)
    return func
import tensorflow as tf
from tensorflow.keras import datasets, layers, optimizers, Sequential, metrics
((xs, ys), _) = custom_method(
datasets.mnist.load_data(), imports='from    tensorflow.keras import datasets, layers, optimizers, Sequential, metrics;import  tensorflow as tf', function_to_run='datasets.mnist.load_data()', method_object=None, function_args=[], function_kwargs={}, max_wait_secs=0)
print('datasets:', xs.shape, ys.shape, xs.min(), xs.max())
xs = custom_method(
tf.convert_to_tensor(xs, dtype=tf.float32), imports='from    tensorflow.keras import datasets, layers, optimizers, Sequential, metrics;import  tensorflow as tf', function_to_run='tf.convert_to_tensor(*args, **kwargs)', method_object=None, function_args=[eval('xs')], function_kwargs={'dtype': eval('tf.float32')}, max_wait_secs=0) / 255.0
db = custom_method(
tf.data.Dataset.from_tensor_slices((xs, ys)), imports='from    tensorflow.keras import datasets, layers, optimizers, Sequential, metrics;import  tensorflow as tf', function_to_run='tf.data.Dataset.from_tensor_slices(*args)', method_object=None, function_args=[eval('(xs,ys)')], function_kwargs={}, max_wait_secs=0)
db = custom_method(
db.batch(32).repeat(10), imports='from    tensorflow.keras import datasets, layers, optimizers, Sequential, metrics;import  tensorflow as tf', function_to_run='obj.batch(32).repeat(*args)', method_object='db', function_args=[eval('10')], function_kwargs={}, max_wait_secs=0, custom_class=None)
network = custom_method(
Sequential([layers.Dense(256, activation='relu'), layers.Dense(256, activation='relu'), layers.Dense(256, activation='relu'), layers.Dense(10)]), imports='from    tensorflow.keras import datasets, layers, optimizers, Sequential, metrics;import  tensorflow as tf', function_to_run='Sequential(*args)', method_object=None, function_args=[eval("[layers.Dense(256, activation='relu'),\n                     layers.Dense(256, activation='relu'),\n                     layers.Dense(256, activation='relu'),\n                     layers.Dense(10)]")], function_kwargs={}, max_wait_secs=0)
custom_method(
network.build(input_shape=(None, 28 * 28)), imports='from    tensorflow.keras import datasets, layers, optimizers, Sequential, metrics;import  tensorflow as tf', function_to_run='obj.build(**kwargs)', method_object='network', function_args=[], function_kwargs={'input_shape': eval('(None, 28*28)')}, max_wait_secs=0, custom_class=None)
custom_method(
network.summary(), imports='from    tensorflow.keras import datasets, layers, optimizers, Sequential, metrics;import  tensorflow as tf', function_to_run='obj.summary()', method_object='network', function_args=[], function_kwargs={}, max_wait_secs=0, custom_class=None)
optimizer = custom_method(
optimizers.SGD(lr=0.01), imports='from    tensorflow.keras import datasets, layers, optimizers, Sequential, metrics;import  tensorflow as tf', function_to_run='optimizers.SGD(**kwargs)', method_object=None, function_args=[], function_kwargs={'lr': eval('0.01')}, max_wait_secs=0)
acc_meter = custom_method(
metrics.Accuracy(), imports='from    tensorflow.keras import datasets, layers, optimizers, Sequential, metrics;import  tensorflow as tf', function_to_run='metrics.Accuracy()', method_object=None, function_args=[], function_kwargs={}, max_wait_secs=0)
for (step, (x, y)) in enumerate(db):
    with custom_method(
    tf.GradientTape(), imports='from    tensorflow.keras import datasets, layers, optimizers, Sequential, metrics;import  tensorflow as tf', function_to_run='tf.GradientTape()', method_object=None, function_args=[], function_kwargs={}, max_wait_secs=0) as tape:
        x = custom_method(
        tf.reshape(x, (-1, 28 * 28)), imports='from    tensorflow.keras import datasets, layers, optimizers, Sequential, metrics;import  tensorflow as tf', function_to_run='tf.reshape(*args)', method_object=None, function_args=[eval('x'), eval('(-1, 28*28)')], function_kwargs={}, max_wait_secs=0)
        out = custom_method(
        network(x), imports='from    tensorflow.keras import datasets, layers, optimizers, Sequential, metrics;import  tensorflow as tf', function_to_run='obj(*args)', method_object='network', function_args=[eval('x')], function_kwargs={}, max_wait_secs=0, custom_class=None)
        y_onehot = custom_method(
        tf.one_hot(y, depth=10), imports='from    tensorflow.keras import datasets, layers, optimizers, Sequential, metrics;import  tensorflow as tf', function_to_run='tf.one_hot(*args, **kwargs)', method_object=None, function_args=[eval('y')], function_kwargs={'depth': eval('10')}, max_wait_secs=0)
        loss = custom_method(
        tf.square(out - y_onehot), imports='from    tensorflow.keras import datasets, layers, optimizers, Sequential, metrics;import  tensorflow as tf', function_to_run='tf.square(*args)', method_object=None, function_args=[eval('out-y_onehot')], function_kwargs={}, max_wait_secs=0)
        loss = custom_method(
        tf.reduce_sum(loss), imports='from    tensorflow.keras import datasets, layers, optimizers, Sequential, metrics;import  tensorflow as tf', function_to_run='tf.reduce_sum(*args)', method_object=None, function_args=[eval('loss')], function_kwargs={}, max_wait_secs=0) / 32
    custom_method(
    acc_meter.update_state(tf.argmax(out, axis=1), y), imports='from    tensorflow.keras import datasets, layers, optimizers, Sequential, metrics;import  tensorflow as tf', function_to_run='obj.update_state(*args)', method_object='acc_meter', function_args=[eval('tf.argmax(out, axis=1)'), eval('y')], function_kwargs={}, max_wait_secs=0, custom_class=None)
    grads = tape.gradient(loss, network.trainable_variables)
    custom_method(
    optimizer.apply_gradients(zip(grads, network.trainable_variables)), imports='from    tensorflow.keras import datasets, layers, optimizers, Sequential, metrics;import  tensorflow as tf', function_to_run='obj.apply_gradients(*args)', method_object='optimizer', function_args=[eval('zip(grads, network.trainable_variables)')], function_kwargs={}, max_wait_secs=0, custom_class=None)
    if step % 200 == 0:
        print(step, 'loss:', float(loss), 'acc:', acc_meter.result().numpy())
        custom_method(
        acc_meter.reset_states(), imports='from    tensorflow.keras import datasets, layers, optimizers, Sequential, metrics;import  tensorflow as tf', function_to_run='obj.reset_states()', method_object='acc_meter', function_args=[], function_kwargs={}, max_wait_secs=0, custom_class=None)
