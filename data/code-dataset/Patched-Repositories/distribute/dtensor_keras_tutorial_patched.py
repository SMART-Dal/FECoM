import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow.experimental import dtensor
import os
from pathlib import Path
import dill as pickle
import sys
from tool.client.client_config import EXPERIMENT_DIR, MAX_WAIT_S, WAIT_AFTER_RUN_S
from tool.server.send_request import send_request
from tool.server.function_details import FunctionDetails
current_path = os.path.abspath(__file__)
(immediate_folder, file_name) = os.path.split(current_path)
immediate_folder = os.path.basename(immediate_folder)
experiment_number = int(sys.argv[0])
experiment_file_name = os.path.splitext(file_name)[0]
EXPERIMENT_FILE_PATH = EXPERIMENT_DIR / 'method-level' / immediate_folder / experiment_file_name / f'experiment-{experiment_number}.json'

def custom_method(func, imports: str, function_to_run: str, method_object=None, function_args: list=None, function_kwargs: dict=None, max_wait_secs=MAX_WAIT_S, custom_class=None, wait_after_run_secs=WAIT_AFTER_RUN_S):
    result = send_request(imports=imports, function_to_run=function_to_run, function_args=function_args, function_kwargs=function_kwargs, max_wait_secs=max_wait_secs, method_object=method_object, custom_class=custom_class, experiment_file_path=EXPERIMENT_FILE_PATH)
    return func
if __name__ == '__main__':
    print(EXPERIMENT_FILE_PATH)

def configure_virtual_cpus(ncpu):
    phy_devices = custom_method(
    tf.config.list_physical_devices('CPU'), imports='import tensorflow as tf;from tensorflow.experimental import dtensor;import tensorflow_datasets as tfds', function_to_run='tf.config.list_physical_devices(*args)', method_object=None, function_args=[eval("'CPU'")], function_kwargs={}, max_wait_secs=0)
    custom_method(
    tf.config.set_logical_device_configuration(phy_devices[0], [tf.config.LogicalDeviceConfiguration()] * ncpu), imports='import tensorflow as tf;from tensorflow.experimental import dtensor;import tensorflow_datasets as tfds', function_to_run='tf.config.set_logical_device_configuration(*args)', method_object=None, function_args=[eval('phy_devices[0]'), eval('[tf.config.LogicalDeviceConfiguration()] * ncpu')], function_kwargs={}, max_wait_secs=0)
configure_virtual_cpus(8)
custom_method(
tf.config.list_logical_devices('CPU'), imports='import tensorflow as tf;from tensorflow.experimental import dtensor;import tensorflow_datasets as tfds', function_to_run='tf.config.list_logical_devices(*args)', method_object=None, function_args=[eval("'CPU'")], function_kwargs={}, max_wait_secs=0)
devices = [f'CPU:{i}' for i in range(8)]
custom_method(
tf.keras.backend.experimental.enable_tf_random_generator(), imports='import tensorflow as tf;from tensorflow.experimental import dtensor;import tensorflow_datasets as tfds', function_to_run='tf.keras.backend.experimental.enable_tf_random_generator()', method_object=None, function_args=[], function_kwargs={}, max_wait_secs=0)
custom_method(
tf.keras.utils.set_random_seed(1337), imports='import tensorflow as tf;from tensorflow.experimental import dtensor;import tensorflow_datasets as tfds', function_to_run='tf.keras.utils.set_random_seed(*args)', method_object=None, function_args=[eval('1337')], function_kwargs={}, max_wait_secs=0)
mesh = custom_method(
dtensor.create_mesh([('batch', 8)], devices=devices), imports='import tensorflow as tf;from tensorflow.experimental import dtensor;import tensorflow_datasets as tfds', function_to_run='dtensor.create_mesh(*args, **kwargs)', method_object=None, function_args=[eval('[("batch", 8)]')], function_kwargs={'devices': eval('devices')}, max_wait_secs=0)
example_weight_layout = custom_method(
dtensor.Layout([dtensor.UNSHARDED, dtensor.UNSHARDED], mesh), imports='import tensorflow as tf;from tensorflow.experimental import dtensor;import tensorflow_datasets as tfds', function_to_run='dtensor.Layout(*args)', method_object=None, function_args=[eval('[dtensor.UNSHARDED, dtensor.UNSHARDED]'), eval('mesh')], function_kwargs={}, max_wait_secs=0)
example_weight_layout = custom_method(
dtensor.Layout.replicated(mesh, rank=2), imports='import tensorflow as tf;from tensorflow.experimental import dtensor;import tensorflow_datasets as tfds', function_to_run='dtensor.Layout.replicated(*args, **kwargs)', method_object=None, function_args=[eval('mesh')], function_kwargs={'rank': eval('2')}, max_wait_secs=0)
example_data_layout = custom_method(
dtensor.Layout(['batch', dtensor.UNSHARDED], mesh), imports='import tensorflow as tf;from tensorflow.experimental import dtensor;import tensorflow_datasets as tfds', function_to_run='dtensor.Layout(*args)', method_object=None, function_args=[eval("['batch', dtensor.UNSHARDED]"), eval('mesh')], function_kwargs={}, max_wait_secs=0)
example_data_layout = custom_method(
dtensor.Layout.batch_sharded(mesh, 'batch', rank=2), imports='import tensorflow as tf;from tensorflow.experimental import dtensor;import tensorflow_datasets as tfds', function_to_run='dtensor.Layout.batch_sharded(*args, **kwargs)', method_object=None, function_args=[eval('mesh'), eval("'batch'")], function_kwargs={'rank': eval('2')}, max_wait_secs=0)
unsharded_layout_2d = custom_method(
dtensor.Layout.replicated(mesh, 2), imports='import tensorflow as tf;from tensorflow.experimental import dtensor;import tensorflow_datasets as tfds', function_to_run='dtensor.Layout.replicated(*args)', method_object=None, function_args=[eval('mesh'), eval('2')], function_kwargs={}, max_wait_secs=0)
unsharded_layout_1d = custom_method(
dtensor.Layout.replicated(mesh, 1), imports='import tensorflow as tf;from tensorflow.experimental import dtensor;import tensorflow_datasets as tfds', function_to_run='dtensor.Layout.replicated(*args)', method_object=None, function_args=[eval('mesh'), eval('1')], function_kwargs={}, max_wait_secs=0)
model = custom_method(
tf.keras.models.Sequential([tf.keras.layers.Flatten(input_shape=(28, 28)), tf.keras.layers.Dense(128, activation='relu', name='d1', kernel_layout=unsharded_layout_2d, bias_layout=unsharded_layout_1d), tf.keras.layers.Dense(10, name='d2', kernel_layout=unsharded_layout_2d, bias_layout=unsharded_layout_1d)]), imports='import tensorflow as tf;from tensorflow.experimental import dtensor;import tensorflow_datasets as tfds', function_to_run='tf.keras.models.Sequential(*args)', method_object=None, function_args=[eval("[\n  tf.keras.layers.Flatten(input_shape=(28, 28)),\n  tf.keras.layers.Dense(128, \n                        activation='relu',\n                        name='d1',\n                        kernel_layout=unsharded_layout_2d, \n                        bias_layout=unsharded_layout_1d),\n  tf.keras.layers.Dense(10,\n                        name='d2',\n                        kernel_layout=unsharded_layout_2d, \n                        bias_layout=unsharded_layout_1d)\n]")], function_kwargs={}, max_wait_secs=0)
for weight in model.weights:
    print(f'Weight name: {weight.name} with layout: {weight.layout}')
    break
((ds_train, ds_test), ds_info) = custom_method(
tfds.load('mnist', split=['train', 'test'], shuffle_files=True, as_supervised=True, with_info=True), imports='import tensorflow as tf;from tensorflow.experimental import dtensor;import tensorflow_datasets as tfds', function_to_run='tfds.load(*args, **kwargs)', method_object=None, function_args=[eval("'mnist'")], function_kwargs={'split': eval("['train', 'test']"), 'shuffle_files': eval('True'), 'as_supervised': eval('True'), 'with_info': eval('True')}, max_wait_secs=0)

def normalize_img(image, label):
    """Normalizes images: `uint8` -> `float32`."""
    return (custom_method(
    tf.cast(image, tf.float32), imports='import tensorflow as tf;from tensorflow.experimental import dtensor;import tensorflow_datasets as tfds', function_to_run='tf.cast(*args)', method_object=None, function_args=[eval('image'), eval('tf.float32')], function_kwargs={}, max_wait_secs=0) / 255.0, label)
batch_size = 128
ds_train = ds_train.map(normalize_img, num_parallel_calls=tf.data.AUTOTUNE)
ds_train = ds_train.cache()
ds_train = ds_train.shuffle(ds_info.splits['train'].num_examples)
ds_train = ds_train.batch(batch_size)
ds_train = ds_train.prefetch(tf.data.AUTOTUNE)
ds_test = ds_test.map(normalize_img, num_parallel_calls=tf.data.AUTOTUNE)
ds_test = ds_test.batch(batch_size)
ds_test = ds_test.cache()
ds_test = ds_test.prefetch(tf.data.AUTOTUNE)

@tf.function
def train_step(model, x, y, optimizer, metrics):
    with custom_method(
    tf.GradientTape(), imports='import tensorflow as tf;from tensorflow.experimental import dtensor;import tensorflow_datasets as tfds', function_to_run='tf.GradientTape()', method_object=None, function_args=[], function_kwargs={}, max_wait_secs=0) as tape:
        logits = custom_method(
        model(x, training=True), imports='import tensorflow as tf;from tensorflow.experimental import dtensor;import tensorflow_datasets as tfds', function_to_run='obj(*args, **kwargs)', method_object=eval('model'), function_args=[eval('x')], function_kwargs={'training': eval('True')}, max_wait_secs=0, custom_class=None)
        loss = custom_method(
        tf.reduce_sum(tf.keras.losses.sparse_categorical_crossentropy(y, logits, from_logits=True)), imports='import tensorflow as tf;from tensorflow.experimental import dtensor;import tensorflow_datasets as tfds', function_to_run='tf.reduce_sum(*args)', method_object=None, function_args=[eval('tf.keras.losses.sparse_categorical_crossentropy(\n        y, logits, from_logits=True)')], function_kwargs={}, max_wait_secs=0)
    gradients = tape.gradient(loss, model.trainable_variables)
    custom_method(
    optimizer.apply_gradients(zip(gradients, model.trainable_variables)), imports='import tensorflow as tf;from tensorflow.experimental import dtensor;import tensorflow_datasets as tfds', function_to_run='obj.apply_gradients(*args)', method_object=eval('optimizer'), function_args=[eval('zip(gradients, model.trainable_variables)')], function_kwargs={}, max_wait_secs=0, custom_class=None)
    for metric in metrics.values():
        metric.update_state(y_true=y, y_pred=logits)
    loss_per_sample = loss / len(x)
    results = {'loss': loss_per_sample}
    return results

@tf.function
def eval_step(model, x, y, metrics):
    logits = custom_method(
    model(x, training=False), imports='import tensorflow as tf;from tensorflow.experimental import dtensor;import tensorflow_datasets as tfds', function_to_run='obj(*args, **kwargs)', method_object=eval('model'), function_args=[eval('x')], function_kwargs={'training': eval('False')}, max_wait_secs=0, custom_class=None)
    loss = custom_method(
    tf.reduce_sum(tf.keras.losses.sparse_categorical_crossentropy(y, logits, from_logits=True)), imports='import tensorflow as tf;from tensorflow.experimental import dtensor;import tensorflow_datasets as tfds', function_to_run='tf.reduce_sum(*args)', method_object=None, function_args=[eval('tf.keras.losses.sparse_categorical_crossentropy(\n        y, logits, from_logits=True)')], function_kwargs={}, max_wait_secs=0)
    for metric in metrics.values():
        metric.update_state(y_true=y, y_pred=logits)
    loss_per_sample = loss / len(x)
    results = {'eval_loss': loss_per_sample}
    return results

def pack_dtensor_inputs(images, labels, image_layout, label_layout):
    num_local_devices = custom_method(
    image_layout.mesh.num_local_devices(), imports='import tensorflow as tf;from tensorflow.experimental import dtensor;import tensorflow_datasets as tfds', function_to_run='obj.mesh.num_local_devices()', method_object=eval('image_layout'), function_args=[], function_kwargs={}, max_wait_secs=0, custom_class=None)
    images = custom_method(
    tf.split(images, num_local_devices), imports='import tensorflow as tf;from tensorflow.experimental import dtensor;import tensorflow_datasets as tfds', function_to_run='tf.split(*args)', method_object=None, function_args=[eval('images'), eval('num_local_devices')], function_kwargs={}, max_wait_secs=0)
    labels = custom_method(
    tf.split(labels, num_local_devices), imports='import tensorflow as tf;from tensorflow.experimental import dtensor;import tensorflow_datasets as tfds', function_to_run='tf.split(*args)', method_object=None, function_args=[eval('labels'), eval('num_local_devices')], function_kwargs={}, max_wait_secs=0)
    images = custom_method(
    dtensor.pack(images, image_layout), imports='import tensorflow as tf;from tensorflow.experimental import dtensor;import tensorflow_datasets as tfds', function_to_run='dtensor.pack(*args)', method_object=None, function_args=[eval('images'), eval('image_layout')], function_kwargs={}, max_wait_secs=0)
    labels = custom_method(
    dtensor.pack(labels, label_layout), imports='import tensorflow as tf;from tensorflow.experimental import dtensor;import tensorflow_datasets as tfds', function_to_run='dtensor.pack(*args)', method_object=None, function_args=[eval('labels'), eval('label_layout')], function_kwargs={}, max_wait_secs=0)
    return (images, labels)
optimizer = custom_method(
tf.keras.dtensor.experimental.optimizers.Adam(0.01, mesh=mesh), imports='import tensorflow as tf;from tensorflow.experimental import dtensor;import tensorflow_datasets as tfds', function_to_run='tf.keras.dtensor.experimental.optimizers.Adam(*args, **kwargs)', method_object=None, function_args=[eval('0.01')], function_kwargs={'mesh': eval('mesh')}, max_wait_secs=0)
metrics = {'accuracy': custom_method(
tf.keras.metrics.SparseCategoricalAccuracy(mesh=mesh), imports='import tensorflow as tf;from tensorflow.experimental import dtensor;import tensorflow_datasets as tfds', function_to_run='tf.keras.metrics.SparseCategoricalAccuracy(**kwargs)', method_object=None, function_args=[], function_kwargs={'mesh': eval('mesh')}, max_wait_secs=0)}
eval_metrics = {'eval_accuracy': custom_method(
tf.keras.metrics.SparseCategoricalAccuracy(mesh=mesh), imports='import tensorflow as tf;from tensorflow.experimental import dtensor;import tensorflow_datasets as tfds', function_to_run='tf.keras.metrics.SparseCategoricalAccuracy(**kwargs)', method_object=None, function_args=[], function_kwargs={'mesh': eval('mesh')}, max_wait_secs=0)}
num_epochs = 3
image_layout = custom_method(
dtensor.Layout.batch_sharded(mesh, 'batch', rank=4), imports='import tensorflow as tf;from tensorflow.experimental import dtensor;import tensorflow_datasets as tfds', function_to_run='dtensor.Layout.batch_sharded(*args, **kwargs)', method_object=None, function_args=[eval('mesh'), eval("'batch'")], function_kwargs={'rank': eval('4')}, max_wait_secs=0)
label_layout = custom_method(
dtensor.Layout.batch_sharded(mesh, 'batch', rank=1), imports='import tensorflow as tf;from tensorflow.experimental import dtensor;import tensorflow_datasets as tfds', function_to_run='dtensor.Layout.batch_sharded(*args, **kwargs)', method_object=None, function_args=[eval('mesh'), eval("'batch'")], function_kwargs={'rank': eval('1')}, max_wait_secs=0)
for epoch in range(num_epochs):
    print('============================')
    print('Epoch: ', epoch)
    for metric in metrics.values():
        metric.reset_state()
    step = 0
    results = {}
    pbar = custom_method(
    tf.keras.utils.Progbar(target=None, stateful_metrics=[]), imports='import tensorflow as tf;from tensorflow.experimental import dtensor;import tensorflow_datasets as tfds', function_to_run='tf.keras.utils.Progbar(**kwargs)', method_object=None, function_args=[], function_kwargs={'target': eval('None'), 'stateful_metrics': eval('[]')}, max_wait_secs=0)
    for input in ds_train:
        (images, labels) = (input[0], input[1])
        (images, labels) = custom_method(
        pack_dtensor_inputs(images, labels, image_layout, label_layout), imports='import tensorflow as tf;from tensorflow.experimental import dtensor;import tensorflow_datasets as tfds', function_to_run='pack_dtensor_inputs(*args)', method_object=None, function_args=[eval('images'), eval('labels'), eval('image_layout'), eval('label_layout')], function_kwargs={}, max_wait_secs=0)
        results.update(train_step(model, images, labels, optimizer, metrics))
        for (metric_name, metric) in metrics.items():
            results[metric_name] = metric.result()
        custom_method(
        pbar.update(step, values=results.items(), finalize=False), imports='import tensorflow as tf;from tensorflow.experimental import dtensor;import tensorflow_datasets as tfds', function_to_run='obj.update(*args, **kwargs)', method_object=eval('pbar'), function_args=[eval('step')], function_kwargs={'values': eval('results.items()'), 'finalize': eval('False')}, max_wait_secs=0, custom_class=None)
        step += 1
    custom_method(
    pbar.update(step, values=results.items(), finalize=True), imports='import tensorflow as tf;from tensorflow.experimental import dtensor;import tensorflow_datasets as tfds', function_to_run='obj.update(*args, **kwargs)', method_object=eval('pbar'), function_args=[eval('step')], function_kwargs={'values': eval('results.items()'), 'finalize': eval('True')}, max_wait_secs=0, custom_class=None)
    for metric in eval_metrics.values():
        metric.reset_state()
    for input in ds_test:
        (images, labels) = (input[0], input[1])
        (images, labels) = custom_method(
        pack_dtensor_inputs(images, labels, image_layout, label_layout), imports='import tensorflow as tf;from tensorflow.experimental import dtensor;import tensorflow_datasets as tfds', function_to_run='pack_dtensor_inputs(*args)', method_object=None, function_args=[eval('images'), eval('labels'), eval('image_layout'), eval('label_layout')], function_kwargs={}, max_wait_secs=0)
        results.update(eval_step(model, images, labels, eval_metrics))
    for (metric_name, metric) in eval_metrics.items():
        results[metric_name] = metric.result()
    for (metric_name, metric) in results.items():
        print(f'{metric_name}: {metric.numpy()}')

class SubclassedModel(tf.keras.Model):

    def __init__(self, name=None):
        super().__init__(name=name)
        self.feature = custom_method(
        tf.keras.layers.Dense(16), imports='import tensorflow as tf;from tensorflow.experimental import dtensor;import tensorflow_datasets as tfds', function_to_run='tf.keras.layers.Dense(*args)', method_object=None, function_args=[eval('16')], function_kwargs={}, max_wait_secs=0)
        self.feature_2 = custom_method(
        tf.keras.layers.Dense(24), imports='import tensorflow as tf;from tensorflow.experimental import dtensor;import tensorflow_datasets as tfds', function_to_run='tf.keras.layers.Dense(*args)', method_object=None, function_args=[eval('24')], function_kwargs={}, max_wait_secs=0)
        self.dropout = custom_method(
        tf.keras.layers.Dropout(0.1), imports='import tensorflow as tf;from tensorflow.experimental import dtensor;import tensorflow_datasets as tfds', function_to_run='tf.keras.layers.Dropout(*args)', method_object=None, function_args=[eval('0.1')], function_kwargs={}, max_wait_secs=0)

    def call(self, inputs, training=None):
        x = self.feature(inputs)
        x = self.dropout(x, training=training)
        return self.feature_2(x)
layout_map = custom_method(
tf.keras.dtensor.experimental.LayoutMap(mesh=mesh), imports='import tensorflow as tf;from tensorflow.experimental import dtensor;import tensorflow_datasets as tfds', function_to_run='tf.keras.dtensor.experimental.LayoutMap(**kwargs)', method_object=None, function_args=[], function_kwargs={'mesh': eval('mesh')}, max_wait_secs=0)
layout_map['feature.*kernel'] = custom_method(
dtensor.Layout.batch_sharded(mesh, 'batch', rank=2), imports='import tensorflow as tf;from tensorflow.experimental import dtensor;import tensorflow_datasets as tfds', function_to_run='dtensor.Layout.batch_sharded(*args, **kwargs)', method_object=None, function_args=[eval('mesh'), eval("'batch'")], function_kwargs={'rank': eval('2')}, max_wait_secs=0)
layout_map['feature.*bias'] = custom_method(
dtensor.Layout.batch_sharded(mesh, 'batch', rank=1), imports='import tensorflow as tf;from tensorflow.experimental import dtensor;import tensorflow_datasets as tfds', function_to_run='dtensor.Layout.batch_sharded(*args, **kwargs)', method_object=None, function_args=[eval('mesh'), eval("'batch'")], function_kwargs={'rank': eval('1')}, max_wait_secs=0)
with custom_method(
tf.keras.dtensor.experimental.layout_map_scope(layout_map), imports='import tensorflow as tf;from tensorflow.experimental import dtensor;import tensorflow_datasets as tfds', function_to_run='tf.keras.dtensor.experimental.layout_map_scope(*args)', method_object=None, function_args=[eval('layout_map')], function_kwargs={}, max_wait_secs=0):
    subclassed_model = SubclassedModel()
dtensor_input = custom_method(
dtensor.copy_to_mesh(tf.zeros((16, 16)), layout=unsharded_layout_2d), imports='import tensorflow as tf;from tensorflow.experimental import dtensor;import tensorflow_datasets as tfds', function_to_run='dtensor.copy_to_mesh(*args, **kwargs)', method_object=None, function_args=[eval('tf.zeros((16, 16))')], function_kwargs={'layout': eval('unsharded_layout_2d')}, max_wait_secs=0)
custom_method(
subclassed_model(dtensor_input), imports='import tensorflow as tf;from tensorflow.experimental import dtensor;import tensorflow_datasets as tfds', function_to_run='obj(*args)', method_object=eval('subclassed_model'), function_args=[eval('dtensor_input')], function_kwargs={}, max_wait_secs=0, custom_class='class SubclassedModel(tf.keras.Model):\n\n  def __init__(self, name=None):\n    super().__init__(name=name)\n    self.feature = tf.keras.layers.Dense(16)\n    self.feature_2 = tf.keras.layers.Dense(24)\n    self.dropout = tf.keras.layers.Dropout(0.1)\n\n  def call(self, inputs, training=None):\n    x = self.feature(inputs)\n    x = self.dropout(x, training=training)\n    return self.feature_2(x)')
print(subclassed_model.feature.kernel.layout)
layout_map = custom_method(
tf.keras.dtensor.experimental.LayoutMap(mesh=mesh), imports='import tensorflow as tf;from tensorflow.experimental import dtensor;import tensorflow_datasets as tfds', function_to_run='tf.keras.dtensor.experimental.LayoutMap(**kwargs)', method_object=None, function_args=[], function_kwargs={'mesh': eval('mesh')}, max_wait_secs=0)
layout_map['feature.*kernel'] = custom_method(
dtensor.Layout.batch_sharded(mesh, 'batch', rank=2), imports='import tensorflow as tf;from tensorflow.experimental import dtensor;import tensorflow_datasets as tfds', function_to_run='dtensor.Layout.batch_sharded(*args, **kwargs)', method_object=None, function_args=[eval('mesh'), eval("'batch'")], function_kwargs={'rank': eval('2')}, max_wait_secs=0)
layout_map['feature.*bias'] = custom_method(
dtensor.Layout.batch_sharded(mesh, 'batch', rank=1), imports='import tensorflow as tf;from tensorflow.experimental import dtensor;import tensorflow_datasets as tfds', function_to_run='dtensor.Layout.batch_sharded(*args, **kwargs)', method_object=None, function_args=[eval('mesh'), eval("'batch'")], function_kwargs={'rank': eval('1')}, max_wait_secs=0)
with custom_method(
tf.keras.dtensor.experimental.layout_map_scope(layout_map), imports='import tensorflow as tf;from tensorflow.experimental import dtensor;import tensorflow_datasets as tfds', function_to_run='tf.keras.dtensor.experimental.layout_map_scope(*args)', method_object=None, function_args=[eval('layout_map')], function_kwargs={}, max_wait_secs=0):
    inputs = custom_method(
    tf.keras.Input((16,), batch_size=16), imports='import tensorflow as tf;from tensorflow.experimental import dtensor;import tensorflow_datasets as tfds', function_to_run='tf.keras.Input(*args, **kwargs)', method_object=None, function_args=[eval('(16,)')], function_kwargs={'batch_size': eval('16')}, max_wait_secs=0)
    x = custom_method(
    tf.keras.layers.Dense(16, name='feature')(inputs), imports='import tensorflow as tf;from tensorflow.experimental import dtensor;import tensorflow_datasets as tfds', function_to_run="tf.keras.layers.Dense(16, name='feature')(*args)", method_object=None, function_args=[eval('inputs')], function_kwargs={}, max_wait_secs=0)
    x = custom_method(
    tf.keras.layers.Dropout(0.1)(x), imports='import tensorflow as tf;from tensorflow.experimental import dtensor;import tensorflow_datasets as tfds', function_to_run='tf.keras.layers.Dropout(0.1)(*args)', method_object=None, function_args=[eval('x')], function_kwargs={}, max_wait_secs=0)
    output = custom_method(
    tf.keras.layers.Dense(32, name='feature_2')(x), imports='import tensorflow as tf;from tensorflow.experimental import dtensor;import tensorflow_datasets as tfds', function_to_run="tf.keras.layers.Dense(32, name='feature_2')(*args)", method_object=None, function_args=[eval('x')], function_kwargs={}, max_wait_secs=0)
    model = custom_method(
    tf.keras.Model(inputs, output), imports='import tensorflow as tf;from tensorflow.experimental import dtensor;import tensorflow_datasets as tfds', function_to_run='tf.keras.Model(*args)', method_object=None, function_args=[eval('inputs'), eval('output')], function_kwargs={}, max_wait_secs=0)
print(model.layers[1].kernel.layout)
with custom_method(
tf.keras.dtensor.experimental.layout_map_scope(layout_map), imports='import tensorflow as tf;from tensorflow.experimental import dtensor;import tensorflow_datasets as tfds', function_to_run='tf.keras.dtensor.experimental.layout_map_scope(*args)', method_object=None, function_args=[eval('layout_map')], function_kwargs={}, max_wait_secs=0):
    model = custom_method(
    tf.keras.Sequential([tf.keras.layers.Dense(16, name='feature', input_shape=(16,)), tf.keras.layers.Dropout(0.1), tf.keras.layers.Dense(32, name='feature_2')]), imports='import tensorflow as tf;from tensorflow.experimental import dtensor;import tensorflow_datasets as tfds', function_to_run='tf.keras.Sequential(*args)', method_object=None, function_args=[eval("[\n      tf.keras.layers.Dense(16, name='feature', input_shape=(16,)),\n      tf.keras.layers.Dropout(0.1),\n      tf.keras.layers.Dense(32, name='feature_2')\n  ]")], function_kwargs={}, max_wait_secs=0)
print(model.layers[2].kernel.layout)
