import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow.experimental import dtensor
import sys
from tool.patching.patching_config import EXPERIMENT_DIR
from tool.measurement.execution import before_execution as before_execution_INSERTED_INTO_SCRIPT
from tool.measurement.execution import after_execution as after_execution_INSERTED_INTO_SCRIPT
from tool.experiment.experiments import ExperimentKinds
experiment_number = sys.argv[1]
experiment_project = sys.argv[2]
EXPERIMENT_FILE_PATH = EXPERIMENT_DIR / ExperimentKinds.METHOD_LEVEL.value / experiment_project / f'experiment-{experiment_number}.json'

def configure_virtual_cpus(ncpu):
    start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT()
    phy_devices = tf.config.list_physical_devices('CPU')
    after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tensorflow.config.list_physical_devices()', method_object=None, function_args=['CPU'], function_kwargs=None)
    start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT()
    tf.config.set_logical_device_configuration(phy_devices[0], [tf.config.LogicalDeviceConfiguration()] * ncpu)
    after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tensorflow.config.set_logical_device_configuration()', method_object=None, function_args=[phy_devices[0], [tf.config.LogicalDeviceConfiguration()] * ncpu], function_kwargs=None)
configure_virtual_cpus(8)
start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT()
tf.config.list_logical_devices('CPU')
after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tensorflow.config.list_logical_devices()', method_object=None, function_args=['CPU'], function_kwargs=None)
devices = [f'CPU:{i}' for i in range(8)]
start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT()
tf.keras.backend.experimental.enable_tf_random_generator()
after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tensorflow.keras.backend.experimental.enable_tf_random_generator()', method_object=None, function_args=None, function_kwargs=None)
start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT()
tf.keras.utils.set_random_seed(1337)
after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tensorflow.keras.utils.set_random_seed()', method_object=None, function_args=[1337], function_kwargs=None)
start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT()
mesh = dtensor.create_mesh([('batch', 8)], devices=devices)
after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tensorflow.experimental.dtensor.create_mesh()', method_object=None, function_args=[[('batch', 8)]], function_kwargs={'devices': devices})
start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT()
example_weight_layout = dtensor.Layout([dtensor.UNSHARDED, dtensor.UNSHARDED], mesh)
after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tensorflow.experimental.dtensor.Layout()', method_object=None, function_args=[[dtensor.UNSHARDED, dtensor.UNSHARDED], mesh], function_kwargs=None)
start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT()
example_weight_layout = dtensor.Layout.replicated(mesh, rank=2)
after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tensorflow.experimental.dtensor.Layout.replicated()', method_object=None, function_args=[mesh], function_kwargs={'rank': 2})
start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT()
example_data_layout = dtensor.Layout(['batch', dtensor.UNSHARDED], mesh)
after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tensorflow.experimental.dtensor.Layout()', method_object=None, function_args=[['batch', dtensor.UNSHARDED], mesh], function_kwargs=None)
start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT()
example_data_layout = dtensor.Layout.batch_sharded(mesh, 'batch', rank=2)
after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tensorflow.experimental.dtensor.Layout.batch_sharded()', method_object=None, function_args=[mesh, 'batch'], function_kwargs={'rank': 2})
start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT()
unsharded_layout_2d = dtensor.Layout.replicated(mesh, 2)
after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tensorflow.experimental.dtensor.Layout.replicated()', method_object=None, function_args=[mesh, 2], function_kwargs=None)
start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT()
unsharded_layout_1d = dtensor.Layout.replicated(mesh, 1)
after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tensorflow.experimental.dtensor.Layout.replicated()', method_object=None, function_args=[mesh, 1], function_kwargs=None)
start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT()
model = tf.keras.models.Sequential([tf.keras.layers.Flatten(input_shape=(28, 28)), tf.keras.layers.Dense(128, activation='relu', name='d1', kernel_layout=unsharded_layout_2d, bias_layout=unsharded_layout_1d), tf.keras.layers.Dense(10, name='d2', kernel_layout=unsharded_layout_2d, bias_layout=unsharded_layout_1d)])
after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tensorflow.keras.models.Sequential()', method_object=None, function_args=[[tf.keras.layers.Flatten(input_shape=(28, 28)), tf.keras.layers.Dense(128, activation='relu', name='d1', kernel_layout=unsharded_layout_2d, bias_layout=unsharded_layout_1d), tf.keras.layers.Dense(10, name='d2', kernel_layout=unsharded_layout_2d, bias_layout=unsharded_layout_1d)]], function_kwargs=None)
for weight in model.weights:
    print(f'Weight name: {weight.name} with layout: {weight.layout}')
    break
((ds_train, ds_test), ds_info) = tfds.load('mnist', split=['train', 'test'], shuffle_files=True, as_supervised=True, with_info=True)

def normalize_img(image, label):
    """Normalizes images: `uint8` -> `float32`."""
    return (tf.cast(image, tf.float32) / 255.0, label)
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
    with tf.GradientTape() as tape:
        start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT()
        logits = model(x, training=True)
        after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tensorflow.keras.models.Sequential()', method_object=model, function_args=[x], function_kwargs={'training': True})
        start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT()
        loss = tf.reduce_sum(tf.keras.losses.sparse_categorical_crossentropy(y, logits, from_logits=True))
        after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tensorflow.reduce_sum()', method_object=None, function_args=[tf.keras.losses.sparse_categorical_crossentropy(y, logits, from_logits=True)], function_kwargs=None)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    for metric in metrics.values():
        metric.update_state(y_true=y, y_pred=logits)
    loss_per_sample = loss / len(x)
    results = {'loss': loss_per_sample}
    return results

@tf.function
def eval_step(model, x, y, metrics):
    start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT()
    logits = model(x, training=False)
    after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tensorflow.keras.models.Sequential()', method_object=model, function_args=[x], function_kwargs={'training': False})
    start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT()
    loss = tf.reduce_sum(tf.keras.losses.sparse_categorical_crossentropy(y, logits, from_logits=True))
    after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tensorflow.reduce_sum()', method_object=None, function_args=[tf.keras.losses.sparse_categorical_crossentropy(y, logits, from_logits=True)], function_kwargs=None)
    for metric in metrics.values():
        metric.update_state(y_true=y, y_pred=logits)
    loss_per_sample = loss / len(x)
    results = {'eval_loss': loss_per_sample}
    return results

def pack_dtensor_inputs(images, labels, image_layout, label_layout):
    num_local_devices = image_layout.mesh.num_local_devices()
    start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT()
    images = tf.split(images, num_local_devices)
    after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tensorflow.split()', method_object=None, function_args=[images, num_local_devices], function_kwargs=None)
    start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT()
    labels = tf.split(labels, num_local_devices)
    after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tensorflow.split()', method_object=None, function_args=[labels, num_local_devices], function_kwargs=None)
    start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT()
    images = dtensor.pack(images, image_layout)
    after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tensorflow.experimental.dtensor.pack()', method_object=None, function_args=[images, image_layout], function_kwargs=None)
    start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT()
    labels = dtensor.pack(labels, label_layout)
    after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tensorflow.experimental.dtensor.pack()', method_object=None, function_args=[labels, label_layout], function_kwargs=None)
    return (images, labels)
start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT()
optimizer = tf.keras.dtensor.experimental.optimizers.Adam(0.01, mesh=mesh)
after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tensorflow.keras.dtensor.experimental.optimizers.Adam()', method_object=None, function_args=[0.01], function_kwargs={'mesh': mesh})
metrics = {'accuracy': tf.keras.metrics.SparseCategoricalAccuracy(mesh=mesh)}
eval_metrics = {'eval_accuracy': tf.keras.metrics.SparseCategoricalAccuracy(mesh=mesh)}
num_epochs = 3
start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT()
image_layout = dtensor.Layout.batch_sharded(mesh, 'batch', rank=4)
after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tensorflow.experimental.dtensor.Layout.batch_sharded()', method_object=None, function_args=[mesh, 'batch'], function_kwargs={'rank': 4})
start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT()
label_layout = dtensor.Layout.batch_sharded(mesh, 'batch', rank=1)
after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tensorflow.experimental.dtensor.Layout.batch_sharded()', method_object=None, function_args=[mesh, 'batch'], function_kwargs={'rank': 1})
for epoch in range(num_epochs):
    print('============================')
    print('Epoch: ', epoch)
    for metric in metrics.values():
        metric.reset_state()
    step = 0
    results = {}
    start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT()
    pbar = tf.keras.utils.Progbar(target=None, stateful_metrics=[])
    after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tensorflow.keras.utils.Progbar()', method_object=None, function_args=None, function_kwargs={'target': None, 'stateful_metrics': []})
    for input in ds_train:
        (images, labels) = (input[0], input[1])
        (images, labels) = pack_dtensor_inputs(images, labels, image_layout, label_layout)
        results.update(train_step(model, images, labels, optimizer, metrics))
        for (metric_name, metric) in metrics.items():
            results[metric_name] = metric.result()
        start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT()
        pbar.update(step, values=results.items(), finalize=False)
        after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tensorflow.keras.utils.Progbar.update()', method_object=pbar, function_args=[step], function_kwargs={'values': results.items(), 'finalize': False})
        step += 1
    start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT()
    pbar.update(step, values=results.items(), finalize=True)
    after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tensorflow.keras.utils.Progbar.update()', method_object=pbar, function_args=[step], function_kwargs={'values': results.items(), 'finalize': True})
    for metric in eval_metrics.values():
        metric.reset_state()
    for input in ds_test:
        (images, labels) = (input[0], input[1])
        (images, labels) = pack_dtensor_inputs(images, labels, image_layout, label_layout)
        results.update(eval_step(model, images, labels, eval_metrics))
    for (metric_name, metric) in eval_metrics.items():
        results[metric_name] = metric.result()
    for (metric_name, metric) in results.items():
        print(f'{metric_name}: {metric.numpy()}')

class SubclassedModel(tf.keras.Model):

    def __init__(self, name=None):
        super().__init__(name=name)
        start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT()
        self.feature = tf.keras.layers.Dense(16)
        after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tensorflow.keras.layers.Dense()', method_object=None, function_args=[16], function_kwargs=None)
        start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT()
        self.feature_2 = tf.keras.layers.Dense(24)
        after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tensorflow.keras.layers.Dense()', method_object=None, function_args=[24], function_kwargs=None)
        start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT()
        self.dropout = tf.keras.layers.Dropout(0.1)
        after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tensorflow.keras.layers.Dropout()', method_object=None, function_args=[0.1], function_kwargs=None)

    def call(self, inputs, training=None):
        x = self.feature(inputs)
        x = self.dropout(x, training=training)
        return self.feature_2(x)
start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT()
layout_map = tf.keras.dtensor.experimental.LayoutMap(mesh=mesh)
after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tensorflow.keras.dtensor.experimental.LayoutMap()', method_object=None, function_args=None, function_kwargs={'mesh': mesh})
start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT()
layout_map['feature.*kernel'] = dtensor.Layout.batch_sharded(mesh, 'batch', rank=2)
after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tensorflow.experimental.dtensor.Layout.batch_sharded()', method_object=None, function_args=[mesh, 'batch'], function_kwargs={'rank': 2})
start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT()
layout_map['feature.*bias'] = dtensor.Layout.batch_sharded(mesh, 'batch', rank=1)
after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tensorflow.experimental.dtensor.Layout.batch_sharded()', method_object=None, function_args=[mesh, 'batch'], function_kwargs={'rank': 1})
with tf.keras.dtensor.experimental.layout_map_scope(layout_map):
    subclassed_model = SubclassedModel()
start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT()
dtensor_input = dtensor.copy_to_mesh(tf.zeros((16, 16)), layout=unsharded_layout_2d)
after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tensorflow.experimental.dtensor.copy_to_mesh()', method_object=None, function_args=[tf.zeros((16, 16))], function_kwargs={'layout': unsharded_layout_2d})
start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT()
subclassed_model(dtensor_input)
after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tensorflow.keras.Model()', method_object=subclassed_model, function_args=[dtensor_input], function_kwargs=None)
print(subclassed_model.feature.kernel.layout)
start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT()
layout_map = tf.keras.dtensor.experimental.LayoutMap(mesh=mesh)
after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tensorflow.keras.dtensor.experimental.LayoutMap()', method_object=None, function_args=None, function_kwargs={'mesh': mesh})
start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT()
layout_map['feature.*kernel'] = dtensor.Layout.batch_sharded(mesh, 'batch', rank=2)
after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tensorflow.experimental.dtensor.Layout.batch_sharded()', method_object=None, function_args=[mesh, 'batch'], function_kwargs={'rank': 2})
start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT()
layout_map['feature.*bias'] = dtensor.Layout.batch_sharded(mesh, 'batch', rank=1)
after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tensorflow.experimental.dtensor.Layout.batch_sharded()', method_object=None, function_args=[mesh, 'batch'], function_kwargs={'rank': 1})
with tf.keras.dtensor.experimental.layout_map_scope(layout_map):
    start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT()
    inputs = tf.keras.Input((16,), batch_size=16)
    after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tensorflow.keras.Input()', method_object=None, function_args=[(16,)], function_kwargs={'batch_size': 16})
    start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT()
    x = tf.keras.layers.Dense(16, name='feature')(inputs)
    after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run="tensorflow.keras.layers.Dense(16, name='feature')()", method_object=None, function_args=[inputs], function_kwargs=None)
    start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT()
    x = tf.keras.layers.Dropout(0.1)(x)
    after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tensorflow.keras.layers.Dropout(0.1)()', method_object=None, function_args=[x], function_kwargs=None)
    start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT()
    output = tf.keras.layers.Dense(32, name='feature_2')(x)
    after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run="tensorflow.keras.layers.Dense(32, name='feature_2')()", method_object=None, function_args=[x], function_kwargs=None)
    start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT()
    model = tf.keras.Model(inputs, output)
    after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tensorflow.keras.Model()', method_object=None, function_args=[inputs, output], function_kwargs=None)
print(model.layers[1].kernel.layout)
with tf.keras.dtensor.experimental.layout_map_scope(layout_map):
    start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT()
    model = tf.keras.Sequential([tf.keras.layers.Dense(16, name='feature', input_shape=(16,)), tf.keras.layers.Dropout(0.1), tf.keras.layers.Dense(32, name='feature_2')])
    after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tensorflow.keras.Sequential()', method_object=None, function_args=[[tf.keras.layers.Dense(16, name='feature', input_shape=(16,)), tf.keras.layers.Dropout(0.1), tf.keras.layers.Dense(32, name='feature_2')]], function_kwargs=None)
print(model.layers[2].kernel.layout)
