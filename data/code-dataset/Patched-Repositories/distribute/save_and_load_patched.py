import tensorflow_datasets as tfds
import tensorflow as tf
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
mirrored_strategy = custom_method(
tf.distribute.MirroredStrategy(), imports='import tensorflow as tf;import tensorflow_hub as hub;import tensorflow_datasets as tfds', function_to_run='tf.distribute.MirroredStrategy()', method_object=None, object_signature=None, function_args=[], function_kwargs={}, max_wait_secs=0)

def get_data():
    datasets = custom_method(
    tfds.load(name='mnist', as_supervised=True), imports='import tensorflow as tf;import tensorflow_hub as hub;import tensorflow_datasets as tfds', function_to_run='tfds.load(**kwargs)', method_object=None, object_signature=None, function_args=[], function_kwargs={'name': eval("'mnist'"), 'as_supervised': eval('True')}, max_wait_secs=0)
    (mnist_train, mnist_test) = (datasets['train'], datasets['test'])
    BUFFER_SIZE = 10000
    BATCH_SIZE_PER_REPLICA = 64
    BATCH_SIZE = BATCH_SIZE_PER_REPLICA * mirrored_strategy.num_replicas_in_sync

    def scale(image, label):
        image = custom_method(
        tf.cast(image, tf.float32), imports='import tensorflow as tf;import tensorflow_hub as hub;import tensorflow_datasets as tfds', function_to_run='tf.cast(*args)', method_object=None, object_signature=None, function_args=[eval('image'), eval('tf.float32')], function_kwargs={}, max_wait_secs=0)
        image /= 255
        return (image, label)
    train_dataset = mnist_train.map(scale).cache().shuffle(BUFFER_SIZE).batch(BATCH_SIZE)
    eval_dataset = mnist_test.map(scale).batch(BATCH_SIZE)
    return (train_dataset, eval_dataset)

def get_model():
    with custom_method(
    mirrored_strategy.scope(), imports='import tensorflow as tf;import tensorflow_hub as hub;import tensorflow_datasets as tfds', function_to_run='obj.scope()', method_object=eval('mirrored_strategy'), object_signature='tf.distribute.MirroredStrategy', function_args=[], function_kwargs={}, max_wait_secs=0, custom_class=None):
        model = custom_method(
        tf.keras.Sequential([tf.keras.layers.Conv2D(32, 3, activation='relu', input_shape=(28, 28, 1)), tf.keras.layers.MaxPooling2D(), tf.keras.layers.Flatten(), tf.keras.layers.Dense(64, activation='relu'), tf.keras.layers.Dense(10)]), imports='import tensorflow as tf;import tensorflow_hub as hub;import tensorflow_datasets as tfds', function_to_run='tf.keras.Sequential(*args)', method_object=None, object_signature=None, function_args=[eval("[\n        tf.keras.layers.Conv2D(32, 3, activation='relu', input_shape=(28, 28, 1)),\n        tf.keras.layers.MaxPooling2D(),\n        tf.keras.layers.Flatten(),\n        tf.keras.layers.Dense(64, activation='relu'),\n        tf.keras.layers.Dense(10)\n    ]")], function_kwargs={}, max_wait_secs=0)
        custom_method(
        model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), optimizer=tf.keras.optimizers.Adam(), metrics=[tf.metrics.SparseCategoricalAccuracy()]), imports='import tensorflow as tf;import tensorflow_hub as hub;import tensorflow_datasets as tfds', function_to_run='obj.compile(**kwargs)', method_object=eval('model'), object_signature='tf.keras.Sequential', function_args=[], function_kwargs={'loss': eval('tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)'), 'optimizer': eval('tf.keras.optimizers.Adam()'), 'metrics': eval('[tf.metrics.SparseCategoricalAccuracy()]')}, max_wait_secs=0, custom_class=None)
        return model
model = custom_method(
get_model(), imports='import tensorflow as tf;import tensorflow_hub as hub;import tensorflow_datasets as tfds', function_to_run='obj()', method_object=eval('get_model'), object_signature='tf.keras.Sequential', function_args=[], function_kwargs={}, max_wait_secs=0, custom_class=None)
(train_dataset, eval_dataset) = get_data()
custom_method(
model.fit(train_dataset, epochs=2), imports='import tensorflow as tf;import tensorflow_hub as hub;import tensorflow_datasets as tfds', function_to_run='obj.fit(*args, **kwargs)', method_object=eval('model'), object_signature='tf.keras.Sequential', function_args=[eval('train_dataset')], function_kwargs={'epochs': eval('2')}, max_wait_secs=0, custom_class=None)
keras_model_path = '/tmp/keras_save'
custom_method(
model.save(keras_model_path), imports='import tensorflow as tf;import tensorflow_hub as hub;import tensorflow_datasets as tfds', function_to_run='obj.save(*args)', method_object=eval('model'), object_signature='tf.keras.Sequential', function_args=[eval('keras_model_path')], function_kwargs={}, max_wait_secs=0, custom_class=None)
restored_keras_model = custom_method(
tf.keras.models.load_model(keras_model_path), imports='import tensorflow as tf;import tensorflow_hub as hub;import tensorflow_datasets as tfds', function_to_run='tf.keras.models.load_model(*args)', method_object=None, object_signature=None, function_args=[eval('keras_model_path')], function_kwargs={}, max_wait_secs=0)
custom_method(
restored_keras_model.fit(train_dataset, epochs=2), imports='import tensorflow as tf;import tensorflow_hub as hub;import tensorflow_datasets as tfds', function_to_run='obj.fit(*args, **kwargs)', method_object=eval('restored_keras_model'), object_signature='tf.keras.Sequential', function_args=[eval('train_dataset')], function_kwargs={'epochs': eval('2')}, max_wait_secs=0, custom_class=None)
another_strategy = custom_method(
tf.distribute.OneDeviceStrategy('/cpu:0'), imports='import tensorflow as tf;import tensorflow_hub as hub;import tensorflow_datasets as tfds', function_to_run='tf.distribute.OneDeviceStrategy(*args)', method_object=None, object_signature=None, function_args=[eval("'/cpu:0'")], function_kwargs={}, max_wait_secs=0)
with custom_method(
another_strategy.scope(), imports='import tensorflow as tf;import tensorflow_hub as hub;import tensorflow_datasets as tfds', function_to_run='obj.scope()', method_object=eval('another_strategy'), object_signature='tf.distribute.OneDeviceStrategy', function_args=[], function_kwargs={}, max_wait_secs=0, custom_class=None):
    restored_keras_model_ds = custom_method(
    tf.keras.models.load_model(keras_model_path), imports='import tensorflow as tf;import tensorflow_hub as hub;import tensorflow_datasets as tfds', function_to_run='tf.keras.models.load_model(*args)', method_object=None, object_signature=None, function_args=[eval('keras_model_path')], function_kwargs={}, max_wait_secs=0)
    custom_method(
    restored_keras_model_ds.fit(train_dataset, epochs=2), imports='import tensorflow as tf;import tensorflow_hub as hub;import tensorflow_datasets as tfds', function_to_run='obj.fit(*args, **kwargs)', method_object=eval('restored_keras_model_ds'), object_signature='tf.keras.Sequential', function_args=[eval('train_dataset')], function_kwargs={'epochs': eval('2')}, max_wait_secs=0, custom_class=None)
model = custom_method(
get_model(), imports='import tensorflow as tf;import tensorflow_hub as hub;import tensorflow_datasets as tfds', function_to_run='obj()', method_object=eval('get_model'), object_signature='tf.keras.Sequential', function_args=[], function_kwargs={}, max_wait_secs=0, custom_class=None)
saved_model_path = '/tmp/tf_save'
custom_method(
tf.saved_model.save(model, saved_model_path), imports='import tensorflow as tf;import tensorflow_hub as hub;import tensorflow_datasets as tfds', function_to_run='tf.saved_model.save(*args)', method_object=None, object_signature=None, function_args=[eval('model'), eval('saved_model_path')], function_kwargs={}, max_wait_secs=0)
DEFAULT_FUNCTION_KEY = 'serving_default'
loaded = custom_method(
tf.saved_model.load(saved_model_path), imports='import tensorflow as tf;import tensorflow_hub as hub;import tensorflow_datasets as tfds', function_to_run='tf.saved_model.load(*args)', method_object=None, object_signature=None, function_args=[eval('saved_model_path')], function_kwargs={}, max_wait_secs=0)
inference_func = loaded.signatures[DEFAULT_FUNCTION_KEY]
predict_dataset = custom_method(
eval_dataset.map(lambda image, label: image), imports='import tensorflow as tf;import tensorflow_hub as hub;import tensorflow_datasets as tfds', function_to_run='obj.map(*args)', method_object=eval('eval_dataset'), object_signature='tf.data.Dataset.from_tensors(\n    (tf.range(5, dtype=tf.float32), tf.range(5, dtype=tf.float32))\n    ).repeat(dataset_size).batch', function_args=[eval('lambda image, label: image')], function_kwargs={}, max_wait_secs=0, custom_class=None)
for batch in custom_method(
predict_dataset.take(1), imports='import tensorflow as tf;import tensorflow_hub as hub;import tensorflow_datasets as tfds', function_to_run='obj.take(*args)', method_object=eval('predict_dataset'), object_signature='tf.data.Dataset.from_tensors(\n    (tf.range(5, dtype=tf.float32), tf.range(5, dtype=tf.float32))\n    ).repeat(dataset_size).batch', function_args=[eval('1')], function_kwargs={}, max_wait_secs=0, custom_class=None):
    print(inference_func(batch))
another_strategy = custom_method(
tf.distribute.MirroredStrategy(), imports='import tensorflow as tf;import tensorflow_hub as hub;import tensorflow_datasets as tfds', function_to_run='tf.distribute.MirroredStrategy()', method_object=None, object_signature=None, function_args=[], function_kwargs={}, max_wait_secs=0)
with custom_method(
another_strategy.scope(), imports='import tensorflow as tf;import tensorflow_hub as hub;import tensorflow_datasets as tfds', function_to_run='obj.scope()', method_object=eval('another_strategy'), object_signature='tf.distribute.OneDeviceStrategy', function_args=[], function_kwargs={}, max_wait_secs=0, custom_class=None):
    loaded = custom_method(
    tf.saved_model.load(saved_model_path), imports='import tensorflow as tf;import tensorflow_hub as hub;import tensorflow_datasets as tfds', function_to_run='tf.saved_model.load(*args)', method_object=None, object_signature=None, function_args=[eval('saved_model_path')], function_kwargs={}, max_wait_secs=0)
    inference_func = loaded.signatures[DEFAULT_FUNCTION_KEY]
    dist_predict_dataset = custom_method(
    another_strategy.experimental_distribute_dataset(predict_dataset), imports='import tensorflow as tf;import tensorflow_hub as hub;import tensorflow_datasets as tfds', function_to_run='obj.experimental_distribute_dataset(*args)', method_object=eval('another_strategy'), object_signature='tf.distribute.OneDeviceStrategy', function_args=[eval('predict_dataset')], function_kwargs={}, max_wait_secs=0, custom_class=None)
    for batch in dist_predict_dataset:
        result = custom_method(
        another_strategy.run(inference_func, args=(batch,)), imports='import tensorflow as tf;import tensorflow_hub as hub;import tensorflow_datasets as tfds', function_to_run='obj.run(*args, **kwargs)', method_object=eval('another_strategy'), object_signature='tf.distribute.OneDeviceStrategy', function_args=[eval('inference_func')], function_kwargs={'args': eval('(batch,)')}, max_wait_secs=0, custom_class=None)
        print(result)
        break
import tensorflow_hub as hub

def build_model(loaded):
    x = custom_method(
    tf.keras.layers.Input(shape=(28, 28, 1), name='input_x'), imports='import tensorflow as tf;import tensorflow_hub as hub;import tensorflow_datasets as tfds', function_to_run='tf.keras.layers.Input(**kwargs)', method_object=None, object_signature=None, function_args=[], function_kwargs={'shape': eval('(28, 28, 1)'), 'name': eval("'input_x'")}, max_wait_secs=0)
    keras_layer = custom_method(
    hub.KerasLayer(loaded, trainable=True)(x), imports='import tensorflow as tf;import tensorflow_hub as hub;import tensorflow_datasets as tfds', function_to_run='hub.KerasLayer(obj, trainable=True)(*args)', method_object=eval('loaded'), object_signature='tf.saved_model.load', function_args=[eval('x')], function_kwargs={}, max_wait_secs=0, custom_class=None)
    model = custom_method(
    tf.keras.Model(x, keras_layer), imports='import tensorflow as tf;import tensorflow_hub as hub;import tensorflow_datasets as tfds', function_to_run='tf.keras.Model(*args)', method_object=None, object_signature=None, function_args=[eval('x'), eval('keras_layer')], function_kwargs={}, max_wait_secs=0)
    return model
another_strategy = custom_method(
tf.distribute.MirroredStrategy(), imports='import tensorflow as tf;import tensorflow_hub as hub;import tensorflow_datasets as tfds', function_to_run='tf.distribute.MirroredStrategy()', method_object=None, object_signature=None, function_args=[], function_kwargs={}, max_wait_secs=0)
with custom_method(
another_strategy.scope(), imports='import tensorflow as tf;import tensorflow_hub as hub;import tensorflow_datasets as tfds', function_to_run='obj.scope()', method_object=eval('another_strategy'), object_signature='tf.distribute.OneDeviceStrategy', function_args=[], function_kwargs={}, max_wait_secs=0, custom_class=None):
    loaded = custom_method(
    tf.saved_model.load(saved_model_path), imports='import tensorflow as tf;import tensorflow_hub as hub;import tensorflow_datasets as tfds', function_to_run='tf.saved_model.load(*args)', method_object=None, object_signature=None, function_args=[eval('saved_model_path')], function_kwargs={}, max_wait_secs=0)
    model = custom_method(
    build_model(loaded), imports='import tensorflow as tf;import tensorflow_hub as hub;import tensorflow_datasets as tfds', function_to_run='obj(*args)', method_object=eval('build_model'), object_signature='tf.keras.Sequential', function_args=[eval('loaded')], function_kwargs={}, max_wait_secs=0, custom_class=None)
    custom_method(
    model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), optimizer=tf.keras.optimizers.Adam(), metrics=[tf.metrics.SparseCategoricalAccuracy()]), imports='import tensorflow as tf;import tensorflow_hub as hub;import tensorflow_datasets as tfds', function_to_run='obj.compile(**kwargs)', method_object=eval('model'), object_signature='tf.keras.Sequential', function_args=[], function_kwargs={'loss': eval('tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)'), 'optimizer': eval('tf.keras.optimizers.Adam()'), 'metrics': eval('[tf.metrics.SparseCategoricalAccuracy()]')}, max_wait_secs=0, custom_class=None)
    custom_method(
    model.fit(train_dataset, epochs=2), imports='import tensorflow as tf;import tensorflow_hub as hub;import tensorflow_datasets as tfds', function_to_run='obj.fit(*args, **kwargs)', method_object=eval('model'), object_signature='tf.keras.Sequential', function_args=[eval('train_dataset')], function_kwargs={'epochs': eval('2')}, max_wait_secs=0, custom_class=None)
model = custom_method(
get_model(), imports='import tensorflow as tf;import tensorflow_hub as hub;import tensorflow_datasets as tfds', function_to_run='obj()', method_object=eval('get_model'), object_signature='tf.keras.Sequential', function_args=[], function_kwargs={}, max_wait_secs=0, custom_class=None)
custom_method(
model.save(keras_model_path), imports='import tensorflow as tf;import tensorflow_hub as hub;import tensorflow_datasets as tfds', function_to_run='obj.save(*args)', method_object=eval('model'), object_signature='tf.keras.Sequential', function_args=[eval('keras_model_path')], function_kwargs={}, max_wait_secs=0, custom_class=None)
another_strategy = custom_method(
tf.distribute.MirroredStrategy(), imports='import tensorflow as tf;import tensorflow_hub as hub;import tensorflow_datasets as tfds', function_to_run='tf.distribute.MirroredStrategy()', method_object=None, object_signature=None, function_args=[], function_kwargs={}, max_wait_secs=0)
with custom_method(
another_strategy.scope(), imports='import tensorflow as tf;import tensorflow_hub as hub;import tensorflow_datasets as tfds', function_to_run='obj.scope()', method_object=eval('another_strategy'), object_signature='tf.distribute.OneDeviceStrategy', function_args=[], function_kwargs={}, max_wait_secs=0, custom_class=None):
    loaded = custom_method(
    tf.saved_model.load(keras_model_path), imports='import tensorflow as tf;import tensorflow_hub as hub;import tensorflow_datasets as tfds', function_to_run='tf.saved_model.load(*args)', method_object=None, object_signature=None, function_args=[eval('keras_model_path')], function_kwargs={}, max_wait_secs=0)
model = custom_method(
get_model(), imports='import tensorflow as tf;import tensorflow_hub as hub;import tensorflow_datasets as tfds', function_to_run='obj()', method_object=eval('get_model'), object_signature='tf.keras.Sequential', function_args=[], function_kwargs={}, max_wait_secs=0, custom_class=None)
saved_model_path = '/tmp/tf_save'
save_options = custom_method(
tf.saved_model.SaveOptions(experimental_io_device='/job:localhost'), imports='import tensorflow as tf;import tensorflow_hub as hub;import tensorflow_datasets as tfds', function_to_run='tf.saved_model.SaveOptions(**kwargs)', method_object=None, object_signature=None, function_args=[], function_kwargs={'experimental_io_device': eval("'/job:localhost'")}, max_wait_secs=0)
custom_method(
model.save(saved_model_path, options=save_options), imports='import tensorflow as tf;import tensorflow_hub as hub;import tensorflow_datasets as tfds', function_to_run='obj.save(*args, **kwargs)', method_object=eval('model'), object_signature='tf.keras.Sequential', function_args=[eval('saved_model_path')], function_kwargs={'options': eval('save_options')}, max_wait_secs=0, custom_class=None)
another_strategy = custom_method(
tf.distribute.MirroredStrategy(), imports='import tensorflow as tf;import tensorflow_hub as hub;import tensorflow_datasets as tfds', function_to_run='tf.distribute.MirroredStrategy()', method_object=None, object_signature=None, function_args=[], function_kwargs={}, max_wait_secs=0)
with custom_method(
another_strategy.scope(), imports='import tensorflow as tf;import tensorflow_hub as hub;import tensorflow_datasets as tfds', function_to_run='obj.scope()', method_object=eval('another_strategy'), object_signature='tf.distribute.OneDeviceStrategy', function_args=[], function_kwargs={}, max_wait_secs=0, custom_class=None):
    load_options = custom_method(
    tf.saved_model.LoadOptions(experimental_io_device='/job:localhost'), imports='import tensorflow as tf;import tensorflow_hub as hub;import tensorflow_datasets as tfds', function_to_run='tf.saved_model.LoadOptions(**kwargs)', method_object=None, object_signature=None, function_args=[], function_kwargs={'experimental_io_device': eval("'/job:localhost'")}, max_wait_secs=0)
    loaded = custom_method(
    tf.keras.models.load_model(saved_model_path, options=load_options), imports='import tensorflow as tf;import tensorflow_hub as hub;import tensorflow_datasets as tfds', function_to_run='tf.keras.models.load_model(*args, **kwargs)', method_object=None, object_signature=None, function_args=[eval('saved_model_path')], function_kwargs={'options': eval('load_options')}, max_wait_secs=0)

class SubclassedModel(tf.keras.Model):
    """Example model defined by subclassing `tf.keras.Model`."""
    output_name = 'output_layer'

    def __init__(self):
        super(SubclassedModel, self).__init__()
        self._dense_layer = custom_method(
        tf.keras.layers.Dense(5, dtype=tf.dtypes.float32, name=self.output_name), imports='import tensorflow as tf;import tensorflow_hub as hub;import tensorflow_datasets as tfds', function_to_run='tf.keras.layers.Dense(*args, **kwargs)', method_object=None, object_signature=None, function_args=[eval('5')], function_kwargs={'dtype': eval('tf.dtypes.float32'), 'name': eval('self.output_name')}, max_wait_secs=0)

    def call(self, inputs):
        return self._dense_layer(inputs)
my_model = SubclassedModel()
try:
    custom_method(
    my_model.save(keras_model_path), imports='import tensorflow as tf;import tensorflow_hub as hub;import tensorflow_datasets as tfds', function_to_run='obj.save(*args)', method_object=eval('my_model'), object_signature='tf.keras.Sequential', function_args=[eval('keras_model_path')], function_kwargs={}, max_wait_secs=0, custom_class='class SubclassedModel(tf.keras.Model):\n  """Example model defined by subclassing `tf.keras.Model`."""\n\n  output_name = \'output_layer\'\n\n  def __init__(self):\n    super(SubclassedModel, self).__init__()\n    self._dense_layer = tf.keras.layers.Dense(\n        5, dtype=tf.dtypes.float32, name=self.output_name)\n\n  def call(self, inputs):\n    return self._dense_layer(inputs)')
except ValueError as e:
    print(f'{type(e).__name__}: ', *e.args)
custom_method(
tf.saved_model.save(my_model, saved_model_path), imports='import tensorflow as tf;import tensorflow_hub as hub;import tensorflow_datasets as tfds', function_to_run='tf.saved_model.save(*args)', method_object=None, object_signature=None, function_args=[eval('my_model'), eval('saved_model_path')], function_kwargs={}, max_wait_secs=0)
x = custom_method(
tf.saved_model.load(saved_model_path), imports='import tensorflow as tf;import tensorflow_hub as hub;import tensorflow_datasets as tfds', function_to_run='tf.saved_model.load(*args)', method_object=None, object_signature=None, function_args=[eval('saved_model_path')], function_kwargs={}, max_wait_secs=0)
x.signatures
print(my_model.save_spec() is None)
BATCH_SIZE_PER_REPLICA = 4
BATCH_SIZE = BATCH_SIZE_PER_REPLICA * mirrored_strategy.num_replicas_in_sync
dataset_size = 100
dataset = custom_method(
tf.data.Dataset.from_tensors((tf.range(5, dtype=tf.float32), tf.range(5, dtype=tf.float32))).repeat(dataset_size).batch(BATCH_SIZE), imports='import tensorflow as tf;import tensorflow_hub as hub;import tensorflow_datasets as tfds', function_to_run='tf.data.Dataset.from_tensors((tf.range(5, dtype=tf.float32), tf.range(5, dtype=tf.float32))).repeat(dataset_size).batch(*args)', method_object=None, object_signature=None, function_args=[eval('BATCH_SIZE')], function_kwargs={}, max_wait_secs=0)
custom_method(
my_model.compile(optimizer='adam', loss='mean_squared_error'), imports='import tensorflow as tf;import tensorflow_hub as hub;import tensorflow_datasets as tfds', function_to_run='obj.compile(**kwargs)', method_object=eval('my_model'), object_signature='tf.keras.Sequential', function_args=[], function_kwargs={'optimizer': eval("'adam'"), 'loss': eval("'mean_squared_error'")}, max_wait_secs=0, custom_class='class SubclassedModel(tf.keras.Model):\n  """Example model defined by subclassing `tf.keras.Model`."""\n\n  output_name = \'output_layer\'\n\n  def __init__(self):\n    super(SubclassedModel, self).__init__()\n    self._dense_layer = tf.keras.layers.Dense(\n        5, dtype=tf.dtypes.float32, name=self.output_name)\n\n  def call(self, inputs):\n    return self._dense_layer(inputs)')
custom_method(
my_model.fit(dataset, epochs=2), imports='import tensorflow as tf;import tensorflow_hub as hub;import tensorflow_datasets as tfds', function_to_run='obj.fit(*args, **kwargs)', method_object=eval('my_model'), object_signature='tf.keras.Sequential', function_args=[eval('dataset')], function_kwargs={'epochs': eval('2')}, max_wait_secs=0, custom_class='class SubclassedModel(tf.keras.Model):\n  """Example model defined by subclassing `tf.keras.Model`."""\n\n  output_name = \'output_layer\'\n\n  def __init__(self):\n    super(SubclassedModel, self).__init__()\n    self._dense_layer = tf.keras.layers.Dense(\n        5, dtype=tf.dtypes.float32, name=self.output_name)\n\n  def call(self, inputs):\n    return self._dense_layer(inputs)')
print(my_model.save_spec() is None)
custom_method(
my_model.save(keras_model_path), imports='import tensorflow as tf;import tensorflow_hub as hub;import tensorflow_datasets as tfds', function_to_run='obj.save(*args)', method_object=eval('my_model'), object_signature='tf.keras.Sequential', function_args=[eval('keras_model_path')], function_kwargs={}, max_wait_secs=0, custom_class='class SubclassedModel(tf.keras.Model):\n  """Example model defined by subclassing `tf.keras.Model`."""\n\n  output_name = \'output_layer\'\n\n  def __init__(self):\n    super(SubclassedModel, self).__init__()\n    self._dense_layer = tf.keras.layers.Dense(\n        5, dtype=tf.dtypes.float32, name=self.output_name)\n\n  def call(self, inputs):\n    return self._dense_layer(inputs)')
