import tensorflow_datasets as tfds
import tensorflow as tf
import sys
from tool.client.client_config import EXPERIMENT_DIR
from tool.server.local_execution import before_execution as before_execution_INSERTED_INTO_SCRIPT
from tool.server.local_execution import after_execution as after_execution_INSERTED_INTO_SCRIPT
experiment_number = sys.argv[1]
experiment_project = sys.argv[2]
EXPERIMENT_FILE_PATH = EXPERIMENT_DIR / 'local-execution' / experiment_project / f'experiment-{experiment_number}.json'
start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT()
mirrored_strategy = tf.distribute.MirroredStrategy()
after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tf.distribute.MirroredStrategy()', method_object=None, function_args=None, function_kwargs=None)

def get_data():
    datasets = tfds.load(name='mnist', as_supervised=True)
    (mnist_train, mnist_test) = (datasets['train'], datasets['test'])
    BUFFER_SIZE = 10000
    BATCH_SIZE_PER_REPLICA = 64
    BATCH_SIZE = BATCH_SIZE_PER_REPLICA * mirrored_strategy.num_replicas_in_sync

    def scale(image, label):
        start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT()
        image = tf.cast(image, tf.float32)
        after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tf.cast()', method_object=None, function_args=[image, tf.float32], function_kwargs=None)
        image /= 255
        return (image, label)
    train_dataset = mnist_train.map(scale).cache().shuffle(BUFFER_SIZE).batch(BATCH_SIZE)
    eval_dataset = mnist_test.map(scale).batch(BATCH_SIZE)
    return (train_dataset, eval_dataset)

def get_model():
    with mirrored_strategy.scope():
        start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT()
        model = tf.keras.Sequential([tf.keras.layers.Conv2D(32, 3, activation='relu', input_shape=(28, 28, 1)), tf.keras.layers.MaxPooling2D(), tf.keras.layers.Flatten(), tf.keras.layers.Dense(64, activation='relu'), tf.keras.layers.Dense(10)])
        after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tf.keras.Sequential()', method_object=None, function_args=[[tf.keras.layers.Conv2D(32, 3, activation='relu', input_shape=(28, 28, 1)), tf.keras.layers.MaxPooling2D(), tf.keras.layers.Flatten(), tf.keras.layers.Dense(64, activation='relu'), tf.keras.layers.Dense(10)]], function_kwargs=None)
        start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT()
        model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), optimizer=tf.keras.optimizers.Adam(), metrics=[tf.metrics.SparseCategoricalAccuracy()])
        after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tf.keras.Sequential.compile()', method_object=model, function_args=None, function_kwargs={'loss': tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), 'optimizer': tf.keras.optimizers.Adam(), 'metrics': [tf.metrics.SparseCategoricalAccuracy()]})
        return model
model = get_model()
(train_dataset, eval_dataset) = get_data()
start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT()
model.fit(train_dataset, epochs=2)
after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='get_model.fit()', method_object=model, function_args=[train_dataset], function_kwargs={'epochs': 2})
keras_model_path = '/tmp/keras_save'
start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT()
model.save(keras_model_path)
after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='get_model.save()', method_object=model, function_args=[keras_model_path], function_kwargs=None)
start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT()
restored_keras_model = tf.keras.models.load_model(keras_model_path)
after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tf.keras.models.load_model()', method_object=None, function_args=[keras_model_path], function_kwargs=None)
start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT()
restored_keras_model.fit(train_dataset, epochs=2)
after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tf.keras.models.load_model.fit()', method_object=restored_keras_model, function_args=[train_dataset], function_kwargs={'epochs': 2})
start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT()
another_strategy = tf.distribute.OneDeviceStrategy('/cpu:0')
after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tf.distribute.OneDeviceStrategy()', method_object=None, function_args=['/cpu:0'], function_kwargs=None)
with another_strategy.scope():
    start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT()
    restored_keras_model_ds = tf.keras.models.load_model(keras_model_path)
    after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tf.keras.models.load_model()', method_object=None, function_args=[keras_model_path], function_kwargs=None)
    start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT()
    restored_keras_model_ds.fit(train_dataset, epochs=2)
    after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tf.keras.models.load_model.fit()', method_object=restored_keras_model_ds, function_args=[train_dataset], function_kwargs={'epochs': 2})
model = get_model()
saved_model_path = '/tmp/tf_save'
start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT()
tf.saved_model.save(model, saved_model_path)
after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tf.saved_model.save()', method_object=None, function_args=[model, saved_model_path], function_kwargs=None)
DEFAULT_FUNCTION_KEY = 'serving_default'
start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT()
loaded = tf.saved_model.load(saved_model_path)
after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tf.saved_model.load()', method_object=None, function_args=[saved_model_path], function_kwargs=None)
inference_func = loaded.signatures[DEFAULT_FUNCTION_KEY]
predict_dataset = eval_dataset.map(lambda image, label: image)
for batch in predict_dataset.take(1):
    print(inference_func(batch))
start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT()
another_strategy = tf.distribute.MirroredStrategy()
after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tf.distribute.MirroredStrategy()', method_object=None, function_args=None, function_kwargs=None)
with another_strategy.scope():
    start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT()
    loaded = tf.saved_model.load(saved_model_path)
    after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tf.saved_model.load()', method_object=None, function_args=[saved_model_path], function_kwargs=None)
    inference_func = loaded.signatures[DEFAULT_FUNCTION_KEY]
    start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT()
    dist_predict_dataset = another_strategy.experimental_distribute_dataset(predict_dataset)
    after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tf.distribute.MirroredStrategy.experimental_distribute_dataset()', method_object=another_strategy, function_args=[predict_dataset], function_kwargs=None)
    for batch in dist_predict_dataset:
        start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT()
        result = another_strategy.run(inference_func, args=(batch,))
        after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tf.distribute.MirroredStrategy.run()', method_object=another_strategy, function_args=[inference_func], function_kwargs={'args': (batch,)})
        print(result)
        break
import tensorflow_hub as hub

def build_model(loaded):
    start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT()
    x = tf.keras.layers.Input(shape=(28, 28, 1), name='input_x')
    after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tf.keras.layers.Input()', method_object=None, function_args=None, function_kwargs={'shape': (28, 28, 1), 'name': 'input_x'})
    keras_layer = hub.KerasLayer(loaded, trainable=True)(x)
    start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT()
    model = tf.keras.Model(x, keras_layer)
    after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tf.keras.Model()', method_object=None, function_args=[x, keras_layer], function_kwargs=None)
    return model
start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT()
another_strategy = tf.distribute.MirroredStrategy()
after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tf.distribute.MirroredStrategy()', method_object=None, function_args=None, function_kwargs=None)
with another_strategy.scope():
    start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT()
    loaded = tf.saved_model.load(saved_model_path)
    after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tf.saved_model.load()', method_object=None, function_args=[saved_model_path], function_kwargs=None)
    model = build_model(loaded)
    start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT()
    model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), optimizer=tf.keras.optimizers.Adam(), metrics=[tf.metrics.SparseCategoricalAccuracy()])
    after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='build_model.compile()', method_object=model, function_args=None, function_kwargs={'loss': tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), 'optimizer': tf.keras.optimizers.Adam(), 'metrics': [tf.metrics.SparseCategoricalAccuracy()]})
    start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT()
    model.fit(train_dataset, epochs=2)
    after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='build_model.fit()', method_object=model, function_args=[train_dataset], function_kwargs={'epochs': 2})
model = get_model()
start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT()
model.save(keras_model_path)
after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='get_model.save()', method_object=model, function_args=[keras_model_path], function_kwargs=None)
start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT()
another_strategy = tf.distribute.MirroredStrategy()
after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tf.distribute.MirroredStrategy()', method_object=None, function_args=None, function_kwargs=None)
with another_strategy.scope():
    start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT()
    loaded = tf.saved_model.load(keras_model_path)
    after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tf.saved_model.load()', method_object=None, function_args=[keras_model_path], function_kwargs=None)
model = get_model()
saved_model_path = '/tmp/tf_save'
start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT()
save_options = tf.saved_model.SaveOptions(experimental_io_device='/job:localhost')
after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tf.saved_model.SaveOptions()', method_object=None, function_args=None, function_kwargs={'experimental_io_device': '/job:localhost'})
start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT()
model.save(saved_model_path, options=save_options)
after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='get_model.save()', method_object=model, function_args=[saved_model_path], function_kwargs={'options': save_options})
start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT()
another_strategy = tf.distribute.MirroredStrategy()
after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tf.distribute.MirroredStrategy()', method_object=None, function_args=None, function_kwargs=None)
with another_strategy.scope():
    start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT()
    load_options = tf.saved_model.LoadOptions(experimental_io_device='/job:localhost')
    after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tf.saved_model.LoadOptions()', method_object=None, function_args=None, function_kwargs={'experimental_io_device': '/job:localhost'})
    start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT()
    loaded = tf.keras.models.load_model(saved_model_path, options=load_options)
    after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tf.keras.models.load_model()', method_object=None, function_args=[saved_model_path], function_kwargs={'options': load_options})

class SubclassedModel(tf.keras.Model):
    """Example model defined by subclassing `tf.keras.Model`."""
    output_name = 'output_layer'

    def __init__(self):
        super(SubclassedModel, self).__init__()
        start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT()
        self._dense_layer = tf.keras.layers.Dense(5, dtype=tf.dtypes.float32, name=self.output_name)
        after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tf.keras.layers.Dense()', method_object=None, function_args=[5], function_kwargs={'dtype': tf.dtypes.float32, 'name': self.output_name})

    def call(self, inputs):
        return self._dense_layer(inputs)
my_model = SubclassedModel()
try:
    start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT()
    my_model.save(keras_model_path)
    after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='SubclassedModel.save()', method_object=my_model, function_args=[keras_model_path], function_kwargs=None)
except ValueError as e:
    print(f'{type(e).__name__}: ', *e.args)
start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT()
tf.saved_model.save(my_model, saved_model_path)
after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tf.saved_model.save()', method_object=None, function_args=[my_model, saved_model_path], function_kwargs=None)
start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT()
x = tf.saved_model.load(saved_model_path)
after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tf.saved_model.load()', method_object=None, function_args=[saved_model_path], function_kwargs=None)
x.signatures
print(my_model.save_spec() is None)
BATCH_SIZE_PER_REPLICA = 4
BATCH_SIZE = BATCH_SIZE_PER_REPLICA * mirrored_strategy.num_replicas_in_sync
dataset_size = 100
start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT()
dataset = tf.data.Dataset.from_tensors((tf.range(5, dtype=tf.float32), tf.range(5, dtype=tf.float32))).repeat(dataset_size).batch(BATCH_SIZE)
after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tf.data.Dataset.from_tensors((tf.range(5, dtype=tf.float32), tf.range(5, dtype=tf.float32))).repeat(dataset_size).batch()', method_object=None, function_args=[BATCH_SIZE], function_kwargs=None)
start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT()
my_model.compile(optimizer='adam', loss='mean_squared_error')
after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='SubclassedModel.compile()', method_object=my_model, function_args=None, function_kwargs={'optimizer': 'adam', 'loss': 'mean_squared_error'})
start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT()
my_model.fit(dataset, epochs=2)
after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='SubclassedModel.fit()', method_object=my_model, function_args=[dataset], function_kwargs={'epochs': 2})
print(my_model.save_spec() is None)
start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT()
my_model.save(keras_model_path)
after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='SubclassedModel.save()', method_object=my_model, function_args=[keras_model_path], function_kwargs=None)
