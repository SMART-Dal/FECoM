import sys
from fecom.patching.patching_config import EXPERIMENT_DIR
from fecom.measurement.execution import before_execution as before_execution_INSERTED_INTO_SCRIPT
from fecom.measurement.execution import after_execution as after_execution_INSERTED_INTO_SCRIPT
from fecom.experiment.experiment_kinds import ExperimentKinds
experiment_number = sys.argv[1]
experiment_project = sys.argv[2]
EXPERIMENT_FILE_PATH = EXPERIMENT_DIR / ExperimentKinds.PROJECT_LEVEL.value / experiment_project / f'experiment-{experiment_number}.json'
start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT(experiment_file_path=EXPERIMENT_FILE_PATH, enable_skip_calls=False)
import tensorflow_datasets as tfds
import tensorflow as tf
mirrored_strategy = tf.distribute.MirroredStrategy()

def get_data():
    datasets = tfds.load(name='mnist', as_supervised=True)
    (mnist_train, mnist_test) = (datasets['train'], datasets['test'])
    BUFFER_SIZE = 10000
    BATCH_SIZE_PER_REPLICA = 64
    BATCH_SIZE = BATCH_SIZE_PER_REPLICA * mirrored_strategy.num_replicas_in_sync

    def scale(image, label):
        image = tf.cast(image, tf.float32)
        image /= 255
        return (image, label)
    train_dataset = mnist_train.map(scale).cache().shuffle(BUFFER_SIZE).batch(BATCH_SIZE)
    eval_dataset = mnist_test.map(scale).batch(BATCH_SIZE)
    return (train_dataset, eval_dataset)

def get_model():
    with mirrored_strategy.scope():
        model = tf.keras.Sequential([tf.keras.layers.Conv2D(32, 3, activation='relu', input_shape=(28, 28, 1)), tf.keras.layers.MaxPooling2D(), tf.keras.layers.Flatten(), tf.keras.layers.Dense(64, activation='relu'), tf.keras.layers.Dense(10)])
        model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), optimizer=tf.keras.optimizers.Adam(), metrics=[tf.metrics.SparseCategoricalAccuracy()])
        return model
model = get_model()
(train_dataset, eval_dataset) = get_data()
model.fit(train_dataset, epochs=2)
keras_model_path = '/tmp/keras_save'
model.save(keras_model_path)
restored_keras_model = tf.keras.models.load_model(keras_model_path)
restored_keras_model.fit(train_dataset, epochs=2)
another_strategy = tf.distribute.OneDeviceStrategy('/cpu:0')
with another_strategy.scope():
    restored_keras_model_ds = tf.keras.models.load_model(keras_model_path)
    restored_keras_model_ds.fit(train_dataset, epochs=2)
model = get_model()
saved_model_path = '/tmp/tf_save'
tf.saved_model.save(model, saved_model_path)
DEFAULT_FUNCTION_KEY = 'serving_default'
loaded = tf.saved_model.load(saved_model_path)
inference_func = loaded.signatures[DEFAULT_FUNCTION_KEY]
predict_dataset = eval_dataset.map(lambda image, label: image)
for batch in predict_dataset.take(1):
    print(inference_func(batch))
another_strategy = tf.distribute.MirroredStrategy()
with another_strategy.scope():
    loaded = tf.saved_model.load(saved_model_path)
    inference_func = loaded.signatures[DEFAULT_FUNCTION_KEY]
    dist_predict_dataset = another_strategy.experimental_distribute_dataset(predict_dataset)
    for batch in dist_predict_dataset:
        result = another_strategy.run(inference_func, args=(batch,))
        print(result)
        break
import tensorflow_hub as hub

def build_model(loaded):
    x = tf.keras.layers.Input(shape=(28, 28, 1), name='input_x')
    keras_layer = hub.KerasLayer(loaded, trainable=True)(x)
    model = tf.keras.Model(x, keras_layer)
    return model
another_strategy = tf.distribute.MirroredStrategy()
with another_strategy.scope():
    loaded = tf.saved_model.load(saved_model_path)
    model = build_model(loaded)
    model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), optimizer=tf.keras.optimizers.Adam(), metrics=[tf.metrics.SparseCategoricalAccuracy()])
    model.fit(train_dataset, epochs=2)
model = get_model()
model.save(keras_model_path)
another_strategy = tf.distribute.MirroredStrategy()
with another_strategy.scope():
    loaded = tf.saved_model.load(keras_model_path)
model = get_model()
saved_model_path = '/tmp/tf_save'
save_options = tf.saved_model.SaveOptions(experimental_io_device='/job:localhost')
model.save(saved_model_path, options=save_options)
another_strategy = tf.distribute.MirroredStrategy()
with another_strategy.scope():
    load_options = tf.saved_model.LoadOptions(experimental_io_device='/job:localhost')
    loaded = tf.keras.models.load_model(saved_model_path, options=load_options)

class SubclassedModel(tf.keras.Model):
    """Example model defined by subclassing `tf.keras.Model`."""
    output_name = 'output_layer'

    def __init__(self):
        super(SubclassedModel, self).__init__()
        self._dense_layer = tf.keras.layers.Dense(5, dtype=tf.dtypes.float32, name=self.output_name)

    def call(self, inputs):
        return self._dense_layer(inputs)
my_model = SubclassedModel()
try:
    my_model.save(keras_model_path)
except ValueError as e:
    print(f'{type(e).__name__}: ', *e.args)
tf.saved_model.save(my_model, saved_model_path)
x = tf.saved_model.load(saved_model_path)
x.signatures
print(my_model.save_spec() is None)
BATCH_SIZE_PER_REPLICA = 4
BATCH_SIZE = BATCH_SIZE_PER_REPLICA * mirrored_strategy.num_replicas_in_sync
dataset_size = 100
dataset = tf.data.Dataset.from_tensors((tf.range(5, dtype=tf.float32), tf.range(5, dtype=tf.float32))).repeat(dataset_size).batch(BATCH_SIZE)
my_model.compile(optimizer='adam', loss='mean_squared_error')
my_model.fit(dataset, epochs=2)
print(my_model.save_spec() is None)
my_model.save(keras_model_path)
after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, enable_skip_calls=False)
