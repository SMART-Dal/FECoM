import pandas as pd
import tensorflow as tf
import sys
from tool.client.client_config import EXPERIMENT_DIR
from tool.server.local_execution import before_execution as before_execution_INSERTED_INTO_SCRIPT
from tool.server.local_execution import after_execution as after_execution_INSERTED_INTO_SCRIPT
experiment_number = sys.argv[1]
experiment_project = sys.argv[2]
EXPERIMENT_FILE_PATH = EXPERIMENT_DIR / 'method-level' / experiment_project / f'experiment-{experiment_number}.json'
SHUFFLE_BUFFER = 500
BATCH_SIZE = 2
start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT()
csv_file = tf.keras.utils.get_file('heart.csv', 'https://storage.googleapis.com/download.tensorflow.org/data/heart.csv')
after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tf.keras.utils.get_file(*args)', method_object=None, object_signature=None, function_args=['heart.csv', 'https://storage.googleapis.com/download.tensorflow.org/data/heart.csv'], function_kwargs={})
df = pd.read_csv(csv_file)
df.head()
df.dtypes
target = df.pop('target')
numeric_feature_names = ['age', 'thalach', 'trestbps', 'chol', 'oldpeak']
numeric_features = df[numeric_feature_names]
numeric_features.head()
start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT()
tf.convert_to_tensor(numeric_features)
after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tf.convert_to_tensor(*args)', method_object=None, object_signature=None, function_args=[numeric_features], function_kwargs={})
start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT()
normalizer = tf.keras.layers.Normalization(axis=-1)
after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tf.keras.layers.Normalization(**kwargs)', method_object=None, object_signature=None, function_args=[], function_kwargs={'axis': -1})
start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT()
normalizer.adapt(numeric_features)
after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='obj.adapt(*args)', method_object=normalizer, object_signature=None, function_args=[numeric_features], function_kwargs={})
start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT()
normalizer(numeric_features.iloc[:3])
after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='obj(*args)', method_object=normalizer, object_signature=None, function_args=[numeric_features.iloc[:3]], function_kwargs={})

def get_basic_model():
    start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT()
    model = tf.keras.Sequential([normalizer, tf.keras.layers.Dense(10, activation='relu'), tf.keras.layers.Dense(10, activation='relu'), tf.keras.layers.Dense(1)])
    after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tf.keras.Sequential(*args)', method_object=None, object_signature=None, function_args=[[normalizer, tf.keras.layers.Dense(10, activation='relu'), tf.keras.layers.Dense(10, activation='relu'), tf.keras.layers.Dense(1)]], function_kwargs={})
    start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT()
    model.compile(optimizer='adam', loss=tf.keras.losses.BinaryCrossentropy(from_logits=True), metrics=['accuracy'])
    after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='obj.compile(**kwargs)', method_object=model, object_signature=None, function_args=[], function_kwargs={'optimizer': 'adam', 'loss': tf.keras.losses.BinaryCrossentropy(from_logits=True), 'metrics': ['accuracy']})
    return model
model = get_basic_model()
start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT()
model.fit(numeric_features, target, epochs=15, batch_size=BATCH_SIZE)
after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='obj.fit(*args, **kwargs)', method_object=model, object_signature=None, function_args=[numeric_features, target], function_kwargs={'epochs': 15, 'batch_size': BATCH_SIZE})
start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT()
numeric_dataset = tf.data.Dataset.from_tensor_slices((numeric_features, target))
after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tf.data.Dataset.from_tensor_slices(*args)', method_object=None, object_signature=None, function_args=[(numeric_features, target)], function_kwargs={})
for row in numeric_dataset.take(3):
    print(row)
start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT()
numeric_batches = numeric_dataset.shuffle(1000).batch(BATCH_SIZE)
after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='obj.shuffle(1000).batch(*args)', method_object=numeric_dataset, object_signature=None, function_args=[BATCH_SIZE], function_kwargs={})
model = get_basic_model()
start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT()
model.fit(numeric_batches, epochs=15)
after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='obj.fit(*args, **kwargs)', method_object=model, object_signature=None, function_args=[numeric_batches], function_kwargs={'epochs': 15})
start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT()
numeric_dict_ds = tf.data.Dataset.from_tensor_slices((dict(numeric_features), target))
after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tf.data.Dataset.from_tensor_slices(*args)', method_object=None, object_signature=None, function_args=[(dict(numeric_features), target)], function_kwargs={})
for row in numeric_dict_ds.take(3):
    print(row)

    def stack_dict(inputs, fun=tf.stack):
        values = []
        for key in sorted(inputs.keys()):
            values.append(tf.cast(inputs[key], tf.float32))
        return fun(values, axis=-1)

class MyModel(tf.keras.Model):

    def __init__(self):
        super().__init__(self)
        start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT()
        self.normalizer = tf.keras.layers.Normalization(axis=-1)
        after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tf.keras.layers.Normalization(**kwargs)', method_object=None, object_signature=None, function_args=[], function_kwargs={'axis': -1})
        start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT()
        self.seq = tf.keras.Sequential([self.normalizer, tf.keras.layers.Dense(10, activation='relu'), tf.keras.layers.Dense(10, activation='relu'), tf.keras.layers.Dense(1)])
        after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tf.keras.Sequential(*args)', method_object=None, object_signature=None, function_args=[[self.normalizer, tf.keras.layers.Dense(10, activation='relu'), tf.keras.layers.Dense(10, activation='relu'), tf.keras.layers.Dense(1)]], function_kwargs={})

    def adapt(self, inputs):
        inputs = stack_dict(inputs)
        self.normalizer.adapt(inputs)

    def call(self, inputs):
        inputs = stack_dict(inputs)
        result = self.seq(inputs)
        return result
model = MyModel()
start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT()
model.adapt(dict(numeric_features))
after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='obj.adapt(*args)', method_object='model', object_signature='MyModel()', function_args=[dict(numeric_features)], function_kwargs={})
start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT()
model.compile(optimizer='adam', loss=tf.keras.losses.BinaryCrossentropy(from_logits=True), metrics=['accuracy'], run_eagerly=True)
after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='obj.compile(**kwargs)', method_object='model', object_signature='MyModel()', function_args=[], function_kwargs={'optimizer': 'adam', 'loss': tf.keras.losses.BinaryCrossentropy(from_logits=True), 'metrics': ['accuracy'], 'run_eagerly': True})
start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT()
model.fit(dict(numeric_features), target, epochs=5, batch_size=BATCH_SIZE)
after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='obj.fit(*args, **kwargs)', method_object='model', object_signature='MyModel()', function_args=[dict(numeric_features), target], function_kwargs={'epochs': 5, 'batch_size': BATCH_SIZE})
start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT()
numeric_dict_batches = numeric_dict_ds.shuffle(SHUFFLE_BUFFER).batch(BATCH_SIZE)
after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='obj.shuffle(SHUFFLE_BUFFER).batch(*args)', method_object=numeric_dict_ds, object_signature=None, function_args=[BATCH_SIZE], function_kwargs={})
start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT()
model.fit(numeric_dict_batches, epochs=5)
after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='obj.fit(*args, **kwargs)', method_object='model', object_signature=None, function_args=[numeric_dict_batches], function_kwargs={'epochs': 5})
start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT()
model.predict(dict(numeric_features.iloc[:3]))
after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='obj.predict(*args)', method_object='model', object_signature=None, function_args=[dict(numeric_features.iloc[:3])], function_kwargs={})
inputs = {}
for (name, column) in numeric_features.items():
    start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT()
    inputs[name] = tf.keras.Input(shape=(1,), name=name, dtype=tf.float32)
    after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tf.keras.Input(**kwargs)', method_object=None, object_signature=None, function_args=[], function_kwargs={'shape': (1,), 'name': name, 'dtype': tf.float32})
inputs
x = stack_dict(inputs, fun=tf.concat)
start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT()
normalizer = tf.keras.layers.Normalization(axis=-1)
after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tf.keras.layers.Normalization(**kwargs)', method_object=None, object_signature=None, function_args=[], function_kwargs={'axis': -1})
start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT()
normalizer.adapt(stack_dict(dict(numeric_features)))
after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='obj.adapt(*args)', method_object=normalizer, object_signature=None, function_args=[stack_dict(dict(numeric_features))], function_kwargs={})
start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT()
x = normalizer(x)
after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='obj(*args)', method_object=normalizer, object_signature=None, function_args=[x], function_kwargs={})
start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT()
x = tf.keras.layers.Dense(10, activation='relu')(x)
after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run="tf.keras.layers.Dense(10, activation='relu')(*args)", method_object=None, object_signature=None, function_args=[x], function_kwargs={})
start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT()
x = tf.keras.layers.Dense(10, activation='relu')(x)
after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run="tf.keras.layers.Dense(10, activation='relu')(*args)", method_object=None, object_signature=None, function_args=[x], function_kwargs={})
start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT()
x = tf.keras.layers.Dense(1)(x)
after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tf.keras.layers.Dense(1)(*args)', method_object=None, object_signature=None, function_args=[x], function_kwargs={})
start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT()
model = tf.keras.Model(inputs, x)
after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tf.keras.Model(*args)', method_object=None, object_signature=None, function_args=[inputs, x], function_kwargs={})
start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT()
model.compile(optimizer='adam', loss=tf.keras.losses.BinaryCrossentropy(from_logits=True), metrics=['accuracy'], run_eagerly=True)
after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='obj.compile(**kwargs)', method_object='model', object_signature='MyModel()', function_args=[], function_kwargs={'optimizer': 'adam', 'loss': tf.keras.losses.BinaryCrossentropy(from_logits=True), 'metrics': ['accuracy'], 'run_eagerly': True})
start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT()
tf.keras.utils.plot_model(model, rankdir='LR', show_shapes=True)
after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tf.keras.utils.plot_model(*args, **kwargs)', method_object=None, object_signature=None, function_args=[model], function_kwargs={'rankdir': 'LR', 'show_shapes': True})
start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT()
model.fit(dict(numeric_features), target, epochs=5, batch_size=BATCH_SIZE)
after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='obj.fit(*args, **kwargs)', method_object='model', object_signature='MyModel()', function_args=[dict(numeric_features), target], function_kwargs={'epochs': 5, 'batch_size': BATCH_SIZE})
start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT()
numeric_dict_batches = numeric_dict_ds.shuffle(SHUFFLE_BUFFER).batch(BATCH_SIZE)
after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='obj.shuffle(SHUFFLE_BUFFER).batch(*args)', method_object=numeric_dict_ds, object_signature=None, function_args=[BATCH_SIZE], function_kwargs={})
start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT()
model.fit(numeric_dict_batches, epochs=5)
after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='obj.fit(*args, **kwargs)', method_object='model', object_signature=None, function_args=[numeric_dict_batches], function_kwargs={'epochs': 5})
binary_feature_names = ['sex', 'fbs', 'exang']
categorical_feature_names = ['cp', 'restecg', 'slope', 'thal', 'ca']
inputs = {}
for (name, column) in df.items():
    if type(column[0]) == str:
        dtype = tf.string
    elif name in categorical_feature_names or name in binary_feature_names:
        dtype = tf.int64
    else:
        dtype = tf.float32
    start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT()
    inputs[name] = tf.keras.Input(shape=(), name=name, dtype=dtype)
    after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tf.keras.Input(**kwargs)', method_object=None, object_signature=None, function_args=[], function_kwargs={'shape': (), 'name': name, 'dtype': dtype})
inputs
preprocessed = []
for name in binary_feature_names:
    inp = inputs[name]
    inp = inp[:, tf.newaxis]
    start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT()
    float_value = tf.cast(inp, tf.float32)
    after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tf.cast(*args)', method_object=None, object_signature=None, function_args=[inp, tf.float32], function_kwargs={})
    preprocessed.append(float_value)
preprocessed
start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT()
normalizer = tf.keras.layers.Normalization(axis=-1)
after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tf.keras.layers.Normalization(**kwargs)', method_object=None, object_signature=None, function_args=[], function_kwargs={'axis': -1})
start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT()
normalizer.adapt(stack_dict(dict(numeric_features)))
after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='obj.adapt(*args)', method_object=normalizer, object_signature=None, function_args=[stack_dict(dict(numeric_features))], function_kwargs={})
numeric_inputs = {}
for name in numeric_feature_names:
    numeric_inputs[name] = inputs[name]
numeric_inputs = stack_dict(numeric_inputs)
start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT()
numeric_normalized = normalizer(numeric_inputs)
after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='obj(*args)', method_object=normalizer, object_signature=None, function_args=[numeric_inputs], function_kwargs={})
preprocessed.append(numeric_normalized)
preprocessed
vocab = ['a', 'b', 'c']
start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT()
lookup = tf.keras.layers.StringLookup(vocabulary=vocab, output_mode='one_hot')
after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tf.keras.layers.StringLookup(**kwargs)', method_object=None, object_signature=None, function_args=[], function_kwargs={'vocabulary': vocab, 'output_mode': 'one_hot'})
start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT()
lookup(['c', 'a', 'a', 'b', 'zzz'])
after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='obj(*args)', method_object=lookup, object_signature=None, function_args=[['c', 'a', 'a', 'b', 'zzz']], function_kwargs={})
vocab = [1, 4, 7, 99]
start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT()
lookup = tf.keras.layers.IntegerLookup(vocabulary=vocab, output_mode='one_hot')
after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tf.keras.layers.IntegerLookup(**kwargs)', method_object=None, object_signature=None, function_args=[], function_kwargs={'vocabulary': vocab, 'output_mode': 'one_hot'})
start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT()
lookup([-1, 4, 1])
after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='obj(*args)', method_object=lookup, object_signature=None, function_args=[[-1, 4, 1]], function_kwargs={})
for name in categorical_feature_names:
    vocab = sorted(set(df[name]))
    print(f'name: {name}')
    print(f'vocab: {vocab}\n')
    if type(vocab[0]) is str:
        start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT()
        lookup = tf.keras.layers.StringLookup(vocabulary=vocab, output_mode='one_hot')
        after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tf.keras.layers.StringLookup(**kwargs)', method_object=None, object_signature=None, function_args=[], function_kwargs={'vocabulary': vocab, 'output_mode': 'one_hot'})
    else:
        start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT()
        lookup = tf.keras.layers.IntegerLookup(vocabulary=vocab, output_mode='one_hot')
        after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tf.keras.layers.IntegerLookup(**kwargs)', method_object=None, object_signature=None, function_args=[], function_kwargs={'vocabulary': vocab, 'output_mode': 'one_hot'})
    x = inputs[name][:, tf.newaxis]
    start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT()
    x = lookup(x)
    after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='obj(*args)', method_object=lookup, object_signature=None, function_args=[x], function_kwargs={})
    preprocessed.append(x)
preprocessed
start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT()
preprocesssed_result = tf.concat(preprocessed, axis=-1)
after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tf.concat(*args, **kwargs)', method_object=None, object_signature=None, function_args=[preprocessed], function_kwargs={'axis': -1})
preprocesssed_result
start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT()
preprocessor = tf.keras.Model(inputs, preprocesssed_result)
after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tf.keras.Model(*args)', method_object=None, object_signature=None, function_args=[inputs, preprocesssed_result], function_kwargs={})
start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT()
tf.keras.utils.plot_model(preprocessor, rankdir='LR', show_shapes=True)
after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tf.keras.utils.plot_model(*args, **kwargs)', method_object=None, object_signature=None, function_args=[preprocessor], function_kwargs={'rankdir': 'LR', 'show_shapes': True})
start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT()
preprocessor(dict(df.iloc[:1]))
after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='obj(*args)', method_object=preprocessor, object_signature=None, function_args=[dict(df.iloc[:1])], function_kwargs={})
start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT()
body = tf.keras.Sequential([tf.keras.layers.Dense(10, activation='relu'), tf.keras.layers.Dense(10, activation='relu'), tf.keras.layers.Dense(1)])
after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tf.keras.Sequential(*args)', method_object=None, object_signature=None, function_args=[[tf.keras.layers.Dense(10, activation='relu'), tf.keras.layers.Dense(10, activation='relu'), tf.keras.layers.Dense(1)]], function_kwargs={})
inputs
start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT()
x = preprocessor(inputs)
after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='obj(*args)', method_object=preprocessor, object_signature=None, function_args=[inputs], function_kwargs={})
x
start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT()
result = body(x)
after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='obj(*args)', method_object=body, object_signature=None, function_args=[x], function_kwargs={})
result
start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT()
model = tf.keras.Model(inputs, result)
after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tf.keras.Model(*args)', method_object=None, object_signature=None, function_args=[inputs, result], function_kwargs={})
start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT()
model.compile(optimizer='adam', loss=tf.keras.losses.BinaryCrossentropy(from_logits=True), metrics=['accuracy'])
after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='obj.compile(**kwargs)', method_object='model', object_signature='MyModel()', function_args=[], function_kwargs={'optimizer': 'adam', 'loss': tf.keras.losses.BinaryCrossentropy(from_logits=True), 'metrics': ['accuracy']})
start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT()
history = model.fit(dict(df), target, epochs=5, batch_size=BATCH_SIZE)
after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='obj.fit(*args, **kwargs)', method_object='model', object_signature=None, function_args=[dict(df), target], function_kwargs={'epochs': 5, 'batch_size': BATCH_SIZE})
start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT()
ds = tf.data.Dataset.from_tensor_slices((dict(df), target))
after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tf.data.Dataset.from_tensor_slices(*args)', method_object=None, object_signature=None, function_args=[(dict(df), target)], function_kwargs={})
start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT()
ds = ds.batch(BATCH_SIZE)
after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='obj.batch(*args)', method_object=ds, object_signature=None, function_args=[BATCH_SIZE], function_kwargs={})
import pprint
for (x, y) in ds.take(1):
    pprint.pprint(x)
    print()
    print(y)
start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT()
history = model.fit(ds, epochs=5)
after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='obj.fit(*args, **kwargs)', method_object='model', object_signature=None, function_args=[ds], function_kwargs={'epochs': 5})
