import pandas as pd
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

def custom_method(func, imports: str, function_to_run: str, method_object=None, object_signature=None, function_args: list=None, function_kwargs: dict=None, custom_class=None):
    result = send_request(imports=imports, function_to_run=function_to_run, function_args=function_args, function_kwargs=function_kwargs, max_wait_secs=MAX_WAIT_S, wait_after_run_secs=WAIT_AFTER_RUN_S, method_object=method_object, object_signature=object_signature, custom_class=custom_class, experiment_file_path=EXPERIMENT_FILE_PATH)
    return func
SHUFFLE_BUFFER = 500
BATCH_SIZE = 2
csv_file = custom_method(
tf.keras.utils.get_file('heart.csv', 'https://storage.googleapis.com/download.tensorflow.org/data/heart.csv'), imports='import pprint;import tensorflow as tf;import pandas as pd', function_to_run='tf.keras.utils.get_file(*args)', method_object=None, object_signature=None, function_args=[eval("'heart.csv'"), eval("'https://storage.googleapis.com/download.tensorflow.org/data/heart.csv'")], function_kwargs={})
df = pd.read_csv(csv_file)
df.head()
df.dtypes
target = df.pop('target')
numeric_feature_names = ['age', 'thalach', 'trestbps', 'chol', 'oldpeak']
numeric_features = df[numeric_feature_names]
numeric_features.head()
custom_method(
tf.convert_to_tensor(numeric_features), imports='import pprint;import tensorflow as tf;import pandas as pd', function_to_run='tf.convert_to_tensor(*args)', method_object=None, object_signature=None, function_args=[eval('numeric_features')], function_kwargs={})
normalizer = custom_method(
tf.keras.layers.Normalization(axis=-1), imports='import pprint;import tensorflow as tf;import pandas as pd', function_to_run='tf.keras.layers.Normalization(**kwargs)', method_object=None, object_signature=None, function_args=[], function_kwargs={'axis': eval('-1')})
custom_method(
normalizer.adapt(numeric_features), imports='import pprint;import tensorflow as tf;import pandas as pd', function_to_run='obj.adapt(*args)', method_object=eval('normalizer'), object_signature=None, function_args=[eval('numeric_features')], function_kwargs={}, custom_class=None)
custom_method(
normalizer(numeric_features.iloc[:3]), imports='import pprint;import tensorflow as tf;import pandas as pd', function_to_run='obj(*args)', method_object=eval('normalizer'), object_signature=None, function_args=[eval('numeric_features.iloc[:3]')], function_kwargs={}, custom_class=None)

def get_basic_model():
    model = custom_method(
    tf.keras.Sequential([normalizer, tf.keras.layers.Dense(10, activation='relu'), tf.keras.layers.Dense(10, activation='relu'), tf.keras.layers.Dense(1)]), imports='import pprint;import tensorflow as tf;import pandas as pd', function_to_run='tf.keras.Sequential(*args)', method_object=None, object_signature=None, function_args=[eval("[\n    normalizer,\n    tf.keras.layers.Dense(10, activation='relu'),\n    tf.keras.layers.Dense(10, activation='relu'),\n    tf.keras.layers.Dense(1)\n  ]")], function_kwargs={})
    custom_method(
    model.compile(optimizer='adam', loss=tf.keras.losses.BinaryCrossentropy(from_logits=True), metrics=['accuracy']), imports='import pprint;import tensorflow as tf;import pandas as pd', function_to_run='obj.compile(**kwargs)', method_object=eval('model'), object_signature=None, function_args=[], function_kwargs={'optimizer': eval("'adam'"), 'loss': eval('tf.keras.losses.BinaryCrossentropy(from_logits=True)'), 'metrics': eval("['accuracy']")}, custom_class=None)
    return model
model = get_basic_model()
custom_method(
model.fit(numeric_features, target, epochs=15, batch_size=BATCH_SIZE), imports='import pprint;import tensorflow as tf;import pandas as pd', function_to_run='obj.fit(*args, **kwargs)', method_object=eval('model'), object_signature=None, function_args=[eval('numeric_features'), eval('target')], function_kwargs={'epochs': eval('15'), 'batch_size': eval('BATCH_SIZE')}, custom_class=None)
numeric_dataset = custom_method(
tf.data.Dataset.from_tensor_slices((numeric_features, target)), imports='import pprint;import tensorflow as tf;import pandas as pd', function_to_run='tf.data.Dataset.from_tensor_slices(*args)', method_object=None, object_signature=None, function_args=[eval('(numeric_features, target)')], function_kwargs={})
for row in custom_method(
numeric_dataset.take(3), imports='import pprint;import tensorflow as tf;import pandas as pd', function_to_run='obj.take(*args)', method_object=eval('numeric_dataset'), object_signature=None, function_args=[eval('3')], function_kwargs={}, custom_class=None):
    print(row)
numeric_batches = custom_method(
numeric_dataset.shuffle(1000).batch(BATCH_SIZE), imports='import pprint;import tensorflow as tf;import pandas as pd', function_to_run='obj.shuffle(1000).batch(*args)', method_object=eval('numeric_dataset'), object_signature=None, function_args=[eval('BATCH_SIZE')], function_kwargs={}, custom_class=None)
model = get_basic_model()
custom_method(
model.fit(numeric_batches, epochs=15), imports='import pprint;import tensorflow as tf;import pandas as pd', function_to_run='obj.fit(*args, **kwargs)', method_object=eval('model'), object_signature=None, function_args=[eval('numeric_batches')], function_kwargs={'epochs': eval('15')}, custom_class=None)
numeric_dict_ds = custom_method(
tf.data.Dataset.from_tensor_slices((dict(numeric_features), target)), imports='import pprint;import tensorflow as tf;import pandas as pd', function_to_run='tf.data.Dataset.from_tensor_slices(*args)', method_object=None, object_signature=None, function_args=[eval('(dict(numeric_features), target)')], function_kwargs={})
for row in custom_method(
numeric_dict_ds.take(3), imports='import pprint;import tensorflow as tf;import pandas as pd', function_to_run='obj.take(*args)', method_object=eval('numeric_dict_ds'), object_signature=None, function_args=[eval('3')], function_kwargs={}, custom_class=None):
    print(row)

    def stack_dict(inputs, fun=tf.stack):
        values = []
        for key in sorted(inputs.keys()):
            values.append(tf.cast(inputs[key], tf.float32))
        return fun(values, axis=-1)

class MyModel(tf.keras.Model):

    def __init__(self):
        super().__init__(self)
        self.normalizer = custom_method(
        tf.keras.layers.Normalization(axis=-1), imports='import pprint;import tensorflow as tf;import pandas as pd', function_to_run='tf.keras.layers.Normalization(**kwargs)', method_object=None, object_signature=None, function_args=[], function_kwargs={'axis': eval('-1')})
        self.seq = custom_method(
        tf.keras.Sequential([self.normalizer, tf.keras.layers.Dense(10, activation='relu'), tf.keras.layers.Dense(10, activation='relu'), tf.keras.layers.Dense(1)]), imports='import pprint;import tensorflow as tf;import pandas as pd', function_to_run='tf.keras.Sequential(*args)', method_object=None, object_signature=None, function_args=[eval("[\n      self.normalizer,\n      tf.keras.layers.Dense(10, activation='relu'),\n      tf.keras.layers.Dense(10, activation='relu'),\n      tf.keras.layers.Dense(1)\n    ]")], function_kwargs={})

    def adapt(self, inputs):
        inputs = stack_dict(inputs)
        self.normalizer.adapt(inputs)

    def call(self, inputs):
        inputs = stack_dict(inputs)
        result = self.seq(inputs)
        return result
model = MyModel()
custom_method(
model.adapt(dict(numeric_features)), imports='import pprint;import tensorflow as tf;import pandas as pd', function_to_run='obj.adapt(*args)', method_object=eval('model'), object_signature='MyModel()', function_args=[eval('dict(numeric_features)')], function_kwargs={}, custom_class="class MyModel(tf.keras.Model):\n  def __init__(self):\n    super().__init__(self)\n\n    self.normalizer = tf.keras.layers.Normalization(axis=-1)\n\n    self.seq = tf.keras.Sequential([\n      self.normalizer,\n      tf.keras.layers.Dense(10, activation='relu'),\n      tf.keras.layers.Dense(10, activation='relu'),\n      tf.keras.layers.Dense(1)\n    ])\n\n  def adapt(self, inputs):\n    inputs = stack_dict(inputs)\n    self.normalizer.adapt(inputs)\n\n  def call(self, inputs):\n    inputs = stack_dict(inputs)\n    result = self.seq(inputs)\n\n    return result")
custom_method(
model.compile(optimizer='adam', loss=tf.keras.losses.BinaryCrossentropy(from_logits=True), metrics=['accuracy'], run_eagerly=True), imports='import pprint;import tensorflow as tf;import pandas as pd', function_to_run='obj.compile(**kwargs)', method_object=eval('model'), object_signature='MyModel()', function_args=[], function_kwargs={'optimizer': eval("'adam'"), 'loss': eval('tf.keras.losses.BinaryCrossentropy(from_logits=True)'), 'metrics': eval("['accuracy']"), 'run_eagerly': eval('True')}, custom_class="class MyModel(tf.keras.Model):\n  def __init__(self):\n    super().__init__(self)\n\n    self.normalizer = tf.keras.layers.Normalization(axis=-1)\n\n    self.seq = tf.keras.Sequential([\n      self.normalizer,\n      tf.keras.layers.Dense(10, activation='relu'),\n      tf.keras.layers.Dense(10, activation='relu'),\n      tf.keras.layers.Dense(1)\n    ])\n\n  def adapt(self, inputs):\n    inputs = stack_dict(inputs)\n    self.normalizer.adapt(inputs)\n\n  def call(self, inputs):\n    inputs = stack_dict(inputs)\n    result = self.seq(inputs)\n\n    return result")
custom_method(
model.fit(dict(numeric_features), target, epochs=5, batch_size=BATCH_SIZE), imports='import pprint;import tensorflow as tf;import pandas as pd', function_to_run='obj.fit(*args, **kwargs)', method_object=eval('model'), object_signature='MyModel()', function_args=[eval('dict(numeric_features)'), eval('target')], function_kwargs={'epochs': eval('5'), 'batch_size': eval('BATCH_SIZE')}, custom_class="class MyModel(tf.keras.Model):\n  def __init__(self):\n    super().__init__(self)\n\n    self.normalizer = tf.keras.layers.Normalization(axis=-1)\n\n    self.seq = tf.keras.Sequential([\n      self.normalizer,\n      tf.keras.layers.Dense(10, activation='relu'),\n      tf.keras.layers.Dense(10, activation='relu'),\n      tf.keras.layers.Dense(1)\n    ])\n\n  def adapt(self, inputs):\n    inputs = stack_dict(inputs)\n    self.normalizer.adapt(inputs)\n\n  def call(self, inputs):\n    inputs = stack_dict(inputs)\n    result = self.seq(inputs)\n\n    return result")
numeric_dict_batches = custom_method(
numeric_dict_ds.shuffle(SHUFFLE_BUFFER).batch(BATCH_SIZE), imports='import pprint;import tensorflow as tf;import pandas as pd', function_to_run='obj.shuffle(SHUFFLE_BUFFER).batch(*args)', method_object=eval('numeric_dict_ds'), object_signature=None, function_args=[eval('BATCH_SIZE')], function_kwargs={}, custom_class=None)
custom_method(
model.fit(numeric_dict_batches, epochs=5), imports='import pprint;import tensorflow as tf;import pandas as pd', function_to_run='obj.fit(*args, **kwargs)', method_object=eval('model'), object_signature=None, function_args=[eval('numeric_dict_batches')], function_kwargs={'epochs': eval('5')}, custom_class="class MyModel(tf.keras.Model):\n  def __init__(self):\n    super().__init__(self)\n\n    self.normalizer = tf.keras.layers.Normalization(axis=-1)\n\n    self.seq = tf.keras.Sequential([\n      self.normalizer,\n      tf.keras.layers.Dense(10, activation='relu'),\n      tf.keras.layers.Dense(10, activation='relu'),\n      tf.keras.layers.Dense(1)\n    ])\n\n  def adapt(self, inputs):\n    inputs = stack_dict(inputs)\n    self.normalizer.adapt(inputs)\n\n  def call(self, inputs):\n    inputs = stack_dict(inputs)\n    result = self.seq(inputs)\n\n    return result")
custom_method(
model.predict(dict(numeric_features.iloc[:3])), imports='import pprint;import tensorflow as tf;import pandas as pd', function_to_run='obj.predict(*args)', method_object=eval('model'), object_signature=None, function_args=[eval('dict(numeric_features.iloc[:3])')], function_kwargs={}, custom_class="class MyModel(tf.keras.Model):\n  def __init__(self):\n    super().__init__(self)\n\n    self.normalizer = tf.keras.layers.Normalization(axis=-1)\n\n    self.seq = tf.keras.Sequential([\n      self.normalizer,\n      tf.keras.layers.Dense(10, activation='relu'),\n      tf.keras.layers.Dense(10, activation='relu'),\n      tf.keras.layers.Dense(1)\n    ])\n\n  def adapt(self, inputs):\n    inputs = stack_dict(inputs)\n    self.normalizer.adapt(inputs)\n\n  def call(self, inputs):\n    inputs = stack_dict(inputs)\n    result = self.seq(inputs)\n\n    return result")
inputs = {}
for (name, column) in numeric_features.items():
    inputs[name] = custom_method(
    tf.keras.Input(shape=(1,), name=name, dtype=tf.float32), imports='import pprint;import tensorflow as tf;import pandas as pd', function_to_run='tf.keras.Input(**kwargs)', method_object=None, object_signature=None, function_args=[], function_kwargs={'shape': eval('(1,)'), 'name': eval('name'), 'dtype': eval('tf.float32')})
inputs
x = stack_dict(inputs, fun=tf.concat)
normalizer = custom_method(
tf.keras.layers.Normalization(axis=-1), imports='import pprint;import tensorflow as tf;import pandas as pd', function_to_run='tf.keras.layers.Normalization(**kwargs)', method_object=None, object_signature=None, function_args=[], function_kwargs={'axis': eval('-1')})
custom_method(
normalizer.adapt(stack_dict(dict(numeric_features))), imports='import pprint;import tensorflow as tf;import pandas as pd', function_to_run='obj.adapt(*args)', method_object=eval('normalizer'), object_signature=None, function_args=[eval('stack_dict(dict(numeric_features))')], function_kwargs={}, custom_class=None)
x = custom_method(
normalizer(x), imports='import pprint;import tensorflow as tf;import pandas as pd', function_to_run='obj(*args)', method_object=eval('normalizer'), object_signature=None, function_args=[eval('x')], function_kwargs={}, custom_class=None)
x = custom_method(
tf.keras.layers.Dense(10, activation='relu')(x), imports='import pprint;import tensorflow as tf;import pandas as pd', function_to_run="tf.keras.layers.Dense(10, activation='relu')(*args)", method_object=None, object_signature=None, function_args=[eval('x')], function_kwargs={})
x = custom_method(
tf.keras.layers.Dense(10, activation='relu')(x), imports='import pprint;import tensorflow as tf;import pandas as pd', function_to_run="tf.keras.layers.Dense(10, activation='relu')(*args)", method_object=None, object_signature=None, function_args=[eval('x')], function_kwargs={})
x = custom_method(
tf.keras.layers.Dense(1)(x), imports='import pprint;import tensorflow as tf;import pandas as pd', function_to_run='tf.keras.layers.Dense(1)(*args)', method_object=None, object_signature=None, function_args=[eval('x')], function_kwargs={})
model = custom_method(
tf.keras.Model(inputs, x), imports='import pprint;import tensorflow as tf;import pandas as pd', function_to_run='tf.keras.Model(*args)', method_object=None, object_signature=None, function_args=[eval('inputs'), eval('x')], function_kwargs={})
custom_method(
model.compile(optimizer='adam', loss=tf.keras.losses.BinaryCrossentropy(from_logits=True), metrics=['accuracy'], run_eagerly=True), imports='import pprint;import tensorflow as tf;import pandas as pd', function_to_run='obj.compile(**kwargs)', method_object=eval('model'), object_signature='MyModel()', function_args=[], function_kwargs={'optimizer': eval("'adam'"), 'loss': eval('tf.keras.losses.BinaryCrossentropy(from_logits=True)'), 'metrics': eval("['accuracy']"), 'run_eagerly': eval('True')}, custom_class="class MyModel(tf.keras.Model):\n  def __init__(self):\n    super().__init__(self)\n\n    self.normalizer = tf.keras.layers.Normalization(axis=-1)\n\n    self.seq = tf.keras.Sequential([\n      self.normalizer,\n      tf.keras.layers.Dense(10, activation='relu'),\n      tf.keras.layers.Dense(10, activation='relu'),\n      tf.keras.layers.Dense(1)\n    ])\n\n  def adapt(self, inputs):\n    inputs = stack_dict(inputs)\n    self.normalizer.adapt(inputs)\n\n  def call(self, inputs):\n    inputs = stack_dict(inputs)\n    result = self.seq(inputs)\n\n    return result")
custom_method(
tf.keras.utils.plot_model(model, rankdir='LR', show_shapes=True), imports='import pprint;import tensorflow as tf;import pandas as pd', function_to_run='tf.keras.utils.plot_model(*args, **kwargs)', method_object=None, object_signature=None, function_args=[eval('model')], function_kwargs={'rankdir': eval('"LR"'), 'show_shapes': eval('True')})
custom_method(
model.fit(dict(numeric_features), target, epochs=5, batch_size=BATCH_SIZE), imports='import pprint;import tensorflow as tf;import pandas as pd', function_to_run='obj.fit(*args, **kwargs)', method_object=eval('model'), object_signature='MyModel()', function_args=[eval('dict(numeric_features)'), eval('target')], function_kwargs={'epochs': eval('5'), 'batch_size': eval('BATCH_SIZE')}, custom_class="class MyModel(tf.keras.Model):\n  def __init__(self):\n    super().__init__(self)\n\n    self.normalizer = tf.keras.layers.Normalization(axis=-1)\n\n    self.seq = tf.keras.Sequential([\n      self.normalizer,\n      tf.keras.layers.Dense(10, activation='relu'),\n      tf.keras.layers.Dense(10, activation='relu'),\n      tf.keras.layers.Dense(1)\n    ])\n\n  def adapt(self, inputs):\n    inputs = stack_dict(inputs)\n    self.normalizer.adapt(inputs)\n\n  def call(self, inputs):\n    inputs = stack_dict(inputs)\n    result = self.seq(inputs)\n\n    return result")
numeric_dict_batches = custom_method(
numeric_dict_ds.shuffle(SHUFFLE_BUFFER).batch(BATCH_SIZE), imports='import pprint;import tensorflow as tf;import pandas as pd', function_to_run='obj.shuffle(SHUFFLE_BUFFER).batch(*args)', method_object=eval('numeric_dict_ds'), object_signature=None, function_args=[eval('BATCH_SIZE')], function_kwargs={}, custom_class=None)
custom_method(
model.fit(numeric_dict_batches, epochs=5), imports='import pprint;import tensorflow as tf;import pandas as pd', function_to_run='obj.fit(*args, **kwargs)', method_object=eval('model'), object_signature=None, function_args=[eval('numeric_dict_batches')], function_kwargs={'epochs': eval('5')}, custom_class="class MyModel(tf.keras.Model):\n  def __init__(self):\n    super().__init__(self)\n\n    self.normalizer = tf.keras.layers.Normalization(axis=-1)\n\n    self.seq = tf.keras.Sequential([\n      self.normalizer,\n      tf.keras.layers.Dense(10, activation='relu'),\n      tf.keras.layers.Dense(10, activation='relu'),\n      tf.keras.layers.Dense(1)\n    ])\n\n  def adapt(self, inputs):\n    inputs = stack_dict(inputs)\n    self.normalizer.adapt(inputs)\n\n  def call(self, inputs):\n    inputs = stack_dict(inputs)\n    result = self.seq(inputs)\n\n    return result")
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
    inputs[name] = custom_method(
    tf.keras.Input(shape=(), name=name, dtype=dtype), imports='import pprint;import tensorflow as tf;import pandas as pd', function_to_run='tf.keras.Input(**kwargs)', method_object=None, object_signature=None, function_args=[], function_kwargs={'shape': eval('()'), 'name': eval('name'), 'dtype': eval('dtype')})
inputs
preprocessed = []
for name in binary_feature_names:
    inp = inputs[name]
    inp = inp[:, tf.newaxis]
    float_value = custom_method(
    tf.cast(inp, tf.float32), imports='import pprint;import tensorflow as tf;import pandas as pd', function_to_run='tf.cast(*args)', method_object=None, object_signature=None, function_args=[eval('inp'), eval('tf.float32')], function_kwargs={})
    preprocessed.append(float_value)
preprocessed
normalizer = custom_method(
tf.keras.layers.Normalization(axis=-1), imports='import pprint;import tensorflow as tf;import pandas as pd', function_to_run='tf.keras.layers.Normalization(**kwargs)', method_object=None, object_signature=None, function_args=[], function_kwargs={'axis': eval('-1')})
custom_method(
normalizer.adapt(stack_dict(dict(numeric_features))), imports='import pprint;import tensorflow as tf;import pandas as pd', function_to_run='obj.adapt(*args)', method_object=eval('normalizer'), object_signature=None, function_args=[eval('stack_dict(dict(numeric_features))')], function_kwargs={}, custom_class=None)
numeric_inputs = {}
for name in numeric_feature_names:
    numeric_inputs[name] = inputs[name]
numeric_inputs = stack_dict(numeric_inputs)
numeric_normalized = custom_method(
normalizer(numeric_inputs), imports='import pprint;import tensorflow as tf;import pandas as pd', function_to_run='obj(*args)', method_object=eval('normalizer'), object_signature=None, function_args=[eval('numeric_inputs')], function_kwargs={}, custom_class=None)
preprocessed.append(numeric_normalized)
preprocessed
vocab = ['a', 'b', 'c']
lookup = custom_method(
tf.keras.layers.StringLookup(vocabulary=vocab, output_mode='one_hot'), imports='import pprint;import tensorflow as tf;import pandas as pd', function_to_run='tf.keras.layers.StringLookup(**kwargs)', method_object=None, object_signature=None, function_args=[], function_kwargs={'vocabulary': eval('vocab'), 'output_mode': eval("'one_hot'")})
custom_method(
lookup(['c', 'a', 'a', 'b', 'zzz']), imports='import pprint;import tensorflow as tf;import pandas as pd', function_to_run='obj(*args)', method_object=eval('lookup'), object_signature=None, function_args=[eval("['c','a','a','b','zzz']")], function_kwargs={}, custom_class=None)
vocab = [1, 4, 7, 99]
lookup = custom_method(
tf.keras.layers.IntegerLookup(vocabulary=vocab, output_mode='one_hot'), imports='import pprint;import tensorflow as tf;import pandas as pd', function_to_run='tf.keras.layers.IntegerLookup(**kwargs)', method_object=None, object_signature=None, function_args=[], function_kwargs={'vocabulary': eval('vocab'), 'output_mode': eval("'one_hot'")})
custom_method(
lookup([-1, 4, 1]), imports='import pprint;import tensorflow as tf;import pandas as pd', function_to_run='obj(*args)', method_object=eval('lookup'), object_signature=None, function_args=[eval('[-1,4,1]')], function_kwargs={}, custom_class=None)
for name in categorical_feature_names:
    vocab = sorted(set(df[name]))
    print(f'name: {name}')
    print(f'vocab: {vocab}\n')
    if type(vocab[0]) is str:
        lookup = custom_method(
        tf.keras.layers.StringLookup(vocabulary=vocab, output_mode='one_hot'), imports='import pprint;import tensorflow as tf;import pandas as pd', function_to_run='tf.keras.layers.StringLookup(**kwargs)', method_object=None, object_signature=None, function_args=[], function_kwargs={'vocabulary': eval('vocab'), 'output_mode': eval("'one_hot'")})
    else:
        lookup = custom_method(
        tf.keras.layers.IntegerLookup(vocabulary=vocab, output_mode='one_hot'), imports='import pprint;import tensorflow as tf;import pandas as pd', function_to_run='tf.keras.layers.IntegerLookup(**kwargs)', method_object=None, object_signature=None, function_args=[], function_kwargs={'vocabulary': eval('vocab'), 'output_mode': eval("'one_hot'")})
    x = inputs[name][:, tf.newaxis]
    x = custom_method(
    lookup(x), imports='import pprint;import tensorflow as tf;import pandas as pd', function_to_run='obj(*args)', method_object=eval('lookup'), object_signature=None, function_args=[eval('x')], function_kwargs={}, custom_class=None)
    preprocessed.append(x)
preprocessed
preprocesssed_result = custom_method(
tf.concat(preprocessed, axis=-1), imports='import pprint;import tensorflow as tf;import pandas as pd', function_to_run='tf.concat(*args, **kwargs)', method_object=None, object_signature=None, function_args=[eval('preprocessed')], function_kwargs={'axis': eval('-1')})
preprocesssed_result
preprocessor = custom_method(
tf.keras.Model(inputs, preprocesssed_result), imports='import pprint;import tensorflow as tf;import pandas as pd', function_to_run='tf.keras.Model(*args)', method_object=None, object_signature=None, function_args=[eval('inputs'), eval('preprocesssed_result')], function_kwargs={})
custom_method(
tf.keras.utils.plot_model(preprocessor, rankdir='LR', show_shapes=True), imports='import pprint;import tensorflow as tf;import pandas as pd', function_to_run='tf.keras.utils.plot_model(*args, **kwargs)', method_object=None, object_signature=None, function_args=[eval('preprocessor')], function_kwargs={'rankdir': eval('"LR"'), 'show_shapes': eval('True')})
custom_method(
preprocessor(dict(df.iloc[:1])), imports='import pprint;import tensorflow as tf;import pandas as pd', function_to_run='obj(*args)', method_object=eval('preprocessor'), object_signature=None, function_args=[eval('dict(df.iloc[:1])')], function_kwargs={}, custom_class=None)
body = custom_method(
tf.keras.Sequential([tf.keras.layers.Dense(10, activation='relu'), tf.keras.layers.Dense(10, activation='relu'), tf.keras.layers.Dense(1)]), imports='import pprint;import tensorflow as tf;import pandas as pd', function_to_run='tf.keras.Sequential(*args)', method_object=None, object_signature=None, function_args=[eval("[\n  tf.keras.layers.Dense(10, activation='relu'),\n  tf.keras.layers.Dense(10, activation='relu'),\n  tf.keras.layers.Dense(1)\n]")], function_kwargs={})
inputs
x = custom_method(
preprocessor(inputs), imports='import pprint;import tensorflow as tf;import pandas as pd', function_to_run='obj(*args)', method_object=eval('preprocessor'), object_signature=None, function_args=[eval('inputs')], function_kwargs={}, custom_class=None)
x
result = custom_method(
body(x), imports='import pprint;import tensorflow as tf;import pandas as pd', function_to_run='obj(*args)', method_object=eval('body'), object_signature=None, function_args=[eval('x')], function_kwargs={}, custom_class=None)
result
model = custom_method(
tf.keras.Model(inputs, result), imports='import pprint;import tensorflow as tf;import pandas as pd', function_to_run='tf.keras.Model(*args)', method_object=None, object_signature=None, function_args=[eval('inputs'), eval('result')], function_kwargs={})
custom_method(
model.compile(optimizer='adam', loss=tf.keras.losses.BinaryCrossentropy(from_logits=True), metrics=['accuracy']), imports='import pprint;import tensorflow as tf;import pandas as pd', function_to_run='obj.compile(**kwargs)', method_object=eval('model'), object_signature='MyModel()', function_args=[], function_kwargs={'optimizer': eval("'adam'"), 'loss': eval('tf.keras.losses.BinaryCrossentropy(from_logits=True)'), 'metrics': eval("['accuracy']")}, custom_class="class MyModel(tf.keras.Model):\n  def __init__(self):\n    super().__init__(self)\n\n    self.normalizer = tf.keras.layers.Normalization(axis=-1)\n\n    self.seq = tf.keras.Sequential([\n      self.normalizer,\n      tf.keras.layers.Dense(10, activation='relu'),\n      tf.keras.layers.Dense(10, activation='relu'),\n      tf.keras.layers.Dense(1)\n    ])\n\n  def adapt(self, inputs):\n    inputs = stack_dict(inputs)\n    self.normalizer.adapt(inputs)\n\n  def call(self, inputs):\n    inputs = stack_dict(inputs)\n    result = self.seq(inputs)\n\n    return result")
history = custom_method(
model.fit(dict(df), target, epochs=5, batch_size=BATCH_SIZE), imports='import pprint;import tensorflow as tf;import pandas as pd', function_to_run='obj.fit(*args, **kwargs)', method_object=eval('model'), object_signature=None, function_args=[eval('dict(df)'), eval('target')], function_kwargs={'epochs': eval('5'), 'batch_size': eval('BATCH_SIZE')}, custom_class="class MyModel(tf.keras.Model):\n  def __init__(self):\n    super().__init__(self)\n\n    self.normalizer = tf.keras.layers.Normalization(axis=-1)\n\n    self.seq = tf.keras.Sequential([\n      self.normalizer,\n      tf.keras.layers.Dense(10, activation='relu'),\n      tf.keras.layers.Dense(10, activation='relu'),\n      tf.keras.layers.Dense(1)\n    ])\n\n  def adapt(self, inputs):\n    inputs = stack_dict(inputs)\n    self.normalizer.adapt(inputs)\n\n  def call(self, inputs):\n    inputs = stack_dict(inputs)\n    result = self.seq(inputs)\n\n    return result")
ds = custom_method(
tf.data.Dataset.from_tensor_slices((dict(df), target)), imports='import pprint;import tensorflow as tf;import pandas as pd', function_to_run='tf.data.Dataset.from_tensor_slices(*args)', method_object=None, object_signature=None, function_args=[eval('(\n    dict(df),\n    target\n)')], function_kwargs={})
ds = custom_method(
ds.batch(BATCH_SIZE), imports='import pprint;import tensorflow as tf;import pandas as pd', function_to_run='obj.batch(*args)', method_object=eval('ds'), object_signature=None, function_args=[eval('BATCH_SIZE')], function_kwargs={}, custom_class=None)
import pprint
for (x, y) in custom_method(
ds.take(1), imports='import pprint;import tensorflow as tf;import pandas as pd', function_to_run='obj.take(*args)', method_object=eval('ds'), object_signature=None, function_args=[eval('1')], function_kwargs={}, custom_class=None):
    pprint.pprint(x)
    print()
    print(y)
history = custom_method(
model.fit(ds, epochs=5), imports='import pprint;import tensorflow as tf;import pandas as pd', function_to_run='obj.fit(*args, **kwargs)', method_object=eval('model'), object_signature=None, function_args=[eval('ds')], function_kwargs={'epochs': eval('5')}, custom_class="class MyModel(tf.keras.Model):\n  def __init__(self):\n    super().__init__(self)\n\n    self.normalizer = tf.keras.layers.Normalization(axis=-1)\n\n    self.seq = tf.keras.Sequential([\n      self.normalizer,\n      tf.keras.layers.Dense(10, activation='relu'),\n      tf.keras.layers.Dense(10, activation='relu'),\n      tf.keras.layers.Dense(1)\n    ])\n\n  def adapt(self, inputs):\n    inputs = stack_dict(inputs)\n    self.normalizer.adapt(inputs)\n\n  def call(self, inputs):\n    inputs = stack_dict(inputs)\n    result = self.seq(inputs)\n\n    return result")
