import pandas as pd
import numpy as np
import sys
from tool.client.client_config import EXPERIMENT_DIR
from tool.measurement.execution import before_execution as before_execution_INSERTED_INTO_SCRIPT
from tool.measurement.execution import after_execution as after_execution_INSERTED_INTO_SCRIPT
experiment_number = sys.argv[1]
experiment_project = sys.argv[2]
EXPERIMENT_FILE_PATH = EXPERIMENT_DIR / 'local-execution' / experiment_project / f'experiment-{experiment_number}.json'
np.set_printoptions(precision=3, suppress=True)
import tensorflow as tf
from tensorflow.keras import layers
abalone_train = pd.read_csv('https://storage.googleapis.com/download.tensorflow.org/data/abalone_train.csv', names=['Length', 'Diameter', 'Height', 'Whole weight', 'Shucked weight', 'Viscera weight', 'Shell weight', 'Age'])
abalone_train.head()
abalone_features = abalone_train.copy()
abalone_labels = abalone_features.pop('Age')
abalone_features = np.array(abalone_features)
abalone_features
start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT()
abalone_model = tf.keras.Sequential([layers.Dense(64), layers.Dense(1)])
after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tf.keras.Sequential()', method_object=None, function_args=[[layers.Dense(64), layers.Dense(1)]], function_kwargs=None)
start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT()
abalone_model.compile(loss=tf.keras.losses.MeanSquaredError(), optimizer=tf.keras.optimizers.Adam())
after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tf.keras.Sequential.compile()', method_object=abalone_model, function_args=None, function_kwargs={'loss': tf.keras.losses.MeanSquaredError(), 'optimizer': tf.keras.optimizers.Adam()})
start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT()
abalone_model.fit(abalone_features, abalone_labels, epochs=10)
after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tf.keras.Sequential.fit()', method_object=abalone_model, function_args=[abalone_features, abalone_labels], function_kwargs={'epochs': 10})
start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT()
normalize = layers.Normalization()
after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='layers.Normalization()', method_object=None, function_args=None, function_kwargs=None)
start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT()
normalize.adapt(abalone_features)
after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='layers.Normalization.adapt()', method_object=normalize, function_args=[abalone_features], function_kwargs=None)
start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT()
norm_abalone_model = tf.keras.Sequential([normalize, layers.Dense(64), layers.Dense(1)])
after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tf.keras.Sequential()', method_object=None, function_args=[[normalize, layers.Dense(64), layers.Dense(1)]], function_kwargs=None)
start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT()
norm_abalone_model.compile(loss=tf.keras.losses.MeanSquaredError(), optimizer=tf.keras.optimizers.Adam())
after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tf.keras.Sequential.compile()', method_object=norm_abalone_model, function_args=None, function_kwargs={'loss': tf.keras.losses.MeanSquaredError(), 'optimizer': tf.keras.optimizers.Adam()})
start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT()
norm_abalone_model.fit(abalone_features, abalone_labels, epochs=10)
after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tf.keras.Sequential.fit()', method_object=norm_abalone_model, function_args=[abalone_features, abalone_labels], function_kwargs={'epochs': 10})
titanic = pd.read_csv('https://storage.googleapis.com/tf-datasets/titanic/train.csv')
titanic.head()
titanic_features = titanic.copy()
titanic_labels = titanic_features.pop('survived')
start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT()
input = tf.keras.Input(shape=(), dtype=tf.float32)
after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tf.keras.Input()', method_object=None, function_args=None, function_kwargs={'shape': (), 'dtype': tf.float32})
result = 2 * input + 1
result
start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT()
calc = tf.keras.Model(inputs=input, outputs=result)
after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tf.keras.Model()', method_object=None, function_args=None, function_kwargs={'inputs': input, 'outputs': result})
print(calc(1).numpy())
print(calc(2).numpy())
inputs = {}
for (name, column) in titanic_features.items():
    dtype = column.dtype
    if dtype == object:
        dtype = tf.string
    else:
        dtype = tf.float32
    start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT()
    inputs[name] = tf.keras.Input(shape=(1,), name=name, dtype=dtype)
    after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tf.keras.Input()', method_object=None, function_args=None, function_kwargs={'shape': (1,), 'name': name, 'dtype': dtype})
inputs
numeric_inputs = {name: input for (name, input) in inputs.items() if input.dtype == tf.float32}
start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT()
x = layers.Concatenate()(list(numeric_inputs.values()))
after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='layers.Concatenate()()', method_object=None, function_args=[list(numeric_inputs.values())], function_kwargs=None)
start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT()
norm = layers.Normalization()
after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='layers.Normalization()', method_object=None, function_args=None, function_kwargs=None)
start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT()
norm.adapt(np.array(titanic[numeric_inputs.keys()]))
after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='layers.Normalization.adapt()', method_object=norm, function_args=[np.array(titanic[numeric_inputs.keys()])], function_kwargs=None)
start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT()
all_numeric_inputs = norm(x)
after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='layers.Normalization()', method_object=norm, function_args=[x], function_kwargs=None)
all_numeric_inputs
preprocessed_inputs = [all_numeric_inputs]
for (name, input) in inputs.items():
    if input.dtype == tf.float32:
        continue
    start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT()
    lookup = layers.StringLookup(vocabulary=np.unique(titanic_features[name]))
    after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='layers.StringLookup()', method_object=None, function_args=None, function_kwargs={'vocabulary': np.unique(titanic_features[name])})
    start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT()
    one_hot = layers.CategoryEncoding(num_tokens=lookup.vocabulary_size())
    after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='layers.CategoryEncoding()', method_object=None, function_args=None, function_kwargs={'num_tokens': lookup.vocabulary_size()})
    start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT()
    x = lookup(input)
    after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='layers.StringLookup()', method_object=lookup, function_args=[input], function_kwargs=None)
    start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT()
    x = one_hot(x)
    after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='layers.CategoryEncoding()', method_object=one_hot, function_args=[x], function_kwargs=None)
    preprocessed_inputs.append(x)
start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT()
preprocessed_inputs_cat = layers.Concatenate()(preprocessed_inputs)
after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='layers.Concatenate()()', method_object=None, function_args=[preprocessed_inputs], function_kwargs=None)
start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT()
titanic_preprocessing = tf.keras.Model(inputs, preprocessed_inputs_cat)
after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tf.keras.Model()', method_object=None, function_args=[inputs, preprocessed_inputs_cat], function_kwargs=None)
start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT()
tf.keras.utils.plot_model(model=titanic_preprocessing, rankdir='LR', dpi=72, show_shapes=True)
after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tf.keras.utils.plot_model()', method_object=None, function_args=None, function_kwargs={'model': titanic_preprocessing, 'rankdir': 'LR', 'dpi': 72, 'show_shapes': True})
titanic_features_dict = {name: np.array(value) for (name, value) in titanic_features.items()}
features_dict = {name: values[:1] for (name, values) in titanic_features_dict.items()}
start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT()
titanic_preprocessing(features_dict)
after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tf.keras.Model()', method_object=titanic_preprocessing, function_args=[features_dict], function_kwargs=None)

def titanic_model(preprocessing_head, inputs):
    start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT()
    body = tf.keras.Sequential([layers.Dense(64), layers.Dense(1)])
    after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tf.keras.Sequential()', method_object=None, function_args=[[layers.Dense(64), layers.Dense(1)]], function_kwargs=None)
    preprocessed_inputs = preprocessing_head(inputs)
    start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT()
    result = body(preprocessed_inputs)
    after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tf.keras.Sequential()', method_object=body, function_args=[preprocessed_inputs], function_kwargs=None)
    start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT()
    model = tf.keras.Model(inputs, result)
    after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tf.keras.Model()', method_object=None, function_args=[inputs, result], function_kwargs=None)
    start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT()
    model.compile(loss=tf.keras.losses.BinaryCrossentropy(from_logits=True), optimizer=tf.keras.optimizers.Adam())
    after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tf.keras.Model.compile()', method_object=model, function_args=None, function_kwargs={'loss': tf.keras.losses.BinaryCrossentropy(from_logits=True), 'optimizer': tf.keras.optimizers.Adam()})
    return model
titanic_model = titanic_model(titanic_preprocessing, inputs)
titanic_model.fit(x=titanic_features_dict, y=titanic_labels, epochs=10)
titanic_model.save('test')
start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT()
reloaded = tf.keras.models.load_model('test')
after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tf.keras.models.load_model()', method_object=None, function_args=['test'], function_kwargs=None)
features_dict = {name: values[:1] for (name, values) in titanic_features_dict.items()}
before = titanic_model(features_dict)
start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT()
after = reloaded(features_dict)
after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tf.keras.models.load_model()', method_object=reloaded, function_args=[features_dict], function_kwargs=None)
assert before - after < 0.001
print(before)
print(after)
import itertools

def slices(features):
    for i in itertools.count():
        example = {name: values[i] for (name, values) in features.items()}
        yield example
for example in slices(titanic_features_dict):
    for (name, value) in example.items():
        print(f'{name:19s}: {value}')
    break
start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT()
features_ds = tf.data.Dataset.from_tensor_slices(titanic_features_dict)
after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tf.data.Dataset.from_tensor_slices()', method_object=None, function_args=[titanic_features_dict], function_kwargs=None)
for example in features_ds:
    for (name, value) in example.items():
        print(f'{name:19s}: {value}')
    break
start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT()
titanic_ds = tf.data.Dataset.from_tensor_slices((titanic_features_dict, titanic_labels))
after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tf.data.Dataset.from_tensor_slices()', method_object=None, function_args=[(titanic_features_dict, titanic_labels)], function_kwargs=None)
start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT()
titanic_batches = titanic_ds.shuffle(len(titanic_labels)).batch(32)
after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tf.data.Dataset.from_tensor_slices.shuffle(len(titanic_labels)).batch()', method_object=titanic_ds, function_args=[32], function_kwargs=None)
titanic_model.fit(titanic_batches, epochs=5)
start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT()
titanic_file_path = tf.keras.utils.get_file('train.csv', 'https://storage.googleapis.com/tf-datasets/titanic/train.csv')
after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tf.keras.utils.get_file()', method_object=None, function_args=['train.csv', 'https://storage.googleapis.com/tf-datasets/titanic/train.csv'], function_kwargs=None)
start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT()
titanic_csv_ds = tf.data.experimental.make_csv_dataset(titanic_file_path, batch_size=5, label_name='survived', num_epochs=1, ignore_errors=True)
after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tf.data.experimental.make_csv_dataset()', method_object=None, function_args=[titanic_file_path], function_kwargs={'batch_size': 5, 'label_name': 'survived', 'num_epochs': 1, 'ignore_errors': True})
for (batch, label) in titanic_csv_ds.take(1):
    for (key, value) in batch.items():
        print(f'{key:20s}: {value}')
    print()
    print(f"{'label':20s}: {label}")
start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT()
traffic_volume_csv_gz = tf.keras.utils.get_file('Metro_Interstate_Traffic_Volume.csv.gz', 'https://archive.ics.uci.edu/ml/machine-learning-databases/00492/Metro_Interstate_Traffic_Volume.csv.gz', cache_dir='.', cache_subdir='traffic')
after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tf.keras.utils.get_file()', method_object=None, function_args=['Metro_Interstate_Traffic_Volume.csv.gz', 'https://archive.ics.uci.edu/ml/machine-learning-databases/00492/Metro_Interstate_Traffic_Volume.csv.gz'], function_kwargs={'cache_dir': '.', 'cache_subdir': 'traffic'})
start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT()
traffic_volume_csv_gz_ds = tf.data.experimental.make_csv_dataset(traffic_volume_csv_gz, batch_size=256, label_name='traffic_volume', num_epochs=1, compression_type='GZIP')
after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tf.data.experimental.make_csv_dataset()', method_object=None, function_args=[traffic_volume_csv_gz], function_kwargs={'batch_size': 256, 'label_name': 'traffic_volume', 'num_epochs': 1, 'compression_type': 'GZIP'})
for (batch, label) in traffic_volume_csv_gz_ds.take(1):
    for (key, value) in batch.items():
        print(f'{key:20s}: {value[:5]}')
    print()
    print(f"{'label':20s}: {label[:5]}")
for (i, (batch, label)) in enumerate(traffic_volume_csv_gz_ds.repeat(20)):
    if i % 40 == 0:
        print('.', end='')
print()
start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT()
caching = traffic_volume_csv_gz_ds.cache().shuffle(1000)
after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tf.data.experimental.make_csv_dataset.cache().shuffle()', method_object=traffic_volume_csv_gz_ds, function_args=[1000], function_kwargs=None)
for (i, (batch, label)) in enumerate(caching.shuffle(1000).repeat(20)):
    if i % 40 == 0:
        print('.', end='')
print()
start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT()
snapshotting = traffic_volume_csv_gz_ds.snapshot('titanic.tfsnap').shuffle(1000)
after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run="tf.data.experimental.make_csv_dataset.snapshot('titanic.tfsnap').shuffle()", method_object=traffic_volume_csv_gz_ds, function_args=[1000], function_kwargs=None)
for (i, (batch, label)) in enumerate(snapshotting.shuffle(1000).repeat(20)):
    if i % 40 == 0:
        print('.', end='')
print()
start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT()
fonts_zip = tf.keras.utils.get_file('fonts.zip', 'https://archive.ics.uci.edu/ml/machine-learning-databases/00417/fonts.zip', cache_dir='.', cache_subdir='fonts', extract=True)
after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tf.keras.utils.get_file()', method_object=None, function_args=['fonts.zip', 'https://archive.ics.uci.edu/ml/machine-learning-databases/00417/fonts.zip'], function_kwargs={'cache_dir': '.', 'cache_subdir': 'fonts', 'extract': True})
import pathlib
font_csvs = sorted((str(p) for p in pathlib.Path('fonts').glob('*.csv')))
font_csvs[:10]
len(font_csvs)
start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT()
fonts_ds = tf.data.experimental.make_csv_dataset(file_pattern='fonts/*.csv', batch_size=10, num_epochs=1, num_parallel_reads=20, shuffle_buffer_size=10000)
after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tf.data.experimental.make_csv_dataset()', method_object=None, function_args=None, function_kwargs={'file_pattern': 'fonts/*.csv', 'batch_size': 10, 'num_epochs': 1, 'num_parallel_reads': 20, 'shuffle_buffer_size': 10000})
for features in fonts_ds.take(1):
    for (i, (name, value)) in enumerate(features.items()):
        if i > 15:
            break
        print(f'{name:20s}: {value}')
print('...')
print(f'[total: {len(features)} features]')
import re

def make_images(features):
    image = [None] * 400
    new_feats = {}
    for (name, value) in features.items():
        match = re.match('r(\\d+)c(\\d+)', name)
        if match:
            image[int(match.group(1)) * 20 + int(match.group(2))] = value
        else:
            new_feats[name] = value
    start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT()
    image = tf.stack(image, axis=0)
    after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tf.stack()', method_object=None, function_args=[image], function_kwargs={'axis': 0})
    start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT()
    image = tf.reshape(image, [20, 20, -1])
    after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tf.reshape()', method_object=None, function_args=[image, [20, 20, -1]], function_kwargs=None)
    new_feats['image'] = image
    return new_feats
start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT()
fonts_image_ds = fonts_ds.map(make_images)
after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tf.data.experimental.make_csv_dataset.map()', method_object=fonts_ds, function_args=[make_images], function_kwargs=None)
for features in fonts_image_ds.take(1):
    break
from matplotlib import pyplot as plt
plt.figure(figsize=(6, 6), dpi=120)
for n in range(9):
    plt.subplot(3, 3, n + 1)
    plt.imshow(features['image'][..., n])
    plt.title(chr(features['m_label'][n]))
    plt.axis('off')
text = pathlib.Path(titanic_file_path).read_text()
lines = text.split('\n')[1:-1]
all_strings = [str()] * 10
all_strings
start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT()
features = tf.io.decode_csv(lines, record_defaults=all_strings)
after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tf.io.decode_csv()', method_object=None, function_args=[lines], function_kwargs={'record_defaults': all_strings})
for f in features:
    print(f'type: {f.dtype.name}, shape: {f.shape}')
print(lines[0])
titanic_types = [int(), str(), float(), int(), int(), float(), str(), str(), str(), str()]
titanic_types
start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT()
features = tf.io.decode_csv(lines, record_defaults=titanic_types)
after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tf.io.decode_csv()', method_object=None, function_args=[lines], function_kwargs={'record_defaults': titanic_types})
for f in features:
    print(f'type: {f.dtype.name}, shape: {f.shape}')
start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT()
simple_titanic = tf.data.experimental.CsvDataset(titanic_file_path, record_defaults=titanic_types, header=True)
after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tf.data.experimental.CsvDataset()', method_object=None, function_args=[titanic_file_path], function_kwargs={'record_defaults': titanic_types, 'header': True})
for example in simple_titanic.take(1):
    print([e.numpy() for e in example])

def decode_titanic_line(line):
    return tf.io.decode_csv(line, titanic_types)
start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT()
manual_titanic = tf.data.TextLineDataset(titanic_file_path).skip(1).map(decode_titanic_line)
after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tf.data.TextLineDataset(titanic_file_path).skip(1).map()', method_object=None, function_args=[decode_titanic_line], function_kwargs=None)
for example in manual_titanic.take(1):
    print([e.numpy() for e in example])
font_line = pathlib.Path(font_csvs[0]).read_text().splitlines()[1]
print(font_line)
num_font_features = font_line.count(',') + 1
font_column_types = [str(), str()] + [float()] * (num_font_features - 2)
font_csvs[0]
start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT()
simple_font_ds = tf.data.experimental.CsvDataset(font_csvs, record_defaults=font_column_types, header=True)
after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tf.data.experimental.CsvDataset()', method_object=None, function_args=[font_csvs], function_kwargs={'record_defaults': font_column_types, 'header': True})
for row in simple_font_ds.take(10):
    print(row[0].numpy())
start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT()
font_files = tf.data.Dataset.list_files('fonts/*.csv')
after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tf.data.Dataset.list_files()', method_object=None, function_args=['fonts/*.csv'], function_kwargs=None)
print('Epoch 1:')
for f in list(font_files)[:5]:
    print('    ', f.numpy())
print('    ...')
print()
print('Epoch 2:')
for f in list(font_files)[:5]:
    print('    ', f.numpy())
print('    ...')

def make_font_csv_ds(path):
    return tf.data.experimental.CsvDataset(path, record_defaults=font_column_types, header=True)
start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT()
font_rows = font_files.interleave(make_font_csv_ds, cycle_length=3)
after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tf.data.Dataset.list_files.interleave()', method_object=font_files, function_args=[make_font_csv_ds], function_kwargs={'cycle_length': 3})
fonts_dict = {'font_name': [], 'character': []}
for row in font_rows.take(10):
    fonts_dict['font_name'].append(row[0].numpy().decode())
    fonts_dict['character'].append(chr(row[2].numpy()))
pd.DataFrame(fonts_dict)
BATCH_SIZE = 2048
start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT()
fonts_ds = tf.data.experimental.make_csv_dataset(file_pattern='fonts/*.csv', batch_size=BATCH_SIZE, num_epochs=1, num_parallel_reads=100)
after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tf.data.experimental.make_csv_dataset()', method_object=None, function_args=None, function_kwargs={'file_pattern': 'fonts/*.csv', 'batch_size': BATCH_SIZE, 'num_epochs': 1, 'num_parallel_reads': 100})
for (i, batch) in enumerate(fonts_ds.take(20)):
    print('.', end='')
print()
start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT()
fonts_files = tf.data.Dataset.list_files('fonts/*.csv')
after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tf.data.Dataset.list_files()', method_object=None, function_args=['fonts/*.csv'], function_kwargs=None)
start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT()
fonts_lines = fonts_files.interleave(lambda fname: tf.data.TextLineDataset(fname).skip(1), cycle_length=100).batch(BATCH_SIZE)
after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tf.data.Dataset.list_files.interleave(lambda fname: tf.data.TextLineDataset(fname).skip(1), cycle_length=100).batch()', method_object=fonts_files, function_args=[BATCH_SIZE], function_kwargs=None)
fonts_fast = fonts_lines.map(lambda x: tf.io.decode_csv(x, record_defaults=font_column_types))
for (i, batch) in enumerate(fonts_fast.take(20)):
    print('.', end='')
print()
