import sys
from tool.patching.patching_config import EXPERIMENT_DIR
from tool.measurement.execution import before_execution as before_execution_INSERTED_INTO_SCRIPT
from tool.measurement.execution import after_execution as after_execution_INSERTED_INTO_SCRIPT
from tool.experiment.experiment_kinds import ExperimentKinds
experiment_number = sys.argv[1]
experiment_project = sys.argv[2]
EXPERIMENT_FILE_PATH = EXPERIMENT_DIR / ExperimentKinds.PROJECT_LEVEL.value / experiment_project / f'experiment-{experiment_number}.json'
start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT(enable_skip_calls=False)
import pandas as pd
import numpy as np
np.set_printoptions(precision=3, suppress=True)
import tensorflow as tf
from tensorflow.keras import layers
abalone_train = pd.read_csv('https://storage.googleapis.com/download.tensorflow.org/data/abalone_train.csv', names=['Length', 'Diameter', 'Height', 'Whole weight', 'Shucked weight', 'Viscera weight', 'Shell weight', 'Age'])
abalone_train.head()
abalone_features = abalone_train.copy()
abalone_labels = abalone_features.pop('Age')
abalone_features = np.array(abalone_features)
abalone_features
abalone_model = tf.keras.Sequential([layers.Dense(64), layers.Dense(1)])
abalone_model.compile(loss=tf.keras.losses.MeanSquaredError(), optimizer=tf.keras.optimizers.Adam())
abalone_model.fit(abalone_features, abalone_labels, epochs=10)
normalize = layers.Normalization()
normalize.adapt(abalone_features)
norm_abalone_model = tf.keras.Sequential([normalize, layers.Dense(64), layers.Dense(1)])
norm_abalone_model.compile(loss=tf.keras.losses.MeanSquaredError(), optimizer=tf.keras.optimizers.Adam())
norm_abalone_model.fit(abalone_features, abalone_labels, epochs=10)
titanic = pd.read_csv('https://storage.googleapis.com/tf-datasets/titanic/train.csv')
titanic.head()
titanic_features = titanic.copy()
titanic_labels = titanic_features.pop('survived')
input = tf.keras.Input(shape=(), dtype=tf.float32)
result = 2 * input + 1
result
calc = tf.keras.Model(inputs=input, outputs=result)
print(calc(1).numpy())
print(calc(2).numpy())
inputs = {}
for (name, column) in titanic_features.items():
    dtype = column.dtype
    if dtype == object:
        dtype = tf.string
    else:
        dtype = tf.float32
    inputs[name] = tf.keras.Input(shape=(1,), name=name, dtype=dtype)
inputs
numeric_inputs = {name: input for (name, input) in inputs.items() if input.dtype == tf.float32}
x = layers.Concatenate()(list(numeric_inputs.values()))
norm = layers.Normalization()
norm.adapt(np.array(titanic[numeric_inputs.keys()]))
all_numeric_inputs = norm(x)
all_numeric_inputs
preprocessed_inputs = [all_numeric_inputs]
for (name, input) in inputs.items():
    if input.dtype == tf.float32:
        continue
    lookup = layers.StringLookup(vocabulary=np.unique(titanic_features[name]))
    one_hot = layers.CategoryEncoding(num_tokens=lookup.vocabulary_size())
    x = lookup(input)
    x = one_hot(x)
    preprocessed_inputs.append(x)
preprocessed_inputs_cat = layers.Concatenate()(preprocessed_inputs)
titanic_preprocessing = tf.keras.Model(inputs, preprocessed_inputs_cat)
tf.keras.utils.plot_model(model=titanic_preprocessing, rankdir='LR', dpi=72, show_shapes=True)
titanic_features_dict = {name: np.array(value) for (name, value) in titanic_features.items()}
features_dict = {name: values[:1] for (name, values) in titanic_features_dict.items()}
titanic_preprocessing(features_dict)

def titanic_model(preprocessing_head, inputs):
    body = tf.keras.Sequential([layers.Dense(64), layers.Dense(1)])
    preprocessed_inputs = preprocessing_head(inputs)
    result = body(preprocessed_inputs)
    model = tf.keras.Model(inputs, result)
    model.compile(loss=tf.keras.losses.BinaryCrossentropy(from_logits=True), optimizer=tf.keras.optimizers.Adam())
    return model
titanic_model = titanic_model(titanic_preprocessing, inputs)
titanic_model.fit(x=titanic_features_dict, y=titanic_labels, epochs=10)
titanic_model.save('test')
reloaded = tf.keras.models.load_model('test')
features_dict = {name: values[:1] for (name, values) in titanic_features_dict.items()}
before = titanic_model(features_dict)
after = reloaded(features_dict)
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
features_ds = tf.data.Dataset.from_tensor_slices(titanic_features_dict)
for example in features_ds:
    for (name, value) in example.items():
        print(f'{name:19s}: {value}')
    break
titanic_ds = tf.data.Dataset.from_tensor_slices((titanic_features_dict, titanic_labels))
titanic_batches = titanic_ds.shuffle(len(titanic_labels)).batch(32)
titanic_model.fit(titanic_batches, epochs=5)
titanic_file_path = tf.keras.utils.get_file('train.csv', 'https://storage.googleapis.com/tf-datasets/titanic/train.csv')
titanic_csv_ds = tf.data.experimental.make_csv_dataset(titanic_file_path, batch_size=5, label_name='survived', num_epochs=1, ignore_errors=True)
for (batch, label) in titanic_csv_ds.take(1):
    for (key, value) in batch.items():
        print(f'{key:20s}: {value}')
    print()
    print(f"{'label':20s}: {label}")
traffic_volume_csv_gz = tf.keras.utils.get_file('Metro_Interstate_Traffic_Volume.csv.gz', 'https://archive.ics.uci.edu/ml/machine-learning-databases/00492/Metro_Interstate_Traffic_Volume.csv.gz', cache_dir='.', cache_subdir='traffic')
traffic_volume_csv_gz_ds = tf.data.experimental.make_csv_dataset(traffic_volume_csv_gz, batch_size=256, label_name='traffic_volume', num_epochs=1, compression_type='GZIP')
for (batch, label) in traffic_volume_csv_gz_ds.take(1):
    for (key, value) in batch.items():
        print(f'{key:20s}: {value[:5]}')
    print()
    print(f"{'label':20s}: {label[:5]}")
for (i, (batch, label)) in enumerate(traffic_volume_csv_gz_ds.repeat(20)):
    if i % 40 == 0:
        print('.', end='')
print()
caching = traffic_volume_csv_gz_ds.cache().shuffle(1000)
for (i, (batch, label)) in enumerate(caching.shuffle(1000).repeat(20)):
    if i % 40 == 0:
        print('.', end='')
print()
snapshotting = traffic_volume_csv_gz_ds.snapshot('titanic.tfsnap').shuffle(1000)
for (i, (batch, label)) in enumerate(snapshotting.shuffle(1000).repeat(20)):
    if i % 40 == 0:
        print('.', end='')
print()
fonts_zip = tf.keras.utils.get_file('fonts.zip', 'https://archive.ics.uci.edu/ml/machine-learning-databases/00417/fonts.zip', cache_dir='.', cache_subdir='fonts', extract=True)
import pathlib
font_csvs = sorted((str(p) for p in pathlib.Path('fonts').glob('*.csv')))
font_csvs[:10]
len(font_csvs)
fonts_ds = tf.data.experimental.make_csv_dataset(file_pattern='fonts/*.csv', batch_size=10, num_epochs=1, num_parallel_reads=20, shuffle_buffer_size=10000)
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
    image = tf.stack(image, axis=0)
    image = tf.reshape(image, [20, 20, -1])
    new_feats['image'] = image
    return new_feats
fonts_image_ds = fonts_ds.map(make_images)
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
features = tf.io.decode_csv(lines, record_defaults=all_strings)
for f in features:
    print(f'type: {f.dtype.name}, shape: {f.shape}')
print(lines[0])
titanic_types = [int(), str(), float(), int(), int(), float(), str(), str(), str(), str()]
titanic_types
features = tf.io.decode_csv(lines, record_defaults=titanic_types)
for f in features:
    print(f'type: {f.dtype.name}, shape: {f.shape}')
simple_titanic = tf.data.experimental.CsvDataset(titanic_file_path, record_defaults=titanic_types, header=True)
for example in simple_titanic.take(1):
    print([e.numpy() for e in example])

def decode_titanic_line(line):
    return tf.io.decode_csv(line, titanic_types)
manual_titanic = tf.data.TextLineDataset(titanic_file_path).skip(1).map(decode_titanic_line)
for example in manual_titanic.take(1):
    print([e.numpy() for e in example])
font_line = pathlib.Path(font_csvs[0]).read_text().splitlines()[1]
print(font_line)
num_font_features = font_line.count(',') + 1
font_column_types = [str(), str()] + [float()] * (num_font_features - 2)
font_csvs[0]
simple_font_ds = tf.data.experimental.CsvDataset(font_csvs, record_defaults=font_column_types, header=True)
for row in simple_font_ds.take(10):
    print(row[0].numpy())
font_files = tf.data.Dataset.list_files('fonts/*.csv')
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
font_rows = font_files.interleave(make_font_csv_ds, cycle_length=3)
fonts_dict = {'font_name': [], 'character': []}
for row in font_rows.take(10):
    fonts_dict['font_name'].append(row[0].numpy().decode())
    fonts_dict['character'].append(chr(row[2].numpy()))
pd.DataFrame(fonts_dict)
BATCH_SIZE = 2048
fonts_ds = tf.data.experimental.make_csv_dataset(file_pattern='fonts/*.csv', batch_size=BATCH_SIZE, num_epochs=1, num_parallel_reads=100)
for (i, batch) in enumerate(fonts_ds.take(20)):
    print('.', end='')
print()
fonts_files = tf.data.Dataset.list_files('fonts/*.csv')
fonts_lines = fonts_files.interleave(lambda fname: tf.data.TextLineDataset(fname).skip(1), cycle_length=100).batch(BATCH_SIZE)
fonts_fast = fonts_lines.map(lambda x: tf.io.decode_csv(x, record_defaults=font_column_types))
for (i, batch) in enumerate(fonts_fast.take(20)):
    print('.', end='')
print()
after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, enable_skip_calls=False)
