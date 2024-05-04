import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers
import sys
from fecom.patching.patching_config import EXPERIMENT_DIR
from fecom.measurement.execution import before_execution as before_execution_INSERTED_INTO_SCRIPT
from fecom.measurement.execution import after_execution as after_execution_INSERTED_INTO_SCRIPT
from fecom.experiment.experiment_kinds import ExperimentKinds
experiment_number = sys.argv[1]
experiment_project = sys.argv[2]
EXPERIMENT_FILE_PATH = EXPERIMENT_DIR / ExperimentKinds.METHOD_LEVEL.value / experiment_project / f'experiment-{experiment_number}.json'
tf.__version__
dataset_url = 'http://storage.googleapis.com/download.tensorflow.org/data/petfinder-mini.zip'
csv_file = 'datasets/petfinder-mini/petfinder-mini.csv'
start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT(experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tensorflow.keras.utils.get_file()')
tf.keras.utils.get_file('petfinder_mini.zip', dataset_url, extract=True, cache_dir='.')
after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tensorflow.keras.utils.get_file()', method_object=None, function_args=['petfinder_mini.zip', dataset_url], function_kwargs={'extract': True, 'cache_dir': '.'})
dataframe = pd.read_csv(csv_file)
dataframe.head()
dataframe['target'] = np.where(dataframe['AdoptionSpeed'] == 4, 0, 1)
dataframe = dataframe.drop(columns=['AdoptionSpeed', 'Description'])
(train, val, test) = np.split(dataframe.sample(frac=1), [int(0.8 * len(dataframe)), int(0.9 * len(dataframe))])
print(len(train), 'training examples')
print(len(val), 'validation examples')
print(len(test), 'test examples')

def df_to_dataset(dataframe, shuffle=True, batch_size=32):
    df = dataframe.copy()
    labels = df.pop('target')
    df = {key: value[:, tf.newaxis] for (key, value) in dataframe.items()}
    start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT(experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tensorflow.data.Dataset.from_tensor_slices()')
    ds = tf.data.Dataset.from_tensor_slices((dict(df), labels))
    after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tensorflow.data.Dataset.from_tensor_slices()', method_object=None, function_args=[(dict(df), labels)], function_kwargs=None)
    if shuffle:
        start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT(experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tensorflow.data.Dataset.from_tensor_slices.shuffle()')
        ds = ds.shuffle(buffer_size=len(dataframe))
        after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tensorflow.data.Dataset.from_tensor_slices.shuffle()', method_object=ds, function_args=None, function_kwargs={'buffer_size': len(dataframe)})
    start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT(experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tensorflow.data.Dataset.from_tensor_slices.shuffle.batch()')
    ds = ds.batch(batch_size)
    after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tensorflow.data.Dataset.from_tensor_slices.shuffle.batch()', method_object=ds, function_args=[batch_size], function_kwargs=None)
    start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT(experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tensorflow.data.Dataset.from_tensor_slices.shuffle.batch.prefetch()')
    ds = ds.prefetch(batch_size)
    after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tensorflow.data.Dataset.from_tensor_slices.shuffle.batch.prefetch()', method_object=ds, function_args=[batch_size], function_kwargs=None)
    return ds
batch_size = 5
train_ds = df_to_dataset(train, batch_size=batch_size)
[(train_features, label_batch)] = train_ds.take(1)
print('Every feature:', list(train_features.keys()))
print('A batch of ages:', train_features['Age'])
print('A batch of targets:', label_batch)

def get_normalization_layer(name, dataset):
    start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT(experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tensorflow.keras.layers.Normalization()')
    normalizer = layers.Normalization(axis=None)
    after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tensorflow.keras.layers.Normalization()', method_object=None, function_args=None, function_kwargs={'axis': None})
    feature_ds = dataset.map(lambda x, y: x[name])
    start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT(experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tensorflow.keras.layers.Normalization.adapt()')
    normalizer.adapt(feature_ds)
    after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tensorflow.keras.layers.Normalization.adapt()', method_object=normalizer, function_args=[feature_ds], function_kwargs=None)
    return normalizer
photo_count_col = train_features['PhotoAmt']
layer = get_normalization_layer('PhotoAmt', train_ds)
layer(photo_count_col)

def get_category_encoding_layer(name, dataset, dtype, max_tokens=None):
    if dtype == 'string':
        start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT(experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tensorflow.keras.layers.StringLookup()')
        index = layers.StringLookup(max_tokens=max_tokens)
        after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tensorflow.keras.layers.StringLookup()', method_object=None, function_args=None, function_kwargs={'max_tokens': max_tokens})
    else:
        start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT(experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tensorflow.keras.layers.IntegerLookup()')
        index = layers.IntegerLookup(max_tokens=max_tokens)
        after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tensorflow.keras.layers.IntegerLookup()', method_object=None, function_args=None, function_kwargs={'max_tokens': max_tokens})
    feature_ds = dataset.map(lambda x, y: x[name])
    start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT(experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tensorflow.keras.layers.IntegerLookup.adapt()')
    index.adapt(feature_ds)
    after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tensorflow.keras.layers.IntegerLookup.adapt()', method_object=index, function_args=[feature_ds], function_kwargs=None)
    start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT(experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tensorflow.keras.layers.CategoryEncoding()')
    encoder = layers.CategoryEncoding(num_tokens=index.vocabulary_size())
    after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tensorflow.keras.layers.CategoryEncoding()', method_object=None, function_args=None, function_kwargs={'num_tokens': index.vocabulary_size()})
    return lambda feature: encoder(index(feature))
test_type_col = train_features['Type']
test_type_layer = get_category_encoding_layer(name='Type', dataset=train_ds, dtype='string')
test_type_layer(test_type_col)
test_age_col = train_features['Age']
test_age_layer = get_category_encoding_layer(name='Age', dataset=train_ds, dtype='int64', max_tokens=5)
test_age_layer(test_age_col)
batch_size = 256
train_ds = df_to_dataset(train, batch_size=batch_size)
val_ds = df_to_dataset(val, shuffle=False, batch_size=batch_size)
test_ds = df_to_dataset(test, shuffle=False, batch_size=batch_size)
all_inputs = []
encoded_features = []
for header in ['PhotoAmt', 'Fee']:
    start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT(experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tensorflow.keras.Input()')
    numeric_col = tf.keras.Input(shape=(1,), name=header)
    after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tensorflow.keras.Input()', method_object=None, function_args=None, function_kwargs={'shape': (1,), 'name': header})
    normalization_layer = get_normalization_layer(header, train_ds)
    encoded_numeric_col = normalization_layer(numeric_col)
    all_inputs.append(numeric_col)
    encoded_features.append(encoded_numeric_col)
start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT(experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tensorflow.keras.Input()')
age_col = tf.keras.Input(shape=(1,), name='Age', dtype='int64')
after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tensorflow.keras.Input()', method_object=None, function_args=None, function_kwargs={'shape': (1,), 'name': 'Age', 'dtype': 'int64'})
encoding_layer = get_category_encoding_layer(name='Age', dataset=train_ds, dtype='int64', max_tokens=5)
encoded_age_col = encoding_layer(age_col)
all_inputs.append(age_col)
encoded_features.append(encoded_age_col)
categorical_cols = ['Type', 'Color1', 'Color2', 'Gender', 'MaturitySize', 'FurLength', 'Vaccinated', 'Sterilized', 'Health', 'Breed1']
for header in categorical_cols:
    start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT(experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tensorflow.keras.Input()')
    categorical_col = tf.keras.Input(shape=(1,), name=header, dtype='string')
    after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tensorflow.keras.Input()', method_object=None, function_args=None, function_kwargs={'shape': (1,), 'name': header, 'dtype': 'string'})
    encoding_layer = get_category_encoding_layer(name=header, dataset=train_ds, dtype='string', max_tokens=5)
    encoded_categorical_col = encoding_layer(categorical_col)
    all_inputs.append(categorical_col)
    encoded_features.append(encoded_categorical_col)
start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT(experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tensorflow.keras.layers.concatenate()')
all_features = tf.keras.layers.concatenate(encoded_features)
after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tensorflow.keras.layers.concatenate()', method_object=None, function_args=[encoded_features], function_kwargs=None)
start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT(experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run="tensorflow.keras.layers.Dense(32, activation='relu')()")
x = tf.keras.layers.Dense(32, activation='relu')(all_features)
after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run="tensorflow.keras.layers.Dense(32, activation='relu')()", method_object=None, function_args=[all_features], function_kwargs=None)
start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT(experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tensorflow.keras.layers.Dropout(0.5)()')
x = tf.keras.layers.Dropout(0.5)(x)
after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tensorflow.keras.layers.Dropout(0.5)()', method_object=None, function_args=[x], function_kwargs=None)
start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT(experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tensorflow.keras.layers.Dense(1)()')
output = tf.keras.layers.Dense(1)(x)
after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tensorflow.keras.layers.Dense(1)()', method_object=None, function_args=[x], function_kwargs=None)
start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT(experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tensorflow.keras.Model()')
model = tf.keras.Model(all_inputs, output)
after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tensorflow.keras.Model()', method_object=None, function_args=[all_inputs, output], function_kwargs=None)
start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT(experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tensorflow.keras.Model.compile()')
model.compile(optimizer='adam', loss=tf.keras.losses.BinaryCrossentropy(from_logits=True), metrics=['accuracy'])
after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tensorflow.keras.Model.compile()', method_object=model, function_args=None, function_kwargs={'optimizer': 'adam', 'loss': tf.keras.losses.BinaryCrossentropy(from_logits=True), 'metrics': ['accuracy']})
start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT(experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tensorflow.keras.utils.plot_model()')
tf.keras.utils.plot_model(model, show_shapes=True, rankdir='LR')
after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tensorflow.keras.utils.plot_model()', method_object=None, function_args=[model], function_kwargs={'show_shapes': True, 'rankdir': 'LR'})
start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT(experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tensorflow.keras.Model.fit()')
model.fit(train_ds, epochs=10, validation_data=val_ds)
after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tensorflow.keras.Model.fit()', method_object=model, function_args=[train_ds], function_kwargs={'epochs': 10, 'validation_data': val_ds})
start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT(experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tensorflow.keras.Model.evaluate()')
(loss, accuracy) = model.evaluate(test_ds)
after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tensorflow.keras.Model.evaluate()', method_object=model, function_args=[test_ds], function_kwargs=None)
print('Accuracy', accuracy)
start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT(experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tensorflow.keras.Model.save()')
model.save('my_pet_classifier')
after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tensorflow.keras.Model.save()', method_object=model, function_args=['my_pet_classifier'], function_kwargs=None)
start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT(experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tensorflow.keras.models.load_model()')
reloaded_model = tf.keras.models.load_model('my_pet_classifier')
after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tensorflow.keras.models.load_model()', method_object=None, function_args=['my_pet_classifier'], function_kwargs=None)
sample = {'Type': 'Cat', 'Age': 3, 'Breed1': 'Tabby', 'Gender': 'Male', 'Color1': 'Black', 'Color2': 'White', 'MaturitySize': 'Small', 'FurLength': 'Short', 'Vaccinated': 'No', 'Sterilized': 'No', 'Health': 'Healthy', 'Fee': 100, 'PhotoAmt': 2}
input_dict = {name: tf.convert_to_tensor([value]) for (name, value) in sample.items()}
start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT(experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tensorflow.keras.models.load_model.predict()')
predictions = reloaded_model.predict(input_dict)
after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tensorflow.keras.models.load_model.predict()', method_object=reloaded_model, function_args=[input_dict], function_kwargs=None)
start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT(experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tensorflow.nn.sigmoid()')
prob = tf.nn.sigmoid(predictions[0])
after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tensorflow.nn.sigmoid()', method_object=None, function_args=[predictions[0]], function_kwargs=None)
print('This particular pet had a %.1f percent probability of getting adopted.' % (100 * prob))
