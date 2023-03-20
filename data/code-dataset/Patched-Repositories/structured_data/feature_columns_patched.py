import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import feature_column
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
import pathlib
import os
from pathlib import Path
import dill as pickle
from tool.client.client_config import EXPERIMENT_DIR, EXPERIMENT_TAG, MAX_WAIT_S, WAIT_AFTER_RUN_S
from tool.server.send_request import send_request
from tool.server.function_details import FunctionDetails
current_path = os.path.abspath(__file__)
(immediate_folder, file_name) = os.path.split(current_path)
immediate_folder = os.path.basename(immediate_folder)
experiment_file_name = os.path.splitext(file_name)[0]
EXPERIMENT_FILE_PATH = EXPERIMENT_DIR / immediate_folder / EXPERIMENT_TAG / (experiment_file_name + '-energy.json')

def custom_method(func, imports: str, function_to_run: str, method_object=None, function_args: list=None, function_kwargs: dict=None, max_wait_secs=MAX_WAIT_S, custom_class=None, wait_after_run_secs=WAIT_AFTER_RUN_S):
    result = send_request(imports=imports, function_to_run=function_to_run, function_args=function_args, function_kwargs=function_kwargs, max_wait_secs=max_wait_secs, method_object=method_object, custom_class=custom_class, experiment_file_path=EXPERIMENT_FILE_PATH)
    return func
if __name__ == '__main__':
    print(EXPERIMENT_FILE_PATH)
dataset_url = 'http://storage.googleapis.com/download.tensorflow.org/data/petfinder-mini.zip'
csv_file = 'datasets/petfinder-mini/petfinder-mini.csv'
custom_method(
tf.keras.utils.get_file('petfinder_mini.zip', dataset_url, extract=True, cache_dir='.'), imports='import pandas as pd;from sklearn.model_selection import train_test_split;import tensorflow as tf;from tensorflow import feature_column;from tensorflow.keras import layers;import numpy as np;import pathlib', function_to_run='tf.keras.utils.get_file(*args, **kwargs)', method_object=None, function_args=[eval("'petfinder_mini.zip'"), eval('dataset_url')], function_kwargs={'extract': eval('True'), 'cache_dir': eval("'.'")}, max_wait_secs=0)
dataframe = pd.read_csv(csv_file)
dataframe.head()
dataframe['target'] = np.where(dataframe['AdoptionSpeed'] == 4, 0, 1)
dataframe = dataframe.drop(columns=['AdoptionSpeed', 'Description'])
(train, test) = train_test_split(dataframe, test_size=0.2)
(train, val) = train_test_split(train, test_size=0.2)
print(len(train), 'train examples')
print(len(val), 'validation examples')
print(len(test), 'test examples')

def df_to_dataset(dataframe, shuffle=True, batch_size=32):
    dataframe = dataframe.copy()
    labels = dataframe.pop('target')
    ds = custom_method(
    tf.data.Dataset.from_tensor_slices((dict(dataframe), labels)), imports='import pandas as pd;from sklearn.model_selection import train_test_split;import tensorflow as tf;from tensorflow import feature_column;from tensorflow.keras import layers;import numpy as np;import pathlib', function_to_run='tf.data.Dataset.from_tensor_slices(*args)', method_object=None, function_args=[eval('(dict(dataframe), labels)')], function_kwargs={}, max_wait_secs=0)
    if shuffle:
        ds = custom_method(
        ds.shuffle(buffer_size=len(dataframe)), imports='import pandas as pd;from sklearn.model_selection import train_test_split;import tensorflow as tf;from tensorflow import feature_column;from tensorflow.keras import layers;import numpy as np;import pathlib', function_to_run='obj.shuffle(**kwargs)', method_object=eval('ds'), function_args=[], function_kwargs={'buffer_size': eval('len(dataframe)')}, max_wait_secs=0, custom_class=None)
    ds = custom_method(
    ds.batch(batch_size), imports='import pandas as pd;from sklearn.model_selection import train_test_split;import tensorflow as tf;from tensorflow import feature_column;from tensorflow.keras import layers;import numpy as np;import pathlib', function_to_run='obj.batch(*args)', method_object=eval('ds'), function_args=[eval('batch_size')], function_kwargs={}, max_wait_secs=0, custom_class=None)
    return ds
batch_size = 5
train_ds = df_to_dataset(train, batch_size=batch_size)
val_ds = df_to_dataset(val, shuffle=False, batch_size=batch_size)
test_ds = df_to_dataset(test, shuffle=False, batch_size=batch_size)
for (feature_batch, label_batch) in custom_method(
train_ds.take(1), imports='import pandas as pd;from sklearn.model_selection import train_test_split;import tensorflow as tf;from tensorflow import feature_column;from tensorflow.keras import layers;import numpy as np;import pathlib', function_to_run='obj.take(*args)', method_object=eval('train_ds'), function_args=[eval('1')], function_kwargs={}, max_wait_secs=0, custom_class=None):
    print('Every feature:', list(feature_batch.keys()))
    print('A batch of ages:', feature_batch['Age'])
    print('A batch of targets:', label_batch)
example_batch = next(iter(train_ds))[0]

def demo(feature_column):
    feature_layer = custom_method(
    layers.DenseFeatures(feature_column), imports='import pandas as pd;from sklearn.model_selection import train_test_split;import tensorflow as tf;from tensorflow import feature_column;from tensorflow.keras import layers;import numpy as np;import pathlib', function_to_run='layers.DenseFeatures(*args)', method_object=None, function_args=[eval('feature_column')], function_kwargs={}, max_wait_secs=0)
    print(feature_layer(example_batch).numpy())
photo_count = custom_method(
feature_column.numeric_column('PhotoAmt'), imports='import pandas as pd;from sklearn.model_selection import train_test_split;import tensorflow as tf;from tensorflow import feature_column;from tensorflow.keras import layers;import numpy as np;import pathlib', function_to_run='feature_column.numeric_column(*args)', method_object=None, function_args=[eval("'PhotoAmt'")], function_kwargs={}, max_wait_secs=0)
demo(photo_count)
age = custom_method(
feature_column.numeric_column('Age'), imports='import pandas as pd;from sklearn.model_selection import train_test_split;import tensorflow as tf;from tensorflow import feature_column;from tensorflow.keras import layers;import numpy as np;import pathlib', function_to_run='feature_column.numeric_column(*args)', method_object=None, function_args=[eval("'Age'")], function_kwargs={}, max_wait_secs=0)
age_buckets = custom_method(
feature_column.bucketized_column(age, boundaries=[1, 3, 5]), imports='import pandas as pd;from sklearn.model_selection import train_test_split;import tensorflow as tf;from tensorflow import feature_column;from tensorflow.keras import layers;import numpy as np;import pathlib', function_to_run='feature_column.bucketized_column(*args, **kwargs)', method_object=None, function_args=[eval('age')], function_kwargs={'boundaries': eval('[1, 3, 5]')}, max_wait_secs=0)
demo(age_buckets)
animal_type = custom_method(
feature_column.categorical_column_with_vocabulary_list('Type', ['Cat', 'Dog']), imports='import pandas as pd;from sklearn.model_selection import train_test_split;import tensorflow as tf;from tensorflow import feature_column;from tensorflow.keras import layers;import numpy as np;import pathlib', function_to_run='feature_column.categorical_column_with_vocabulary_list(*args)', method_object=None, function_args=[eval("'Type'"), eval("['Cat', 'Dog']")], function_kwargs={}, max_wait_secs=0)
animal_type_one_hot = custom_method(
feature_column.indicator_column(animal_type), imports='import pandas as pd;from sklearn.model_selection import train_test_split;import tensorflow as tf;from tensorflow import feature_column;from tensorflow.keras import layers;import numpy as np;import pathlib', function_to_run='feature_column.indicator_column(*args)', method_object=None, function_args=[eval('animal_type')], function_kwargs={}, max_wait_secs=0)
demo(animal_type_one_hot)
breed1 = custom_method(
feature_column.categorical_column_with_vocabulary_list('Breed1', dataframe.Breed1.unique()), imports='import pandas as pd;from sklearn.model_selection import train_test_split;import tensorflow as tf;from tensorflow import feature_column;from tensorflow.keras import layers;import numpy as np;import pathlib', function_to_run='feature_column.categorical_column_with_vocabulary_list(*args)', method_object=None, function_args=[eval("'Breed1'"), eval('dataframe.Breed1.unique()')], function_kwargs={}, max_wait_secs=0)
breed1_embedding = custom_method(
feature_column.embedding_column(breed1, dimension=8), imports='import pandas as pd;from sklearn.model_selection import train_test_split;import tensorflow as tf;from tensorflow import feature_column;from tensorflow.keras import layers;import numpy as np;import pathlib', function_to_run='feature_column.embedding_column(*args, **kwargs)', method_object=None, function_args=[eval('breed1')], function_kwargs={'dimension': eval('8')}, max_wait_secs=0)
demo(breed1_embedding)
breed1_hashed = custom_method(
feature_column.categorical_column_with_hash_bucket('Breed1', hash_bucket_size=10), imports='import pandas as pd;from sklearn.model_selection import train_test_split;import tensorflow as tf;from tensorflow import feature_column;from tensorflow.keras import layers;import numpy as np;import pathlib', function_to_run='feature_column.categorical_column_with_hash_bucket(*args, **kwargs)', method_object=None, function_args=[eval("'Breed1'")], function_kwargs={'hash_bucket_size': eval('10')}, max_wait_secs=0)
demo(feature_column.indicator_column(breed1_hashed))
crossed_feature = custom_method(
feature_column.crossed_column([age_buckets, animal_type], hash_bucket_size=10), imports='import pandas as pd;from sklearn.model_selection import train_test_split;import tensorflow as tf;from tensorflow import feature_column;from tensorflow.keras import layers;import numpy as np;import pathlib', function_to_run='feature_column.crossed_column(*args, **kwargs)', method_object=None, function_args=[eval('[age_buckets, animal_type]')], function_kwargs={'hash_bucket_size': eval('10')}, max_wait_secs=0)
demo(feature_column.indicator_column(crossed_feature))
feature_columns = []
for header in ['PhotoAmt', 'Fee', 'Age']:
    custom_method(
    feature_columns.append(feature_column.numeric_column(header)), imports='import pandas as pd;from sklearn.model_selection import train_test_split;import tensorflow as tf;from tensorflow import feature_column;from tensorflow.keras import layers;import numpy as np;import pathlib', function_to_run='feature_columns.append(*args)', method_object=None, function_args=[eval('feature_column.numeric_column(header)')], function_kwargs={}, max_wait_secs=0)
age = custom_method(
feature_column.numeric_column('Age'), imports='import pandas as pd;from sklearn.model_selection import train_test_split;import tensorflow as tf;from tensorflow import feature_column;from tensorflow.keras import layers;import numpy as np;import pathlib', function_to_run='feature_column.numeric_column(*args)', method_object=None, function_args=[eval("'Age'")], function_kwargs={}, max_wait_secs=0)
age_buckets = custom_method(
feature_column.bucketized_column(age, boundaries=[1, 2, 3, 4, 5]), imports='import pandas as pd;from sklearn.model_selection import train_test_split;import tensorflow as tf;from tensorflow import feature_column;from tensorflow.keras import layers;import numpy as np;import pathlib', function_to_run='feature_column.bucketized_column(*args, **kwargs)', method_object=None, function_args=[eval('age')], function_kwargs={'boundaries': eval('[1, 2, 3, 4, 5]')}, max_wait_secs=0)
custom_method(
feature_columns.append(age_buckets), imports='import pandas as pd;from sklearn.model_selection import train_test_split;import tensorflow as tf;from tensorflow import feature_column;from tensorflow.keras import layers;import numpy as np;import pathlib', function_to_run='feature_columns.append(*args)', method_object=None, function_args=[eval('age_buckets')], function_kwargs={}, max_wait_secs=0)
indicator_column_names = ['Type', 'Color1', 'Color2', 'Gender', 'MaturitySize', 'FurLength', 'Vaccinated', 'Sterilized', 'Health']
for col_name in indicator_column_names:
    categorical_column = custom_method(
    feature_column.categorical_column_with_vocabulary_list(col_name, dataframe[col_name].unique()), imports='import pandas as pd;from sklearn.model_selection import train_test_split;import tensorflow as tf;from tensorflow import feature_column;from tensorflow.keras import layers;import numpy as np;import pathlib', function_to_run='feature_column.categorical_column_with_vocabulary_list(*args)', method_object=None, function_args=[eval('col_name'), eval('dataframe[col_name].unique()')], function_kwargs={}, max_wait_secs=0)
    indicator_column = custom_method(
    feature_column.indicator_column(categorical_column), imports='import pandas as pd;from sklearn.model_selection import train_test_split;import tensorflow as tf;from tensorflow import feature_column;from tensorflow.keras import layers;import numpy as np;import pathlib', function_to_run='feature_column.indicator_column(*args)', method_object=None, function_args=[eval('categorical_column')], function_kwargs={}, max_wait_secs=0)
    custom_method(
    feature_columns.append(indicator_column), imports='import pandas as pd;from sklearn.model_selection import train_test_split;import tensorflow as tf;from tensorflow import feature_column;from tensorflow.keras import layers;import numpy as np;import pathlib', function_to_run='feature_columns.append(*args)', method_object=None, function_args=[eval('indicator_column')], function_kwargs={}, max_wait_secs=0)
breed1 = custom_method(
feature_column.categorical_column_with_vocabulary_list('Breed1', dataframe.Breed1.unique()), imports='import pandas as pd;from sklearn.model_selection import train_test_split;import tensorflow as tf;from tensorflow import feature_column;from tensorflow.keras import layers;import numpy as np;import pathlib', function_to_run='feature_column.categorical_column_with_vocabulary_list(*args)', method_object=None, function_args=[eval("'Breed1'"), eval('dataframe.Breed1.unique()')], function_kwargs={}, max_wait_secs=0)
breed1_embedding = custom_method(
feature_column.embedding_column(breed1, dimension=8), imports='import pandas as pd;from sklearn.model_selection import train_test_split;import tensorflow as tf;from tensorflow import feature_column;from tensorflow.keras import layers;import numpy as np;import pathlib', function_to_run='feature_column.embedding_column(*args, **kwargs)', method_object=None, function_args=[eval('breed1')], function_kwargs={'dimension': eval('8')}, max_wait_secs=0)
custom_method(
feature_columns.append(breed1_embedding), imports='import pandas as pd;from sklearn.model_selection import train_test_split;import tensorflow as tf;from tensorflow import feature_column;from tensorflow.keras import layers;import numpy as np;import pathlib', function_to_run='feature_columns.append(*args)', method_object=None, function_args=[eval('breed1_embedding')], function_kwargs={}, max_wait_secs=0)
age_type_feature = custom_method(
feature_column.crossed_column([age_buckets, animal_type], hash_bucket_size=100), imports='import pandas as pd;from sklearn.model_selection import train_test_split;import tensorflow as tf;from tensorflow import feature_column;from tensorflow.keras import layers;import numpy as np;import pathlib', function_to_run='feature_column.crossed_column(*args, **kwargs)', method_object=None, function_args=[eval('[age_buckets, animal_type]')], function_kwargs={'hash_bucket_size': eval('100')}, max_wait_secs=0)
custom_method(
feature_columns.append(feature_column.indicator_column(age_type_feature)), imports='import pandas as pd;from sklearn.model_selection import train_test_split;import tensorflow as tf;from tensorflow import feature_column;from tensorflow.keras import layers;import numpy as np;import pathlib', function_to_run='feature_columns.append(*args)', method_object=None, function_args=[eval('feature_column.indicator_column(age_type_feature)')], function_kwargs={}, max_wait_secs=0)
feature_layer = custom_method(
tf.keras.layers.DenseFeatures(feature_columns), imports='import pandas as pd;from sklearn.model_selection import train_test_split;import tensorflow as tf;from tensorflow import feature_column;from tensorflow.keras import layers;import numpy as np;import pathlib', function_to_run='tf.keras.layers.DenseFeatures(*args)', method_object=None, function_args=[eval('feature_columns')], function_kwargs={}, max_wait_secs=0)
batch_size = 32
train_ds = df_to_dataset(train, batch_size=batch_size)
val_ds = df_to_dataset(val, shuffle=False, batch_size=batch_size)
test_ds = df_to_dataset(test, shuffle=False, batch_size=batch_size)
model = custom_method(
tf.keras.Sequential([feature_layer, layers.Dense(128, activation='relu'), layers.Dense(128, activation='relu'), layers.Dropout(0.1), layers.Dense(1)]), imports='import pandas as pd;from sklearn.model_selection import train_test_split;import tensorflow as tf;from tensorflow import feature_column;from tensorflow.keras import layers;import numpy as np;import pathlib', function_to_run='tf.keras.Sequential(*args)', method_object=None, function_args=[eval("[\n  feature_layer,\n  layers.Dense(128, activation='relu'),\n  layers.Dense(128, activation='relu'),\n  layers.Dropout(.1),\n  layers.Dense(1)\n]")], function_kwargs={}, max_wait_secs=0)
custom_method(
model.compile(optimizer='adam', loss=tf.keras.losses.BinaryCrossentropy(from_logits=True), metrics=['accuracy']), imports='import pandas as pd;from sklearn.model_selection import train_test_split;import tensorflow as tf;from tensorflow import feature_column;from tensorflow.keras import layers;import numpy as np;import pathlib', function_to_run='obj.compile(**kwargs)', method_object=eval('model'), function_args=[], function_kwargs={'optimizer': eval("'adam'"), 'loss': eval('tf.keras.losses.BinaryCrossentropy(from_logits=True)'), 'metrics': eval("['accuracy']")}, max_wait_secs=0, custom_class=None)
custom_method(
model.fit(train_ds, validation_data=val_ds, epochs=10), imports='import pandas as pd;from sklearn.model_selection import train_test_split;import tensorflow as tf;from tensorflow import feature_column;from tensorflow.keras import layers;import numpy as np;import pathlib', function_to_run='obj.fit(*args, **kwargs)', method_object=eval('model'), function_args=[eval('train_ds')], function_kwargs={'validation_data': eval('val_ds'), 'epochs': eval('10')}, max_wait_secs=0, custom_class=None)
(loss, accuracy) = custom_method(
model.evaluate(test_ds), imports='import pandas as pd;from sklearn.model_selection import train_test_split;import tensorflow as tf;from tensorflow import feature_column;from tensorflow.keras import layers;import numpy as np;import pathlib', function_to_run='obj.evaluate(*args)', method_object=eval('model'), function_args=[eval('test_ds')], function_kwargs={}, max_wait_secs=0, custom_class=None)
print('Accuracy', accuracy)
