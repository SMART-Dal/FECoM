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
dataset_url = 'http://storage.googleapis.com/download.tensorflow.org/data/petfinder-mini.zip'
csv_file = 'datasets/petfinder-mini/petfinder-mini.csv'
custom_method(
tf.keras.utils.get_file('petfinder_mini.zip', dataset_url, extract=True, cache_dir='.'), imports='from sklearn.model_selection import train_test_split;import tensorflow as tf;import pathlib;import numpy as np;import pandas as pd;from tensorflow import feature_column;from tensorflow.keras import layers', function_to_run='tf.keras.utils.get_file(*args, **kwargs)', method_object=None, object_signature=None, function_args=[eval("'petfinder_mini.zip'"), eval('dataset_url')], function_kwargs={'extract': eval('True'), 'cache_dir': eval("'.'")})
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
    tf.data.Dataset.from_tensor_slices((dict(dataframe), labels)), imports='from sklearn.model_selection import train_test_split;import tensorflow as tf;import pathlib;import numpy as np;import pandas as pd;from tensorflow import feature_column;from tensorflow.keras import layers', function_to_run='tf.data.Dataset.from_tensor_slices(*args)', method_object=None, object_signature=None, function_args=[eval('(dict(dataframe), labels)')], function_kwargs={})
    if shuffle:
        ds = ds.shuffle(buffer_size=len(dataframe))
    ds = ds.batch(batch_size)
    return ds
batch_size = 5
train_ds = df_to_dataset(train, batch_size=batch_size)
val_ds = df_to_dataset(val, shuffle=False, batch_size=batch_size)
test_ds = df_to_dataset(test, shuffle=False, batch_size=batch_size)
for (feature_batch, label_batch) in train_ds.take(1):
    print('Every feature:', list(feature_batch.keys()))
    print('A batch of ages:', feature_batch['Age'])
    print('A batch of targets:', label_batch)
example_batch = next(iter(train_ds))[0]

def demo(feature_column):
    feature_layer = custom_method(
    layers.DenseFeatures(feature_column), imports='from sklearn.model_selection import train_test_split;import tensorflow as tf;import pathlib;import numpy as np;import pandas as pd;from tensorflow import feature_column;from tensorflow.keras import layers', function_to_run='layers.DenseFeatures(*args)', method_object=None, object_signature=None, function_args=[eval('feature_column')], function_kwargs={})
    print(feature_layer(example_batch).numpy())
photo_count = custom_method(
feature_column.numeric_column('PhotoAmt'), imports='from sklearn.model_selection import train_test_split;import tensorflow as tf;import pathlib;import numpy as np;import pandas as pd;from tensorflow import feature_column;from tensorflow.keras import layers', function_to_run='feature_column.numeric_column(*args)', method_object=None, object_signature=None, function_args=[eval("'PhotoAmt'")], function_kwargs={})
demo(photo_count)
age = custom_method(
feature_column.numeric_column('Age'), imports='from sklearn.model_selection import train_test_split;import tensorflow as tf;import pathlib;import numpy as np;import pandas as pd;from tensorflow import feature_column;from tensorflow.keras import layers', function_to_run='feature_column.numeric_column(*args)', method_object=None, object_signature=None, function_args=[eval("'Age'")], function_kwargs={})
age_buckets = custom_method(
feature_column.bucketized_column(age, boundaries=[1, 3, 5]), imports='from sklearn.model_selection import train_test_split;import tensorflow as tf;import pathlib;import numpy as np;import pandas as pd;from tensorflow import feature_column;from tensorflow.keras import layers', function_to_run='feature_column.bucketized_column(*args, **kwargs)', method_object=None, object_signature=None, function_args=[eval('age')], function_kwargs={'boundaries': eval('[1, 3, 5]')})
demo(age_buckets)
animal_type = custom_method(
feature_column.categorical_column_with_vocabulary_list('Type', ['Cat', 'Dog']), imports='from sklearn.model_selection import train_test_split;import tensorflow as tf;import pathlib;import numpy as np;import pandas as pd;from tensorflow import feature_column;from tensorflow.keras import layers', function_to_run='feature_column.categorical_column_with_vocabulary_list(*args)', method_object=None, object_signature=None, function_args=[eval("'Type'"), eval("['Cat', 'Dog']")], function_kwargs={})
animal_type_one_hot = custom_method(
feature_column.indicator_column(animal_type), imports='from sklearn.model_selection import train_test_split;import tensorflow as tf;import pathlib;import numpy as np;import pandas as pd;from tensorflow import feature_column;from tensorflow.keras import layers', function_to_run='feature_column.indicator_column(*args)', method_object=None, object_signature=None, function_args=[eval('animal_type')], function_kwargs={})
demo(animal_type_one_hot)
breed1 = custom_method(
feature_column.categorical_column_with_vocabulary_list('Breed1', dataframe.Breed1.unique()), imports='from sklearn.model_selection import train_test_split;import tensorflow as tf;import pathlib;import numpy as np;import pandas as pd;from tensorflow import feature_column;from tensorflow.keras import layers', function_to_run='feature_column.categorical_column_with_vocabulary_list(*args)', method_object=None, object_signature=None, function_args=[eval("'Breed1'"), eval('dataframe.Breed1.unique()')], function_kwargs={})
breed1_embedding = custom_method(
feature_column.embedding_column(breed1, dimension=8), imports='from sklearn.model_selection import train_test_split;import tensorflow as tf;import pathlib;import numpy as np;import pandas as pd;from tensorflow import feature_column;from tensorflow.keras import layers', function_to_run='feature_column.embedding_column(*args, **kwargs)', method_object=None, object_signature=None, function_args=[eval('breed1')], function_kwargs={'dimension': eval('8')})
demo(breed1_embedding)
breed1_hashed = custom_method(
feature_column.categorical_column_with_hash_bucket('Breed1', hash_bucket_size=10), imports='from sklearn.model_selection import train_test_split;import tensorflow as tf;import pathlib;import numpy as np;import pandas as pd;from tensorflow import feature_column;from tensorflow.keras import layers', function_to_run='feature_column.categorical_column_with_hash_bucket(*args, **kwargs)', method_object=None, object_signature=None, function_args=[eval("'Breed1'")], function_kwargs={'hash_bucket_size': eval('10')})
demo(feature_column.indicator_column(breed1_hashed))
crossed_feature = custom_method(
feature_column.crossed_column([age_buckets, animal_type], hash_bucket_size=10), imports='from sklearn.model_selection import train_test_split;import tensorflow as tf;import pathlib;import numpy as np;import pandas as pd;from tensorflow import feature_column;from tensorflow.keras import layers', function_to_run='feature_column.crossed_column(*args, **kwargs)', method_object=None, object_signature=None, function_args=[eval('[age_buckets, animal_type]')], function_kwargs={'hash_bucket_size': eval('10')})
demo(feature_column.indicator_column(crossed_feature))
feature_columns = []
for header in ['PhotoAmt', 'Fee', 'Age']:
    feature_columns.append(feature_column.numeric_column(header))
age = custom_method(
feature_column.numeric_column('Age'), imports='from sklearn.model_selection import train_test_split;import tensorflow as tf;import pathlib;import numpy as np;import pandas as pd;from tensorflow import feature_column;from tensorflow.keras import layers', function_to_run='feature_column.numeric_column(*args)', method_object=None, object_signature=None, function_args=[eval("'Age'")], function_kwargs={})
age_buckets = custom_method(
feature_column.bucketized_column(age, boundaries=[1, 2, 3, 4, 5]), imports='from sklearn.model_selection import train_test_split;import tensorflow as tf;import pathlib;import numpy as np;import pandas as pd;from tensorflow import feature_column;from tensorflow.keras import layers', function_to_run='feature_column.bucketized_column(*args, **kwargs)', method_object=None, object_signature=None, function_args=[eval('age')], function_kwargs={'boundaries': eval('[1, 2, 3, 4, 5]')})
feature_columns.append(age_buckets)
indicator_column_names = ['Type', 'Color1', 'Color2', 'Gender', 'MaturitySize', 'FurLength', 'Vaccinated', 'Sterilized', 'Health']
for col_name in indicator_column_names:
    categorical_column = custom_method(
    feature_column.categorical_column_with_vocabulary_list(col_name, dataframe[col_name].unique()), imports='from sklearn.model_selection import train_test_split;import tensorflow as tf;import pathlib;import numpy as np;import pandas as pd;from tensorflow import feature_column;from tensorflow.keras import layers', function_to_run='feature_column.categorical_column_with_vocabulary_list(*args)', method_object=None, object_signature=None, function_args=[eval('col_name'), eval('dataframe[col_name].unique()')], function_kwargs={})
    indicator_column = custom_method(
    feature_column.indicator_column(categorical_column), imports='from sklearn.model_selection import train_test_split;import tensorflow as tf;import pathlib;import numpy as np;import pandas as pd;from tensorflow import feature_column;from tensorflow.keras import layers', function_to_run='feature_column.indicator_column(*args)', method_object=None, object_signature=None, function_args=[eval('categorical_column')], function_kwargs={})
    feature_columns.append(indicator_column)
breed1 = custom_method(
feature_column.categorical_column_with_vocabulary_list('Breed1', dataframe.Breed1.unique()), imports='from sklearn.model_selection import train_test_split;import tensorflow as tf;import pathlib;import numpy as np;import pandas as pd;from tensorflow import feature_column;from tensorflow.keras import layers', function_to_run='feature_column.categorical_column_with_vocabulary_list(*args)', method_object=None, object_signature=None, function_args=[eval("'Breed1'"), eval('dataframe.Breed1.unique()')], function_kwargs={})
breed1_embedding = custom_method(
feature_column.embedding_column(breed1, dimension=8), imports='from sklearn.model_selection import train_test_split;import tensorflow as tf;import pathlib;import numpy as np;import pandas as pd;from tensorflow import feature_column;from tensorflow.keras import layers', function_to_run='feature_column.embedding_column(*args, **kwargs)', method_object=None, object_signature=None, function_args=[eval('breed1')], function_kwargs={'dimension': eval('8')})
feature_columns.append(breed1_embedding)
age_type_feature = custom_method(
feature_column.crossed_column([age_buckets, animal_type], hash_bucket_size=100), imports='from sklearn.model_selection import train_test_split;import tensorflow as tf;import pathlib;import numpy as np;import pandas as pd;from tensorflow import feature_column;from tensorflow.keras import layers', function_to_run='feature_column.crossed_column(*args, **kwargs)', method_object=None, object_signature=None, function_args=[eval('[age_buckets, animal_type]')], function_kwargs={'hash_bucket_size': eval('100')})
feature_columns.append(feature_column.indicator_column(age_type_feature))
feature_layer = custom_method(
tf.keras.layers.DenseFeatures(feature_columns), imports='from sklearn.model_selection import train_test_split;import tensorflow as tf;import pathlib;import numpy as np;import pandas as pd;from tensorflow import feature_column;from tensorflow.keras import layers', function_to_run='tf.keras.layers.DenseFeatures(*args)', method_object=None, object_signature=None, function_args=[eval('feature_columns')], function_kwargs={})
batch_size = 32
train_ds = df_to_dataset(train, batch_size=batch_size)
val_ds = df_to_dataset(val, shuffle=False, batch_size=batch_size)
test_ds = df_to_dataset(test, shuffle=False, batch_size=batch_size)
model = custom_method(
tf.keras.Sequential([feature_layer, layers.Dense(128, activation='relu'), layers.Dense(128, activation='relu'), layers.Dropout(0.1), layers.Dense(1)]), imports='from sklearn.model_selection import train_test_split;import tensorflow as tf;import pathlib;import numpy as np;import pandas as pd;from tensorflow import feature_column;from tensorflow.keras import layers', function_to_run='tf.keras.Sequential(*args)', method_object=None, object_signature=None, function_args=[eval("[\n  feature_layer,\n  layers.Dense(128, activation='relu'),\n  layers.Dense(128, activation='relu'),\n  layers.Dropout(.1),\n  layers.Dense(1)\n]")], function_kwargs={})
model.compile(optimizer='adam', loss=tf.keras.losses.BinaryCrossentropy(from_logits=True), metrics=['accuracy'])
model.fit(train_ds, validation_data=val_ds, epochs=10)
(loss, accuracy) = model.evaluate(test_ds)
print('Accuracy', accuracy)
