import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers
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
tf.__version__
dataset_url = 'http://storage.googleapis.com/download.tensorflow.org/data/petfinder-mini.zip'
csv_file = 'datasets/petfinder-mini/petfinder-mini.csv'
custom_method(
tf.keras.utils.get_file('petfinder_mini.zip', dataset_url, extract=True, cache_dir='.'), imports='import tensorflow as tf;import pandas as pd;import numpy as np;from tensorflow.keras import layers', function_to_run='tf.keras.utils.get_file(*args, **kwargs)', method_object=None, object_signature=None, function_args=[eval("'petfinder_mini.zip'"), eval('dataset_url')], function_kwargs={'extract': eval('True'), 'cache_dir': eval("'.'")})
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
    ds = custom_method(
    tf.data.Dataset.from_tensor_slices((dict(df), labels)), imports='import tensorflow as tf;import pandas as pd;import numpy as np;from tensorflow.keras import layers', function_to_run='tf.data.Dataset.from_tensor_slices(*args)', method_object=None, object_signature=None, function_args=[eval('(dict(df), labels)')], function_kwargs={})
    if shuffle:
        ds = ds.shuffle(buffer_size=len(dataframe))
    ds = ds.batch(batch_size)
    ds = ds.prefetch(batch_size)
    return ds
batch_size = 5
train_ds = df_to_dataset(train, batch_size=batch_size)
[(train_features, label_batch)] = train_ds.take(1)
print('Every feature:', list(train_features.keys()))
print('A batch of ages:', train_features['Age'])
print('A batch of targets:', label_batch)

def get_normalization_layer(name, dataset):
    normalizer = custom_method(
    layers.Normalization(axis=None), imports='import tensorflow as tf;import pandas as pd;import numpy as np;from tensorflow.keras import layers', function_to_run='layers.Normalization(**kwargs)', method_object=None, object_signature=None, function_args=[], function_kwargs={'axis': eval('None')})
    feature_ds = dataset.map(lambda x, y: x[name])
    normalizer.adapt(feature_ds)
    return normalizer
photo_count_col = train_features['PhotoAmt']
layer = get_normalization_layer('PhotoAmt', train_ds)
layer(photo_count_col)

def get_category_encoding_layer(name, dataset, dtype, max_tokens=None):
    if dtype == 'string':
        index = custom_method(
        layers.StringLookup(max_tokens=max_tokens), imports='import tensorflow as tf;import pandas as pd;import numpy as np;from tensorflow.keras import layers', function_to_run='layers.StringLookup(**kwargs)', method_object=None, object_signature=None, function_args=[], function_kwargs={'max_tokens': eval('max_tokens')})
    else:
        index = custom_method(
        layers.IntegerLookup(max_tokens=max_tokens), imports='import tensorflow as tf;import pandas as pd;import numpy as np;from tensorflow.keras import layers', function_to_run='layers.IntegerLookup(**kwargs)', method_object=None, object_signature=None, function_args=[], function_kwargs={'max_tokens': eval('max_tokens')})
    feature_ds = dataset.map(lambda x, y: x[name])
    index.adapt(feature_ds)
    encoder = custom_method(
    layers.CategoryEncoding(num_tokens=index.vocabulary_size()), imports='import tensorflow as tf;import pandas as pd;import numpy as np;from tensorflow.keras import layers', function_to_run='layers.CategoryEncoding(**kwargs)', method_object=None, object_signature=None, function_args=[], function_kwargs={'num_tokens': eval('index.vocabulary_size()')})
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
    numeric_col = custom_method(
    tf.keras.Input(shape=(1,), name=header), imports='import tensorflow as tf;import pandas as pd;import numpy as np;from tensorflow.keras import layers', function_to_run='tf.keras.Input(**kwargs)', method_object=None, object_signature=None, function_args=[], function_kwargs={'shape': eval('(1,)'), 'name': eval('header')})
    normalization_layer = get_normalization_layer(header, train_ds)
    encoded_numeric_col = normalization_layer(numeric_col)
    all_inputs.append(numeric_col)
    encoded_features.append(encoded_numeric_col)
age_col = custom_method(
tf.keras.Input(shape=(1,), name='Age', dtype='int64'), imports='import tensorflow as tf;import pandas as pd;import numpy as np;from tensorflow.keras import layers', function_to_run='tf.keras.Input(**kwargs)', method_object=None, object_signature=None, function_args=[], function_kwargs={'shape': eval('(1,)'), 'name': eval("'Age'"), 'dtype': eval("'int64'")})
encoding_layer = get_category_encoding_layer(name='Age', dataset=train_ds, dtype='int64', max_tokens=5)
encoded_age_col = encoding_layer(age_col)
all_inputs.append(age_col)
encoded_features.append(encoded_age_col)
categorical_cols = ['Type', 'Color1', 'Color2', 'Gender', 'MaturitySize', 'FurLength', 'Vaccinated', 'Sterilized', 'Health', 'Breed1']
for header in categorical_cols:
    categorical_col = custom_method(
    tf.keras.Input(shape=(1,), name=header, dtype='string'), imports='import tensorflow as tf;import pandas as pd;import numpy as np;from tensorflow.keras import layers', function_to_run='tf.keras.Input(**kwargs)', method_object=None, object_signature=None, function_args=[], function_kwargs={'shape': eval('(1,)'), 'name': eval('header'), 'dtype': eval("'string'")})
    encoding_layer = get_category_encoding_layer(name=header, dataset=train_ds, dtype='string', max_tokens=5)
    encoded_categorical_col = encoding_layer(categorical_col)
    all_inputs.append(categorical_col)
    encoded_features.append(encoded_categorical_col)
all_features = custom_method(
tf.keras.layers.concatenate(encoded_features), imports='import tensorflow as tf;import pandas as pd;import numpy as np;from tensorflow.keras import layers', function_to_run='tf.keras.layers.concatenate(*args)', method_object=None, object_signature=None, function_args=[eval('encoded_features')], function_kwargs={})
x = custom_method(
tf.keras.layers.Dense(32, activation='relu')(all_features), imports='import tensorflow as tf;import pandas as pd;import numpy as np;from tensorflow.keras import layers', function_to_run="tf.keras.layers.Dense(32, activation='relu')(*args)", method_object=None, object_signature=None, function_args=[eval('all_features')], function_kwargs={})
x = custom_method(
tf.keras.layers.Dropout(0.5)(x), imports='import tensorflow as tf;import pandas as pd;import numpy as np;from tensorflow.keras import layers', function_to_run='tf.keras.layers.Dropout(0.5)(*args)', method_object=None, object_signature=None, function_args=[eval('x')], function_kwargs={})
output = custom_method(
tf.keras.layers.Dense(1)(x), imports='import tensorflow as tf;import pandas as pd;import numpy as np;from tensorflow.keras import layers', function_to_run='tf.keras.layers.Dense(1)(*args)', method_object=None, object_signature=None, function_args=[eval('x')], function_kwargs={})
model = custom_method(
tf.keras.Model(all_inputs, output), imports='import tensorflow as tf;import pandas as pd;import numpy as np;from tensorflow.keras import layers', function_to_run='tf.keras.Model(*args)', method_object=None, object_signature=None, function_args=[eval('all_inputs'), eval('output')], function_kwargs={})
model.compile(optimizer='adam', loss=tf.keras.losses.BinaryCrossentropy(from_logits=True), metrics=['accuracy'])
custom_method(
tf.keras.utils.plot_model(model, show_shapes=True, rankdir='LR'), imports='import tensorflow as tf;import pandas as pd;import numpy as np;from tensorflow.keras import layers', function_to_run='tf.keras.utils.plot_model(*args, **kwargs)', method_object=None, object_signature=None, function_args=[eval('model')], function_kwargs={'show_shapes': eval('True'), 'rankdir': eval('"LR"')})
model.fit(train_ds, epochs=10, validation_data=val_ds)
(loss, accuracy) = model.evaluate(test_ds)
print('Accuracy', accuracy)
model.save('my_pet_classifier')
reloaded_model = custom_method(
tf.keras.models.load_model('my_pet_classifier'), imports='import tensorflow as tf;import pandas as pd;import numpy as np;from tensorflow.keras import layers', function_to_run='tf.keras.models.load_model(*args)', method_object=None, object_signature=None, function_args=[eval("'my_pet_classifier'")], function_kwargs={})
sample = {'Type': 'Cat', 'Age': 3, 'Breed1': 'Tabby', 'Gender': 'Male', 'Color1': 'Black', 'Color2': 'White', 'MaturitySize': 'Small', 'FurLength': 'Short', 'Vaccinated': 'No', 'Sterilized': 'No', 'Health': 'Healthy', 'Fee': 100, 'PhotoAmt': 2}
input_dict = {name: tf.convert_to_tensor([value]) for (name, value) in sample.items()}
predictions = reloaded_model.predict(input_dict)
prob = custom_method(
tf.nn.sigmoid(predictions[0]), imports='import tensorflow as tf;import pandas as pd;import numpy as np;from tensorflow.keras import layers', function_to_run='tf.nn.sigmoid(*args)', method_object=None, object_signature=None, function_args=[eval('predictions[0]')], function_kwargs={})
print('This particular pet had a %.1f percent probability of getting adopted.' % (100 * prob))
