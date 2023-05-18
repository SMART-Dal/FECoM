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
tf.__version__
dataset_url = 'http://storage.googleapis.com/download.tensorflow.org/data/petfinder-mini.zip'
csv_file = 'datasets/petfinder-mini/petfinder-mini.csv'
custom_method(imports='from tensorflow.keras import layers;import pandas as pd;import tensorflow as tf;import numpy as np', function_to_run='tf.keras.utils.get_file(*args, **kwargs)', method_object=None, object_signature=None, function_args=[eval("'petfinder_mini.zip'"), eval('dataset_url')], function_kwargs={'extract': eval('True'), 'cache_dir': eval("'.'")})
tf.keras.utils.get_file('petfinder_mini.zip', dataset_url, extract=True, cache_dir='.')
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
    custom_method(imports='from tensorflow.keras import layers;import pandas as pd;import tensorflow as tf;import numpy as np', function_to_run='tf.data.Dataset.from_tensor_slices(*args)', method_object=None, object_signature=None, function_args=[eval('(dict(df), labels)')], function_kwargs={})
    ds = tf.data.Dataset.from_tensor_slices((dict(df), labels))
    if shuffle:
        custom_method(imports='from tensorflow.keras import layers;import pandas as pd;import tensorflow as tf;import numpy as np', function_to_run='obj.shuffle(**kwargs)', method_object=eval('ds'), object_signature=None, function_args=[], function_kwargs={'buffer_size': eval('len(dataframe)')}, custom_class=None)
        ds = ds.shuffle(buffer_size=len(dataframe))
    custom_method(imports='from tensorflow.keras import layers;import pandas as pd;import tensorflow as tf;import numpy as np', function_to_run='obj.batch(*args)', method_object=eval('ds'), object_signature=None, function_args=[eval('batch_size')], function_kwargs={}, custom_class=None)
    ds = ds.batch(batch_size)
    custom_method(imports='from tensorflow.keras import layers;import pandas as pd;import tensorflow as tf;import numpy as np', function_to_run='obj.prefetch(*args)', method_object=eval('ds'), object_signature=None, function_args=[eval('batch_size')], function_kwargs={}, custom_class=None)
    ds = ds.prefetch(batch_size)
    return ds
batch_size = 5
train_ds = df_to_dataset(train, batch_size=batch_size)
[(train_features, label_batch)] = train_ds.take(1)
print('Every feature:', list(train_features.keys()))
print('A batch of ages:', train_features['Age'])
print('A batch of targets:', label_batch)

def get_normalization_layer(name, dataset):
    custom_method(imports='from tensorflow.keras import layers;import pandas as pd;import tensorflow as tf;import numpy as np', function_to_run='layers.Normalization(**kwargs)', method_object=None, object_signature=None, function_args=[], function_kwargs={'axis': eval('None')})
    normalizer = layers.Normalization(axis=None)
    feature_ds = dataset.map(lambda x, y: x[name])
    custom_method(imports='from tensorflow.keras import layers;import pandas as pd;import tensorflow as tf;import numpy as np', function_to_run='obj.adapt(*args)', method_object=eval('normalizer'), object_signature=None, function_args=[eval('feature_ds')], function_kwargs={}, custom_class=None)
    normalizer.adapt(feature_ds)
    return normalizer
photo_count_col = train_features['PhotoAmt']
layer = get_normalization_layer('PhotoAmt', train_ds)
layer(photo_count_col)

def get_category_encoding_layer(name, dataset, dtype, max_tokens=None):
    if dtype == 'string':
        custom_method(imports='from tensorflow.keras import layers;import pandas as pd;import tensorflow as tf;import numpy as np', function_to_run='layers.StringLookup(**kwargs)', method_object=None, object_signature=None, function_args=[], function_kwargs={'max_tokens': eval('max_tokens')})
        index = layers.StringLookup(max_tokens=max_tokens)
    else:
        custom_method(imports='from tensorflow.keras import layers;import pandas as pd;import tensorflow as tf;import numpy as np', function_to_run='layers.IntegerLookup(**kwargs)', method_object=None, object_signature=None, function_args=[], function_kwargs={'max_tokens': eval('max_tokens')})
        index = layers.IntegerLookup(max_tokens=max_tokens)
    feature_ds = dataset.map(lambda x, y: x[name])
    custom_method(imports='from tensorflow.keras import layers;import pandas as pd;import tensorflow as tf;import numpy as np', function_to_run='obj.adapt(*args)', method_object=eval('index'), object_signature=None, function_args=[eval('feature_ds')], function_kwargs={}, custom_class=None)
    index.adapt(feature_ds)
    custom_method(imports='from tensorflow.keras import layers;import pandas as pd;import tensorflow as tf;import numpy as np', function_to_run='layers.CategoryEncoding(**kwargs)', method_object=None, object_signature=None, function_args=[], function_kwargs={'num_tokens': eval('index.vocabulary_size()')})
    encoder = layers.CategoryEncoding(num_tokens=index.vocabulary_size())
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
    custom_method(imports='from tensorflow.keras import layers;import pandas as pd;import tensorflow as tf;import numpy as np', function_to_run='tf.keras.Input(**kwargs)', method_object=None, object_signature=None, function_args=[], function_kwargs={'shape': eval('(1,)'), 'name': eval('header')})
    numeric_col = tf.keras.Input(shape=(1,), name=header)
    normalization_layer = get_normalization_layer(header, train_ds)
    encoded_numeric_col = normalization_layer(numeric_col)
    all_inputs.append(numeric_col)
    encoded_features.append(encoded_numeric_col)
custom_method(imports='from tensorflow.keras import layers;import pandas as pd;import tensorflow as tf;import numpy as np', function_to_run='tf.keras.Input(**kwargs)', method_object=None, object_signature=None, function_args=[], function_kwargs={'shape': eval('(1,)'), 'name': eval("'Age'"), 'dtype': eval("'int64'")})
age_col = tf.keras.Input(shape=(1,), name='Age', dtype='int64')
encoding_layer = get_category_encoding_layer(name='Age', dataset=train_ds, dtype='int64', max_tokens=5)
encoded_age_col = encoding_layer(age_col)
all_inputs.append(age_col)
encoded_features.append(encoded_age_col)
categorical_cols = ['Type', 'Color1', 'Color2', 'Gender', 'MaturitySize', 'FurLength', 'Vaccinated', 'Sterilized', 'Health', 'Breed1']
for header in categorical_cols:
    custom_method(imports='from tensorflow.keras import layers;import pandas as pd;import tensorflow as tf;import numpy as np', function_to_run='tf.keras.Input(**kwargs)', method_object=None, object_signature=None, function_args=[], function_kwargs={'shape': eval('(1,)'), 'name': eval('header'), 'dtype': eval("'string'")})
    categorical_col = tf.keras.Input(shape=(1,), name=header, dtype='string')
    encoding_layer = get_category_encoding_layer(name=header, dataset=train_ds, dtype='string', max_tokens=5)
    encoded_categorical_col = encoding_layer(categorical_col)
    all_inputs.append(categorical_col)
    encoded_features.append(encoded_categorical_col)
custom_method(imports='from tensorflow.keras import layers;import pandas as pd;import tensorflow as tf;import numpy as np', function_to_run='tf.keras.layers.concatenate(*args)', method_object=None, object_signature=None, function_args=[eval('encoded_features')], function_kwargs={})
all_features = tf.keras.layers.concatenate(encoded_features)
custom_method(imports='from tensorflow.keras import layers;import pandas as pd;import tensorflow as tf;import numpy as np', function_to_run="tf.keras.layers.Dense(32, activation='relu')(*args)", method_object=None, object_signature=None, function_args=[eval('all_features')], function_kwargs={})
x = tf.keras.layers.Dense(32, activation='relu')(all_features)
custom_method(imports='from tensorflow.keras import layers;import pandas as pd;import tensorflow as tf;import numpy as np', function_to_run='tf.keras.layers.Dropout(0.5)(*args)', method_object=None, object_signature=None, function_args=[eval('x')], function_kwargs={})
x = tf.keras.layers.Dropout(0.5)(x)
custom_method(imports='from tensorflow.keras import layers;import pandas as pd;import tensorflow as tf;import numpy as np', function_to_run='tf.keras.layers.Dense(1)(*args)', method_object=None, object_signature=None, function_args=[eval('x')], function_kwargs={})
output = tf.keras.layers.Dense(1)(x)
custom_method(imports='from tensorflow.keras import layers;import pandas as pd;import tensorflow as tf;import numpy as np', function_to_run='tf.keras.Model(*args)', method_object=None, object_signature=None, function_args=[eval('all_inputs'), eval('output')], function_kwargs={})
model = tf.keras.Model(all_inputs, output)
custom_method(imports='from tensorflow.keras import layers;import pandas as pd;import tensorflow as tf;import numpy as np', function_to_run='obj.compile(**kwargs)', method_object=eval('model'), object_signature=None, function_args=[], function_kwargs={'optimizer': eval("'adam'"), 'loss': eval('tf.keras.losses.BinaryCrossentropy(from_logits=True)'), 'metrics': eval('["accuracy"]')}, custom_class=None)
model.compile(optimizer='adam', loss=tf.keras.losses.BinaryCrossentropy(from_logits=True), metrics=['accuracy'])
custom_method(imports='from tensorflow.keras import layers;import pandas as pd;import tensorflow as tf;import numpy as np', function_to_run='tf.keras.utils.plot_model(*args, **kwargs)', method_object=None, object_signature=None, function_args=[eval('model')], function_kwargs={'show_shapes': eval('True'), 'rankdir': eval('"LR"')})
tf.keras.utils.plot_model(model, show_shapes=True, rankdir='LR')
custom_method(imports='from tensorflow.keras import layers;import pandas as pd;import tensorflow as tf;import numpy as np', function_to_run='obj.fit(*args, **kwargs)', method_object=eval('model'), object_signature=None, function_args=[eval('train_ds')], function_kwargs={'epochs': eval('10'), 'validation_data': eval('val_ds')}, custom_class=None)
model.fit(train_ds, epochs=10, validation_data=val_ds)
custom_method(imports='from tensorflow.keras import layers;import pandas as pd;import tensorflow as tf;import numpy as np', function_to_run='obj.evaluate(*args)', method_object=eval('model'), object_signature=None, function_args=[eval('test_ds')], function_kwargs={}, custom_class=None)
(loss, accuracy) = model.evaluate(test_ds)
print('Accuracy', accuracy)
custom_method(imports='from tensorflow.keras import layers;import pandas as pd;import tensorflow as tf;import numpy as np', function_to_run='obj.save(*args)', method_object=eval('model'), object_signature=None, function_args=[eval("'my_pet_classifier'")], function_kwargs={}, custom_class=None)
model.save('my_pet_classifier')
custom_method(imports='from tensorflow.keras import layers;import pandas as pd;import tensorflow as tf;import numpy as np', function_to_run='tf.keras.models.load_model(*args)', method_object=None, object_signature=None, function_args=[eval("'my_pet_classifier'")], function_kwargs={})
reloaded_model = tf.keras.models.load_model('my_pet_classifier')
sample = {'Type': 'Cat', 'Age': 3, 'Breed1': 'Tabby', 'Gender': 'Male', 'Color1': 'Black', 'Color2': 'White', 'MaturitySize': 'Small', 'FurLength': 'Short', 'Vaccinated': 'No', 'Sterilized': 'No', 'Health': 'Healthy', 'Fee': 100, 'PhotoAmt': 2}
input_dict = {name: tf.convert_to_tensor([value]) for (name, value) in sample.items()}
custom_method(imports='from tensorflow.keras import layers;import pandas as pd;import tensorflow as tf;import numpy as np', function_to_run='obj.predict(*args)', method_object=eval('reloaded_model'), object_signature=None, function_args=[eval('input_dict')], function_kwargs={}, custom_class=None)
predictions = reloaded_model.predict(input_dict)
custom_method(imports='from tensorflow.keras import layers;import pandas as pd;import tensorflow as tf;import numpy as np', function_to_run='tf.nn.sigmoid(*args)', method_object=None, object_signature=None, function_args=[eval('predictions[0]')], function_kwargs={})
prob = tf.nn.sigmoid(predictions[0])
print('This particular pet had a %.1f percent probability of getting adopted.' % (100 * prob))
