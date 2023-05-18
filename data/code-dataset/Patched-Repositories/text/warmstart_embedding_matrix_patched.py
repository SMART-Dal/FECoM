import io
import numpy as np
import os
import re
import shutil
import string
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, Embedding, GlobalAveragePooling1D
from tensorflow.keras.layers import TextVectorization
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
url = 'https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz'
custom_method(imports='import re;import os;import io;import tensorflow as tf;import numpy as np;import shutil;from tensorflow.keras.layers import TextVectorization;from tensorflow.keras import Model;import string;from tensorflow.keras.layers import Dense, Embedding, GlobalAveragePooling1D', function_to_run='tf.keras.utils.get_file(*args, **kwargs)', method_object=None, object_signature=None, function_args=[eval('"aclImdb_v1.tar.gz"'), eval('url')], function_kwargs={'untar': eval('True'), 'cache_dir': eval('"."'), 'cache_subdir': eval('""')})
dataset = tf.keras.utils.get_file('aclImdb_v1.tar.gz', url, untar=True, cache_dir='.', cache_subdir='')
dataset_dir = os.path.join(os.path.dirname(dataset), 'aclImdb')
os.listdir(dataset_dir)
train_dir = os.path.join(dataset_dir, 'train')
os.listdir(train_dir)
remove_dir = os.path.join(train_dir, 'unsup')
shutil.rmtree(remove_dir)
batch_size = 1024
seed = 123
custom_method(imports='import re;import os;import io;import tensorflow as tf;import numpy as np;import shutil;from tensorflow.keras.layers import TextVectorization;from tensorflow.keras import Model;import string;from tensorflow.keras.layers import Dense, Embedding, GlobalAveragePooling1D', function_to_run='tf.keras.utils.text_dataset_from_directory(*args, **kwargs)', method_object=None, object_signature=None, function_args=[eval('"aclImdb/train"')], function_kwargs={'batch_size': eval('batch_size'), 'validation_split': eval('0.2'), 'subset': eval('"training"'), 'seed': eval('seed')})
train_ds = tf.keras.utils.text_dataset_from_directory('aclImdb/train', batch_size=batch_size, validation_split=0.2, subset='training', seed=seed)
custom_method(imports='import re;import os;import io;import tensorflow as tf;import numpy as np;import shutil;from tensorflow.keras.layers import TextVectorization;from tensorflow.keras import Model;import string;from tensorflow.keras.layers import Dense, Embedding, GlobalAveragePooling1D', function_to_run='tf.keras.utils.text_dataset_from_directory(*args, **kwargs)', method_object=None, object_signature=None, function_args=[eval('"aclImdb/train"')], function_kwargs={'batch_size': eval('batch_size'), 'validation_split': eval('0.2'), 'subset': eval('"validation"'), 'seed': eval('seed')})
val_ds = tf.keras.utils.text_dataset_from_directory('aclImdb/train', batch_size=batch_size, validation_split=0.2, subset='validation', seed=seed)
AUTOTUNE = tf.data.AUTOTUNE
custom_method(imports='import re;import os;import io;import tensorflow as tf;import numpy as np;import shutil;from tensorflow.keras.layers import TextVectorization;from tensorflow.keras import Model;import string;from tensorflow.keras.layers import Dense, Embedding, GlobalAveragePooling1D', function_to_run='obj.cache().prefetch(**kwargs)', method_object=eval('train_ds'), object_signature=None, function_args=[], function_kwargs={'buffer_size': eval('AUTOTUNE')}, custom_class=None)
train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
custom_method(imports='import re;import os;import io;import tensorflow as tf;import numpy as np;import shutil;from tensorflow.keras.layers import TextVectorization;from tensorflow.keras import Model;import string;from tensorflow.keras.layers import Dense, Embedding, GlobalAveragePooling1D', function_to_run='obj.cache().prefetch(**kwargs)', method_object=eval('val_ds'), object_signature=None, function_args=[], function_kwargs={'buffer_size': eval('AUTOTUNE')}, custom_class=None)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

def custom_standardization(input_data):
    custom_method(imports='import re;import os;import io;import tensorflow as tf;import numpy as np;import shutil;from tensorflow.keras.layers import TextVectorization;from tensorflow.keras import Model;import string;from tensorflow.keras.layers import Dense, Embedding, GlobalAveragePooling1D', function_to_run='tf.strings.lower(*args)', method_object=None, object_signature=None, function_args=[eval('input_data')], function_kwargs={})
    lowercase = tf.strings.lower(input_data)
    custom_method(imports='import re;import os;import io;import tensorflow as tf;import numpy as np;import shutil;from tensorflow.keras.layers import TextVectorization;from tensorflow.keras import Model;import string;from tensorflow.keras.layers import Dense, Embedding, GlobalAveragePooling1D', function_to_run='tf.strings.regex_replace(*args)', method_object=None, object_signature=None, function_args=[eval('lowercase'), eval('"<br />"'), eval('" "')], function_kwargs={})
    stripped_html = tf.strings.regex_replace(lowercase, '<br />', ' ')
    return tf.strings.regex_replace(stripped_html, '[%s]' % re.escape(string.punctuation), '')
vocab_size = 10000
sequence_length = 100
custom_method(imports='import re;import os;import io;import tensorflow as tf;import numpy as np;import shutil;from tensorflow.keras.layers import TextVectorization;from tensorflow.keras import Model;import string;from tensorflow.keras.layers import Dense, Embedding, GlobalAveragePooling1D', function_to_run='TextVectorization(**kwargs)', method_object=None, object_signature=None, function_args=[], function_kwargs={'standardize': eval('custom_standardization'), 'max_tokens': eval('vocab_size'), 'output_mode': eval('"int"'), 'output_sequence_length': eval('sequence_length')})
vectorize_layer = TextVectorization(standardize=custom_standardization, max_tokens=vocab_size, output_mode='int', output_sequence_length=sequence_length)
custom_method(imports='import re;import os;import io;import tensorflow as tf;import numpy as np;import shutil;from tensorflow.keras.layers import TextVectorization;from tensorflow.keras import Model;import string;from tensorflow.keras.layers import Dense, Embedding, GlobalAveragePooling1D', function_to_run='obj.map(*args)', method_object=eval('train_ds'), object_signature=None, function_args=[eval('lambda x, y: x')], function_kwargs={}, custom_class=None)
text_ds = train_ds.map(lambda x, y: x)
custom_method(imports='import re;import os;import io;import tensorflow as tf;import numpy as np;import shutil;from tensorflow.keras.layers import TextVectorization;from tensorflow.keras import Model;import string;from tensorflow.keras.layers import Dense, Embedding, GlobalAveragePooling1D', function_to_run='obj.adapt(*args)', method_object=eval('vectorize_layer'), object_signature=None, function_args=[eval('text_ds')], function_kwargs={}, custom_class=None)
vectorize_layer.adapt(text_ds)
embedding_dim = 16
custom_method(imports='import re;import os;import io;import tensorflow as tf;import numpy as np;import shutil;from tensorflow.keras.layers import TextVectorization;from tensorflow.keras import Model;import string;from tensorflow.keras.layers import Dense, Embedding, GlobalAveragePooling1D', function_to_run='Embedding(*args, **kwargs)', method_object=None, object_signature=None, function_args=[eval('vocab_size'), eval('embedding_dim')], function_kwargs={'name': eval('"embedding"')})
text_embedding = Embedding(vocab_size, embedding_dim, name='embedding')
custom_method(imports='import re;import os;import io;import tensorflow as tf;import numpy as np;import shutil;from tensorflow.keras.layers import TextVectorization;from tensorflow.keras import Model;import string;from tensorflow.keras.layers import Dense, Embedding, GlobalAveragePooling1D', function_to_run='tf.keras.Sequential(*args, **kwargs)', method_object=None, object_signature=None, function_args=[eval('[vectorize_layer, text_embedding]')], function_kwargs={'name': eval('"text_input"')})
text_input = tf.keras.Sequential([vectorize_layer, text_embedding], name='text_input')
custom_method(imports='import re;import os;import io;import tensorflow as tf;import numpy as np;import shutil;from tensorflow.keras.layers import TextVectorization;from tensorflow.keras import Model;import string;from tensorflow.keras.layers import Dense, Embedding, GlobalAveragePooling1D', function_to_run='tf.keras.Sequential(*args, **kwargs)', method_object=None, object_signature=None, function_args=[eval('[GlobalAveragePooling1D(), Dense(16, activation="relu"), Dense(1)]')], function_kwargs={'name': eval('"classifier_head"')})
classifier_head = tf.keras.Sequential([GlobalAveragePooling1D(), Dense(16, activation='relu'), Dense(1)], name='classifier_head')
custom_method(imports='import re;import os;import io;import tensorflow as tf;import numpy as np;import shutil;from tensorflow.keras.layers import TextVectorization;from tensorflow.keras import Model;import string;from tensorflow.keras.layers import Dense, Embedding, GlobalAveragePooling1D', function_to_run='tf.keras.Sequential(*args)', method_object=None, object_signature=None, function_args=[eval('[text_input, classifier_head]')], function_kwargs={})
model = tf.keras.Sequential([text_input, classifier_head])
custom_method(imports='import re;import os;import io;import tensorflow as tf;import numpy as np;import shutil;from tensorflow.keras.layers import TextVectorization;from tensorflow.keras import Model;import string;from tensorflow.keras.layers import Dense, Embedding, GlobalAveragePooling1D', function_to_run='tf.keras.callbacks.TensorBoard(**kwargs)', method_object=None, object_signature=None, function_args=[], function_kwargs={'log_dir': eval('"logs"')})
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir='logs')
custom_method(imports='import re;import os;import io;import tensorflow as tf;import numpy as np;import shutil;from tensorflow.keras.layers import TextVectorization;from tensorflow.keras import Model;import string;from tensorflow.keras.layers import Dense, Embedding, GlobalAveragePooling1D', function_to_run='obj.compile(**kwargs)', method_object=eval('model'), object_signature=None, function_args=[], function_kwargs={'optimizer': eval('"adam"'), 'loss': eval('tf.keras.losses.BinaryCrossentropy(from_logits=True)'), 'metrics': eval('["accuracy"]')}, custom_class=None)
model.compile(optimizer='adam', loss=tf.keras.losses.BinaryCrossentropy(from_logits=True), metrics=['accuracy'])
custom_method(imports='import re;import os;import io;import tensorflow as tf;import numpy as np;import shutil;from tensorflow.keras.layers import TextVectorization;from tensorflow.keras import Model;import string;from tensorflow.keras.layers import Dense, Embedding, GlobalAveragePooling1D', function_to_run='obj.fit(*args, **kwargs)', method_object=eval('model'), object_signature=None, function_args=[eval('train_ds')], function_kwargs={'validation_data': eval('val_ds'), 'epochs': eval('15'), 'callbacks': eval('[tensorboard_callback]')}, custom_class=None)
model.fit(train_ds, validation_data=val_ds, epochs=15, callbacks=[tensorboard_callback])
custom_method(imports='import re;import os;import io;import tensorflow as tf;import numpy as np;import shutil;from tensorflow.keras.layers import TextVectorization;from tensorflow.keras import Model;import string;from tensorflow.keras.layers import Dense, Embedding, GlobalAveragePooling1D', function_to_run='obj.summary()', method_object=eval('model'), object_signature=None, function_args=[], function_kwargs={}, custom_class=None)
model.summary()
embedding_weights_base = model.get_layer('text_input').get_layer('embedding').get_weights()[0]
custom_method(imports='import re;import os;import io;import tensorflow as tf;import numpy as np;import shutil;from tensorflow.keras.layers import TextVectorization;from tensorflow.keras import Model;import string;from tensorflow.keras.layers import Dense, Embedding, GlobalAveragePooling1D', function_to_run='obj.get_vocabulary()', method_object=eval('vectorize_layer'), object_signature=None, function_args=[], function_kwargs={}, custom_class=None)
vocab_base = vectorize_layer.get_vocabulary()
vocab_size_new = 10200
sequence_length = 100
custom_method(imports='import re;import os;import io;import tensorflow as tf;import numpy as np;import shutil;from tensorflow.keras.layers import TextVectorization;from tensorflow.keras import Model;import string;from tensorflow.keras.layers import Dense, Embedding, GlobalAveragePooling1D', function_to_run='TextVectorization(**kwargs)', method_object=None, object_signature=None, function_args=[], function_kwargs={'standardize': eval('custom_standardization'), 'max_tokens': eval('vocab_size_new'), 'output_mode': eval('"int"'), 'output_sequence_length': eval('sequence_length')})
vectorize_layer_new = TextVectorization(standardize=custom_standardization, max_tokens=vocab_size_new, output_mode='int', output_sequence_length=sequence_length)
custom_method(imports='import re;import os;import io;import tensorflow as tf;import numpy as np;import shutil;from tensorflow.keras.layers import TextVectorization;from tensorflow.keras import Model;import string;from tensorflow.keras.layers import Dense, Embedding, GlobalAveragePooling1D', function_to_run='obj.map(*args)', method_object=eval('train_ds'), object_signature=None, function_args=[eval('lambda x, y: x')], function_kwargs={}, custom_class=None)
text_ds = train_ds.map(lambda x, y: x)
custom_method(imports='import re;import os;import io;import tensorflow as tf;import numpy as np;import shutil;from tensorflow.keras.layers import TextVectorization;from tensorflow.keras import Model;import string;from tensorflow.keras.layers import Dense, Embedding, GlobalAveragePooling1D', function_to_run='obj.adapt(*args)', method_object=eval('vectorize_layer_new'), object_signature=None, function_args=[eval('text_ds')], function_kwargs={}, custom_class=None)
vectorize_layer_new.adapt(text_ds)
custom_method(imports='import re;import os;import io;import tensorflow as tf;import numpy as np;import shutil;from tensorflow.keras.layers import TextVectorization;from tensorflow.keras import Model;import string;from tensorflow.keras.layers import Dense, Embedding, GlobalAveragePooling1D', function_to_run='obj.get_vocabulary()', method_object=eval('vectorize_layer_new'), object_signature=None, function_args=[], function_kwargs={}, custom_class=None)
vocab_new = vectorize_layer_new.get_vocabulary()
set(vocab_base) ^ set(vocab_new)
custom_method(imports='import re;import os;import io;import tensorflow as tf;import numpy as np;import shutil;from tensorflow.keras.layers import TextVectorization;from tensorflow.keras import Model;import string;from tensorflow.keras.layers import Dense, Embedding, GlobalAveragePooling1D', function_to_run='tf.keras.utils.warmstart_embedding_matrix(**kwargs)', method_object=None, object_signature=None, function_args=[], function_kwargs={'base_vocabulary': eval('vocab_base'), 'new_vocabulary': eval('vocab_new'), 'base_embeddings': eval('embedding_weights_base'), 'new_embeddings_initializer': eval('"uniform"')})
updated_embedding = tf.keras.utils.warmstart_embedding_matrix(base_vocabulary=vocab_base, new_vocabulary=vocab_new, base_embeddings=embedding_weights_base, new_embeddings_initializer='uniform')
custom_method(imports='import re;import os;import io;import tensorflow as tf;import numpy as np;import shutil;from tensorflow.keras.layers import TextVectorization;from tensorflow.keras import Model;import string;from tensorflow.keras.layers import Dense, Embedding, GlobalAveragePooling1D', function_to_run='tf.Variable(*args)', method_object=None, object_signature=None, function_args=[eval('updated_embedding')], function_kwargs={})
updated_embedding_variable = tf.Variable(updated_embedding)
updated_embedding_variable.shape
custom_method(imports='import re;import os;import io;import tensorflow as tf;import numpy as np;import shutil;from tensorflow.keras.layers import TextVectorization;from tensorflow.keras import Model;import string;from tensorflow.keras.layers import Dense, Embedding, GlobalAveragePooling1D', function_to_run='Embedding(*args, **kwargs)', method_object=None, object_signature=None, function_args=[eval('vectorize_layer_new.vocabulary_size()'), eval('embedding_dim')], function_kwargs={'name': eval('"embedding"')})
text_embedding_layer_new = Embedding(vectorize_layer_new.vocabulary_size(), embedding_dim, name='embedding')
custom_method(imports='import re;import os;import io;import tensorflow as tf;import numpy as np;import shutil;from tensorflow.keras.layers import TextVectorization;from tensorflow.keras import Model;import string;from tensorflow.keras.layers import Dense, Embedding, GlobalAveragePooling1D', function_to_run='obj.build(**kwargs)', method_object=eval('text_embedding_layer_new'), object_signature=None, function_args=[], function_kwargs={'input_shape': eval('[None]')}, custom_class=None)
text_embedding_layer_new.build(input_shape=[None])
custom_method(imports='import re;import os;import io;import tensorflow as tf;import numpy as np;import shutil;from tensorflow.keras.layers import TextVectorization;from tensorflow.keras import Model;import string;from tensorflow.keras.layers import Dense, Embedding, GlobalAveragePooling1D', function_to_run='obj.embeddings.assign(*args)', method_object=eval('text_embedding_layer_new'), object_signature=None, function_args=[eval('updated_embedding')], function_kwargs={}, custom_class=None)
text_embedding_layer_new.embeddings.assign(updated_embedding)
custom_method(imports='import re;import os;import io;import tensorflow as tf;import numpy as np;import shutil;from tensorflow.keras.layers import TextVectorization;from tensorflow.keras import Model;import string;from tensorflow.keras.layers import Dense, Embedding, GlobalAveragePooling1D', function_to_run='tf.keras.Sequential(*args, **kwargs)', method_object=None, object_signature=None, function_args=[eval('[vectorize_layer_new, text_embedding_layer_new]')], function_kwargs={'name': eval('"text_input_new"')})
text_input_new = tf.keras.Sequential([vectorize_layer_new, text_embedding_layer_new], name='text_input_new')
custom_method(imports='import re;import os;import io;import tensorflow as tf;import numpy as np;import shutil;from tensorflow.keras.layers import TextVectorization;from tensorflow.keras import Model;import string;from tensorflow.keras.layers import Dense, Embedding, GlobalAveragePooling1D', function_to_run='obj.summary()', method_object=eval('text_input_new'), object_signature=None, function_args=[], function_kwargs={}, custom_class=None)
text_input_new.summary()
text_input_new.get_layer('embedding').get_weights()[0].shape
custom_method(imports='import re;import os;import io;import tensorflow as tf;import numpy as np;import shutil;from tensorflow.keras.layers import TextVectorization;from tensorflow.keras import Model;import string;from tensorflow.keras.layers import Dense, Embedding, GlobalAveragePooling1D', function_to_run='tf.keras.Sequential(*args)', method_object=None, object_signature=None, function_args=[eval('[text_input_new, classifier_head]')], function_kwargs={})
warm_started_model = tf.keras.Sequential([text_input_new, classifier_head])
custom_method(imports='import re;import os;import io;import tensorflow as tf;import numpy as np;import shutil;from tensorflow.keras.layers import TextVectorization;from tensorflow.keras import Model;import string;from tensorflow.keras.layers import Dense, Embedding, GlobalAveragePooling1D', function_to_run='obj.summary()', method_object=eval('warm_started_model'), object_signature=None, function_args=[], function_kwargs={}, custom_class=None)
warm_started_model.summary()
base_vocab_index = vectorize_layer('the')[0]
new_vocab_index = vectorize_layer_new('the')[0]
print(warm_started_model.get_layer('text_input_new').get_layer('embedding')(new_vocab_index) == embedding_weights_base[base_vocab_index])
custom_method(imports='import re;import os;import io;import tensorflow as tf;import numpy as np;import shutil;from tensorflow.keras.layers import TextVectorization;from tensorflow.keras import Model;import string;from tensorflow.keras.layers import Dense, Embedding, GlobalAveragePooling1D', function_to_run='obj.compile(**kwargs)', method_object=eval('model'), object_signature=None, function_args=[], function_kwargs={'optimizer': eval('"adam"'), 'loss': eval('tf.keras.losses.BinaryCrossentropy(from_logits=True)'), 'metrics': eval('["accuracy"]')}, custom_class=None)
model.compile(optimizer='adam', loss=tf.keras.losses.BinaryCrossentropy(from_logits=True), metrics=['accuracy'])
custom_method(imports='import re;import os;import io;import tensorflow as tf;import numpy as np;import shutil;from tensorflow.keras.layers import TextVectorization;from tensorflow.keras import Model;import string;from tensorflow.keras.layers import Dense, Embedding, GlobalAveragePooling1D', function_to_run='obj.fit(*args, **kwargs)', method_object=eval('model'), object_signature=None, function_args=[eval('train_ds')], function_kwargs={'validation_data': eval('val_ds'), 'epochs': eval('15'), 'callbacks': eval('[tensorboard_callback]')}, custom_class=None)
model.fit(train_ds, validation_data=val_ds, epochs=15, callbacks=[tensorboard_callback])
