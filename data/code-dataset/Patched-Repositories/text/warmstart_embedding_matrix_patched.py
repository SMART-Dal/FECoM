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
import sys
from tool.client.client_config import EXPERIMENT_DIR
from tool.server.local_execution import before_execution as before_execution_INSERTED_INTO_SCRIPT
from tool.server.local_execution import after_execution as after_execution_INSERTED_INTO_SCRIPT
experiment_number = sys.argv[1]
experiment_project = sys.argv[2]
EXPERIMENT_FILE_PATH = EXPERIMENT_DIR / 'method-level' / experiment_project / f'experiment-{experiment_number}.json'
url = 'https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz'
start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT()
dataset = tf.keras.utils.get_file('aclImdb_v1.tar.gz', url, untar=True, cache_dir='.', cache_subdir='')
after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tf.keras.utils.get_file(*args, **kwargs)', method_object=None, object_signature=None, function_args=['aclImdb_v1.tar.gz', url], function_kwargs={'untar': True, 'cache_dir': '.', 'cache_subdir': ''})
dataset_dir = os.path.join(os.path.dirname(dataset), 'aclImdb')
os.listdir(dataset_dir)
train_dir = os.path.join(dataset_dir, 'train')
os.listdir(train_dir)
remove_dir = os.path.join(train_dir, 'unsup')
shutil.rmtree(remove_dir)
batch_size = 1024
seed = 123
start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT()
train_ds = tf.keras.utils.text_dataset_from_directory('aclImdb/train', batch_size=batch_size, validation_split=0.2, subset='training', seed=seed)
after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tf.keras.utils.text_dataset_from_directory(*args, **kwargs)', method_object=None, object_signature=None, function_args=['aclImdb/train'], function_kwargs={'batch_size': batch_size, 'validation_split': 0.2, 'subset': 'training', 'seed': seed})
start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT()
val_ds = tf.keras.utils.text_dataset_from_directory('aclImdb/train', batch_size=batch_size, validation_split=0.2, subset='validation', seed=seed)
after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tf.keras.utils.text_dataset_from_directory(*args, **kwargs)', method_object=None, object_signature=None, function_args=['aclImdb/train'], function_kwargs={'batch_size': batch_size, 'validation_split': 0.2, 'subset': 'validation', 'seed': seed})
AUTOTUNE = tf.data.AUTOTUNE
start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT()
train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='obj.cache().prefetch(**kwargs)', method_object=train_ds, object_signature=None, function_args=[], function_kwargs={'buffer_size': AUTOTUNE})
start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT()
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)
after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='obj.cache().prefetch(**kwargs)', method_object=val_ds, object_signature=None, function_args=[], function_kwargs={'buffer_size': AUTOTUNE})

def custom_standardization(input_data):
    start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT()
    lowercase = tf.strings.lower(input_data)
    after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tf.strings.lower(*args)', method_object=None, object_signature=None, function_args=[input_data], function_kwargs={})
    start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT()
    stripped_html = tf.strings.regex_replace(lowercase, '<br />', ' ')
    after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tf.strings.regex_replace(*args)', method_object=None, object_signature=None, function_args=[lowercase, '<br />', ' '], function_kwargs={})
    return tf.strings.regex_replace(stripped_html, '[%s]' % re.escape(string.punctuation), '')
vocab_size = 10000
sequence_length = 100
start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT()
vectorize_layer = TextVectorization(standardize=custom_standardization, max_tokens=vocab_size, output_mode='int', output_sequence_length=sequence_length)
after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='TextVectorization(**kwargs)', method_object=None, object_signature=None, function_args=[], function_kwargs={'standardize': custom_standardization, 'max_tokens': vocab_size, 'output_mode': 'int', 'output_sequence_length': sequence_length})
start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT()
text_ds = train_ds.map(lambda x, y: x)
after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='obj.map(*args)', method_object=train_ds, object_signature=None, function_args=[lambda x, y: x], function_kwargs={})
start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT()
vectorize_layer.adapt(text_ds)
after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='obj.adapt(*args)', method_object=vectorize_layer, object_signature=None, function_args=[text_ds], function_kwargs={})
embedding_dim = 16
start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT()
text_embedding = Embedding(vocab_size, embedding_dim, name='embedding')
after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='Embedding(*args, **kwargs)', method_object=None, object_signature=None, function_args=[vocab_size, embedding_dim], function_kwargs={'name': 'embedding'})
start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT()
text_input = tf.keras.Sequential([vectorize_layer, text_embedding], name='text_input')
after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tf.keras.Sequential(*args, **kwargs)', method_object=None, object_signature=None, function_args=[[vectorize_layer, text_embedding]], function_kwargs={'name': 'text_input'})
start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT()
classifier_head = tf.keras.Sequential([GlobalAveragePooling1D(), Dense(16, activation='relu'), Dense(1)], name='classifier_head')
after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tf.keras.Sequential(*args, **kwargs)', method_object=None, object_signature=None, function_args=[[GlobalAveragePooling1D(), Dense(16, activation='relu'), Dense(1)]], function_kwargs={'name': 'classifier_head'})
start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT()
model = tf.keras.Sequential([text_input, classifier_head])
after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tf.keras.Sequential(*args)', method_object=None, object_signature=None, function_args=[[text_input, classifier_head]], function_kwargs={})
start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT()
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir='logs')
after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tf.keras.callbacks.TensorBoard(**kwargs)', method_object=None, object_signature=None, function_args=[], function_kwargs={'log_dir': 'logs'})
start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT()
model.compile(optimizer='adam', loss=tf.keras.losses.BinaryCrossentropy(from_logits=True), metrics=['accuracy'])
after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='obj.compile(**kwargs)', method_object=model, object_signature=None, function_args=[], function_kwargs={'optimizer': 'adam', 'loss': tf.keras.losses.BinaryCrossentropy(from_logits=True), 'metrics': ['accuracy']})
start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT()
model.fit(train_ds, validation_data=val_ds, epochs=15, callbacks=[tensorboard_callback])
after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='obj.fit(*args, **kwargs)', method_object=model, object_signature=None, function_args=[train_ds], function_kwargs={'validation_data': val_ds, 'epochs': 15, 'callbacks': [tensorboard_callback]})
start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT()
model.summary()
after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='obj.summary()', method_object=model, object_signature=None, function_args=[], function_kwargs={})
embedding_weights_base = model.get_layer('text_input').get_layer('embedding').get_weights()[0]
start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT()
vocab_base = vectorize_layer.get_vocabulary()
after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='obj.get_vocabulary()', method_object=vectorize_layer, object_signature=None, function_args=[], function_kwargs={})
vocab_size_new = 10200
sequence_length = 100
start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT()
vectorize_layer_new = TextVectorization(standardize=custom_standardization, max_tokens=vocab_size_new, output_mode='int', output_sequence_length=sequence_length)
after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='TextVectorization(**kwargs)', method_object=None, object_signature=None, function_args=[], function_kwargs={'standardize': custom_standardization, 'max_tokens': vocab_size_new, 'output_mode': 'int', 'output_sequence_length': sequence_length})
start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT()
text_ds = train_ds.map(lambda x, y: x)
after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='obj.map(*args)', method_object=train_ds, object_signature=None, function_args=[lambda x, y: x], function_kwargs={})
start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT()
vectorize_layer_new.adapt(text_ds)
after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='obj.adapt(*args)', method_object=vectorize_layer_new, object_signature=None, function_args=[text_ds], function_kwargs={})
start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT()
vocab_new = vectorize_layer_new.get_vocabulary()
after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='obj.get_vocabulary()', method_object=vectorize_layer_new, object_signature=None, function_args=[], function_kwargs={})
set(vocab_base) ^ set(vocab_new)
start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT()
updated_embedding = tf.keras.utils.warmstart_embedding_matrix(base_vocabulary=vocab_base, new_vocabulary=vocab_new, base_embeddings=embedding_weights_base, new_embeddings_initializer='uniform')
after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tf.keras.utils.warmstart_embedding_matrix(**kwargs)', method_object=None, object_signature=None, function_args=[], function_kwargs={'base_vocabulary': vocab_base, 'new_vocabulary': vocab_new, 'base_embeddings': embedding_weights_base, 'new_embeddings_initializer': 'uniform'})
start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT()
updated_embedding_variable = tf.Variable(updated_embedding)
after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tf.Variable(*args)', method_object=None, object_signature=None, function_args=[updated_embedding], function_kwargs={})
updated_embedding_variable.shape
start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT()
text_embedding_layer_new = Embedding(vectorize_layer_new.vocabulary_size(), embedding_dim, name='embedding')
after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='Embedding(*args, **kwargs)', method_object=None, object_signature=None, function_args=[vectorize_layer_new.vocabulary_size(), embedding_dim], function_kwargs={'name': 'embedding'})
start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT()
text_embedding_layer_new.build(input_shape=[None])
after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='obj.build(**kwargs)', method_object=text_embedding_layer_new, object_signature=None, function_args=[], function_kwargs={'input_shape': [None]})
start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT()
text_embedding_layer_new.embeddings.assign(updated_embedding)
after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='obj.embeddings.assign(*args)', method_object=text_embedding_layer_new, object_signature=None, function_args=[updated_embedding], function_kwargs={})
start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT()
text_input_new = tf.keras.Sequential([vectorize_layer_new, text_embedding_layer_new], name='text_input_new')
after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tf.keras.Sequential(*args, **kwargs)', method_object=None, object_signature=None, function_args=[[vectorize_layer_new, text_embedding_layer_new]], function_kwargs={'name': 'text_input_new'})
start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT()
text_input_new.summary()
after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='obj.summary()', method_object=text_input_new, object_signature=None, function_args=[], function_kwargs={})
text_input_new.get_layer('embedding').get_weights()[0].shape
start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT()
warm_started_model = tf.keras.Sequential([text_input_new, classifier_head])
after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tf.keras.Sequential(*args)', method_object=None, object_signature=None, function_args=[[text_input_new, classifier_head]], function_kwargs={})
start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT()
warm_started_model.summary()
after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='obj.summary()', method_object=warm_started_model, object_signature=None, function_args=[], function_kwargs={})
base_vocab_index = vectorize_layer('the')[0]
new_vocab_index = vectorize_layer_new('the')[0]
print(warm_started_model.get_layer('text_input_new').get_layer('embedding')(new_vocab_index) == embedding_weights_base[base_vocab_index])
start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT()
model.compile(optimizer='adam', loss=tf.keras.losses.BinaryCrossentropy(from_logits=True), metrics=['accuracy'])
after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='obj.compile(**kwargs)', method_object=model, object_signature=None, function_args=[], function_kwargs={'optimizer': 'adam', 'loss': tf.keras.losses.BinaryCrossentropy(from_logits=True), 'metrics': ['accuracy']})
start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT()
model.fit(train_ds, validation_data=val_ds, epochs=15, callbacks=[tensorboard_callback])
after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='obj.fit(*args, **kwargs)', method_object=model, object_signature=None, function_args=[train_ds], function_kwargs={'validation_data': val_ds, 'epochs': 15, 'callbacks': [tensorboard_callback]})
