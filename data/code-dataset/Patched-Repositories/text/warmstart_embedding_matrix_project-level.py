import sys
from tool.patching.patching_config import EXPERIMENT_DIR
from tool.measurement.execution import before_execution as before_execution_INSERTED_INTO_SCRIPT
from tool.measurement.execution import after_execution as after_execution_INSERTED_INTO_SCRIPT
from tool.experiment.experiment_kinds import ExperimentKinds
experiment_number = sys.argv[1]
experiment_project = sys.argv[2]
EXPERIMENT_FILE_PATH = EXPERIMENT_DIR / ExperimentKinds.PROJECT_LEVEL.value / experiment_project / f'experiment-{experiment_number}.json'
start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT(enable_skip_calls=False)
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
url = 'https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz'
dataset = tf.keras.utils.get_file('aclImdb_v1.tar.gz', url, untar=True, cache_dir='.', cache_subdir='')
dataset_dir = os.path.join(os.path.dirname(dataset), 'aclImdb')
os.listdir(dataset_dir)
train_dir = os.path.join(dataset_dir, 'train')
os.listdir(train_dir)
remove_dir = os.path.join(train_dir, 'unsup')
shutil.rmtree(remove_dir)
batch_size = 1024
seed = 123
train_ds = tf.keras.utils.text_dataset_from_directory('aclImdb/train', batch_size=batch_size, validation_split=0.2, subset='training', seed=seed)
val_ds = tf.keras.utils.text_dataset_from_directory('aclImdb/train', batch_size=batch_size, validation_split=0.2, subset='validation', seed=seed)
AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

def custom_standardization(input_data):
    lowercase = tf.strings.lower(input_data)
    stripped_html = tf.strings.regex_replace(lowercase, '<br />', ' ')
    return tf.strings.regex_replace(stripped_html, '[%s]' % re.escape(string.punctuation), '')
vocab_size = 10000
sequence_length = 100
vectorize_layer = TextVectorization(standardize=custom_standardization, max_tokens=vocab_size, output_mode='int', output_sequence_length=sequence_length)
text_ds = train_ds.map(lambda x, y: x)
vectorize_layer.adapt(text_ds)
embedding_dim = 16
text_embedding = Embedding(vocab_size, embedding_dim, name='embedding')
text_input = tf.keras.Sequential([vectorize_layer, text_embedding], name='text_input')
classifier_head = tf.keras.Sequential([GlobalAveragePooling1D(), Dense(16, activation='relu'), Dense(1)], name='classifier_head')
model = tf.keras.Sequential([text_input, classifier_head])
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir='logs')
model.compile(optimizer='adam', loss=tf.keras.losses.BinaryCrossentropy(from_logits=True), metrics=['accuracy'])
model.fit(train_ds, validation_data=val_ds, epochs=15, callbacks=[tensorboard_callback])
model.summary()
embedding_weights_base = model.get_layer('text_input').get_layer('embedding').get_weights()[0]
vocab_base = vectorize_layer.get_vocabulary()
vocab_size_new = 10200
sequence_length = 100
vectorize_layer_new = TextVectorization(standardize=custom_standardization, max_tokens=vocab_size_new, output_mode='int', output_sequence_length=sequence_length)
text_ds = train_ds.map(lambda x, y: x)
vectorize_layer_new.adapt(text_ds)
vocab_new = vectorize_layer_new.get_vocabulary()
set(vocab_base) ^ set(vocab_new)
updated_embedding = tf.keras.utils.warmstart_embedding_matrix(base_vocabulary=vocab_base, new_vocabulary=vocab_new, base_embeddings=embedding_weights_base, new_embeddings_initializer='uniform')
updated_embedding_variable = tf.Variable(updated_embedding)
updated_embedding_variable.shape
text_embedding_layer_new = Embedding(vectorize_layer_new.vocabulary_size(), embedding_dim, name='embedding')
text_embedding_layer_new.build(input_shape=[None])
text_embedding_layer_new.embeddings.assign(updated_embedding)
text_input_new = tf.keras.Sequential([vectorize_layer_new, text_embedding_layer_new], name='text_input_new')
text_input_new.summary()
text_input_new.get_layer('embedding').get_weights()[0].shape
warm_started_model = tf.keras.Sequential([text_input_new, classifier_head])
warm_started_model.summary()
base_vocab_index = vectorize_layer('the')[0]
new_vocab_index = vectorize_layer_new('the')[0]
print(warm_started_model.get_layer('text_input_new').get_layer('embedding')(new_vocab_index) == embedding_weights_base[base_vocab_index])
model.compile(optimizer='adam', loss=tf.keras.losses.BinaryCrossentropy(from_logits=True), metrics=['accuracy'])
model.fit(train_ds, validation_data=val_ds, epochs=15, callbacks=[tensorboard_callback])
after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, enable_skip_calls=False)
