import collections
import pathlib
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import losses
from tensorflow.keras import utils
from tensorflow.keras.layers import TextVectorization
import tensorflow_datasets as tfds
import tensorflow_text as tf_text
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

def custom_method(func, imports: str, function_to_run: str, method_object=None, object_signature=None, function_args: list=None, function_kwargs: dict=None, max_wait_secs=MAX_WAIT_S, custom_class=None, wait_after_run_secs=WAIT_AFTER_RUN_S):
    result = send_request(imports=imports, function_to_run=function_to_run, function_args=function_args, function_kwargs=function_kwargs, max_wait_secs=max_wait_secs, wait_after_run_secs=wait_after_run_secs, method_object=method_object, object_signature=object_signature, custom_class=custom_class, experiment_file_path=EXPERIMENT_FILE_PATH)
    return func
data_url = 'https://storage.googleapis.com/download.tensorflow.org/data/stack_overflow_16k.tar.gz'
dataset_dir = custom_method(
utils.get_file(origin=data_url, untar=True, cache_dir='stack_overflow', cache_subdir=''), imports='from tensorflow.keras import utils;import tensorflow_datasets as tfds;import pathlib;import collections;import tensorflow_text as tf_text;from tensorflow.keras import layers;from tensorflow.keras import losses;from tensorflow.keras.layers import TextVectorization;import tensorflow as tf', function_to_run='utils.get_file(**kwargs)', method_object=None, object_signature=None, function_args=[], function_kwargs={'origin': eval('data_url'), 'untar': eval('True'), 'cache_dir': eval("'stack_overflow'"), 'cache_subdir': eval("''")}, max_wait_secs=0)
dataset_dir = pathlib.Path(dataset_dir).parent
list(dataset_dir.iterdir())
train_dir = dataset_dir / 'train'
list(train_dir.iterdir())
sample_file = train_dir / 'python/1755.txt'
with open(sample_file) as f:
    print(f.read())
batch_size = 32
seed = 42
raw_train_ds = custom_method(
utils.text_dataset_from_directory(train_dir, batch_size=batch_size, validation_split=0.2, subset='training', seed=seed), imports='from tensorflow.keras import utils;import tensorflow_datasets as tfds;import pathlib;import collections;import tensorflow_text as tf_text;from tensorflow.keras import layers;from tensorflow.keras import losses;from tensorflow.keras.layers import TextVectorization;import tensorflow as tf', function_to_run='utils.text_dataset_from_directory(*args, **kwargs)', method_object=None, object_signature=None, function_args=[eval('train_dir')], function_kwargs={'batch_size': eval('batch_size'), 'validation_split': eval('0.2'), 'subset': eval("'training'"), 'seed': eval('seed')}, max_wait_secs=0)
for (text_batch, label_batch) in custom_method(
raw_train_ds.take(1), imports='from tensorflow.keras import utils;import tensorflow_datasets as tfds;import pathlib;import collections;import tensorflow_text as tf_text;from tensorflow.keras import layers;from tensorflow.keras import losses;from tensorflow.keras.layers import TextVectorization;import tensorflow as tf', function_to_run='obj.take(*args)', method_object=eval('raw_train_ds'), object_signature='utils.text_dataset_from_directory', function_args=[eval('1')], function_kwargs={}, max_wait_secs=0, custom_class=None):
    for i in range(10):
        print('Question: ', text_batch.numpy()[i])
        print('Label:', label_batch.numpy()[i])
for (i, label) in enumerate(raw_train_ds.class_names):
    print('Label', i, 'corresponds to', label)
raw_val_ds = custom_method(
utils.text_dataset_from_directory(train_dir, batch_size=batch_size, validation_split=0.2, subset='validation', seed=seed), imports='from tensorflow.keras import utils;import tensorflow_datasets as tfds;import pathlib;import collections;import tensorflow_text as tf_text;from tensorflow.keras import layers;from tensorflow.keras import losses;from tensorflow.keras.layers import TextVectorization;import tensorflow as tf', function_to_run='utils.text_dataset_from_directory(*args, **kwargs)', method_object=None, object_signature=None, function_args=[eval('train_dir')], function_kwargs={'batch_size': eval('batch_size'), 'validation_split': eval('0.2'), 'subset': eval("'validation'"), 'seed': eval('seed')}, max_wait_secs=0)
test_dir = dataset_dir / 'test'
raw_test_ds = custom_method(
utils.text_dataset_from_directory(test_dir, batch_size=batch_size), imports='from tensorflow.keras import utils;import tensorflow_datasets as tfds;import pathlib;import collections;import tensorflow_text as tf_text;from tensorflow.keras import layers;from tensorflow.keras import losses;from tensorflow.keras.layers import TextVectorization;import tensorflow as tf', function_to_run='utils.text_dataset_from_directory(*args, **kwargs)', method_object=None, object_signature=None, function_args=[eval('test_dir')], function_kwargs={'batch_size': eval('batch_size')}, max_wait_secs=0)
VOCAB_SIZE = 10000
binary_vectorize_layer = custom_method(
TextVectorization(max_tokens=VOCAB_SIZE, output_mode='binary'), imports='from tensorflow.keras import utils;import tensorflow_datasets as tfds;import pathlib;import collections;import tensorflow_text as tf_text;from tensorflow.keras import layers;from tensorflow.keras import losses;from tensorflow.keras.layers import TextVectorization;import tensorflow as tf', function_to_run='TextVectorization(**kwargs)', method_object=None, object_signature=None, function_args=[], function_kwargs={'max_tokens': eval('VOCAB_SIZE'), 'output_mode': eval("'binary'")}, max_wait_secs=0)
MAX_SEQUENCE_LENGTH = 250
int_vectorize_layer = custom_method(
TextVectorization(max_tokens=VOCAB_SIZE, output_mode='int', output_sequence_length=MAX_SEQUENCE_LENGTH), imports='from tensorflow.keras import utils;import tensorflow_datasets as tfds;import pathlib;import collections;import tensorflow_text as tf_text;from tensorflow.keras import layers;from tensorflow.keras import losses;from tensorflow.keras.layers import TextVectorization;import tensorflow as tf', function_to_run='TextVectorization(**kwargs)', method_object=None, object_signature=None, function_args=[], function_kwargs={'max_tokens': eval('VOCAB_SIZE'), 'output_mode': eval("'int'"), 'output_sequence_length': eval('MAX_SEQUENCE_LENGTH')}, max_wait_secs=0)
train_text = custom_method(
raw_train_ds.map(lambda text, labels: text), imports='from tensorflow.keras import utils;import tensorflow_datasets as tfds;import pathlib;import collections;import tensorflow_text as tf_text;from tensorflow.keras import layers;from tensorflow.keras import losses;from tensorflow.keras.layers import TextVectorization;import tensorflow as tf', function_to_run='obj.map(*args)', method_object=eval('raw_train_ds'), object_signature='utils.text_dataset_from_directory', function_args=[eval('lambda text, labels: text')], function_kwargs={}, max_wait_secs=0, custom_class=None)
custom_method(
binary_vectorize_layer.adapt(train_text), imports='from tensorflow.keras import utils;import tensorflow_datasets as tfds;import pathlib;import collections;import tensorflow_text as tf_text;from tensorflow.keras import layers;from tensorflow.keras import losses;from tensorflow.keras.layers import TextVectorization;import tensorflow as tf', function_to_run='obj.adapt(*args)', method_object=eval('binary_vectorize_layer'), object_signature='TextVectorization', function_args=[eval('train_text')], function_kwargs={}, max_wait_secs=0, custom_class=None)
custom_method(
int_vectorize_layer.adapt(train_text), imports='from tensorflow.keras import utils;import tensorflow_datasets as tfds;import pathlib;import collections;import tensorflow_text as tf_text;from tensorflow.keras import layers;from tensorflow.keras import losses;from tensorflow.keras.layers import TextVectorization;import tensorflow as tf', function_to_run='obj.adapt(*args)', method_object=eval('int_vectorize_layer'), object_signature='TextVectorization', function_args=[eval('train_text')], function_kwargs={}, max_wait_secs=0, custom_class=None)

def binary_vectorize_text(text, label):
    text = custom_method(
    tf.expand_dims(text, -1), imports='from tensorflow.keras import utils;import tensorflow_datasets as tfds;import pathlib;import collections;import tensorflow_text as tf_text;from tensorflow.keras import layers;from tensorflow.keras import losses;from tensorflow.keras.layers import TextVectorization;import tensorflow as tf', function_to_run='tf.expand_dims(*args)', method_object=None, object_signature=None, function_args=[eval('text'), eval('-1')], function_kwargs={}, max_wait_secs=0)
    return (custom_method(
    binary_vectorize_layer(text), imports='from tensorflow.keras import utils;import tensorflow_datasets as tfds;import pathlib;import collections;import tensorflow_text as tf_text;from tensorflow.keras import layers;from tensorflow.keras import losses;from tensorflow.keras.layers import TextVectorization;import tensorflow as tf', function_to_run='obj(*args)', method_object=eval('binary_vectorize_layer'), object_signature='TextVectorization', function_args=[eval('text')], function_kwargs={}, max_wait_secs=0, custom_class=None), label)

def int_vectorize_text(text, label):
    text = custom_method(
    tf.expand_dims(text, -1), imports='from tensorflow.keras import utils;import tensorflow_datasets as tfds;import pathlib;import collections;import tensorflow_text as tf_text;from tensorflow.keras import layers;from tensorflow.keras import losses;from tensorflow.keras.layers import TextVectorization;import tensorflow as tf', function_to_run='tf.expand_dims(*args)', method_object=None, object_signature=None, function_args=[eval('text'), eval('-1')], function_kwargs={}, max_wait_secs=0)
    return (custom_method(
    int_vectorize_layer(text), imports='from tensorflow.keras import utils;import tensorflow_datasets as tfds;import pathlib;import collections;import tensorflow_text as tf_text;from tensorflow.keras import layers;from tensorflow.keras import losses;from tensorflow.keras.layers import TextVectorization;import tensorflow as tf', function_to_run='obj(*args)', method_object=eval('int_vectorize_layer'), object_signature='TextVectorization', function_args=[eval('text')], function_kwargs={}, max_wait_secs=0, custom_class=None), label)
(text_batch, label_batch) = next(iter(raw_train_ds))
(first_question, first_label) = (text_batch[0], label_batch[0])
print('Question', first_question)
print('Label', first_label)
print("'binary' vectorized question:", binary_vectorize_text(first_question, first_label)[0])
print("'int' vectorized question:", int_vectorize_text(first_question, first_label)[0])
print('1289 ---> ', int_vectorize_layer.get_vocabulary()[1289])
print('313 ---> ', int_vectorize_layer.get_vocabulary()[313])
print('Vocabulary size: {}'.format(len(int_vectorize_layer.get_vocabulary())))
binary_train_ds = custom_method(
raw_train_ds.map(binary_vectorize_text), imports='from tensorflow.keras import utils;import tensorflow_datasets as tfds;import pathlib;import collections;import tensorflow_text as tf_text;from tensorflow.keras import layers;from tensorflow.keras import losses;from tensorflow.keras.layers import TextVectorization;import tensorflow as tf', function_to_run='obj.map(*args)', method_object=eval('raw_train_ds'), object_signature='utils.text_dataset_from_directory', function_args=[eval('binary_vectorize_text')], function_kwargs={}, max_wait_secs=0, custom_class=None)
binary_val_ds = custom_method(
raw_val_ds.map(binary_vectorize_text), imports='from tensorflow.keras import utils;import tensorflow_datasets as tfds;import pathlib;import collections;import tensorflow_text as tf_text;from tensorflow.keras import layers;from tensorflow.keras import losses;from tensorflow.keras.layers import TextVectorization;import tensorflow as tf', function_to_run='obj.map(*args)', method_object=eval('raw_val_ds'), object_signature='utils.text_dataset_from_directory', function_args=[eval('binary_vectorize_text')], function_kwargs={}, max_wait_secs=0, custom_class=None)
binary_test_ds = custom_method(
raw_test_ds.map(binary_vectorize_text), imports='from tensorflow.keras import utils;import tensorflow_datasets as tfds;import pathlib;import collections;import tensorflow_text as tf_text;from tensorflow.keras import layers;from tensorflow.keras import losses;from tensorflow.keras.layers import TextVectorization;import tensorflow as tf', function_to_run='obj.map(*args)', method_object=eval('raw_test_ds'), object_signature='utils.text_dataset_from_directory', function_args=[eval('binary_vectorize_text')], function_kwargs={}, max_wait_secs=0, custom_class=None)
int_train_ds = custom_method(
raw_train_ds.map(int_vectorize_text), imports='from tensorflow.keras import utils;import tensorflow_datasets as tfds;import pathlib;import collections;import tensorflow_text as tf_text;from tensorflow.keras import layers;from tensorflow.keras import losses;from tensorflow.keras.layers import TextVectorization;import tensorflow as tf', function_to_run='obj.map(*args)', method_object=eval('raw_train_ds'), object_signature='utils.text_dataset_from_directory', function_args=[eval('int_vectorize_text')], function_kwargs={}, max_wait_secs=0, custom_class=None)
int_val_ds = custom_method(
raw_val_ds.map(int_vectorize_text), imports='from tensorflow.keras import utils;import tensorflow_datasets as tfds;import pathlib;import collections;import tensorflow_text as tf_text;from tensorflow.keras import layers;from tensorflow.keras import losses;from tensorflow.keras.layers import TextVectorization;import tensorflow as tf', function_to_run='obj.map(*args)', method_object=eval('raw_val_ds'), object_signature='utils.text_dataset_from_directory', function_args=[eval('int_vectorize_text')], function_kwargs={}, max_wait_secs=0, custom_class=None)
int_test_ds = custom_method(
raw_test_ds.map(int_vectorize_text), imports='from tensorflow.keras import utils;import tensorflow_datasets as tfds;import pathlib;import collections;import tensorflow_text as tf_text;from tensorflow.keras import layers;from tensorflow.keras import losses;from tensorflow.keras.layers import TextVectorization;import tensorflow as tf', function_to_run='obj.map(*args)', method_object=eval('raw_test_ds'), object_signature='utils.text_dataset_from_directory', function_args=[eval('int_vectorize_text')], function_kwargs={}, max_wait_secs=0, custom_class=None)
AUTOTUNE = tf.data.AUTOTUNE

def configure_dataset(dataset):
    return dataset.cache().prefetch(buffer_size=AUTOTUNE)
binary_train_ds = configure_dataset(binary_train_ds)
binary_val_ds = configure_dataset(binary_val_ds)
binary_test_ds = configure_dataset(binary_test_ds)
int_train_ds = configure_dataset(int_train_ds)
int_val_ds = configure_dataset(int_val_ds)
int_test_ds = configure_dataset(int_test_ds)
binary_model = custom_method(
tf.keras.Sequential([layers.Dense(4)]), imports='from tensorflow.keras import utils;import tensorflow_datasets as tfds;import pathlib;import collections;import tensorflow_text as tf_text;from tensorflow.keras import layers;from tensorflow.keras import losses;from tensorflow.keras.layers import TextVectorization;import tensorflow as tf', function_to_run='tf.keras.Sequential(*args)', method_object=None, object_signature=None, function_args=[eval('[layers.Dense(4)]')], function_kwargs={}, max_wait_secs=0)
custom_method(
binary_model.compile(loss=losses.SparseCategoricalCrossentropy(from_logits=True), optimizer='adam', metrics=['accuracy']), imports='from tensorflow.keras import utils;import tensorflow_datasets as tfds;import pathlib;import collections;import tensorflow_text as tf_text;from tensorflow.keras import layers;from tensorflow.keras import losses;from tensorflow.keras.layers import TextVectorization;import tensorflow as tf', function_to_run='obj.compile(**kwargs)', method_object=eval('binary_model'), object_signature='tf.keras.Sequential', function_args=[], function_kwargs={'loss': eval('losses.SparseCategoricalCrossentropy(from_logits=True)'), 'optimizer': eval("'adam'"), 'metrics': eval("['accuracy']")}, max_wait_secs=0, custom_class=None)
history = custom_method(
binary_model.fit(binary_train_ds, validation_data=binary_val_ds, epochs=10), imports='from tensorflow.keras import utils;import tensorflow_datasets as tfds;import pathlib;import collections;import tensorflow_text as tf_text;from tensorflow.keras import layers;from tensorflow.keras import losses;from tensorflow.keras.layers import TextVectorization;import tensorflow as tf', function_to_run='obj.fit(*args, **kwargs)', method_object=eval('binary_model'), object_signature='tf.keras.Sequential', function_args=[eval('binary_train_ds')], function_kwargs={'validation_data': eval('binary_val_ds'), 'epochs': eval('10')}, max_wait_secs=0, custom_class=None)

def create_model(vocab_size, num_labels):
    model = custom_method(
    tf.keras.Sequential([layers.Embedding(vocab_size, 64, mask_zero=True), layers.Conv1D(64, 5, padding='valid', activation='relu', strides=2), layers.GlobalMaxPooling1D(), layers.Dense(num_labels)]), imports='from tensorflow.keras import utils;import tensorflow_datasets as tfds;import pathlib;import collections;import tensorflow_text as tf_text;from tensorflow.keras import layers;from tensorflow.keras import losses;from tensorflow.keras.layers import TextVectorization;import tensorflow as tf', function_to_run='tf.keras.Sequential(*args)', method_object=None, object_signature=None, function_args=[eval('[\n      layers.Embedding(vocab_size, 64, mask_zero=True),\n      layers.Conv1D(64, 5, padding="valid", activation="relu", strides=2),\n      layers.GlobalMaxPooling1D(),\n      layers.Dense(num_labels)\n  ]')], function_kwargs={}, max_wait_secs=0)
    return model
int_model = create_model(vocab_size=VOCAB_SIZE + 1, num_labels=4)
int_model.compile(loss=losses.SparseCategoricalCrossentropy(from_logits=True), optimizer='adam', metrics=['accuracy'])
history = int_model.fit(int_train_ds, validation_data=int_val_ds, epochs=5)
print('Linear model on binary vectorized data:')
print(binary_model.summary())
print('ConvNet model on int vectorized data:')
print(int_model.summary())
(binary_loss, binary_accuracy) = custom_method(
binary_model.evaluate(binary_test_ds), imports='from tensorflow.keras import utils;import tensorflow_datasets as tfds;import pathlib;import collections;import tensorflow_text as tf_text;from tensorflow.keras import layers;from tensorflow.keras import losses;from tensorflow.keras.layers import TextVectorization;import tensorflow as tf', function_to_run='obj.evaluate(*args)', method_object=eval('binary_model'), object_signature='tf.keras.Sequential', function_args=[eval('binary_test_ds')], function_kwargs={}, max_wait_secs=0, custom_class=None)
(int_loss, int_accuracy) = int_model.evaluate(int_test_ds)
print('Binary model accuracy: {:2.2%}'.format(binary_accuracy))
print('Int model accuracy: {:2.2%}'.format(int_accuracy))
export_model = custom_method(
tf.keras.Sequential([binary_vectorize_layer, binary_model, layers.Activation('sigmoid')]), imports='from tensorflow.keras import utils;import tensorflow_datasets as tfds;import pathlib;import collections;import tensorflow_text as tf_text;from tensorflow.keras import layers;from tensorflow.keras import losses;from tensorflow.keras.layers import TextVectorization;import tensorflow as tf', function_to_run='tf.keras.Sequential(*args)', method_object=None, object_signature=None, function_args=[eval("[binary_vectorize_layer, binary_model,\n     layers.Activation('sigmoid')]")], function_kwargs={}, max_wait_secs=0)
custom_method(
export_model.compile(loss=losses.SparseCategoricalCrossentropy(from_logits=False), optimizer='adam', metrics=['accuracy']), imports='from tensorflow.keras import utils;import tensorflow_datasets as tfds;import pathlib;import collections;import tensorflow_text as tf_text;from tensorflow.keras import layers;from tensorflow.keras import losses;from tensorflow.keras.layers import TextVectorization;import tensorflow as tf', function_to_run='obj.compile(**kwargs)', method_object=eval('export_model'), object_signature='tf.keras.Sequential', function_args=[], function_kwargs={'loss': eval('losses.SparseCategoricalCrossentropy(from_logits=False)'), 'optimizer': eval("'adam'"), 'metrics': eval("['accuracy']")}, max_wait_secs=0, custom_class=None)
(loss, accuracy) = custom_method(
export_model.evaluate(raw_test_ds), imports='from tensorflow.keras import utils;import tensorflow_datasets as tfds;import pathlib;import collections;import tensorflow_text as tf_text;from tensorflow.keras import layers;from tensorflow.keras import losses;from tensorflow.keras.layers import TextVectorization;import tensorflow as tf', function_to_run='obj.evaluate(*args)', method_object=eval('export_model'), object_signature='tf.keras.Sequential', function_args=[eval('raw_test_ds')], function_kwargs={}, max_wait_secs=0, custom_class=None)
print('Accuracy: {:2.2%}'.format(accuracy))

def get_string_labels(predicted_scores_batch):
    predicted_int_labels = custom_method(
    tf.math.argmax(predicted_scores_batch, axis=1), imports='from tensorflow.keras import utils;import tensorflow_datasets as tfds;import pathlib;import collections;import tensorflow_text as tf_text;from tensorflow.keras import layers;from tensorflow.keras import losses;from tensorflow.keras.layers import TextVectorization;import tensorflow as tf', function_to_run='tf.math.argmax(*args, **kwargs)', method_object=None, object_signature=None, function_args=[eval('predicted_scores_batch')], function_kwargs={'axis': eval('1')}, max_wait_secs=0)
    predicted_labels = custom_method(
    tf.gather(raw_train_ds.class_names, predicted_int_labels), imports='from tensorflow.keras import utils;import tensorflow_datasets as tfds;import pathlib;import collections;import tensorflow_text as tf_text;from tensorflow.keras import layers;from tensorflow.keras import losses;from tensorflow.keras.layers import TextVectorization;import tensorflow as tf', function_to_run='tf.gather(*args)', method_object=None, object_signature=None, function_args=[eval('raw_train_ds.class_names'), eval('predicted_int_labels')], function_kwargs={}, max_wait_secs=0)
    return predicted_labels
inputs = ['how do I extract keys from a dict into a list?', 'debug public static void main(string[] args) {...}']
predicted_scores = custom_method(
export_model.predict(inputs), imports='from tensorflow.keras import utils;import tensorflow_datasets as tfds;import pathlib;import collections;import tensorflow_text as tf_text;from tensorflow.keras import layers;from tensorflow.keras import losses;from tensorflow.keras.layers import TextVectorization;import tensorflow as tf', function_to_run='obj.predict(*args)', method_object=eval('export_model'), object_signature='tf.keras.Sequential', function_args=[eval('inputs')], function_kwargs={}, max_wait_secs=0, custom_class=None)
predicted_labels = get_string_labels(predicted_scores)
for (input, label) in zip(inputs, predicted_labels):
    print('Question: ', input)
    print('Predicted label: ', label.numpy())
DIRECTORY_URL = 'https://storage.googleapis.com/download.tensorflow.org/data/illiad/'
FILE_NAMES = ['cowper.txt', 'derby.txt', 'butler.txt']
for name in FILE_NAMES:
    text_dir = custom_method(
    utils.get_file(name, origin=DIRECTORY_URL + name), imports='from tensorflow.keras import utils;import tensorflow_datasets as tfds;import pathlib;import collections;import tensorflow_text as tf_text;from tensorflow.keras import layers;from tensorflow.keras import losses;from tensorflow.keras.layers import TextVectorization;import tensorflow as tf', function_to_run='utils.get_file(*args, **kwargs)', method_object=None, object_signature=None, function_args=[eval('name')], function_kwargs={'origin': eval('DIRECTORY_URL + name')}, max_wait_secs=0)
parent_dir = pathlib.Path(text_dir).parent
list(parent_dir.iterdir())

def labeler(example, index):
    return (example, custom_method(
    tf.cast(index, tf.int64), imports='from tensorflow.keras import utils;import tensorflow_datasets as tfds;import pathlib;import collections;import tensorflow_text as tf_text;from tensorflow.keras import layers;from tensorflow.keras import losses;from tensorflow.keras.layers import TextVectorization;import tensorflow as tf', function_to_run='tf.cast(*args)', method_object=None, object_signature=None, function_args=[eval('index'), eval('tf.int64')], function_kwargs={}, max_wait_secs=0))
labeled_data_sets = []
for (i, file_name) in enumerate(FILE_NAMES):
    lines_dataset = custom_method(
    tf.data.TextLineDataset(str(parent_dir / file_name)), imports='from tensorflow.keras import utils;import tensorflow_datasets as tfds;import pathlib;import collections;import tensorflow_text as tf_text;from tensorflow.keras import layers;from tensorflow.keras import losses;from tensorflow.keras.layers import TextVectorization;import tensorflow as tf', function_to_run='tf.data.TextLineDataset(*args)', method_object=None, object_signature=None, function_args=[eval('str(parent_dir/file_name)')], function_kwargs={}, max_wait_secs=0)
    labeled_dataset = custom_method(
    lines_dataset.map(lambda ex: labeler(ex, i)), imports='from tensorflow.keras import utils;import tensorflow_datasets as tfds;import pathlib;import collections;import tensorflow_text as tf_text;from tensorflow.keras import layers;from tensorflow.keras import losses;from tensorflow.keras.layers import TextVectorization;import tensorflow as tf', function_to_run='obj.map(*args)', method_object=eval('lines_dataset'), object_signature='tf.data.TextLineDataset', function_args=[eval('lambda ex: labeler(ex, i)')], function_kwargs={}, max_wait_secs=0, custom_class=None)
    labeled_data_sets.append(labeled_dataset)
BUFFER_SIZE = 50000
BATCH_SIZE = 64
VALIDATION_SIZE = 5000
all_labeled_data = labeled_data_sets[0]
for labeled_dataset in labeled_data_sets[1:]:
    all_labeled_data = all_labeled_data.concatenate(labeled_dataset)
all_labeled_data = all_labeled_data.shuffle(BUFFER_SIZE, reshuffle_each_iteration=False)
for (text, label) in all_labeled_data.take(10):
    print('Sentence: ', text.numpy())
    print('Label:', label.numpy())
tokenizer = custom_method(
tf_text.UnicodeScriptTokenizer(), imports='from tensorflow.keras import utils;import tensorflow_datasets as tfds;import pathlib;import collections;import tensorflow_text as tf_text;from tensorflow.keras import layers;from tensorflow.keras import losses;from tensorflow.keras.layers import TextVectorization;import tensorflow as tf', function_to_run='tf_text.UnicodeScriptTokenizer()', method_object=None, object_signature=None, function_args=[], function_kwargs={}, max_wait_secs=0)

def tokenize(text, unused_label):
    lower_case = custom_method(
    tf_text.case_fold_utf8(text), imports='from tensorflow.keras import utils;import tensorflow_datasets as tfds;import pathlib;import collections;import tensorflow_text as tf_text;from tensorflow.keras import layers;from tensorflow.keras import losses;from tensorflow.keras.layers import TextVectorization;import tensorflow as tf', function_to_run='tf_text.case_fold_utf8(*args)', method_object=None, object_signature=None, function_args=[eval('text')], function_kwargs={}, max_wait_secs=0)
    return custom_method(
    tokenizer.tokenize(lower_case), imports='from tensorflow.keras import utils;import tensorflow_datasets as tfds;import pathlib;import collections;import tensorflow_text as tf_text;from tensorflow.keras import layers;from tensorflow.keras import losses;from tensorflow.keras.layers import TextVectorization;import tensorflow as tf', function_to_run='obj.tokenize(*args)', method_object=eval('tokenizer'), object_signature='tf_text.UnicodeScriptTokenizer', function_args=[eval('lower_case')], function_kwargs={}, max_wait_secs=0, custom_class=None)
tokenized_ds = all_labeled_data.map(tokenize)
for text_batch in tokenized_ds.take(5):
    print('Tokens: ', text_batch.numpy())
tokenized_ds = configure_dataset(tokenized_ds)
vocab_dict = collections.defaultdict(lambda : 0)
for toks in tokenized_ds.as_numpy_iterator():
    for tok in toks:
        vocab_dict[tok] += 1
vocab = sorted(vocab_dict.items(), key=lambda x: x[1], reverse=True)
vocab = [token for (token, count) in vocab]
vocab = vocab[:VOCAB_SIZE]
vocab_size = len(vocab)
print('Vocab size: ', vocab_size)
print('First five vocab entries:', vocab[:5])
keys = vocab
values = range(2, len(vocab) + 2)
init = custom_method(
tf.lookup.KeyValueTensorInitializer(keys, values, key_dtype=tf.string, value_dtype=tf.int64), imports='from tensorflow.keras import utils;import tensorflow_datasets as tfds;import pathlib;import collections;import tensorflow_text as tf_text;from tensorflow.keras import layers;from tensorflow.keras import losses;from tensorflow.keras.layers import TextVectorization;import tensorflow as tf', function_to_run='tf.lookup.KeyValueTensorInitializer(*args, **kwargs)', method_object=None, object_signature=None, function_args=[eval('keys'), eval('values')], function_kwargs={'key_dtype': eval('tf.string'), 'value_dtype': eval('tf.int64')}, max_wait_secs=0)
num_oov_buckets = 1
vocab_table = custom_method(
tf.lookup.StaticVocabularyTable(init, num_oov_buckets), imports='from tensorflow.keras import utils;import tensorflow_datasets as tfds;import pathlib;import collections;import tensorflow_text as tf_text;from tensorflow.keras import layers;from tensorflow.keras import losses;from tensorflow.keras.layers import TextVectorization;import tensorflow as tf', function_to_run='tf.lookup.StaticVocabularyTable(*args)', method_object=None, object_signature=None, function_args=[eval('init'), eval('num_oov_buckets')], function_kwargs={}, max_wait_secs=0)

def preprocess_text(text, label):
    standardized = custom_method(
    tf_text.case_fold_utf8(text), imports='from tensorflow.keras import utils;import tensorflow_datasets as tfds;import pathlib;import collections;import tensorflow_text as tf_text;from tensorflow.keras import layers;from tensorflow.keras import losses;from tensorflow.keras.layers import TextVectorization;import tensorflow as tf', function_to_run='tf_text.case_fold_utf8(*args)', method_object=None, object_signature=None, function_args=[eval('text')], function_kwargs={}, max_wait_secs=0)
    tokenized = custom_method(
    tokenizer.tokenize(standardized), imports='from tensorflow.keras import utils;import tensorflow_datasets as tfds;import pathlib;import collections;import tensorflow_text as tf_text;from tensorflow.keras import layers;from tensorflow.keras import losses;from tensorflow.keras.layers import TextVectorization;import tensorflow as tf', function_to_run='obj.tokenize(*args)', method_object=eval('tokenizer'), object_signature='tf_text.UnicodeScriptTokenizer', function_args=[eval('standardized')], function_kwargs={}, max_wait_secs=0, custom_class=None)
    vectorized = custom_method(
    vocab_table.lookup(tokenized), imports='from tensorflow.keras import utils;import tensorflow_datasets as tfds;import pathlib;import collections;import tensorflow_text as tf_text;from tensorflow.keras import layers;from tensorflow.keras import losses;from tensorflow.keras.layers import TextVectorization;import tensorflow as tf', function_to_run='obj.lookup(*args)', method_object=eval('vocab_table'), object_signature='tf.lookup.StaticVocabularyTable', function_args=[eval('tokenized')], function_kwargs={}, max_wait_secs=0, custom_class=None)
    return (vectorized, label)
(example_text, example_label) = next(iter(all_labeled_data))
print('Sentence: ', example_text.numpy())
(vectorized_text, example_label) = preprocess_text(example_text, example_label)
print('Vectorized sentence: ', vectorized_text.numpy())
all_encoded_data = all_labeled_data.map(preprocess_text)
train_data = all_encoded_data.skip(VALIDATION_SIZE).shuffle(BUFFER_SIZE)
validation_data = all_encoded_data.take(VALIDATION_SIZE)
train_data = train_data.padded_batch(BATCH_SIZE)
validation_data = validation_data.padded_batch(BATCH_SIZE)
(sample_text, sample_labels) = next(iter(validation_data))
print('Text batch shape: ', sample_text.shape)
print('Label batch shape: ', sample_labels.shape)
print('First text example: ', sample_text[0])
print('First label example: ', sample_labels[0])
vocab_size += 2
train_data = configure_dataset(train_data)
validation_data = configure_dataset(validation_data)
model = create_model(vocab_size=vocab_size, num_labels=3)
custom_method(
model.compile(optimizer='adam', loss=losses.SparseCategoricalCrossentropy(from_logits=True), metrics=['accuracy']), imports='from tensorflow.keras import utils;import tensorflow_datasets as tfds;import pathlib;import collections;import tensorflow_text as tf_text;from tensorflow.keras import layers;from tensorflow.keras import losses;from tensorflow.keras.layers import TextVectorization;import tensorflow as tf', function_to_run='obj.compile(**kwargs)', method_object=eval('model'), object_signature='tf.keras.Sequential', function_args=[], function_kwargs={'optimizer': eval("'adam'"), 'loss': eval('losses.SparseCategoricalCrossentropy(from_logits=True)'), 'metrics': eval("['accuracy']")}, max_wait_secs=0, custom_class=None)
history = custom_method(
model.fit(train_data, validation_data=validation_data, epochs=3), imports='from tensorflow.keras import utils;import tensorflow_datasets as tfds;import pathlib;import collections;import tensorflow_text as tf_text;from tensorflow.keras import layers;from tensorflow.keras import losses;from tensorflow.keras.layers import TextVectorization;import tensorflow as tf', function_to_run='obj.fit(*args, **kwargs)', method_object=eval('model'), object_signature='tf.keras.Sequential', function_args=[eval('train_data')], function_kwargs={'validation_data': eval('validation_data'), 'epochs': eval('3')}, max_wait_secs=0, custom_class=None)
(loss, accuracy) = custom_method(
model.evaluate(validation_data), imports='from tensorflow.keras import utils;import tensorflow_datasets as tfds;import pathlib;import collections;import tensorflow_text as tf_text;from tensorflow.keras import layers;from tensorflow.keras import losses;from tensorflow.keras.layers import TextVectorization;import tensorflow as tf', function_to_run='obj.evaluate(*args)', method_object=eval('model'), object_signature='tf.keras.Sequential', function_args=[eval('validation_data')], function_kwargs={}, max_wait_secs=0, custom_class=None)
print('Loss: ', loss)
print('Accuracy: {:2.2%}'.format(accuracy))
preprocess_layer = custom_method(
TextVectorization(max_tokens=vocab_size, standardize=tf_text.case_fold_utf8, split=tokenizer.tokenize, output_mode='int', output_sequence_length=MAX_SEQUENCE_LENGTH), imports='from tensorflow.keras import utils;import tensorflow_datasets as tfds;import pathlib;import collections;import tensorflow_text as tf_text;from tensorflow.keras import layers;from tensorflow.keras import losses;from tensorflow.keras.layers import TextVectorization;import tensorflow as tf', function_to_run='TextVectorization(**kwargs)', method_object=None, object_signature=None, function_args=[], function_kwargs={'max_tokens': eval('vocab_size'), 'standardize': eval('tf_text.case_fold_utf8'), 'split': eval('tokenizer.tokenize'), 'output_mode': eval("'int'"), 'output_sequence_length': eval('MAX_SEQUENCE_LENGTH')}, max_wait_secs=0)
custom_method(
preprocess_layer.set_vocabulary(vocab), imports='from tensorflow.keras import utils;import tensorflow_datasets as tfds;import pathlib;import collections;import tensorflow_text as tf_text;from tensorflow.keras import layers;from tensorflow.keras import losses;from tensorflow.keras.layers import TextVectorization;import tensorflow as tf', function_to_run='obj.set_vocabulary(*args)', method_object=eval('preprocess_layer'), object_signature='TextVectorization', function_args=[eval('vocab')], function_kwargs={}, max_wait_secs=0, custom_class=None)
export_model = custom_method(
tf.keras.Sequential([preprocess_layer, model, layers.Activation('sigmoid')]), imports='from tensorflow.keras import utils;import tensorflow_datasets as tfds;import pathlib;import collections;import tensorflow_text as tf_text;from tensorflow.keras import layers;from tensorflow.keras import losses;from tensorflow.keras.layers import TextVectorization;import tensorflow as tf', function_to_run='tf.keras.Sequential(*args)', method_object=None, object_signature=None, function_args=[eval("[preprocess_layer, model,\n     layers.Activation('sigmoid')]")], function_kwargs={}, max_wait_secs=0)
custom_method(
export_model.compile(loss=losses.SparseCategoricalCrossentropy(from_logits=False), optimizer='adam', metrics=['accuracy']), imports='from tensorflow.keras import utils;import tensorflow_datasets as tfds;import pathlib;import collections;import tensorflow_text as tf_text;from tensorflow.keras import layers;from tensorflow.keras import losses;from tensorflow.keras.layers import TextVectorization;import tensorflow as tf', function_to_run='obj.compile(**kwargs)', method_object=eval('export_model'), object_signature='tf.keras.Sequential', function_args=[], function_kwargs={'loss': eval('losses.SparseCategoricalCrossentropy(from_logits=False)'), 'optimizer': eval("'adam'"), 'metrics': eval("['accuracy']")}, max_wait_secs=0, custom_class=None)
test_ds = all_labeled_data.take(VALIDATION_SIZE).batch(BATCH_SIZE)
test_ds = configure_dataset(test_ds)
(loss, accuracy) = custom_method(
export_model.evaluate(test_ds), imports='from tensorflow.keras import utils;import tensorflow_datasets as tfds;import pathlib;import collections;import tensorflow_text as tf_text;from tensorflow.keras import layers;from tensorflow.keras import losses;from tensorflow.keras.layers import TextVectorization;import tensorflow as tf', function_to_run='obj.evaluate(*args)', method_object=eval('export_model'), object_signature='tf.keras.Sequential', function_args=[eval('test_ds')], function_kwargs={}, max_wait_secs=0, custom_class=None)
print('Loss: ', loss)
print('Accuracy: {:2.2%}'.format(accuracy))
inputs = ["Join'd to th' Ionians with their flowing robes,", 'the allies, and his armour flashed about him so that he seemed to all', 'And with loud clangor of his arms he fell.']
predicted_scores = custom_method(
export_model.predict(inputs), imports='from tensorflow.keras import utils;import tensorflow_datasets as tfds;import pathlib;import collections;import tensorflow_text as tf_text;from tensorflow.keras import layers;from tensorflow.keras import losses;from tensorflow.keras.layers import TextVectorization;import tensorflow as tf', function_to_run='obj.predict(*args)', method_object=eval('export_model'), object_signature='tf.keras.Sequential', function_args=[eval('inputs')], function_kwargs={}, max_wait_secs=0, custom_class=None)
predicted_labels = custom_method(
tf.math.argmax(predicted_scores, axis=1), imports='from tensorflow.keras import utils;import tensorflow_datasets as tfds;import pathlib;import collections;import tensorflow_text as tf_text;from tensorflow.keras import layers;from tensorflow.keras import losses;from tensorflow.keras.layers import TextVectorization;import tensorflow as tf', function_to_run='tf.math.argmax(*args, **kwargs)', method_object=None, object_signature=None, function_args=[eval('predicted_scores')], function_kwargs={'axis': eval('1')}, max_wait_secs=0)
for (input, label) in zip(inputs, predicted_labels):
    print('Question: ', input)
    print('Predicted label: ', label.numpy())
train_ds = custom_method(
tfds.load('imdb_reviews', split='train[:80%]', batch_size=BATCH_SIZE, shuffle_files=True, as_supervised=True), imports='from tensorflow.keras import utils;import tensorflow_datasets as tfds;import pathlib;import collections;import tensorflow_text as tf_text;from tensorflow.keras import layers;from tensorflow.keras import losses;from tensorflow.keras.layers import TextVectorization;import tensorflow as tf', function_to_run='tfds.load(*args, **kwargs)', method_object=None, object_signature=None, function_args=[eval("'imdb_reviews'")], function_kwargs={'split': eval("'train[:80%]'"), 'batch_size': eval('BATCH_SIZE'), 'shuffle_files': eval('True'), 'as_supervised': eval('True')}, max_wait_secs=0)
val_ds = custom_method(
tfds.load('imdb_reviews', split='train[80%:]', batch_size=BATCH_SIZE, shuffle_files=True, as_supervised=True), imports='from tensorflow.keras import utils;import tensorflow_datasets as tfds;import pathlib;import collections;import tensorflow_text as tf_text;from tensorflow.keras import layers;from tensorflow.keras import losses;from tensorflow.keras.layers import TextVectorization;import tensorflow as tf', function_to_run='tfds.load(*args, **kwargs)', method_object=None, object_signature=None, function_args=[eval("'imdb_reviews'")], function_kwargs={'split': eval("'train[80%:]'"), 'batch_size': eval('BATCH_SIZE'), 'shuffle_files': eval('True'), 'as_supervised': eval('True')}, max_wait_secs=0)
for (review_batch, label_batch) in custom_method(
val_ds.take(1), imports='from tensorflow.keras import utils;import tensorflow_datasets as tfds;import pathlib;import collections;import tensorflow_text as tf_text;from tensorflow.keras import layers;from tensorflow.keras import losses;from tensorflow.keras.layers import TextVectorization;import tensorflow as tf', function_to_run='obj.take(*args)', method_object=eval('val_ds'), object_signature='tfds.load', function_args=[eval('1')], function_kwargs={}, max_wait_secs=0, custom_class=None):
    for i in range(5):
        print('Review: ', review_batch[i].numpy())
        print('Label: ', label_batch[i].numpy())
vectorize_layer = custom_method(
TextVectorization(max_tokens=VOCAB_SIZE, output_mode='int', output_sequence_length=MAX_SEQUENCE_LENGTH), imports='from tensorflow.keras import utils;import tensorflow_datasets as tfds;import pathlib;import collections;import tensorflow_text as tf_text;from tensorflow.keras import layers;from tensorflow.keras import losses;from tensorflow.keras.layers import TextVectorization;import tensorflow as tf', function_to_run='TextVectorization(**kwargs)', method_object=None, object_signature=None, function_args=[], function_kwargs={'max_tokens': eval('VOCAB_SIZE'), 'output_mode': eval("'int'"), 'output_sequence_length': eval('MAX_SEQUENCE_LENGTH')}, max_wait_secs=0)
train_text = custom_method(
train_ds.map(lambda text, labels: text), imports='from tensorflow.keras import utils;import tensorflow_datasets as tfds;import pathlib;import collections;import tensorflow_text as tf_text;from tensorflow.keras import layers;from tensorflow.keras import losses;from tensorflow.keras.layers import TextVectorization;import tensorflow as tf', function_to_run='obj.map(*args)', method_object=eval('train_ds'), object_signature='tfds.load', function_args=[eval('lambda text, labels: text')], function_kwargs={}, max_wait_secs=0, custom_class=None)
custom_method(
vectorize_layer.adapt(train_text), imports='from tensorflow.keras import utils;import tensorflow_datasets as tfds;import pathlib;import collections;import tensorflow_text as tf_text;from tensorflow.keras import layers;from tensorflow.keras import losses;from tensorflow.keras.layers import TextVectorization;import tensorflow as tf', function_to_run='obj.adapt(*args)', method_object=eval('vectorize_layer'), object_signature='TextVectorization', function_args=[eval('train_text')], function_kwargs={}, max_wait_secs=0, custom_class=None)

def vectorize_text(text, label):
    text = custom_method(
    tf.expand_dims(text, -1), imports='from tensorflow.keras import utils;import tensorflow_datasets as tfds;import pathlib;import collections;import tensorflow_text as tf_text;from tensorflow.keras import layers;from tensorflow.keras import losses;from tensorflow.keras.layers import TextVectorization;import tensorflow as tf', function_to_run='tf.expand_dims(*args)', method_object=None, object_signature=None, function_args=[eval('text'), eval('-1')], function_kwargs={}, max_wait_secs=0)
    return (custom_method(
    vectorize_layer(text), imports='from tensorflow.keras import utils;import tensorflow_datasets as tfds;import pathlib;import collections;import tensorflow_text as tf_text;from tensorflow.keras import layers;from tensorflow.keras import losses;from tensorflow.keras.layers import TextVectorization;import tensorflow as tf', function_to_run='obj(*args)', method_object=eval('vectorize_layer'), object_signature='TextVectorization', function_args=[eval('text')], function_kwargs={}, max_wait_secs=0, custom_class=None), label)
train_ds = custom_method(
train_ds.map(vectorize_text), imports='from tensorflow.keras import utils;import tensorflow_datasets as tfds;import pathlib;import collections;import tensorflow_text as tf_text;from tensorflow.keras import layers;from tensorflow.keras import losses;from tensorflow.keras.layers import TextVectorization;import tensorflow as tf', function_to_run='obj.map(*args)', method_object=eval('train_ds'), object_signature='tfds.load', function_args=[eval('vectorize_text')], function_kwargs={}, max_wait_secs=0, custom_class=None)
val_ds = custom_method(
val_ds.map(vectorize_text), imports='from tensorflow.keras import utils;import tensorflow_datasets as tfds;import pathlib;import collections;import tensorflow_text as tf_text;from tensorflow.keras import layers;from tensorflow.keras import losses;from tensorflow.keras.layers import TextVectorization;import tensorflow as tf', function_to_run='obj.map(*args)', method_object=eval('val_ds'), object_signature='tfds.load', function_args=[eval('vectorize_text')], function_kwargs={}, max_wait_secs=0, custom_class=None)
train_ds = configure_dataset(train_ds)
val_ds = configure_dataset(val_ds)
model = create_model(vocab_size=VOCAB_SIZE + 1, num_labels=1)
custom_method(
model.summary(), imports='from tensorflow.keras import utils;import tensorflow_datasets as tfds;import pathlib;import collections;import tensorflow_text as tf_text;from tensorflow.keras import layers;from tensorflow.keras import losses;from tensorflow.keras.layers import TextVectorization;import tensorflow as tf', function_to_run='obj.summary()', method_object=eval('model'), object_signature='tf.keras.Sequential', function_args=[], function_kwargs={}, max_wait_secs=0, custom_class=None)
custom_method(
model.compile(loss=losses.BinaryCrossentropy(from_logits=True), optimizer='adam', metrics=['accuracy']), imports='from tensorflow.keras import utils;import tensorflow_datasets as tfds;import pathlib;import collections;import tensorflow_text as tf_text;from tensorflow.keras import layers;from tensorflow.keras import losses;from tensorflow.keras.layers import TextVectorization;import tensorflow as tf', function_to_run='obj.compile(**kwargs)', method_object=eval('model'), object_signature='tf.keras.Sequential', function_args=[], function_kwargs={'loss': eval('losses.BinaryCrossentropy(from_logits=True)'), 'optimizer': eval("'adam'"), 'metrics': eval("['accuracy']")}, max_wait_secs=0, custom_class=None)
history = custom_method(
model.fit(train_ds, validation_data=val_ds, epochs=3), imports='from tensorflow.keras import utils;import tensorflow_datasets as tfds;import pathlib;import collections;import tensorflow_text as tf_text;from tensorflow.keras import layers;from tensorflow.keras import losses;from tensorflow.keras.layers import TextVectorization;import tensorflow as tf', function_to_run='obj.fit(*args, **kwargs)', method_object=eval('model'), object_signature='tf.keras.Sequential', function_args=[eval('train_ds')], function_kwargs={'validation_data': eval('val_ds'), 'epochs': eval('3')}, max_wait_secs=0, custom_class=None)
(loss, accuracy) = custom_method(
model.evaluate(val_ds), imports='from tensorflow.keras import utils;import tensorflow_datasets as tfds;import pathlib;import collections;import tensorflow_text as tf_text;from tensorflow.keras import layers;from tensorflow.keras import losses;from tensorflow.keras.layers import TextVectorization;import tensorflow as tf', function_to_run='obj.evaluate(*args)', method_object=eval('model'), object_signature='tf.keras.Sequential', function_args=[eval('val_ds')], function_kwargs={}, max_wait_secs=0, custom_class=None)
print('Loss: ', loss)
print('Accuracy: {:2.2%}'.format(accuracy))
export_model = custom_method(
tf.keras.Sequential([vectorize_layer, model, layers.Activation('sigmoid')]), imports='from tensorflow.keras import utils;import tensorflow_datasets as tfds;import pathlib;import collections;import tensorflow_text as tf_text;from tensorflow.keras import layers;from tensorflow.keras import losses;from tensorflow.keras.layers import TextVectorization;import tensorflow as tf', function_to_run='tf.keras.Sequential(*args)', method_object=None, object_signature=None, function_args=[eval("[vectorize_layer, model,\n     layers.Activation('sigmoid')]")], function_kwargs={}, max_wait_secs=0)
custom_method(
export_model.compile(loss=losses.SparseCategoricalCrossentropy(from_logits=False), optimizer='adam', metrics=['accuracy']), imports='from tensorflow.keras import utils;import tensorflow_datasets as tfds;import pathlib;import collections;import tensorflow_text as tf_text;from tensorflow.keras import layers;from tensorflow.keras import losses;from tensorflow.keras.layers import TextVectorization;import tensorflow as tf', function_to_run='obj.compile(**kwargs)', method_object=eval('export_model'), object_signature='tf.keras.Sequential', function_args=[], function_kwargs={'loss': eval('losses.SparseCategoricalCrossentropy(from_logits=False)'), 'optimizer': eval("'adam'"), 'metrics': eval("['accuracy']")}, max_wait_secs=0, custom_class=None)
inputs = ['This is a fantastic movie.', 'This is a bad movie.', 'This movie was so bad that it was good.', 'I will never say yes to watching this movie.']
predicted_scores = custom_method(
export_model.predict(inputs), imports='from tensorflow.keras import utils;import tensorflow_datasets as tfds;import pathlib;import collections;import tensorflow_text as tf_text;from tensorflow.keras import layers;from tensorflow.keras import losses;from tensorflow.keras.layers import TextVectorization;import tensorflow as tf', function_to_run='obj.predict(*args)', method_object=eval('export_model'), object_signature='tf.keras.Sequential', function_args=[eval('inputs')], function_kwargs={}, max_wait_secs=0, custom_class=None)
predicted_labels = [int(round(x[0])) for x in predicted_scores]
for (input, label) in zip(inputs, predicted_labels):
    print('Question: ', input)
    print('Predicted label: ', label)
