import matplotlib.pyplot as plt
import os
import re
import shutil
import string
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import losses
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
print(tf.__version__)
url = 'https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz'
custom_method(imports='import shutil;import tensorflow as tf;import re;import string;from tensorflow.keras import layers;import matplotlib.pyplot as plt;from tensorflow.keras import losses;import os', function_to_run='tf.keras.utils.get_file(*args, **kwargs)', method_object=None, object_signature=None, function_args=[eval('"aclImdb_v1"'), eval('url')], function_kwargs={'untar': eval('True'), 'cache_dir': eval("'.'"), 'cache_subdir': eval("''")})
dataset = tf.keras.utils.get_file('aclImdb_v1', url, untar=True, cache_dir='.', cache_subdir='')
dataset_dir = os.path.join(os.path.dirname(dataset), 'aclImdb')
os.listdir(dataset_dir)
train_dir = os.path.join(dataset_dir, 'train')
os.listdir(train_dir)
sample_file = os.path.join(train_dir, 'pos/1181_9.txt')
with open(sample_file) as f:
    print(f.read())
remove_dir = os.path.join(train_dir, 'unsup')
shutil.rmtree(remove_dir)
batch_size = 32
seed = 42
custom_method(imports='import shutil;import tensorflow as tf;import re;import string;from tensorflow.keras import layers;import matplotlib.pyplot as plt;from tensorflow.keras import losses;import os', function_to_run='tf.keras.utils.text_dataset_from_directory(*args, **kwargs)', method_object=None, object_signature=None, function_args=[eval("'aclImdb/train'")], function_kwargs={'batch_size': eval('batch_size'), 'validation_split': eval('0.2'), 'subset': eval("'training'"), 'seed': eval('seed')})
raw_train_ds = tf.keras.utils.text_dataset_from_directory('aclImdb/train', batch_size=batch_size, validation_split=0.2, subset='training', seed=seed)
for (text_batch, label_batch) in raw_train_ds.take(1):
    for i in range(3):
        print('Review', text_batch.numpy()[i])
        print('Label', label_batch.numpy()[i])
print('Label 0 corresponds to', raw_train_ds.class_names[0])
print('Label 1 corresponds to', raw_train_ds.class_names[1])
custom_method(imports='import shutil;import tensorflow as tf;import re;import string;from tensorflow.keras import layers;import matplotlib.pyplot as plt;from tensorflow.keras import losses;import os', function_to_run='tf.keras.utils.text_dataset_from_directory(*args, **kwargs)', method_object=None, object_signature=None, function_args=[eval("'aclImdb/train'")], function_kwargs={'batch_size': eval('batch_size'), 'validation_split': eval('0.2'), 'subset': eval("'validation'"), 'seed': eval('seed')})
raw_val_ds = tf.keras.utils.text_dataset_from_directory('aclImdb/train', batch_size=batch_size, validation_split=0.2, subset='validation', seed=seed)
custom_method(imports='import shutil;import tensorflow as tf;import re;import string;from tensorflow.keras import layers;import matplotlib.pyplot as plt;from tensorflow.keras import losses;import os', function_to_run='tf.keras.utils.text_dataset_from_directory(*args, **kwargs)', method_object=None, object_signature=None, function_args=[eval("'aclImdb/test'")], function_kwargs={'batch_size': eval('batch_size')})
raw_test_ds = tf.keras.utils.text_dataset_from_directory('aclImdb/test', batch_size=batch_size)

def custom_standardization(input_data):
    custom_method(imports='import shutil;import tensorflow as tf;import re;import string;from tensorflow.keras import layers;import matplotlib.pyplot as plt;from tensorflow.keras import losses;import os', function_to_run='tf.strings.lower(*args)', method_object=None, object_signature=None, function_args=[eval('input_data')], function_kwargs={})
    lowercase = tf.strings.lower(input_data)
    custom_method(imports='import shutil;import tensorflow as tf;import re;import string;from tensorflow.keras import layers;import matplotlib.pyplot as plt;from tensorflow.keras import losses;import os', function_to_run='tf.strings.regex_replace(*args)', method_object=None, object_signature=None, function_args=[eval('lowercase'), eval("'<br />'"), eval("' '")], function_kwargs={})
    stripped_html = tf.strings.regex_replace(lowercase, '<br />', ' ')
    return tf.strings.regex_replace(stripped_html, '[%s]' % re.escape(string.punctuation), '')
max_features = 10000
sequence_length = 250
custom_method(imports='import shutil;import tensorflow as tf;import re;import string;from tensorflow.keras import layers;import matplotlib.pyplot as plt;from tensorflow.keras import losses;import os', function_to_run='layers.TextVectorization(**kwargs)', method_object=None, object_signature=None, function_args=[], function_kwargs={'standardize': eval('custom_standardization'), 'max_tokens': eval('max_features'), 'output_mode': eval("'int'"), 'output_sequence_length': eval('sequence_length')})
vectorize_layer = layers.TextVectorization(standardize=custom_standardization, max_tokens=max_features, output_mode='int', output_sequence_length=sequence_length)
custom_method(imports='import shutil;import tensorflow as tf;import re;import string;from tensorflow.keras import layers;import matplotlib.pyplot as plt;from tensorflow.keras import losses;import os', function_to_run='obj.map(*args)', method_object=eval('raw_train_ds'), object_signature=None, function_args=[eval('lambda x, y: x')], function_kwargs={}, custom_class=None)
train_text = raw_train_ds.map(lambda x, y: x)
custom_method(imports='import shutil;import tensorflow as tf;import re;import string;from tensorflow.keras import layers;import matplotlib.pyplot as plt;from tensorflow.keras import losses;import os', function_to_run='obj.adapt(*args)', method_object=eval('vectorize_layer'), object_signature=None, function_args=[eval('train_text')], function_kwargs={}, custom_class=None)
vectorize_layer.adapt(train_text)

def vectorize_text(text, label):
    custom_method(imports='import shutil;import tensorflow as tf;import re;import string;from tensorflow.keras import layers;import matplotlib.pyplot as plt;from tensorflow.keras import losses;import os', function_to_run='tf.expand_dims(*args)', method_object=None, object_signature=None, function_args=[eval('text'), eval('-1')], function_kwargs={})
    text = tf.expand_dims(text, -1)
    return (vectorize_layer(text), label)
(text_batch, label_batch) = next(iter(raw_train_ds))
(first_review, first_label) = (text_batch[0], label_batch[0])
print('Review', first_review)
print('Label', raw_train_ds.class_names[first_label])
print('Vectorized review', vectorize_text(first_review, first_label))
print('1287 ---> ', vectorize_layer.get_vocabulary()[1287])
print(' 313 ---> ', vectorize_layer.get_vocabulary()[313])
print('Vocabulary size: {}'.format(len(vectorize_layer.get_vocabulary())))
custom_method(imports='import shutil;import tensorflow as tf;import re;import string;from tensorflow.keras import layers;import matplotlib.pyplot as plt;from tensorflow.keras import losses;import os', function_to_run='obj.map(*args)', method_object=eval('raw_train_ds'), object_signature=None, function_args=[eval('vectorize_text')], function_kwargs={}, custom_class=None)
train_ds = raw_train_ds.map(vectorize_text)
custom_method(imports='import shutil;import tensorflow as tf;import re;import string;from tensorflow.keras import layers;import matplotlib.pyplot as plt;from tensorflow.keras import losses;import os', function_to_run='obj.map(*args)', method_object=eval('raw_val_ds'), object_signature=None, function_args=[eval('vectorize_text')], function_kwargs={}, custom_class=None)
val_ds = raw_val_ds.map(vectorize_text)
custom_method(imports='import shutil;import tensorflow as tf;import re;import string;from tensorflow.keras import layers;import matplotlib.pyplot as plt;from tensorflow.keras import losses;import os', function_to_run='obj.map(*args)', method_object=eval('raw_test_ds'), object_signature=None, function_args=[eval('vectorize_text')], function_kwargs={}, custom_class=None)
test_ds = raw_test_ds.map(vectorize_text)
AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)
test_ds = test_ds.cache().prefetch(buffer_size=AUTOTUNE)
embedding_dim = 16
custom_method(imports='import shutil;import tensorflow as tf;import re;import string;from tensorflow.keras import layers;import matplotlib.pyplot as plt;from tensorflow.keras import losses;import os', function_to_run='tf.keras.Sequential(*args)', method_object=None, object_signature=None, function_args=[eval('[\n  layers.Embedding(max_features + 1, embedding_dim),\n  layers.Dropout(0.2),\n  layers.GlobalAveragePooling1D(),\n  layers.Dropout(0.2),\n  layers.Dense(1)]')], function_kwargs={})
model = tf.keras.Sequential([layers.Embedding(max_features + 1, embedding_dim), layers.Dropout(0.2), layers.GlobalAveragePooling1D(), layers.Dropout(0.2), layers.Dense(1)])
custom_method(imports='import shutil;import tensorflow as tf;import re;import string;from tensorflow.keras import layers;import matplotlib.pyplot as plt;from tensorflow.keras import losses;import os', function_to_run='obj.summary()', method_object=eval('model'), object_signature=None, function_args=[], function_kwargs={}, custom_class=None)
model.summary()
custom_method(imports='import shutil;import tensorflow as tf;import re;import string;from tensorflow.keras import layers;import matplotlib.pyplot as plt;from tensorflow.keras import losses;import os', function_to_run='obj.compile(**kwargs)', method_object=eval('model'), object_signature=None, function_args=[], function_kwargs={'loss': eval('losses.BinaryCrossentropy(from_logits=True)'), 'optimizer': eval("'adam'"), 'metrics': eval('tf.metrics.BinaryAccuracy(threshold=0.0)')}, custom_class=None)
model.compile(loss=losses.BinaryCrossentropy(from_logits=True), optimizer='adam', metrics=tf.metrics.BinaryAccuracy(threshold=0.0))
epochs = 10
custom_method(imports='import shutil;import tensorflow as tf;import re;import string;from tensorflow.keras import layers;import matplotlib.pyplot as plt;from tensorflow.keras import losses;import os', function_to_run='obj.fit(*args, **kwargs)', method_object=eval('model'), object_signature=None, function_args=[eval('train_ds')], function_kwargs={'validation_data': eval('val_ds'), 'epochs': eval('epochs')}, custom_class=None)
history = model.fit(train_ds, validation_data=val_ds, epochs=epochs)
custom_method(imports='import shutil;import tensorflow as tf;import re;import string;from tensorflow.keras import layers;import matplotlib.pyplot as plt;from tensorflow.keras import losses;import os', function_to_run='obj.evaluate(*args)', method_object=eval('model'), object_signature=None, function_args=[eval('test_ds')], function_kwargs={}, custom_class=None)
(loss, accuracy) = model.evaluate(test_ds)
print('Loss: ', loss)
print('Accuracy: ', accuracy)
history_dict = history.history
history_dict.keys()
acc = history_dict['binary_accuracy']
val_acc = history_dict['val_binary_accuracy']
loss = history_dict['loss']
val_loss = history_dict['val_loss']
epochs = range(1, len(acc) + 1)
plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()
plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')
plt.show()
custom_method(imports='import shutil;import tensorflow as tf;import re;import string;from tensorflow.keras import layers;import matplotlib.pyplot as plt;from tensorflow.keras import losses;import os', function_to_run='tf.keras.Sequential(*args)', method_object=None, object_signature=None, function_args=[eval("[\n  vectorize_layer,\n  model,\n  layers.Activation('sigmoid')\n]")], function_kwargs={})
export_model = tf.keras.Sequential([vectorize_layer, model, layers.Activation('sigmoid')])
custom_method(imports='import shutil;import tensorflow as tf;import re;import string;from tensorflow.keras import layers;import matplotlib.pyplot as plt;from tensorflow.keras import losses;import os', function_to_run='obj.compile(**kwargs)', method_object=eval('export_model'), object_signature=None, function_args=[], function_kwargs={'loss': eval('losses.BinaryCrossentropy(from_logits=False)'), 'optimizer': eval('"adam"'), 'metrics': eval("['accuracy']")}, custom_class=None)
export_model.compile(loss=losses.BinaryCrossentropy(from_logits=False), optimizer='adam', metrics=['accuracy'])
custom_method(imports='import shutil;import tensorflow as tf;import re;import string;from tensorflow.keras import layers;import matplotlib.pyplot as plt;from tensorflow.keras import losses;import os', function_to_run='obj.evaluate(*args)', method_object=eval('export_model'), object_signature=None, function_args=[eval('raw_test_ds')], function_kwargs={}, custom_class=None)
(loss, accuracy) = export_model.evaluate(raw_test_ds)
print(accuracy)
examples = ['The movie was great!', 'The movie was okay.', 'The movie was terrible...']
custom_method(imports='import shutil;import tensorflow as tf;import re;import string;from tensorflow.keras import layers;import matplotlib.pyplot as plt;from tensorflow.keras import losses;import os', function_to_run='obj.predict(*args)', method_object=eval('export_model'), object_signature=None, function_args=[eval('examples')], function_kwargs={}, custom_class=None)
export_model.predict(examples)
