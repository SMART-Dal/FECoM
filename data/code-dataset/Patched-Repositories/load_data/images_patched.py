import numpy as np
import os
import PIL
import PIL.Image
import tensorflow as tf
import tensorflow_datasets as tfds
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

def custom_method(func, imports: str, function_to_run: str, method_object=None, function_args: list=None, function_kwargs: dict=None, max_wait_secs=MAX_WAIT_S, custom_class=None, wait_after_run_secs=WAIT_AFTER_RUN_S):
    result = send_request(imports=imports, function_to_run=function_to_run, function_args=function_args, function_kwargs=function_kwargs, max_wait_secs=max_wait_secs, wait_after_run_secs=wait_after_run_secs, method_object=method_object, custom_class=custom_class, experiment_file_path=EXPERIMENT_FILE_PATH)
    return func
if __name__ == '__main__':
    print(EXPERIMENT_FILE_PATH)
print(tf.__version__)
import pathlib
dataset_url = 'https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz'
archive = custom_method(
tf.keras.utils.get_file(origin=dataset_url, extract=True), imports='import numpy as np;import PIL;import pathlib;import matplotlib.pyplot as plt;import PIL.Image;import tensorflow as tf;import os;import tensorflow_datasets as tfds', function_to_run='tf.keras.utils.get_file(**kwargs)', method_object=None, function_args=[], function_kwargs={'origin': eval('dataset_url'), 'extract': eval('True')}, max_wait_secs=0)
data_dir = custom_method(
pathlib.Path(archive).with_suffix(''), imports='import numpy as np;import PIL;import pathlib;import matplotlib.pyplot as plt;import PIL.Image;import tensorflow as tf;import os;import tensorflow_datasets as tfds', function_to_run='pathlib.Path(obj).with_suffix(*args)', method_object=eval('archive'), function_args=[eval("''")], function_kwargs={}, max_wait_secs=0, custom_class=None)
image_count = len(list(data_dir.glob('*/*.jpg')))
print(image_count)
roses = list(data_dir.glob('roses/*'))
PIL.Image.open(str(roses[0]))
roses = list(data_dir.glob('roses/*'))
PIL.Image.open(str(roses[1]))
batch_size = 32
img_height = 180
img_width = 180
train_ds = custom_method(
tf.keras.utils.image_dataset_from_directory(data_dir, validation_split=0.2, subset='training', seed=123, image_size=(img_height, img_width), batch_size=batch_size), imports='import numpy as np;import PIL;import pathlib;import matplotlib.pyplot as plt;import PIL.Image;import tensorflow as tf;import os;import tensorflow_datasets as tfds', function_to_run='tf.keras.utils.image_dataset_from_directory(*args, **kwargs)', method_object=None, function_args=[eval('data_dir')], function_kwargs={'validation_split': eval('0.2'), 'subset': eval('"training"'), 'seed': eval('123'), 'image_size': eval('(img_height, img_width)'), 'batch_size': eval('batch_size')}, max_wait_secs=0)
val_ds = custom_method(
tf.keras.utils.image_dataset_from_directory(data_dir, validation_split=0.2, subset='validation', seed=123, image_size=(img_height, img_width), batch_size=batch_size), imports='import numpy as np;import PIL;import pathlib;import matplotlib.pyplot as plt;import PIL.Image;import tensorflow as tf;import os;import tensorflow_datasets as tfds', function_to_run='tf.keras.utils.image_dataset_from_directory(*args, **kwargs)', method_object=None, function_args=[eval('data_dir')], function_kwargs={'validation_split': eval('0.2'), 'subset': eval('"validation"'), 'seed': eval('123'), 'image_size': eval('(img_height, img_width)'), 'batch_size': eval('batch_size')}, max_wait_secs=0)
class_names = train_ds.class_names
print(class_names)
import matplotlib.pyplot as plt
plt.figure(figsize=(10, 10))
for (images, labels) in custom_method(
train_ds.take(1), imports='import numpy as np;import PIL;import pathlib;import matplotlib.pyplot as plt;import PIL.Image;import tensorflow as tf;import os;import tensorflow_datasets as tfds', function_to_run='obj.take(*args)', method_object=eval('train_ds'), function_args=[eval('1')], function_kwargs={}, max_wait_secs=0, custom_class=None):
    for i in range(9):
        ax = plt.subplot(3, 3, i + 1)
        plt.imshow(images[i].numpy().astype('uint8'))
        plt.title(class_names[labels[i]])
        plt.axis('off')
for (image_batch, labels_batch) in train_ds:
    print(image_batch.shape)
    print(labels_batch.shape)
    break
normalization_layer = custom_method(
tf.keras.layers.Rescaling(1.0 / 255), imports='import numpy as np;import PIL;import pathlib;import matplotlib.pyplot as plt;import PIL.Image;import tensorflow as tf;import os;import tensorflow_datasets as tfds', function_to_run='tf.keras.layers.Rescaling(*args)', method_object=None, function_args=[eval('1./255')], function_kwargs={}, max_wait_secs=0)
normalized_ds = custom_method(
train_ds.map(lambda x, y: (normalization_layer(x), y)), imports='import numpy as np;import PIL;import pathlib;import matplotlib.pyplot as plt;import PIL.Image;import tensorflow as tf;import os;import tensorflow_datasets as tfds', function_to_run='obj.map(*args)', method_object=eval('train_ds'), function_args=[eval('lambda x, y: (normalization_layer(x), y)')], function_kwargs={}, max_wait_secs=0, custom_class=None)
(image_batch, labels_batch) = next(iter(normalized_ds))
first_image = image_batch[0]
print(np.min(first_image), np.max(first_image))
AUTOTUNE = tf.data.AUTOTUNE
train_ds = custom_method(
train_ds.cache().prefetch(buffer_size=AUTOTUNE), imports='import numpy as np;import PIL;import pathlib;import matplotlib.pyplot as plt;import PIL.Image;import tensorflow as tf;import os;import tensorflow_datasets as tfds', function_to_run='obj.cache().prefetch(**kwargs)', method_object=eval('train_ds'), function_args=[], function_kwargs={'buffer_size': eval('AUTOTUNE')}, max_wait_secs=0, custom_class=None)
val_ds = custom_method(
val_ds.cache().prefetch(buffer_size=AUTOTUNE), imports='import numpy as np;import PIL;import pathlib;import matplotlib.pyplot as plt;import PIL.Image;import tensorflow as tf;import os;import tensorflow_datasets as tfds', function_to_run='obj.cache().prefetch(**kwargs)', method_object=eval('val_ds'), function_args=[], function_kwargs={'buffer_size': eval('AUTOTUNE')}, max_wait_secs=0, custom_class=None)
num_classes = 5
model = custom_method(
tf.keras.Sequential([tf.keras.layers.Rescaling(1.0 / 255), tf.keras.layers.Conv2D(32, 3, activation='relu'), tf.keras.layers.MaxPooling2D(), tf.keras.layers.Conv2D(32, 3, activation='relu'), tf.keras.layers.MaxPooling2D(), tf.keras.layers.Conv2D(32, 3, activation='relu'), tf.keras.layers.MaxPooling2D(), tf.keras.layers.Flatten(), tf.keras.layers.Dense(128, activation='relu'), tf.keras.layers.Dense(num_classes)]), imports='import numpy as np;import PIL;import pathlib;import matplotlib.pyplot as plt;import PIL.Image;import tensorflow as tf;import os;import tensorflow_datasets as tfds', function_to_run='tf.keras.Sequential(*args)', method_object=None, function_args=[eval("[\n  tf.keras.layers.Rescaling(1./255),\n  tf.keras.layers.Conv2D(32, 3, activation='relu'),\n  tf.keras.layers.MaxPooling2D(),\n  tf.keras.layers.Conv2D(32, 3, activation='relu'),\n  tf.keras.layers.MaxPooling2D(),\n  tf.keras.layers.Conv2D(32, 3, activation='relu'),\n  tf.keras.layers.MaxPooling2D(),\n  tf.keras.layers.Flatten(),\n  tf.keras.layers.Dense(128, activation='relu'),\n  tf.keras.layers.Dense(num_classes)\n]")], function_kwargs={}, max_wait_secs=0)
custom_method(
model.compile(optimizer='adam', loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=['accuracy']), imports='import numpy as np;import PIL;import pathlib;import matplotlib.pyplot as plt;import PIL.Image;import tensorflow as tf;import os;import tensorflow_datasets as tfds', function_to_run='obj.compile(**kwargs)', method_object=eval('model'), function_args=[], function_kwargs={'optimizer': eval("'adam'"), 'loss': eval('tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)'), 'metrics': eval("['accuracy']")}, max_wait_secs=0, custom_class=None)
custom_method(
model.fit(train_ds, validation_data=val_ds, epochs=3), imports='import numpy as np;import PIL;import pathlib;import matplotlib.pyplot as plt;import PIL.Image;import tensorflow as tf;import os;import tensorflow_datasets as tfds', function_to_run='obj.fit(*args, **kwargs)', method_object=eval('model'), function_args=[eval('train_ds')], function_kwargs={'validation_data': eval('val_ds'), 'epochs': eval('3')}, max_wait_secs=0, custom_class=None)
list_ds = custom_method(
tf.data.Dataset.list_files(str(data_dir / '*/*'), shuffle=False), imports='import numpy as np;import PIL;import pathlib;import matplotlib.pyplot as plt;import PIL.Image;import tensorflow as tf;import os;import tensorflow_datasets as tfds', function_to_run='tf.data.Dataset.list_files(*args, **kwargs)', method_object=None, function_args=[eval("str(data_dir/'*/*')")], function_kwargs={'shuffle': eval('False')}, max_wait_secs=0)
list_ds = custom_method(
list_ds.shuffle(image_count, reshuffle_each_iteration=False), imports='import numpy as np;import PIL;import pathlib;import matplotlib.pyplot as plt;import PIL.Image;import tensorflow as tf;import os;import tensorflow_datasets as tfds', function_to_run='obj.shuffle(*args, **kwargs)', method_object=eval('list_ds'), function_args=[eval('image_count')], function_kwargs={'reshuffle_each_iteration': eval('False')}, max_wait_secs=0, custom_class=None)
for f in custom_method(
list_ds.take(5), imports='import numpy as np;import PIL;import pathlib;import matplotlib.pyplot as plt;import PIL.Image;import tensorflow as tf;import os;import tensorflow_datasets as tfds', function_to_run='obj.take(*args)', method_object=eval('list_ds'), function_args=[eval('5')], function_kwargs={}, max_wait_secs=0, custom_class=None):
    print(f.numpy())
class_names = np.array(sorted([item.name for item in data_dir.glob('*') if item.name != 'LICENSE.txt']))
print(class_names)
val_size = int(image_count * 0.2)
train_ds = custom_method(
list_ds.skip(val_size), imports='import numpy as np;import PIL;import pathlib;import matplotlib.pyplot as plt;import PIL.Image;import tensorflow as tf;import os;import tensorflow_datasets as tfds', function_to_run='obj.skip(*args)', method_object=eval('list_ds'), function_args=[eval('val_size')], function_kwargs={}, max_wait_secs=0, custom_class=None)
val_ds = custom_method(
list_ds.take(val_size), imports='import numpy as np;import PIL;import pathlib;import matplotlib.pyplot as plt;import PIL.Image;import tensorflow as tf;import os;import tensorflow_datasets as tfds', function_to_run='obj.take(*args)', method_object=eval('list_ds'), function_args=[eval('val_size')], function_kwargs={}, max_wait_secs=0, custom_class=None)
print(tf.data.experimental.cardinality(train_ds).numpy())
print(tf.data.experimental.cardinality(val_ds).numpy())

def get_label(file_path):
    parts = custom_method(
    tf.strings.split(file_path, os.path.sep), imports='import numpy as np;import PIL;import pathlib;import matplotlib.pyplot as plt;import PIL.Image;import tensorflow as tf;import os;import tensorflow_datasets as tfds', function_to_run='tf.strings.split(*args)', method_object=None, function_args=[eval('file_path'), eval('os.path.sep')], function_kwargs={}, max_wait_secs=0)
    one_hot = parts[-2] == class_names
    return custom_method(
    tf.argmax(one_hot), imports='import numpy as np;import PIL;import pathlib;import matplotlib.pyplot as plt;import PIL.Image;import tensorflow as tf;import os;import tensorflow_datasets as tfds', function_to_run='tf.argmax(*args)', method_object=None, function_args=[eval('one_hot')], function_kwargs={}, max_wait_secs=0)

def decode_img(img):
    img = custom_method(
    tf.io.decode_jpeg(img, channels=3), imports='import numpy as np;import PIL;import pathlib;import matplotlib.pyplot as plt;import PIL.Image;import tensorflow as tf;import os;import tensorflow_datasets as tfds', function_to_run='tf.io.decode_jpeg(*args, **kwargs)', method_object=None, function_args=[eval('img')], function_kwargs={'channels': eval('3')}, max_wait_secs=0)
    return custom_method(
    tf.image.resize(img, [img_height, img_width]), imports='import numpy as np;import PIL;import pathlib;import matplotlib.pyplot as plt;import PIL.Image;import tensorflow as tf;import os;import tensorflow_datasets as tfds', function_to_run='tf.image.resize(*args)', method_object=None, function_args=[eval('img'), eval('[img_height, img_width]')], function_kwargs={}, max_wait_secs=0)

def process_path(file_path):
    label = get_label(file_path)
    img = custom_method(
    tf.io.read_file(file_path), imports='import numpy as np;import PIL;import pathlib;import matplotlib.pyplot as plt;import PIL.Image;import tensorflow as tf;import os;import tensorflow_datasets as tfds', function_to_run='tf.io.read_file(*args)', method_object=None, function_args=[eval('file_path')], function_kwargs={}, max_wait_secs=0)
    img = custom_method(
    decode_img(img), imports='import numpy as np;import PIL;import pathlib;import matplotlib.pyplot as plt;import PIL.Image;import tensorflow as tf;import os;import tensorflow_datasets as tfds', function_to_run='obj(*args)', method_object=eval('decode_img'), function_args=[eval('img')], function_kwargs={}, max_wait_secs=0, custom_class=None)
    return (img, label)
train_ds = custom_method(
train_ds.map(process_path, num_parallel_calls=AUTOTUNE), imports='import numpy as np;import PIL;import pathlib;import matplotlib.pyplot as plt;import PIL.Image;import tensorflow as tf;import os;import tensorflow_datasets as tfds', function_to_run='obj.map(*args, **kwargs)', method_object=eval('train_ds'), function_args=[eval('process_path')], function_kwargs={'num_parallel_calls': eval('AUTOTUNE')}, max_wait_secs=0, custom_class=None)
val_ds = custom_method(
val_ds.map(process_path, num_parallel_calls=AUTOTUNE), imports='import numpy as np;import PIL;import pathlib;import matplotlib.pyplot as plt;import PIL.Image;import tensorflow as tf;import os;import tensorflow_datasets as tfds', function_to_run='obj.map(*args, **kwargs)', method_object=eval('val_ds'), function_args=[eval('process_path')], function_kwargs={'num_parallel_calls': eval('AUTOTUNE')}, max_wait_secs=0, custom_class=None)
for (image, label) in custom_method(
train_ds.take(1), imports='import numpy as np;import PIL;import pathlib;import matplotlib.pyplot as plt;import PIL.Image;import tensorflow as tf;import os;import tensorflow_datasets as tfds', function_to_run='obj.take(*args)', method_object=eval('train_ds'), function_args=[eval('1')], function_kwargs={}, max_wait_secs=0, custom_class=None):
    print('Image shape: ', image.numpy().shape)
    print('Label: ', label.numpy())

def configure_for_performance(ds):
    ds = ds.cache()
    ds = ds.shuffle(buffer_size=1000)
    ds = ds.batch(batch_size)
    ds = ds.prefetch(buffer_size=AUTOTUNE)
    return ds
train_ds = configure_for_performance(train_ds)
val_ds = configure_for_performance(val_ds)
(image_batch, label_batch) = next(iter(train_ds))
plt.figure(figsize=(10, 10))
for i in range(9):
    ax = plt.subplot(3, 3, i + 1)
    plt.imshow(image_batch[i].numpy().astype('uint8'))
    label = label_batch[i]
    plt.title(class_names[label])
    plt.axis('off')
custom_method(
model.fit(train_ds, validation_data=val_ds, epochs=3), imports='import numpy as np;import PIL;import pathlib;import matplotlib.pyplot as plt;import PIL.Image;import tensorflow as tf;import os;import tensorflow_datasets as tfds', function_to_run='obj.fit(*args, **kwargs)', method_object=eval('model'), function_args=[eval('train_ds')], function_kwargs={'validation_data': eval('val_ds'), 'epochs': eval('3')}, max_wait_secs=0, custom_class=None)
((train_ds, val_ds, test_ds), metadata) = custom_method(
tfds.load('tf_flowers', split=['train[:80%]', 'train[80%:90%]', 'train[90%:]'], with_info=True, as_supervised=True), imports='import numpy as np;import PIL;import pathlib;import matplotlib.pyplot as plt;import PIL.Image;import tensorflow as tf;import os;import tensorflow_datasets as tfds', function_to_run='tfds.load(*args, **kwargs)', method_object=None, function_args=[eval("'tf_flowers'")], function_kwargs={'split': eval("['train[:80%]', 'train[80%:90%]', 'train[90%:]']"), 'with_info': eval('True'), 'as_supervised': eval('True')}, max_wait_secs=0)
num_classes = metadata.features['label'].num_classes
print(num_classes)
get_label_name = metadata.features['label'].int2str
(image, label) = next(iter(train_ds))
_ = plt.imshow(image)
_ = plt.title(get_label_name(label))
train_ds = configure_for_performance(train_ds)
val_ds = configure_for_performance(val_ds)
test_ds = configure_for_performance(test_ds)
