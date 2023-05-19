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

def custom_method(imports: str, function_to_run: str, method_object=None, object_signature=None, function_args: list=None, function_kwargs: dict=None, custom_class=None):
    result = send_request(imports=imports, function_to_run=function_to_run, function_args=function_args, function_kwargs=function_kwargs, max_wait_secs=MAX_WAIT_S, wait_after_run_secs=WAIT_AFTER_RUN_S, method_object=method_object, object_signature=object_signature, custom_class=custom_class, experiment_file_path=EXPERIMENT_FILE_PATH)
print(tf.__version__)
import pathlib
dataset_url = 'https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz'
custom_method(imports='import pathlib;import os;import tensorflow_datasets as tfds;import matplotlib.pyplot as plt;import tensorflow as tf;import PIL;import PIL.Image;import numpy as np', function_to_run='tf.keras.utils.get_file(**kwargs)', method_object=None, object_signature=None, function_args=[], function_kwargs={'origin': eval('dataset_url'), 'extract': eval('True')})
archive = tf.keras.utils.get_file(origin=dataset_url, extract=True)
data_dir = pathlib.Path(archive).with_suffix('')
image_count = len(list(data_dir.glob('*/*.jpg')))
print(image_count)
roses = list(data_dir.glob('roses/*'))
PIL.Image.open(str(roses[0]))
roses = list(data_dir.glob('roses/*'))
PIL.Image.open(str(roses[1]))
batch_size = 32
img_height = 180
img_width = 180
custom_method(imports='import pathlib;import os;import tensorflow_datasets as tfds;import matplotlib.pyplot as plt;import tensorflow as tf;import PIL;import PIL.Image;import numpy as np', function_to_run='tf.keras.utils.image_dataset_from_directory(*args, **kwargs)', method_object=None, object_signature=None, function_args=[eval('data_dir')], function_kwargs={'validation_split': eval('0.2'), 'subset': eval('"training"'), 'seed': eval('123'), 'image_size': eval('(img_height, img_width)'), 'batch_size': eval('batch_size')})
train_ds = tf.keras.utils.image_dataset_from_directory(data_dir, validation_split=0.2, subset='training', seed=123, image_size=(img_height, img_width), batch_size=batch_size)
custom_method(imports='import pathlib;import os;import tensorflow_datasets as tfds;import matplotlib.pyplot as plt;import tensorflow as tf;import PIL;import PIL.Image;import numpy as np', function_to_run='tf.keras.utils.image_dataset_from_directory(*args, **kwargs)', method_object=None, object_signature=None, function_args=[eval('data_dir')], function_kwargs={'validation_split': eval('0.2'), 'subset': eval('"validation"'), 'seed': eval('123'), 'image_size': eval('(img_height, img_width)'), 'batch_size': eval('batch_size')})
val_ds = tf.keras.utils.image_dataset_from_directory(data_dir, validation_split=0.2, subset='validation', seed=123, image_size=(img_height, img_width), batch_size=batch_size)
class_names = train_ds.class_names
print(class_names)
import matplotlib.pyplot as plt
plt.figure(figsize=(10, 10))
for (images, labels) in train_ds.take(1):
    for i in range(9):
        ax = plt.subplot(3, 3, i + 1)
        plt.imshow(images[i].numpy().astype('uint8'))
        plt.title(class_names[labels[i]])
        plt.axis('off')
for (image_batch, labels_batch) in train_ds:
    print(image_batch.shape)
    print(labels_batch.shape)
    break
custom_method(imports='import pathlib;import os;import tensorflow_datasets as tfds;import matplotlib.pyplot as plt;import tensorflow as tf;import PIL;import PIL.Image;import numpy as np', function_to_run='tf.keras.layers.Rescaling(*args)', method_object=None, object_signature=None, function_args=[eval('1./255')], function_kwargs={})
normalization_layer = tf.keras.layers.Rescaling(1.0 / 255)
custom_method(imports='import pathlib;import os;import tensorflow_datasets as tfds;import matplotlib.pyplot as plt;import tensorflow as tf;import PIL;import PIL.Image;import numpy as np', function_to_run='obj.map(*args)', method_object=eval('train_ds'), object_signature=None, function_args=[eval('lambda x, y: (normalization_layer(x), y)')], function_kwargs={}, custom_class=None)
normalized_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
(image_batch, labels_batch) = next(iter(normalized_ds))
first_image = image_batch[0]
print(np.min(first_image), np.max(first_image))
AUTOTUNE = tf.data.AUTOTUNE
custom_method(imports='import pathlib;import os;import tensorflow_datasets as tfds;import matplotlib.pyplot as plt;import tensorflow as tf;import PIL;import PIL.Image;import numpy as np', function_to_run='obj.cache().prefetch(**kwargs)', method_object=eval('train_ds'), object_signature=None, function_args=[], function_kwargs={'buffer_size': eval('AUTOTUNE')}, custom_class=None)
train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
custom_method(imports='import pathlib;import os;import tensorflow_datasets as tfds;import matplotlib.pyplot as plt;import tensorflow as tf;import PIL;import PIL.Image;import numpy as np', function_to_run='obj.cache().prefetch(**kwargs)', method_object=eval('val_ds'), object_signature=None, function_args=[], function_kwargs={'buffer_size': eval('AUTOTUNE')}, custom_class=None)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)
num_classes = 5
custom_method(imports='import pathlib;import os;import tensorflow_datasets as tfds;import matplotlib.pyplot as plt;import tensorflow as tf;import PIL;import PIL.Image;import numpy as np', function_to_run='tf.keras.Sequential(*args)', method_object=None, object_signature=None, function_args=[eval("[\n  tf.keras.layers.Rescaling(1./255),\n  tf.keras.layers.Conv2D(32, 3, activation='relu'),\n  tf.keras.layers.MaxPooling2D(),\n  tf.keras.layers.Conv2D(32, 3, activation='relu'),\n  tf.keras.layers.MaxPooling2D(),\n  tf.keras.layers.Conv2D(32, 3, activation='relu'),\n  tf.keras.layers.MaxPooling2D(),\n  tf.keras.layers.Flatten(),\n  tf.keras.layers.Dense(128, activation='relu'),\n  tf.keras.layers.Dense(num_classes)\n]")], function_kwargs={})
model = tf.keras.Sequential([tf.keras.layers.Rescaling(1.0 / 255), tf.keras.layers.Conv2D(32, 3, activation='relu'), tf.keras.layers.MaxPooling2D(), tf.keras.layers.Conv2D(32, 3, activation='relu'), tf.keras.layers.MaxPooling2D(), tf.keras.layers.Conv2D(32, 3, activation='relu'), tf.keras.layers.MaxPooling2D(), tf.keras.layers.Flatten(), tf.keras.layers.Dense(128, activation='relu'), tf.keras.layers.Dense(num_classes)])
custom_method(imports='import pathlib;import os;import tensorflow_datasets as tfds;import matplotlib.pyplot as plt;import tensorflow as tf;import PIL;import PIL.Image;import numpy as np', function_to_run='obj.compile(**kwargs)', method_object=eval('model'), object_signature=None, function_args=[], function_kwargs={'optimizer': eval("'adam'"), 'loss': eval('tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)'), 'metrics': eval("['accuracy']")}, custom_class=None)
model.compile(optimizer='adam', loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=['accuracy'])
custom_method(imports='import pathlib;import os;import tensorflow_datasets as tfds;import matplotlib.pyplot as plt;import tensorflow as tf;import PIL;import PIL.Image;import numpy as np', function_to_run='obj.fit(*args, **kwargs)', method_object=eval('model'), object_signature=None, function_args=[eval('train_ds')], function_kwargs={'validation_data': eval('val_ds'), 'epochs': eval('3')}, custom_class=None)
model.fit(train_ds, validation_data=val_ds, epochs=3)
custom_method(imports='import pathlib;import os;import tensorflow_datasets as tfds;import matplotlib.pyplot as plt;import tensorflow as tf;import PIL;import PIL.Image;import numpy as np', function_to_run='tf.data.Dataset.list_files(*args, **kwargs)', method_object=None, object_signature=None, function_args=[eval("str(data_dir/'*/*')")], function_kwargs={'shuffle': eval('False')})
list_ds = tf.data.Dataset.list_files(str(data_dir / '*/*'), shuffle=False)
custom_method(imports='import pathlib;import os;import tensorflow_datasets as tfds;import matplotlib.pyplot as plt;import tensorflow as tf;import PIL;import PIL.Image;import numpy as np', function_to_run='obj.shuffle(*args, **kwargs)', method_object=eval('list_ds'), object_signature=None, function_args=[eval('image_count')], function_kwargs={'reshuffle_each_iteration': eval('False')}, custom_class=None)
list_ds = list_ds.shuffle(image_count, reshuffle_each_iteration=False)
for f in list_ds.take(5):
    print(f.numpy())
class_names = np.array(sorted([item.name for item in data_dir.glob('*') if item.name != 'LICENSE.txt']))
print(class_names)
val_size = int(image_count * 0.2)
custom_method(imports='import pathlib;import os;import tensorflow_datasets as tfds;import matplotlib.pyplot as plt;import tensorflow as tf;import PIL;import PIL.Image;import numpy as np', function_to_run='obj.skip(*args)', method_object=eval('list_ds'), object_signature=None, function_args=[eval('val_size')], function_kwargs={}, custom_class=None)
train_ds = list_ds.skip(val_size)
custom_method(imports='import pathlib;import os;import tensorflow_datasets as tfds;import matplotlib.pyplot as plt;import tensorflow as tf;import PIL;import PIL.Image;import numpy as np', function_to_run='obj.take(*args)', method_object=eval('list_ds'), object_signature=None, function_args=[eval('val_size')], function_kwargs={}, custom_class=None)
val_ds = list_ds.take(val_size)
print(tf.data.experimental.cardinality(train_ds).numpy())
print(tf.data.experimental.cardinality(val_ds).numpy())

def get_label(file_path):
    custom_method(imports='import pathlib;import os;import tensorflow_datasets as tfds;import matplotlib.pyplot as plt;import tensorflow as tf;import PIL;import PIL.Image;import numpy as np', function_to_run='tf.strings.split(*args)', method_object=None, object_signature=None, function_args=[eval('file_path'), eval('os.path.sep')], function_kwargs={})
    parts = tf.strings.split(file_path, os.path.sep)
    one_hot = parts[-2] == class_names
    return tf.argmax(one_hot)

def decode_img(img):
    custom_method(imports='import pathlib;import os;import tensorflow_datasets as tfds;import matplotlib.pyplot as plt;import tensorflow as tf;import PIL;import PIL.Image;import numpy as np', function_to_run='tf.io.decode_jpeg(*args, **kwargs)', method_object=None, object_signature=None, function_args=[eval('img')], function_kwargs={'channels': eval('3')})
    img = tf.io.decode_jpeg(img, channels=3)
    return tf.image.resize(img, [img_height, img_width])

def process_path(file_path):
    label = get_label(file_path)
    custom_method(imports='import pathlib;import os;import tensorflow_datasets as tfds;import matplotlib.pyplot as plt;import tensorflow as tf;import PIL;import PIL.Image;import numpy as np', function_to_run='tf.io.read_file(*args)', method_object=None, object_signature=None, function_args=[eval('file_path')], function_kwargs={})
    img = tf.io.read_file(file_path)
    img = decode_img(img)
    return (img, label)
custom_method(imports='import pathlib;import os;import tensorflow_datasets as tfds;import matplotlib.pyplot as plt;import tensorflow as tf;import PIL;import PIL.Image;import numpy as np', function_to_run='obj.map(*args, **kwargs)', method_object=eval('train_ds'), object_signature=None, function_args=[eval('process_path')], function_kwargs={'num_parallel_calls': eval('AUTOTUNE')}, custom_class=None)
train_ds = train_ds.map(process_path, num_parallel_calls=AUTOTUNE)
custom_method(imports='import pathlib;import os;import tensorflow_datasets as tfds;import matplotlib.pyplot as plt;import tensorflow as tf;import PIL;import PIL.Image;import numpy as np', function_to_run='obj.map(*args, **kwargs)', method_object=eval('val_ds'), object_signature=None, function_args=[eval('process_path')], function_kwargs={'num_parallel_calls': eval('AUTOTUNE')}, custom_class=None)
val_ds = val_ds.map(process_path, num_parallel_calls=AUTOTUNE)
for (image, label) in train_ds.take(1):
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
custom_method(imports='import pathlib;import os;import tensorflow_datasets as tfds;import matplotlib.pyplot as plt;import tensorflow as tf;import PIL;import PIL.Image;import numpy as np', function_to_run='obj.fit(*args, **kwargs)', method_object=eval('model'), object_signature=None, function_args=[eval('train_ds')], function_kwargs={'validation_data': eval('val_ds'), 'epochs': eval('3')}, custom_class=None)
model.fit(train_ds, validation_data=val_ds, epochs=3)
((train_ds, val_ds, test_ds), metadata) = tfds.load('tf_flowers', split=['train[:80%]', 'train[80%:90%]', 'train[90%:]'], with_info=True, as_supervised=True)
num_classes = metadata.features['label'].num_classes
print(num_classes)
get_label_name = metadata.features['label'].int2str
(image, label) = next(iter(train_ds))
_ = plt.imshow(image)
_ = plt.title(get_label_name(label))
train_ds = configure_for_performance(train_ds)
val_ds = configure_for_performance(val_ds)
test_ds = configure_for_performance(test_ds)
