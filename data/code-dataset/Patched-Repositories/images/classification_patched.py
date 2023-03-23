import matplotlib.pyplot as plt
import numpy as np
import PIL
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
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
    result = send_request(imports=imports, function_to_run=function_to_run, function_args=function_args, function_kwargs=function_kwargs, max_wait_secs=max_wait_secs, wait_after_run_secs=wait_after_run_secs, method_object=method_object, object_signature=object_signature, custom_class=custom_class, experiment_file_path=EXPERIMENT_FILE_PATH)
    return func
dataset_url = 'https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz'
data_dir = custom_method(
tf.keras.utils.get_file('flower_photos', origin=dataset_url, untar=True), imports='import matplotlib.pyplot as plt;import pathlib;from tensorflow.keras.models import Sequential;from tensorflow import keras;from tensorflow.keras import layers;import numpy as np;import PIL;import tensorflow as tf', function_to_run='tf.keras.utils.get_file(*args, **kwargs)', method_object=None, object_signature=None, function_args=[eval("'flower_photos'")], function_kwargs={'origin': eval('dataset_url'), 'untar': eval('True')})
data_dir = pathlib.Path(data_dir)
image_count = len(list(data_dir.glob('*/*.jpg')))
print(image_count)
roses = list(data_dir.glob('roses/*'))
PIL.Image.open(str(roses[0]))
PIL.Image.open(str(roses[1]))
tulips = list(data_dir.glob('tulips/*'))
PIL.Image.open(str(tulips[0]))
PIL.Image.open(str(tulips[1]))
batch_size = 32
img_height = 180
img_width = 180
train_ds = custom_method(
tf.keras.utils.image_dataset_from_directory(data_dir, validation_split=0.2, subset='training', seed=123, image_size=(img_height, img_width), batch_size=batch_size), imports='import matplotlib.pyplot as plt;import pathlib;from tensorflow.keras.models import Sequential;from tensorflow import keras;from tensorflow.keras import layers;import numpy as np;import PIL;import tensorflow as tf', function_to_run='tf.keras.utils.image_dataset_from_directory(*args, **kwargs)', method_object=None, object_signature=None, function_args=[eval('data_dir')], function_kwargs={'validation_split': eval('0.2'), 'subset': eval('"training"'), 'seed': eval('123'), 'image_size': eval('(img_height, img_width)'), 'batch_size': eval('batch_size')})
val_ds = custom_method(
tf.keras.utils.image_dataset_from_directory(data_dir, validation_split=0.2, subset='validation', seed=123, image_size=(img_height, img_width), batch_size=batch_size), imports='import matplotlib.pyplot as plt;import pathlib;from tensorflow.keras.models import Sequential;from tensorflow import keras;from tensorflow.keras import layers;import numpy as np;import PIL;import tensorflow as tf', function_to_run='tf.keras.utils.image_dataset_from_directory(*args, **kwargs)', method_object=None, object_signature=None, function_args=[eval('data_dir')], function_kwargs={'validation_split': eval('0.2'), 'subset': eval('"validation"'), 'seed': eval('123'), 'image_size': eval('(img_height, img_width)'), 'batch_size': eval('batch_size')})
class_names = train_ds.class_names
print(class_names)
import matplotlib.pyplot as plt
plt.figure(figsize=(10, 10))
for (images, labels) in custom_method(
train_ds.take(1), imports='import matplotlib.pyplot as plt;import pathlib;from tensorflow.keras.models import Sequential;from tensorflow import keras;from tensorflow.keras import layers;import numpy as np;import PIL;import tensorflow as tf', function_to_run='obj.take(*args)', method_object=eval('train_ds'), object_signature='tf.keras.utils.image_dataset_from_directory', function_args=[eval('1')], function_kwargs={}, custom_class=None):
    for i in range(9):
        ax = plt.subplot(3, 3, i + 1)
        plt.imshow(images[i].numpy().astype('uint8'))
        plt.title(class_names[labels[i]])
        plt.axis('off')
for (image_batch, labels_batch) in train_ds:
    print(image_batch.shape)
    print(labels_batch.shape)
    break
AUTOTUNE = tf.data.AUTOTUNE
train_ds = custom_method(
train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE), imports='import matplotlib.pyplot as plt;import pathlib;from tensorflow.keras.models import Sequential;from tensorflow import keras;from tensorflow.keras import layers;import numpy as np;import PIL;import tensorflow as tf', function_to_run='obj.cache().shuffle(1000).prefetch(**kwargs)', method_object=eval('train_ds'), object_signature='tf.keras.utils.image_dataset_from_directory', function_args=[], function_kwargs={'buffer_size': eval('AUTOTUNE')}, custom_class=None)
val_ds = custom_method(
val_ds.cache().prefetch(buffer_size=AUTOTUNE), imports='import matplotlib.pyplot as plt;import pathlib;from tensorflow.keras.models import Sequential;from tensorflow import keras;from tensorflow.keras import layers;import numpy as np;import PIL;import tensorflow as tf', function_to_run='obj.cache().prefetch(**kwargs)', method_object=eval('val_ds'), object_signature='tf.keras.utils.image_dataset_from_directory', function_args=[], function_kwargs={'buffer_size': eval('AUTOTUNE')}, custom_class=None)
normalization_layer = custom_method(
layers.Rescaling(1.0 / 255), imports='import matplotlib.pyplot as plt;import pathlib;from tensorflow.keras.models import Sequential;from tensorflow import keras;from tensorflow.keras import layers;import numpy as np;import PIL;import tensorflow as tf', function_to_run='layers.Rescaling(*args)', method_object=None, object_signature=None, function_args=[eval('1./255')], function_kwargs={})
normalized_ds = custom_method(
train_ds.map(lambda x, y: (normalization_layer(x), y)), imports='import matplotlib.pyplot as plt;import pathlib;from tensorflow.keras.models import Sequential;from tensorflow import keras;from tensorflow.keras import layers;import numpy as np;import PIL;import tensorflow as tf', function_to_run='obj.map(*args)', method_object=eval('train_ds'), object_signature='tf.keras.utils.image_dataset_from_directory', function_args=[eval('lambda x, y: (normalization_layer(x), y)')], function_kwargs={}, custom_class=None)
(image_batch, labels_batch) = next(iter(normalized_ds))
first_image = image_batch[0]
print(np.min(first_image), np.max(first_image))
num_classes = len(class_names)
model = custom_method(
Sequential([layers.Rescaling(1.0 / 255, input_shape=(img_height, img_width, 3)), layers.Conv2D(16, 3, padding='same', activation='relu'), layers.MaxPooling2D(), layers.Conv2D(32, 3, padding='same', activation='relu'), layers.MaxPooling2D(), layers.Conv2D(64, 3, padding='same', activation='relu'), layers.MaxPooling2D(), layers.Flatten(), layers.Dense(128, activation='relu'), layers.Dense(num_classes)]), imports='import matplotlib.pyplot as plt;import pathlib;from tensorflow.keras.models import Sequential;from tensorflow import keras;from tensorflow.keras import layers;import numpy as np;import PIL;import tensorflow as tf', function_to_run='Sequential(*args)', method_object=None, object_signature=None, function_args=[eval("[\n  layers.Rescaling(1./255, input_shape=(img_height, img_width, 3)),\n  layers.Conv2D(16, 3, padding='same', activation='relu'),\n  layers.MaxPooling2D(),\n  layers.Conv2D(32, 3, padding='same', activation='relu'),\n  layers.MaxPooling2D(),\n  layers.Conv2D(64, 3, padding='same', activation='relu'),\n  layers.MaxPooling2D(),\n  layers.Flatten(),\n  layers.Dense(128, activation='relu'),\n  layers.Dense(num_classes)\n]")], function_kwargs={})
custom_method(
model.compile(optimizer='adam', loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=['accuracy']), imports='import matplotlib.pyplot as plt;import pathlib;from tensorflow.keras.models import Sequential;from tensorflow import keras;from tensorflow.keras import layers;import numpy as np;import PIL;import tensorflow as tf', function_to_run='obj.compile(**kwargs)', method_object=eval('model'), object_signature='Sequential', function_args=[], function_kwargs={'optimizer': eval("'adam'"), 'loss': eval('tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)'), 'metrics': eval("['accuracy']")}, custom_class=None)
custom_method(
model.summary(), imports='import matplotlib.pyplot as plt;import pathlib;from tensorflow.keras.models import Sequential;from tensorflow import keras;from tensorflow.keras import layers;import numpy as np;import PIL;import tensorflow as tf', function_to_run='obj.summary()', method_object=eval('model'), object_signature='Sequential', function_args=[], function_kwargs={}, custom_class=None)
epochs = 10
history = custom_method(
model.fit(train_ds, validation_data=val_ds, epochs=epochs), imports='import matplotlib.pyplot as plt;import pathlib;from tensorflow.keras.models import Sequential;from tensorflow import keras;from tensorflow.keras import layers;import numpy as np;import PIL;import tensorflow as tf', function_to_run='obj.fit(*args, **kwargs)', method_object=eval('model'), object_signature='Sequential', function_args=[eval('train_ds')], function_kwargs={'validation_data': eval('val_ds'), 'epochs': eval('epochs')}, custom_class=None)
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs_range = range(epochs)
plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')
plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()
data_augmentation = custom_method(
keras.Sequential([layers.RandomFlip('horizontal', input_shape=(img_height, img_width, 3)), layers.RandomRotation(0.1), layers.RandomZoom(0.1)]), imports='import matplotlib.pyplot as plt;import pathlib;from tensorflow.keras.models import Sequential;from tensorflow import keras;from tensorflow.keras import layers;import numpy as np;import PIL;import tensorflow as tf', function_to_run='keras.Sequential(*args)', method_object=None, object_signature=None, function_args=[eval('[\n    layers.RandomFlip("horizontal",\n                      input_shape=(img_height,\n                                  img_width,\n                                  3)),\n    layers.RandomRotation(0.1),\n    layers.RandomZoom(0.1),\n  ]')], function_kwargs={})
plt.figure(figsize=(10, 10))
for (images, _) in custom_method(
train_ds.take(1), imports='import matplotlib.pyplot as plt;import pathlib;from tensorflow.keras.models import Sequential;from tensorflow import keras;from tensorflow.keras import layers;import numpy as np;import PIL;import tensorflow as tf', function_to_run='obj.take(*args)', method_object=eval('train_ds'), object_signature='tf.keras.utils.image_dataset_from_directory', function_args=[eval('1')], function_kwargs={}, custom_class=None):
    for i in range(9):
        augmented_images = custom_method(
        data_augmentation(images), imports='import matplotlib.pyplot as plt;import pathlib;from tensorflow.keras.models import Sequential;from tensorflow import keras;from tensorflow.keras import layers;import numpy as np;import PIL;import tensorflow as tf', function_to_run='obj(*args)', method_object=eval('data_augmentation'), object_signature='keras.Sequential', function_args=[eval('images')], function_kwargs={}, custom_class=None)
        ax = plt.subplot(3, 3, i + 1)
        plt.imshow(augmented_images[0].numpy().astype('uint8'))
        plt.axis('off')
model = custom_method(
Sequential([data_augmentation, layers.Rescaling(1.0 / 255), layers.Conv2D(16, 3, padding='same', activation='relu'), layers.MaxPooling2D(), layers.Conv2D(32, 3, padding='same', activation='relu'), layers.MaxPooling2D(), layers.Conv2D(64, 3, padding='same', activation='relu'), layers.MaxPooling2D(), layers.Dropout(0.2), layers.Flatten(), layers.Dense(128, activation='relu'), layers.Dense(num_classes, name='outputs')]), imports='import matplotlib.pyplot as plt;import pathlib;from tensorflow.keras.models import Sequential;from tensorflow import keras;from tensorflow.keras import layers;import numpy as np;import PIL;import tensorflow as tf', function_to_run='Sequential(*args)', method_object=None, object_signature=None, function_args=[eval('[\n  data_augmentation,\n  layers.Rescaling(1./255),\n  layers.Conv2D(16, 3, padding=\'same\', activation=\'relu\'),\n  layers.MaxPooling2D(),\n  layers.Conv2D(32, 3, padding=\'same\', activation=\'relu\'),\n  layers.MaxPooling2D(),\n  layers.Conv2D(64, 3, padding=\'same\', activation=\'relu\'),\n  layers.MaxPooling2D(),\n  layers.Dropout(0.2),\n  layers.Flatten(),\n  layers.Dense(128, activation=\'relu\'),\n  layers.Dense(num_classes, name="outputs")\n]')], function_kwargs={})
custom_method(
model.compile(optimizer='adam', loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=['accuracy']), imports='import matplotlib.pyplot as plt;import pathlib;from tensorflow.keras.models import Sequential;from tensorflow import keras;from tensorflow.keras import layers;import numpy as np;import PIL;import tensorflow as tf', function_to_run='obj.compile(**kwargs)', method_object=eval('model'), object_signature='Sequential', function_args=[], function_kwargs={'optimizer': eval("'adam'"), 'loss': eval('tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)'), 'metrics': eval("['accuracy']")}, custom_class=None)
custom_method(
model.summary(), imports='import matplotlib.pyplot as plt;import pathlib;from tensorflow.keras.models import Sequential;from tensorflow import keras;from tensorflow.keras import layers;import numpy as np;import PIL;import tensorflow as tf', function_to_run='obj.summary()', method_object=eval('model'), object_signature='Sequential', function_args=[], function_kwargs={}, custom_class=None)
epochs = 15
history = custom_method(
model.fit(train_ds, validation_data=val_ds, epochs=epochs), imports='import matplotlib.pyplot as plt;import pathlib;from tensorflow.keras.models import Sequential;from tensorflow import keras;from tensorflow.keras import layers;import numpy as np;import PIL;import tensorflow as tf', function_to_run='obj.fit(*args, **kwargs)', method_object=eval('model'), object_signature='Sequential', function_args=[eval('train_ds')], function_kwargs={'validation_data': eval('val_ds'), 'epochs': eval('epochs')}, custom_class=None)
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs_range = range(epochs)
plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')
plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()
sunflower_url = 'https://storage.googleapis.com/download.tensorflow.org/example_images/592px-Red_sunflower.jpg'
sunflower_path = custom_method(
tf.keras.utils.get_file('Red_sunflower', origin=sunflower_url), imports='import matplotlib.pyplot as plt;import pathlib;from tensorflow.keras.models import Sequential;from tensorflow import keras;from tensorflow.keras import layers;import numpy as np;import PIL;import tensorflow as tf', function_to_run='tf.keras.utils.get_file(*args, **kwargs)', method_object=None, object_signature=None, function_args=[eval("'Red_sunflower'")], function_kwargs={'origin': eval('sunflower_url')})
img = custom_method(
tf.keras.utils.load_img(sunflower_path, target_size=(img_height, img_width)), imports='import matplotlib.pyplot as plt;import pathlib;from tensorflow.keras.models import Sequential;from tensorflow import keras;from tensorflow.keras import layers;import numpy as np;import PIL;import tensorflow as tf', function_to_run='tf.keras.utils.load_img(*args, **kwargs)', method_object=None, object_signature=None, function_args=[eval('sunflower_path')], function_kwargs={'target_size': eval('(img_height, img_width)')})
img_array = custom_method(
tf.keras.utils.img_to_array(img), imports='import matplotlib.pyplot as plt;import pathlib;from tensorflow.keras.models import Sequential;from tensorflow import keras;from tensorflow.keras import layers;import numpy as np;import PIL;import tensorflow as tf', function_to_run='tf.keras.utils.img_to_array(*args)', method_object=None, object_signature=None, function_args=[eval('img')], function_kwargs={})
img_array = custom_method(
tf.expand_dims(img_array, 0), imports='import matplotlib.pyplot as plt;import pathlib;from tensorflow.keras.models import Sequential;from tensorflow import keras;from tensorflow.keras import layers;import numpy as np;import PIL;import tensorflow as tf', function_to_run='tf.expand_dims(*args)', method_object=None, object_signature=None, function_args=[eval('img_array'), eval('0')], function_kwargs={})
predictions = custom_method(
model.predict(img_array), imports='import matplotlib.pyplot as plt;import pathlib;from tensorflow.keras.models import Sequential;from tensorflow import keras;from tensorflow.keras import layers;import numpy as np;import PIL;import tensorflow as tf', function_to_run='obj.predict(*args)', method_object=eval('model'), object_signature='Sequential', function_args=[eval('img_array')], function_kwargs={}, custom_class=None)
score = custom_method(
tf.nn.softmax(predictions[0]), imports='import matplotlib.pyplot as plt;import pathlib;from tensorflow.keras.models import Sequential;from tensorflow import keras;from tensorflow.keras import layers;import numpy as np;import PIL;import tensorflow as tf', function_to_run='tf.nn.softmax(*args)', method_object=None, object_signature=None, function_args=[eval('predictions[0]')], function_kwargs={})
print('This image most likely belongs to {} with a {:.2f} percent confidence.'.format(class_names[np.argmax(score)], 100 * np.max(score)))
converter = custom_method(
tf.lite.TFLiteConverter.from_keras_model(model), imports='import matplotlib.pyplot as plt;import pathlib;from tensorflow.keras.models import Sequential;from tensorflow import keras;from tensorflow.keras import layers;import numpy as np;import PIL;import tensorflow as tf', function_to_run='tf.lite.TFLiteConverter.from_keras_model(*args)', method_object=None, object_signature=None, function_args=[eval('model')], function_kwargs={})
tflite_model = custom_method(
converter.convert(), imports='import matplotlib.pyplot as plt;import pathlib;from tensorflow.keras.models import Sequential;from tensorflow import keras;from tensorflow.keras import layers;import numpy as np;import PIL;import tensorflow as tf', function_to_run='obj.convert()', method_object=eval('converter'), object_signature='tf.lite.TFLiteConverter.from_keras_model', function_args=[], function_kwargs={}, custom_class=None)
with open('model.tflite', 'wb') as f:
    f.write(tflite_model)
TF_MODEL_FILE_PATH = 'model.tflite'
interpreter = custom_method(
tf.lite.Interpreter(model_path=TF_MODEL_FILE_PATH), imports='import matplotlib.pyplot as plt;import pathlib;from tensorflow.keras.models import Sequential;from tensorflow import keras;from tensorflow.keras import layers;import numpy as np;import PIL;import tensorflow as tf', function_to_run='tf.lite.Interpreter(**kwargs)', method_object=None, object_signature=None, function_args=[], function_kwargs={'model_path': eval('TF_MODEL_FILE_PATH')})
custom_method(
interpreter.get_signature_list(), imports='import matplotlib.pyplot as plt;import pathlib;from tensorflow.keras.models import Sequential;from tensorflow import keras;from tensorflow.keras import layers;import numpy as np;import PIL;import tensorflow as tf', function_to_run='obj.get_signature_list()', method_object=eval('interpreter'), object_signature='tf.lite.Interpreter', function_args=[], function_kwargs={}, custom_class=None)
classify_lite = custom_method(
interpreter.get_signature_runner('serving_default'), imports='import matplotlib.pyplot as plt;import pathlib;from tensorflow.keras.models import Sequential;from tensorflow import keras;from tensorflow.keras import layers;import numpy as np;import PIL;import tensorflow as tf', function_to_run='obj.get_signature_runner(*args)', method_object=eval('interpreter'), object_signature='tf.lite.Interpreter', function_args=[eval("'serving_default'")], function_kwargs={}, custom_class=None)
classify_lite
predictions_lite = classify_lite(sequential_1_input=img_array)['outputs']
score_lite = custom_method(
tf.nn.softmax(predictions_lite), imports='import matplotlib.pyplot as plt;import pathlib;from tensorflow.keras.models import Sequential;from tensorflow import keras;from tensorflow.keras import layers;import numpy as np;import PIL;import tensorflow as tf', function_to_run='tf.nn.softmax(*args)', method_object=None, object_signature=None, function_args=[eval('predictions_lite')], function_kwargs={})
print('This image most likely belongs to {} with a {:.2f} percent confidence.'.format(class_names[np.argmax(score_lite)], 100 * np.max(score_lite)))
print(np.max(np.abs(predictions - predictions_lite)))
