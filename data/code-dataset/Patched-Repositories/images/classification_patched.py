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
    result = send_request(imports=imports, function_to_run=function_to_run, function_args=function_args, function_kwargs=function_kwargs, max_wait_secs=MAX_WAIT_S, wait_after_run_secs=WAIT_AFTER_RUN_S, method_object=method_object, object_signature=object_signature, custom_class=custom_class, experiment_file_path=EXPERIMENT_FILE_PATH)
dataset_url = 'https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz'
custom_method(imports='import pathlib;import numpy as np;from tensorflow import keras;from tensorflow.keras import layers;from tensorflow.keras.models import Sequential;import matplotlib.pyplot as plt;import tensorflow as tf;import PIL', function_to_run='tf.keras.utils.get_file(*args, **kwargs)', method_object=None, object_signature=None, function_args=[eval("'flower_photos'")], function_kwargs={'origin': eval('dataset_url'), 'untar': eval('True')})
data_dir = tf.keras.utils.get_file('flower_photos', origin=dataset_url, untar=True)
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
custom_method(imports='import pathlib;import numpy as np;from tensorflow import keras;from tensorflow.keras import layers;from tensorflow.keras.models import Sequential;import matplotlib.pyplot as plt;import tensorflow as tf;import PIL', function_to_run='tf.keras.utils.image_dataset_from_directory(*args, **kwargs)', method_object=None, object_signature=None, function_args=[eval('data_dir')], function_kwargs={'validation_split': eval('0.2'), 'subset': eval('"training"'), 'seed': eval('123'), 'image_size': eval('(img_height, img_width)'), 'batch_size': eval('batch_size')})
train_ds = tf.keras.utils.image_dataset_from_directory(data_dir, validation_split=0.2, subset='training', seed=123, image_size=(img_height, img_width), batch_size=batch_size)
custom_method(imports='import pathlib;import numpy as np;from tensorflow import keras;from tensorflow.keras import layers;from tensorflow.keras.models import Sequential;import matplotlib.pyplot as plt;import tensorflow as tf;import PIL', function_to_run='tf.keras.utils.image_dataset_from_directory(*args, **kwargs)', method_object=None, object_signature=None, function_args=[eval('data_dir')], function_kwargs={'validation_split': eval('0.2'), 'subset': eval('"validation"'), 'seed': eval('123'), 'image_size': eval('(img_height, img_width)'), 'batch_size': eval('batch_size')})
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
AUTOTUNE = tf.data.AUTOTUNE
custom_method(imports='import pathlib;import numpy as np;from tensorflow import keras;from tensorflow.keras import layers;from tensorflow.keras.models import Sequential;import matplotlib.pyplot as plt;import tensorflow as tf;import PIL', function_to_run='obj.cache().shuffle(1000).prefetch(**kwargs)', method_object=eval('train_ds'), object_signature=None, function_args=[], function_kwargs={'buffer_size': eval('AUTOTUNE')}, custom_class=None)
train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
custom_method(imports='import pathlib;import numpy as np;from tensorflow import keras;from tensorflow.keras import layers;from tensorflow.keras.models import Sequential;import matplotlib.pyplot as plt;import tensorflow as tf;import PIL', function_to_run='obj.cache().prefetch(**kwargs)', method_object=eval('val_ds'), object_signature=None, function_args=[], function_kwargs={'buffer_size': eval('AUTOTUNE')}, custom_class=None)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)
custom_method(imports='import pathlib;import numpy as np;from tensorflow import keras;from tensorflow.keras import layers;from tensorflow.keras.models import Sequential;import matplotlib.pyplot as plt;import tensorflow as tf;import PIL', function_to_run='layers.Rescaling(*args)', method_object=None, object_signature=None, function_args=[eval('1./255')], function_kwargs={})
normalization_layer = layers.Rescaling(1.0 / 255)
custom_method(imports='import pathlib;import numpy as np;from tensorflow import keras;from tensorflow.keras import layers;from tensorflow.keras.models import Sequential;import matplotlib.pyplot as plt;import tensorflow as tf;import PIL', function_to_run='obj.map(*args)', method_object=eval('train_ds'), object_signature=None, function_args=[eval('lambda x, y: (normalization_layer(x), y)')], function_kwargs={}, custom_class=None)
normalized_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
(image_batch, labels_batch) = next(iter(normalized_ds))
first_image = image_batch[0]
print(np.min(first_image), np.max(first_image))
num_classes = len(class_names)
custom_method(imports='import pathlib;import numpy as np;from tensorflow import keras;from tensorflow.keras import layers;from tensorflow.keras.models import Sequential;import matplotlib.pyplot as plt;import tensorflow as tf;import PIL', function_to_run='Sequential(*args)', method_object=None, object_signature=None, function_args=[eval("[\n  layers.Rescaling(1./255, input_shape=(img_height, img_width, 3)),\n  layers.Conv2D(16, 3, padding='same', activation='relu'),\n  layers.MaxPooling2D(),\n  layers.Conv2D(32, 3, padding='same', activation='relu'),\n  layers.MaxPooling2D(),\n  layers.Conv2D(64, 3, padding='same', activation='relu'),\n  layers.MaxPooling2D(),\n  layers.Flatten(),\n  layers.Dense(128, activation='relu'),\n  layers.Dense(num_classes)\n]")], function_kwargs={})
model = Sequential([layers.Rescaling(1.0 / 255, input_shape=(img_height, img_width, 3)), layers.Conv2D(16, 3, padding='same', activation='relu'), layers.MaxPooling2D(), layers.Conv2D(32, 3, padding='same', activation='relu'), layers.MaxPooling2D(), layers.Conv2D(64, 3, padding='same', activation='relu'), layers.MaxPooling2D(), layers.Flatten(), layers.Dense(128, activation='relu'), layers.Dense(num_classes)])
custom_method(imports='import pathlib;import numpy as np;from tensorflow import keras;from tensorflow.keras import layers;from tensorflow.keras.models import Sequential;import matplotlib.pyplot as plt;import tensorflow as tf;import PIL', function_to_run='obj.compile(**kwargs)', method_object=eval('model'), object_signature=None, function_args=[], function_kwargs={'optimizer': eval("'adam'"), 'loss': eval('tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)'), 'metrics': eval("['accuracy']")}, custom_class=None)
model.compile(optimizer='adam', loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=['accuracy'])
custom_method(imports='import pathlib;import numpy as np;from tensorflow import keras;from tensorflow.keras import layers;from tensorflow.keras.models import Sequential;import matplotlib.pyplot as plt;import tensorflow as tf;import PIL', function_to_run='obj.summary()', method_object=eval('model'), object_signature=None, function_args=[], function_kwargs={}, custom_class=None)
model.summary()
epochs = 10
custom_method(imports='import pathlib;import numpy as np;from tensorflow import keras;from tensorflow.keras import layers;from tensorflow.keras.models import Sequential;import matplotlib.pyplot as plt;import tensorflow as tf;import PIL', function_to_run='obj.fit(*args, **kwargs)', method_object=eval('model'), object_signature=None, function_args=[eval('train_ds')], function_kwargs={'validation_data': eval('val_ds'), 'epochs': eval('epochs')}, custom_class=None)
history = model.fit(train_ds, validation_data=val_ds, epochs=epochs)
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
custom_method(imports='import pathlib;import numpy as np;from tensorflow import keras;from tensorflow.keras import layers;from tensorflow.keras.models import Sequential;import matplotlib.pyplot as plt;import tensorflow as tf;import PIL', function_to_run='keras.Sequential(*args)', method_object=None, object_signature=None, function_args=[eval('[\n    layers.RandomFlip("horizontal",\n                      input_shape=(img_height,\n                                  img_width,\n                                  3)),\n    layers.RandomRotation(0.1),\n    layers.RandomZoom(0.1),\n  ]')], function_kwargs={})
data_augmentation = keras.Sequential([layers.RandomFlip('horizontal', input_shape=(img_height, img_width, 3)), layers.RandomRotation(0.1), layers.RandomZoom(0.1)])
plt.figure(figsize=(10, 10))
for (images, _) in train_ds.take(1):
    for i in range(9):
        custom_method(imports='import pathlib;import numpy as np;from tensorflow import keras;from tensorflow.keras import layers;from tensorflow.keras.models import Sequential;import matplotlib.pyplot as plt;import tensorflow as tf;import PIL', function_to_run='obj(*args)', method_object=eval('data_augmentation'), object_signature=None, function_args=[eval('images')], function_kwargs={}, custom_class=None)
        augmented_images = data_augmentation(images)
        ax = plt.subplot(3, 3, i + 1)
        plt.imshow(augmented_images[0].numpy().astype('uint8'))
        plt.axis('off')
custom_method(imports='import pathlib;import numpy as np;from tensorflow import keras;from tensorflow.keras import layers;from tensorflow.keras.models import Sequential;import matplotlib.pyplot as plt;import tensorflow as tf;import PIL', function_to_run='Sequential(*args)', method_object=None, object_signature=None, function_args=[eval('[\n  data_augmentation,\n  layers.Rescaling(1./255),\n  layers.Conv2D(16, 3, padding=\'same\', activation=\'relu\'),\n  layers.MaxPooling2D(),\n  layers.Conv2D(32, 3, padding=\'same\', activation=\'relu\'),\n  layers.MaxPooling2D(),\n  layers.Conv2D(64, 3, padding=\'same\', activation=\'relu\'),\n  layers.MaxPooling2D(),\n  layers.Dropout(0.2),\n  layers.Flatten(),\n  layers.Dense(128, activation=\'relu\'),\n  layers.Dense(num_classes, name="outputs")\n]')], function_kwargs={})
model = Sequential([data_augmentation, layers.Rescaling(1.0 / 255), layers.Conv2D(16, 3, padding='same', activation='relu'), layers.MaxPooling2D(), layers.Conv2D(32, 3, padding='same', activation='relu'), layers.MaxPooling2D(), layers.Conv2D(64, 3, padding='same', activation='relu'), layers.MaxPooling2D(), layers.Dropout(0.2), layers.Flatten(), layers.Dense(128, activation='relu'), layers.Dense(num_classes, name='outputs')])
custom_method(imports='import pathlib;import numpy as np;from tensorflow import keras;from tensorflow.keras import layers;from tensorflow.keras.models import Sequential;import matplotlib.pyplot as plt;import tensorflow as tf;import PIL', function_to_run='obj.compile(**kwargs)', method_object=eval('model'), object_signature=None, function_args=[], function_kwargs={'optimizer': eval("'adam'"), 'loss': eval('tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)'), 'metrics': eval("['accuracy']")}, custom_class=None)
model.compile(optimizer='adam', loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=['accuracy'])
custom_method(imports='import pathlib;import numpy as np;from tensorflow import keras;from tensorflow.keras import layers;from tensorflow.keras.models import Sequential;import matplotlib.pyplot as plt;import tensorflow as tf;import PIL', function_to_run='obj.summary()', method_object=eval('model'), object_signature=None, function_args=[], function_kwargs={}, custom_class=None)
model.summary()
epochs = 15
custom_method(imports='import pathlib;import numpy as np;from tensorflow import keras;from tensorflow.keras import layers;from tensorflow.keras.models import Sequential;import matplotlib.pyplot as plt;import tensorflow as tf;import PIL', function_to_run='obj.fit(*args, **kwargs)', method_object=eval('model'), object_signature=None, function_args=[eval('train_ds')], function_kwargs={'validation_data': eval('val_ds'), 'epochs': eval('epochs')}, custom_class=None)
history = model.fit(train_ds, validation_data=val_ds, epochs=epochs)
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
custom_method(imports='import pathlib;import numpy as np;from tensorflow import keras;from tensorflow.keras import layers;from tensorflow.keras.models import Sequential;import matplotlib.pyplot as plt;import tensorflow as tf;import PIL', function_to_run='tf.keras.utils.get_file(*args, **kwargs)', method_object=None, object_signature=None, function_args=[eval("'Red_sunflower'")], function_kwargs={'origin': eval('sunflower_url')})
sunflower_path = tf.keras.utils.get_file('Red_sunflower', origin=sunflower_url)
custom_method(imports='import pathlib;import numpy as np;from tensorflow import keras;from tensorflow.keras import layers;from tensorflow.keras.models import Sequential;import matplotlib.pyplot as plt;import tensorflow as tf;import PIL', function_to_run='tf.keras.utils.load_img(*args, **kwargs)', method_object=None, object_signature=None, function_args=[eval('sunflower_path')], function_kwargs={'target_size': eval('(img_height, img_width)')})
img = tf.keras.utils.load_img(sunflower_path, target_size=(img_height, img_width))
custom_method(imports='import pathlib;import numpy as np;from tensorflow import keras;from tensorflow.keras import layers;from tensorflow.keras.models import Sequential;import matplotlib.pyplot as plt;import tensorflow as tf;import PIL', function_to_run='tf.keras.utils.img_to_array(*args)', method_object=None, object_signature=None, function_args=[eval('img')], function_kwargs={})
img_array = tf.keras.utils.img_to_array(img)
custom_method(imports='import pathlib;import numpy as np;from tensorflow import keras;from tensorflow.keras import layers;from tensorflow.keras.models import Sequential;import matplotlib.pyplot as plt;import tensorflow as tf;import PIL', function_to_run='tf.expand_dims(*args)', method_object=None, object_signature=None, function_args=[eval('img_array'), eval('0')], function_kwargs={})
img_array = tf.expand_dims(img_array, 0)
custom_method(imports='import pathlib;import numpy as np;from tensorflow import keras;from tensorflow.keras import layers;from tensorflow.keras.models import Sequential;import matplotlib.pyplot as plt;import tensorflow as tf;import PIL', function_to_run='obj.predict(*args)', method_object=eval('model'), object_signature=None, function_args=[eval('img_array')], function_kwargs={}, custom_class=None)
predictions = model.predict(img_array)
custom_method(imports='import pathlib;import numpy as np;from tensorflow import keras;from tensorflow.keras import layers;from tensorflow.keras.models import Sequential;import matplotlib.pyplot as plt;import tensorflow as tf;import PIL', function_to_run='tf.nn.softmax(*args)', method_object=None, object_signature=None, function_args=[eval('predictions[0]')], function_kwargs={})
score = tf.nn.softmax(predictions[0])
print('This image most likely belongs to {} with a {:.2f} percent confidence.'.format(class_names[np.argmax(score)], 100 * np.max(score)))
custom_method(imports='import pathlib;import numpy as np;from tensorflow import keras;from tensorflow.keras import layers;from tensorflow.keras.models import Sequential;import matplotlib.pyplot as plt;import tensorflow as tf;import PIL', function_to_run='tf.lite.TFLiteConverter.from_keras_model(*args)', method_object=None, object_signature=None, function_args=[eval('model')], function_kwargs={})
converter = tf.lite.TFLiteConverter.from_keras_model(model)
custom_method(imports='import pathlib;import numpy as np;from tensorflow import keras;from tensorflow.keras import layers;from tensorflow.keras.models import Sequential;import matplotlib.pyplot as plt;import tensorflow as tf;import PIL', function_to_run='obj.convert()', method_object=eval('converter'), object_signature=None, function_args=[], function_kwargs={}, custom_class=None)
tflite_model = converter.convert()
with open('model.tflite', 'wb') as f:
    f.write(tflite_model)
TF_MODEL_FILE_PATH = 'model.tflite'
custom_method(imports='import pathlib;import numpy as np;from tensorflow import keras;from tensorflow.keras import layers;from tensorflow.keras.models import Sequential;import matplotlib.pyplot as plt;import tensorflow as tf;import PIL', function_to_run='tf.lite.Interpreter(**kwargs)', method_object=None, object_signature=None, function_args=[], function_kwargs={'model_path': eval('TF_MODEL_FILE_PATH')})
interpreter = tf.lite.Interpreter(model_path=TF_MODEL_FILE_PATH)
custom_method(imports='import pathlib;import numpy as np;from tensorflow import keras;from tensorflow.keras import layers;from tensorflow.keras.models import Sequential;import matplotlib.pyplot as plt;import tensorflow as tf;import PIL', function_to_run='obj.get_signature_list()', method_object=eval('interpreter'), object_signature=None, function_args=[], function_kwargs={}, custom_class=None)
interpreter.get_signature_list()
custom_method(imports='import pathlib;import numpy as np;from tensorflow import keras;from tensorflow.keras import layers;from tensorflow.keras.models import Sequential;import matplotlib.pyplot as plt;import tensorflow as tf;import PIL', function_to_run='obj.get_signature_runner(*args)', method_object=eval('interpreter'), object_signature=None, function_args=[eval("'serving_default'")], function_kwargs={}, custom_class=None)
classify_lite = interpreter.get_signature_runner('serving_default')
classify_lite
predictions_lite = classify_lite(sequential_1_input=img_array)['outputs']
custom_method(imports='import pathlib;import numpy as np;from tensorflow import keras;from tensorflow.keras import layers;from tensorflow.keras.models import Sequential;import matplotlib.pyplot as plt;import tensorflow as tf;import PIL', function_to_run='tf.nn.softmax(*args)', method_object=None, object_signature=None, function_args=[eval('predictions_lite')], function_kwargs={})
score_lite = tf.nn.softmax(predictions_lite)
print('This image most likely belongs to {} with a {:.2f} percent confidence.'.format(class_names[np.argmax(score_lite)], 100 * np.max(score_lite)))
print(np.max(np.abs(predictions - predictions_lite)))
