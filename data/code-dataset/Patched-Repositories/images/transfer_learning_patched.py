import matplotlib.pyplot as plt
import numpy as np
import os
import tensorflow as tf
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
_URL = 'https://storage.googleapis.com/mledu-datasets/cats_and_dogs_filtered.zip'
path_to_zip = custom_method(
tf.keras.utils.get_file('cats_and_dogs.zip', origin=_URL, extract=True), imports='import tensorflow as tf;import matplotlib.pyplot as plt;import os;import numpy as np', function_to_run='tf.keras.utils.get_file(*args, **kwargs)', method_object=None, object_signature=None, function_args=[eval("'cats_and_dogs.zip'")], function_kwargs={'origin': eval('_URL'), 'extract': eval('True')}, max_wait_secs=0)
PATH = os.path.join(os.path.dirname(path_to_zip), 'cats_and_dogs_filtered')
train_dir = os.path.join(PATH, 'train')
validation_dir = os.path.join(PATH, 'validation')
BATCH_SIZE = 32
IMG_SIZE = (160, 160)
train_dataset = custom_method(
tf.keras.utils.image_dataset_from_directory(train_dir, shuffle=True, batch_size=BATCH_SIZE, image_size=IMG_SIZE), imports='import tensorflow as tf;import matplotlib.pyplot as plt;import os;import numpy as np', function_to_run='tf.keras.utils.image_dataset_from_directory(*args, **kwargs)', method_object=None, object_signature=None, function_args=[eval('train_dir')], function_kwargs={'shuffle': eval('True'), 'batch_size': eval('BATCH_SIZE'), 'image_size': eval('IMG_SIZE')}, max_wait_secs=0)
validation_dataset = custom_method(
tf.keras.utils.image_dataset_from_directory(validation_dir, shuffle=True, batch_size=BATCH_SIZE, image_size=IMG_SIZE), imports='import tensorflow as tf;import matplotlib.pyplot as plt;import os;import numpy as np', function_to_run='tf.keras.utils.image_dataset_from_directory(*args, **kwargs)', method_object=None, object_signature=None, function_args=[eval('validation_dir')], function_kwargs={'shuffle': eval('True'), 'batch_size': eval('BATCH_SIZE'), 'image_size': eval('IMG_SIZE')}, max_wait_secs=0)
class_names = train_dataset.class_names
plt.figure(figsize=(10, 10))
for (images, labels) in custom_method(
train_dataset.take(1), imports='import tensorflow as tf;import matplotlib.pyplot as plt;import os;import numpy as np', function_to_run='obj.take(*args)', method_object=eval('train_dataset'), object_signature='tf.keras.utils.image_dataset_from_directory', function_args=[eval('1')], function_kwargs={}, max_wait_secs=0, custom_class=None):
    for i in range(9):
        ax = plt.subplot(3, 3, i + 1)
        plt.imshow(images[i].numpy().astype('uint8'))
        plt.title(class_names[labels[i]])
        plt.axis('off')
val_batches = custom_method(
tf.data.experimental.cardinality(validation_dataset), imports='import tensorflow as tf;import matplotlib.pyplot as plt;import os;import numpy as np', function_to_run='tf.data.experimental.cardinality(*args)', method_object=None, object_signature=None, function_args=[eval('validation_dataset')], function_kwargs={}, max_wait_secs=0)
test_dataset = custom_method(
validation_dataset.take(val_batches // 5), imports='import tensorflow as tf;import matplotlib.pyplot as plt;import os;import numpy as np', function_to_run='obj.take(*args)', method_object=eval('validation_dataset'), object_signature='tf.keras.utils.image_dataset_from_directory', function_args=[eval('val_batches // 5')], function_kwargs={}, max_wait_secs=0, custom_class=None)
validation_dataset = custom_method(
validation_dataset.skip(val_batches // 5), imports='import tensorflow as tf;import matplotlib.pyplot as plt;import os;import numpy as np', function_to_run='obj.skip(*args)', method_object=eval('validation_dataset'), object_signature='tf.keras.utils.image_dataset_from_directory', function_args=[eval('val_batches // 5')], function_kwargs={}, max_wait_secs=0, custom_class=None)
print('Number of validation batches: %d' % tf.data.experimental.cardinality(validation_dataset))
print('Number of test batches: %d' % tf.data.experimental.cardinality(test_dataset))
AUTOTUNE = tf.data.AUTOTUNE
train_dataset = custom_method(
train_dataset.prefetch(buffer_size=AUTOTUNE), imports='import tensorflow as tf;import matplotlib.pyplot as plt;import os;import numpy as np', function_to_run='obj.prefetch(**kwargs)', method_object=eval('train_dataset'), object_signature='tf.keras.utils.image_dataset_from_directory', function_args=[], function_kwargs={'buffer_size': eval('AUTOTUNE')}, max_wait_secs=0, custom_class=None)
validation_dataset = custom_method(
validation_dataset.prefetch(buffer_size=AUTOTUNE), imports='import tensorflow as tf;import matplotlib.pyplot as plt;import os;import numpy as np', function_to_run='obj.prefetch(**kwargs)', method_object=eval('validation_dataset'), object_signature='tf.keras.utils.image_dataset_from_directory', function_args=[], function_kwargs={'buffer_size': eval('AUTOTUNE')}, max_wait_secs=0, custom_class=None)
test_dataset = test_dataset.prefetch(buffer_size=AUTOTUNE)
data_augmentation = custom_method(
tf.keras.Sequential([tf.keras.layers.RandomFlip('horizontal'), tf.keras.layers.RandomRotation(0.2)]), imports='import tensorflow as tf;import matplotlib.pyplot as plt;import os;import numpy as np', function_to_run='tf.keras.Sequential(*args)', method_object=None, object_signature=None, function_args=[eval("[\n  tf.keras.layers.RandomFlip('horizontal'),\n  tf.keras.layers.RandomRotation(0.2),\n]")], function_kwargs={}, max_wait_secs=0)
for (image, _) in custom_method(
train_dataset.take(1), imports='import tensorflow as tf;import matplotlib.pyplot as plt;import os;import numpy as np', function_to_run='obj.take(*args)', method_object=eval('train_dataset'), object_signature='tf.keras.utils.image_dataset_from_directory', function_args=[eval('1')], function_kwargs={}, max_wait_secs=0, custom_class=None):
    plt.figure(figsize=(10, 10))
    first_image = image[0]
    for i in range(9):
        ax = plt.subplot(3, 3, i + 1)
        augmented_image = custom_method(
        data_augmentation(tf.expand_dims(first_image, 0)), imports='import tensorflow as tf;import matplotlib.pyplot as plt;import os;import numpy as np', function_to_run='obj(*args)', method_object=eval('data_augmentation'), object_signature='tf.keras.Sequential', function_args=[eval('tf.expand_dims(first_image, 0)')], function_kwargs={}, max_wait_secs=0, custom_class=None)
        plt.imshow(augmented_image[0] / 255)
        plt.axis('off')
preprocess_input = tf.keras.applications.mobilenet_v2.preprocess_input
rescale = custom_method(
tf.keras.layers.Rescaling(1.0 / 127.5, offset=-1), imports='import tensorflow as tf;import matplotlib.pyplot as plt;import os;import numpy as np', function_to_run='tf.keras.layers.Rescaling(*args, **kwargs)', method_object=None, object_signature=None, function_args=[eval('1./127.5')], function_kwargs={'offset': eval('-1')}, max_wait_secs=0)
IMG_SHAPE = IMG_SIZE + (3,)
base_model = custom_method(
tf.keras.applications.MobileNetV2(input_shape=IMG_SHAPE, include_top=False, weights='imagenet'), imports='import tensorflow as tf;import matplotlib.pyplot as plt;import os;import numpy as np', function_to_run='tf.keras.applications.MobileNetV2(**kwargs)', method_object=None, object_signature=None, function_args=[], function_kwargs={'input_shape': eval('IMG_SHAPE'), 'include_top': eval('False'), 'weights': eval("'imagenet'")}, max_wait_secs=0)
(image_batch, label_batch) = next(iter(train_dataset))
feature_batch = custom_method(
base_model(image_batch), imports='import tensorflow as tf;import matplotlib.pyplot as plt;import os;import numpy as np', function_to_run='obj(*args)', method_object=eval('base_model'), object_signature='tf.keras.applications.MobileNetV2', function_args=[eval('image_batch')], function_kwargs={}, max_wait_secs=0, custom_class=None)
print(feature_batch.shape)
base_model.trainable = False
custom_method(
base_model.summary(), imports='import tensorflow as tf;import matplotlib.pyplot as plt;import os;import numpy as np', function_to_run='obj.summary()', method_object=eval('base_model'), object_signature='tf.keras.applications.MobileNetV2', function_args=[], function_kwargs={}, max_wait_secs=0, custom_class=None)
global_average_layer = custom_method(
tf.keras.layers.GlobalAveragePooling2D(), imports='import tensorflow as tf;import matplotlib.pyplot as plt;import os;import numpy as np', function_to_run='tf.keras.layers.GlobalAveragePooling2D()', method_object=None, object_signature=None, function_args=[], function_kwargs={}, max_wait_secs=0)
feature_batch_average = custom_method(
global_average_layer(feature_batch), imports='import tensorflow as tf;import matplotlib.pyplot as plt;import os;import numpy as np', function_to_run='obj(*args)', method_object=eval('global_average_layer'), object_signature='tf.keras.layers.GlobalAveragePooling2D', function_args=[eval('feature_batch')], function_kwargs={}, max_wait_secs=0, custom_class=None)
print(feature_batch_average.shape)
prediction_layer = custom_method(
tf.keras.layers.Dense(1), imports='import tensorflow as tf;import matplotlib.pyplot as plt;import os;import numpy as np', function_to_run='tf.keras.layers.Dense(*args)', method_object=None, object_signature=None, function_args=[eval('1')], function_kwargs={}, max_wait_secs=0)
prediction_batch = custom_method(
prediction_layer(feature_batch_average), imports='import tensorflow as tf;import matplotlib.pyplot as plt;import os;import numpy as np', function_to_run='obj(*args)', method_object=eval('prediction_layer'), object_signature='tf.keras.layers.Dense', function_args=[eval('feature_batch_average')], function_kwargs={}, max_wait_secs=0, custom_class=None)
print(prediction_batch.shape)
inputs = custom_method(
tf.keras.Input(shape=(160, 160, 3)), imports='import tensorflow as tf;import matplotlib.pyplot as plt;import os;import numpy as np', function_to_run='tf.keras.Input(**kwargs)', method_object=None, object_signature=None, function_args=[], function_kwargs={'shape': eval('(160, 160, 3)')}, max_wait_secs=0)
x = custom_method(
data_augmentation(inputs), imports='import tensorflow as tf;import matplotlib.pyplot as plt;import os;import numpy as np', function_to_run='obj(*args)', method_object=eval('data_augmentation'), object_signature='tf.keras.Sequential', function_args=[eval('inputs')], function_kwargs={}, max_wait_secs=0, custom_class=None)
x = preprocess_input(x)
x = custom_method(
base_model(x, training=False), imports='import tensorflow as tf;import matplotlib.pyplot as plt;import os;import numpy as np', function_to_run='obj(*args, **kwargs)', method_object=eval('base_model'), object_signature='tf.keras.applications.MobileNetV2', function_args=[eval('x')], function_kwargs={'training': eval('False')}, max_wait_secs=0, custom_class=None)
x = custom_method(
global_average_layer(x), imports='import tensorflow as tf;import matplotlib.pyplot as plt;import os;import numpy as np', function_to_run='obj(*args)', method_object=eval('global_average_layer'), object_signature='tf.keras.layers.GlobalAveragePooling2D', function_args=[eval('x')], function_kwargs={}, max_wait_secs=0, custom_class=None)
x = custom_method(
tf.keras.layers.Dropout(0.2)(x), imports='import tensorflow as tf;import matplotlib.pyplot as plt;import os;import numpy as np', function_to_run='tf.keras.layers.Dropout(0.2)(*args)', method_object=None, object_signature=None, function_args=[eval('x')], function_kwargs={}, max_wait_secs=0)
outputs = custom_method(
prediction_layer(x), imports='import tensorflow as tf;import matplotlib.pyplot as plt;import os;import numpy as np', function_to_run='obj(*args)', method_object=eval('prediction_layer'), object_signature='tf.keras.layers.Dense', function_args=[eval('x')], function_kwargs={}, max_wait_secs=0, custom_class=None)
model = custom_method(
tf.keras.Model(inputs, outputs), imports='import tensorflow as tf;import matplotlib.pyplot as plt;import os;import numpy as np', function_to_run='tf.keras.Model(*args)', method_object=None, object_signature=None, function_args=[eval('inputs'), eval('outputs')], function_kwargs={}, max_wait_secs=0)
base_learning_rate = 0.0001
custom_method(
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=base_learning_rate), loss=tf.keras.losses.BinaryCrossentropy(from_logits=True), metrics=['accuracy']), imports='import tensorflow as tf;import matplotlib.pyplot as plt;import os;import numpy as np', function_to_run='obj.compile(**kwargs)', method_object=eval('model'), object_signature='tf.keras.Model', function_args=[], function_kwargs={'optimizer': eval('tf.keras.optimizers.Adam(learning_rate=base_learning_rate)'), 'loss': eval('tf.keras.losses.BinaryCrossentropy(from_logits=True)'), 'metrics': eval("['accuracy']")}, max_wait_secs=0, custom_class=None)
custom_method(
model.summary(), imports='import tensorflow as tf;import matplotlib.pyplot as plt;import os;import numpy as np', function_to_run='obj.summary()', method_object=eval('model'), object_signature='tf.keras.Model', function_args=[], function_kwargs={}, max_wait_secs=0, custom_class=None)
len(model.trainable_variables)
initial_epochs = 10
(loss0, accuracy0) = custom_method(
model.evaluate(validation_dataset), imports='import tensorflow as tf;import matplotlib.pyplot as plt;import os;import numpy as np', function_to_run='obj.evaluate(*args)', method_object=eval('model'), object_signature='tf.keras.Model', function_args=[eval('validation_dataset')], function_kwargs={}, max_wait_secs=0, custom_class=None)
print('initial loss: {:.2f}'.format(loss0))
print('initial accuracy: {:.2f}'.format(accuracy0))
history = custom_method(
model.fit(train_dataset, epochs=initial_epochs, validation_data=validation_dataset), imports='import tensorflow as tf;import matplotlib.pyplot as plt;import os;import numpy as np', function_to_run='obj.fit(*args, **kwargs)', method_object=eval('model'), object_signature='tf.keras.Model', function_args=[eval('train_dataset')], function_kwargs={'epochs': eval('initial_epochs'), 'validation_data': eval('validation_dataset')}, max_wait_secs=0, custom_class=None)
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']
plt.figure(figsize=(8, 8))
plt.subplot(2, 1, 1)
plt.plot(acc, label='Training Accuracy')
plt.plot(val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.ylabel('Accuracy')
plt.ylim([min(plt.ylim()), 1])
plt.title('Training and Validation Accuracy')
plt.subplot(2, 1, 2)
plt.plot(loss, label='Training Loss')
plt.plot(val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.ylabel('Cross Entropy')
plt.ylim([0, 1.0])
plt.title('Training and Validation Loss')
plt.xlabel('epoch')
plt.show()
base_model.trainable = True
print('Number of layers in the base model: ', len(base_model.layers))
fine_tune_at = 100
for layer in base_model.layers[:fine_tune_at]:
    layer.trainable = False
custom_method(
model.compile(loss=tf.keras.losses.BinaryCrossentropy(from_logits=True), optimizer=tf.keras.optimizers.RMSprop(learning_rate=base_learning_rate / 10), metrics=['accuracy']), imports='import tensorflow as tf;import matplotlib.pyplot as plt;import os;import numpy as np', function_to_run='obj.compile(**kwargs)', method_object=eval('model'), object_signature='tf.keras.Model', function_args=[], function_kwargs={'loss': eval('tf.keras.losses.BinaryCrossentropy(from_logits=True)'), 'optimizer': eval('tf.keras.optimizers.RMSprop(learning_rate=base_learning_rate/10)'), 'metrics': eval("['accuracy']")}, max_wait_secs=0, custom_class=None)
custom_method(
model.summary(), imports='import tensorflow as tf;import matplotlib.pyplot as plt;import os;import numpy as np', function_to_run='obj.summary()', method_object=eval('model'), object_signature='tf.keras.Model', function_args=[], function_kwargs={}, max_wait_secs=0, custom_class=None)
len(model.trainable_variables)
fine_tune_epochs = 10
total_epochs = initial_epochs + fine_tune_epochs
history_fine = custom_method(
model.fit(train_dataset, epochs=total_epochs, initial_epoch=history.epoch[-1], validation_data=validation_dataset), imports='import tensorflow as tf;import matplotlib.pyplot as plt;import os;import numpy as np', function_to_run='obj.fit(*args, **kwargs)', method_object=eval('model'), object_signature='tf.keras.Model', function_args=[eval('train_dataset')], function_kwargs={'epochs': eval('total_epochs'), 'initial_epoch': eval('history.epoch[-1]'), 'validation_data': eval('validation_dataset')}, max_wait_secs=0, custom_class=None)
acc += history_fine.history['accuracy']
val_acc += history_fine.history['val_accuracy']
loss += history_fine.history['loss']
val_loss += history_fine.history['val_loss']
plt.figure(figsize=(8, 8))
plt.subplot(2, 1, 1)
plt.plot(acc, label='Training Accuracy')
plt.plot(val_acc, label='Validation Accuracy')
plt.ylim([0.8, 1])
plt.plot([initial_epochs - 1, initial_epochs - 1], plt.ylim(), label='Start Fine Tuning')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')
plt.subplot(2, 1, 2)
plt.plot(loss, label='Training Loss')
plt.plot(val_loss, label='Validation Loss')
plt.ylim([0, 1.0])
plt.plot([initial_epochs - 1, initial_epochs - 1], plt.ylim(), label='Start Fine Tuning')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.xlabel('epoch')
plt.show()
(loss, accuracy) = custom_method(
model.evaluate(test_dataset), imports='import tensorflow as tf;import matplotlib.pyplot as plt;import os;import numpy as np', function_to_run='obj.evaluate(*args)', method_object=eval('model'), object_signature='tf.keras.Model', function_args=[eval('test_dataset')], function_kwargs={}, max_wait_secs=0, custom_class=None)
print('Test accuracy :', accuracy)
(image_batch, label_batch) = test_dataset.as_numpy_iterator().next()
predictions = model.predict_on_batch(image_batch).flatten()
predictions = custom_method(
tf.nn.sigmoid(predictions), imports='import tensorflow as tf;import matplotlib.pyplot as plt;import os;import numpy as np', function_to_run='tf.nn.sigmoid(*args)', method_object=None, object_signature=None, function_args=[eval('predictions')], function_kwargs={}, max_wait_secs=0)
predictions = custom_method(
tf.where(predictions < 0.5, 0, 1), imports='import tensorflow as tf;import matplotlib.pyplot as plt;import os;import numpy as np', function_to_run='tf.where(*args)', method_object=None, object_signature=None, function_args=[eval('predictions < 0.5'), eval('0'), eval('1')], function_kwargs={}, max_wait_secs=0)
print('Predictions:\n', predictions.numpy())
print('Labels:\n', label_batch)
plt.figure(figsize=(10, 10))
for i in range(9):
    ax = plt.subplot(3, 3, i + 1)
    plt.imshow(image_batch[i].astype('uint8'))
    plt.title(class_names[predictions[i]])
    plt.axis('off')
