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

def custom_method(imports: str, function_to_run: str, method_object=None, object_signature=None, function_args: list=None, function_kwargs: dict=None, custom_class=None):
    result = send_request(imports=imports, function_to_run=function_to_run, function_args=function_args, function_kwargs=function_kwargs, max_wait_secs=MAX_WAIT_S, wait_after_run_secs=WAIT_AFTER_RUN_S, method_object=method_object, object_signature=object_signature, custom_class=custom_class, experiment_file_path=EXPERIMENT_FILE_PATH)
_URL = 'https://storage.googleapis.com/mledu-datasets/cats_and_dogs_filtered.zip'
custom_method(imports='import matplotlib.pyplot as plt;import tensorflow as tf;import numpy as np;import os', function_to_run='tf.keras.utils.get_file(*args, **kwargs)', method_object=None, object_signature=None, function_args=[eval("'cats_and_dogs.zip'")], function_kwargs={'origin': eval('_URL'), 'extract': eval('True')})
path_to_zip = tf.keras.utils.get_file('cats_and_dogs.zip', origin=_URL, extract=True)
PATH = os.path.join(os.path.dirname(path_to_zip), 'cats_and_dogs_filtered')
train_dir = os.path.join(PATH, 'train')
validation_dir = os.path.join(PATH, 'validation')
BATCH_SIZE = 32
IMG_SIZE = (160, 160)
custom_method(imports='import matplotlib.pyplot as plt;import tensorflow as tf;import numpy as np;import os', function_to_run='tf.keras.utils.image_dataset_from_directory(*args, **kwargs)', method_object=None, object_signature=None, function_args=[eval('train_dir')], function_kwargs={'shuffle': eval('True'), 'batch_size': eval('BATCH_SIZE'), 'image_size': eval('IMG_SIZE')})
train_dataset = tf.keras.utils.image_dataset_from_directory(train_dir, shuffle=True, batch_size=BATCH_SIZE, image_size=IMG_SIZE)
custom_method(imports='import matplotlib.pyplot as plt;import tensorflow as tf;import numpy as np;import os', function_to_run='tf.keras.utils.image_dataset_from_directory(*args, **kwargs)', method_object=None, object_signature=None, function_args=[eval('validation_dir')], function_kwargs={'shuffle': eval('True'), 'batch_size': eval('BATCH_SIZE'), 'image_size': eval('IMG_SIZE')})
validation_dataset = tf.keras.utils.image_dataset_from_directory(validation_dir, shuffle=True, batch_size=BATCH_SIZE, image_size=IMG_SIZE)
class_names = train_dataset.class_names
plt.figure(figsize=(10, 10))
for (images, labels) in train_dataset.take(1):
    for i in range(9):
        ax = plt.subplot(3, 3, i + 1)
        plt.imshow(images[i].numpy().astype('uint8'))
        plt.title(class_names[labels[i]])
        plt.axis('off')
custom_method(imports='import matplotlib.pyplot as plt;import tensorflow as tf;import numpy as np;import os', function_to_run='tf.data.experimental.cardinality(*args)', method_object=None, object_signature=None, function_args=[eval('validation_dataset')], function_kwargs={})
val_batches = tf.data.experimental.cardinality(validation_dataset)
custom_method(imports='import matplotlib.pyplot as plt;import tensorflow as tf;import numpy as np;import os', function_to_run='obj.take(*args)', method_object=eval('validation_dataset'), object_signature=None, function_args=[eval('val_batches // 5')], function_kwargs={}, custom_class=None)
test_dataset = validation_dataset.take(val_batches // 5)
custom_method(imports='import matplotlib.pyplot as plt;import tensorflow as tf;import numpy as np;import os', function_to_run='obj.skip(*args)', method_object=eval('validation_dataset'), object_signature=None, function_args=[eval('val_batches // 5')], function_kwargs={}, custom_class=None)
validation_dataset = validation_dataset.skip(val_batches // 5)
print('Number of validation batches: %d' % tf.data.experimental.cardinality(validation_dataset))
print('Number of test batches: %d' % tf.data.experimental.cardinality(test_dataset))
AUTOTUNE = tf.data.AUTOTUNE
custom_method(imports='import matplotlib.pyplot as plt;import tensorflow as tf;import numpy as np;import os', function_to_run='obj.prefetch(**kwargs)', method_object=eval('train_dataset'), object_signature=None, function_args=[], function_kwargs={'buffer_size': eval('AUTOTUNE')}, custom_class=None)
train_dataset = train_dataset.prefetch(buffer_size=AUTOTUNE)
custom_method(imports='import matplotlib.pyplot as plt;import tensorflow as tf;import numpy as np;import os', function_to_run='obj.prefetch(**kwargs)', method_object=eval('validation_dataset'), object_signature=None, function_args=[], function_kwargs={'buffer_size': eval('AUTOTUNE')}, custom_class=None)
validation_dataset = validation_dataset.prefetch(buffer_size=AUTOTUNE)
test_dataset = test_dataset.prefetch(buffer_size=AUTOTUNE)
custom_method(imports='import matplotlib.pyplot as plt;import tensorflow as tf;import numpy as np;import os', function_to_run='tf.keras.Sequential(*args)', method_object=None, object_signature=None, function_args=[eval("[\n  tf.keras.layers.RandomFlip('horizontal'),\n  tf.keras.layers.RandomRotation(0.2),\n]")], function_kwargs={})
data_augmentation = tf.keras.Sequential([tf.keras.layers.RandomFlip('horizontal'), tf.keras.layers.RandomRotation(0.2)])
for (image, _) in train_dataset.take(1):
    plt.figure(figsize=(10, 10))
    first_image = image[0]
    for i in range(9):
        ax = plt.subplot(3, 3, i + 1)
        custom_method(imports='import matplotlib.pyplot as plt;import tensorflow as tf;import numpy as np;import os', function_to_run='obj(*args)', method_object=eval('data_augmentation'), object_signature=None, function_args=[eval('tf.expand_dims(first_image, 0)')], function_kwargs={}, custom_class=None)
        augmented_image = data_augmentation(tf.expand_dims(first_image, 0))
        plt.imshow(augmented_image[0] / 255)
        plt.axis('off')
preprocess_input = tf.keras.applications.mobilenet_v2.preprocess_input
custom_method(imports='import matplotlib.pyplot as plt;import tensorflow as tf;import numpy as np;import os', function_to_run='tf.keras.layers.Rescaling(*args, **kwargs)', method_object=None, object_signature=None, function_args=[eval('1./127.5')], function_kwargs={'offset': eval('-1')})
rescale = tf.keras.layers.Rescaling(1.0 / 127.5, offset=-1)
IMG_SHAPE = IMG_SIZE + (3,)
custom_method(imports='import matplotlib.pyplot as plt;import tensorflow as tf;import numpy as np;import os', function_to_run='tf.keras.applications.MobileNetV2(**kwargs)', method_object=None, object_signature=None, function_args=[], function_kwargs={'input_shape': eval('IMG_SHAPE'), 'include_top': eval('False'), 'weights': eval("'imagenet'")})
base_model = tf.keras.applications.MobileNetV2(input_shape=IMG_SHAPE, include_top=False, weights='imagenet')
(image_batch, label_batch) = next(iter(train_dataset))
custom_method(imports='import matplotlib.pyplot as plt;import tensorflow as tf;import numpy as np;import os', function_to_run='obj(*args)', method_object=eval('base_model'), object_signature=None, function_args=[eval('image_batch')], function_kwargs={}, custom_class=None)
feature_batch = base_model(image_batch)
print(feature_batch.shape)
base_model.trainable = False
custom_method(imports='import matplotlib.pyplot as plt;import tensorflow as tf;import numpy as np;import os', function_to_run='obj.summary()', method_object=eval('base_model'), object_signature=None, function_args=[], function_kwargs={}, custom_class=None)
base_model.summary()
custom_method(imports='import matplotlib.pyplot as plt;import tensorflow as tf;import numpy as np;import os', function_to_run='tf.keras.layers.GlobalAveragePooling2D()', method_object=None, object_signature=None, function_args=[], function_kwargs={})
global_average_layer = tf.keras.layers.GlobalAveragePooling2D()
custom_method(imports='import matplotlib.pyplot as plt;import tensorflow as tf;import numpy as np;import os', function_to_run='obj(*args)', method_object=eval('global_average_layer'), object_signature=None, function_args=[eval('feature_batch')], function_kwargs={}, custom_class=None)
feature_batch_average = global_average_layer(feature_batch)
print(feature_batch_average.shape)
custom_method(imports='import matplotlib.pyplot as plt;import tensorflow as tf;import numpy as np;import os', function_to_run='tf.keras.layers.Dense(*args)', method_object=None, object_signature=None, function_args=[eval('1')], function_kwargs={})
prediction_layer = tf.keras.layers.Dense(1)
custom_method(imports='import matplotlib.pyplot as plt;import tensorflow as tf;import numpy as np;import os', function_to_run='obj(*args)', method_object=eval('prediction_layer'), object_signature=None, function_args=[eval('feature_batch_average')], function_kwargs={}, custom_class=None)
prediction_batch = prediction_layer(feature_batch_average)
print(prediction_batch.shape)
custom_method(imports='import matplotlib.pyplot as plt;import tensorflow as tf;import numpy as np;import os', function_to_run='tf.keras.Input(**kwargs)', method_object=None, object_signature=None, function_args=[], function_kwargs={'shape': eval('(160, 160, 3)')})
inputs = tf.keras.Input(shape=(160, 160, 3))
custom_method(imports='import matplotlib.pyplot as plt;import tensorflow as tf;import numpy as np;import os', function_to_run='obj(*args)', method_object=eval('data_augmentation'), object_signature=None, function_args=[eval('inputs')], function_kwargs={}, custom_class=None)
x = data_augmentation(inputs)
x = preprocess_input(x)
custom_method(imports='import matplotlib.pyplot as plt;import tensorflow as tf;import numpy as np;import os', function_to_run='obj(*args, **kwargs)', method_object=eval('base_model'), object_signature=None, function_args=[eval('x')], function_kwargs={'training': eval('False')}, custom_class=None)
x = base_model(x, training=False)
custom_method(imports='import matplotlib.pyplot as plt;import tensorflow as tf;import numpy as np;import os', function_to_run='obj(*args)', method_object=eval('global_average_layer'), object_signature=None, function_args=[eval('x')], function_kwargs={}, custom_class=None)
x = global_average_layer(x)
custom_method(imports='import matplotlib.pyplot as plt;import tensorflow as tf;import numpy as np;import os', function_to_run='tf.keras.layers.Dropout(0.2)(*args)', method_object=None, object_signature=None, function_args=[eval('x')], function_kwargs={})
x = tf.keras.layers.Dropout(0.2)(x)
custom_method(imports='import matplotlib.pyplot as plt;import tensorflow as tf;import numpy as np;import os', function_to_run='obj(*args)', method_object=eval('prediction_layer'), object_signature=None, function_args=[eval('x')], function_kwargs={}, custom_class=None)
outputs = prediction_layer(x)
custom_method(imports='import matplotlib.pyplot as plt;import tensorflow as tf;import numpy as np;import os', function_to_run='tf.keras.Model(*args)', method_object=None, object_signature=None, function_args=[eval('inputs'), eval('outputs')], function_kwargs={})
model = tf.keras.Model(inputs, outputs)
base_learning_rate = 0.0001
custom_method(imports='import matplotlib.pyplot as plt;import tensorflow as tf;import numpy as np;import os', function_to_run='obj.compile(**kwargs)', method_object=eval('model'), object_signature=None, function_args=[], function_kwargs={'optimizer': eval('tf.keras.optimizers.Adam(learning_rate=base_learning_rate)'), 'loss': eval('tf.keras.losses.BinaryCrossentropy(from_logits=True)'), 'metrics': eval("['accuracy']")}, custom_class=None)
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=base_learning_rate), loss=tf.keras.losses.BinaryCrossentropy(from_logits=True), metrics=['accuracy'])
custom_method(imports='import matplotlib.pyplot as plt;import tensorflow as tf;import numpy as np;import os', function_to_run='obj.summary()', method_object=eval('model'), object_signature=None, function_args=[], function_kwargs={}, custom_class=None)
model.summary()
len(model.trainable_variables)
initial_epochs = 10
custom_method(imports='import matplotlib.pyplot as plt;import tensorflow as tf;import numpy as np;import os', function_to_run='obj.evaluate(*args)', method_object=eval('model'), object_signature=None, function_args=[eval('validation_dataset')], function_kwargs={}, custom_class=None)
(loss0, accuracy0) = model.evaluate(validation_dataset)
print('initial loss: {:.2f}'.format(loss0))
print('initial accuracy: {:.2f}'.format(accuracy0))
custom_method(imports='import matplotlib.pyplot as plt;import tensorflow as tf;import numpy as np;import os', function_to_run='obj.fit(*args, **kwargs)', method_object=eval('model'), object_signature=None, function_args=[eval('train_dataset')], function_kwargs={'epochs': eval('initial_epochs'), 'validation_data': eval('validation_dataset')}, custom_class=None)
history = model.fit(train_dataset, epochs=initial_epochs, validation_data=validation_dataset)
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
custom_method(imports='import matplotlib.pyplot as plt;import tensorflow as tf;import numpy as np;import os', function_to_run='obj.compile(**kwargs)', method_object=eval('model'), object_signature=None, function_args=[], function_kwargs={'loss': eval('tf.keras.losses.BinaryCrossentropy(from_logits=True)'), 'optimizer': eval('tf.keras.optimizers.RMSprop(learning_rate=base_learning_rate/10)'), 'metrics': eval("['accuracy']")}, custom_class=None)
model.compile(loss=tf.keras.losses.BinaryCrossentropy(from_logits=True), optimizer=tf.keras.optimizers.RMSprop(learning_rate=base_learning_rate / 10), metrics=['accuracy'])
custom_method(imports='import matplotlib.pyplot as plt;import tensorflow as tf;import numpy as np;import os', function_to_run='obj.summary()', method_object=eval('model'), object_signature=None, function_args=[], function_kwargs={}, custom_class=None)
model.summary()
len(model.trainable_variables)
fine_tune_epochs = 10
total_epochs = initial_epochs + fine_tune_epochs
custom_method(imports='import matplotlib.pyplot as plt;import tensorflow as tf;import numpy as np;import os', function_to_run='obj.fit(*args, **kwargs)', method_object=eval('model'), object_signature=None, function_args=[eval('train_dataset')], function_kwargs={'epochs': eval('total_epochs'), 'initial_epoch': eval('history.epoch[-1]'), 'validation_data': eval('validation_dataset')}, custom_class=None)
history_fine = model.fit(train_dataset, epochs=total_epochs, initial_epoch=history.epoch[-1], validation_data=validation_dataset)
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
custom_method(imports='import matplotlib.pyplot as plt;import tensorflow as tf;import numpy as np;import os', function_to_run='obj.evaluate(*args)', method_object=eval('model'), object_signature=None, function_args=[eval('test_dataset')], function_kwargs={}, custom_class=None)
(loss, accuracy) = model.evaluate(test_dataset)
print('Test accuracy :', accuracy)
(image_batch, label_batch) = test_dataset.as_numpy_iterator().next()
custom_method(imports='import matplotlib.pyplot as plt;import tensorflow as tf;import numpy as np;import os', function_to_run='obj.predict_on_batch(image_batch).flatten()', method_object=eval('model'), object_signature=None, function_args=[], function_kwargs={}, custom_class=None)
predictions = model.predict_on_batch(image_batch).flatten()
custom_method(imports='import matplotlib.pyplot as plt;import tensorflow as tf;import numpy as np;import os', function_to_run='tf.nn.sigmoid(*args)', method_object=None, object_signature=None, function_args=[eval('predictions')], function_kwargs={})
predictions = tf.nn.sigmoid(predictions)
custom_method(imports='import matplotlib.pyplot as plt;import tensorflow as tf;import numpy as np;import os', function_to_run='tf.where(*args)', method_object=None, object_signature=None, function_args=[eval('predictions < 0.5'), eval('0'), eval('1')], function_kwargs={})
predictions = tf.where(predictions < 0.5, 0, 1)
print('Predictions:\n', predictions.numpy())
print('Labels:\n', label_batch)
plt.figure(figsize=(10, 10))
for i in range(9):
    ax = plt.subplot(3, 3, i + 1)
    plt.imshow(image_batch[i].astype('uint8'))
    plt.title(class_names[predictions[i]])
    plt.axis('off')
