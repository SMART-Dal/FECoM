import matplotlib.pyplot as plt
import numpy as np
import PIL
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
import pathlib
import sys
from tool.client.client_config import EXPERIMENT_DIR
from tool.server.local_execution import before_execution as before_execution_INSERTED_INTO_SCRIPT
from tool.server.local_execution import after_execution as after_execution_INSERTED_INTO_SCRIPT
experiment_number = sys.argv[1]
experiment_project = sys.argv[2]
EXPERIMENT_FILE_PATH = EXPERIMENT_DIR / 'method-level' / experiment_project / f'experiment-{experiment_number}.json'
dataset_url = 'https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz'
start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT()
data_dir = tf.keras.utils.get_file('flower_photos', origin=dataset_url, untar=True)
after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tf.keras.utils.get_file(*args, **kwargs)', method_object=None, object_signature=None, function_args=['flower_photos'], function_kwargs={'origin': dataset_url, 'untar': True})
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
start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT()
train_ds = tf.keras.utils.image_dataset_from_directory(data_dir, validation_split=0.2, subset='training', seed=123, image_size=(img_height, img_width), batch_size=batch_size)
after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tf.keras.utils.image_dataset_from_directory(*args, **kwargs)', method_object=None, object_signature=None, function_args=[data_dir], function_kwargs={'validation_split': 0.2, 'subset': 'training', 'seed': 123, 'image_size': (img_height, img_width), 'batch_size': batch_size})
start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT()
val_ds = tf.keras.utils.image_dataset_from_directory(data_dir, validation_split=0.2, subset='validation', seed=123, image_size=(img_height, img_width), batch_size=batch_size)
after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tf.keras.utils.image_dataset_from_directory(*args, **kwargs)', method_object=None, object_signature=None, function_args=[data_dir], function_kwargs={'validation_split': 0.2, 'subset': 'validation', 'seed': 123, 'image_size': (img_height, img_width), 'batch_size': batch_size})
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
start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT()
train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='obj.cache().shuffle(1000).prefetch(**kwargs)', method_object=train_ds, object_signature=None, function_args=[], function_kwargs={'buffer_size': AUTOTUNE})
start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT()
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)
after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='obj.cache().prefetch(**kwargs)', method_object=val_ds, object_signature=None, function_args=[], function_kwargs={'buffer_size': AUTOTUNE})
start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT()
normalization_layer = layers.Rescaling(1.0 / 255)
after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='layers.Rescaling(*args)', method_object=None, object_signature=None, function_args=[1.0 / 255], function_kwargs={})
start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT()
normalized_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='obj.map(*args)', method_object=train_ds, object_signature=None, function_args=[lambda x, y: (normalization_layer(x), y)], function_kwargs={})
(image_batch, labels_batch) = next(iter(normalized_ds))
first_image = image_batch[0]
print(np.min(first_image), np.max(first_image))
num_classes = len(class_names)
start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT()
model = Sequential([layers.Rescaling(1.0 / 255, input_shape=(img_height, img_width, 3)), layers.Conv2D(16, 3, padding='same', activation='relu'), layers.MaxPooling2D(), layers.Conv2D(32, 3, padding='same', activation='relu'), layers.MaxPooling2D(), layers.Conv2D(64, 3, padding='same', activation='relu'), layers.MaxPooling2D(), layers.Flatten(), layers.Dense(128, activation='relu'), layers.Dense(num_classes)])
after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='Sequential(*args)', method_object=None, object_signature=None, function_args=[[layers.Rescaling(1.0 / 255, input_shape=(img_height, img_width, 3)), layers.Conv2D(16, 3, padding='same', activation='relu'), layers.MaxPooling2D(), layers.Conv2D(32, 3, padding='same', activation='relu'), layers.MaxPooling2D(), layers.Conv2D(64, 3, padding='same', activation='relu'), layers.MaxPooling2D(), layers.Flatten(), layers.Dense(128, activation='relu'), layers.Dense(num_classes)]], function_kwargs={})
start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT()
model.compile(optimizer='adam', loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=['accuracy'])
after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='obj.compile(**kwargs)', method_object=model, object_signature=None, function_args=[], function_kwargs={'optimizer': 'adam', 'loss': tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), 'metrics': ['accuracy']})
start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT()
model.summary()
after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='obj.summary()', method_object=model, object_signature=None, function_args=[], function_kwargs={})
epochs = 10
start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT()
history = model.fit(train_ds, validation_data=val_ds, epochs=epochs)
after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='obj.fit(*args, **kwargs)', method_object=model, object_signature=None, function_args=[train_ds], function_kwargs={'validation_data': val_ds, 'epochs': epochs})
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
start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT()
data_augmentation = keras.Sequential([layers.RandomFlip('horizontal', input_shape=(img_height, img_width, 3)), layers.RandomRotation(0.1), layers.RandomZoom(0.1)])
after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='keras.Sequential(*args)', method_object=None, object_signature=None, function_args=[[layers.RandomFlip('horizontal', input_shape=(img_height, img_width, 3)), layers.RandomRotation(0.1), layers.RandomZoom(0.1)]], function_kwargs={})
plt.figure(figsize=(10, 10))
for (images, _) in train_ds.take(1):
    for i in range(9):
        start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT()
        augmented_images = data_augmentation(images)
        after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='obj(*args)', method_object=data_augmentation, object_signature=None, function_args=[images], function_kwargs={})
        ax = plt.subplot(3, 3, i + 1)
        plt.imshow(augmented_images[0].numpy().astype('uint8'))
        plt.axis('off')
start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT()
model = Sequential([data_augmentation, layers.Rescaling(1.0 / 255), layers.Conv2D(16, 3, padding='same', activation='relu'), layers.MaxPooling2D(), layers.Conv2D(32, 3, padding='same', activation='relu'), layers.MaxPooling2D(), layers.Conv2D(64, 3, padding='same', activation='relu'), layers.MaxPooling2D(), layers.Dropout(0.2), layers.Flatten(), layers.Dense(128, activation='relu'), layers.Dense(num_classes, name='outputs')])
after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='Sequential(*args)', method_object=None, object_signature=None, function_args=[[data_augmentation, layers.Rescaling(1.0 / 255), layers.Conv2D(16, 3, padding='same', activation='relu'), layers.MaxPooling2D(), layers.Conv2D(32, 3, padding='same', activation='relu'), layers.MaxPooling2D(), layers.Conv2D(64, 3, padding='same', activation='relu'), layers.MaxPooling2D(), layers.Dropout(0.2), layers.Flatten(), layers.Dense(128, activation='relu'), layers.Dense(num_classes, name='outputs')]], function_kwargs={})
start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT()
model.compile(optimizer='adam', loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=['accuracy'])
after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='obj.compile(**kwargs)', method_object=model, object_signature=None, function_args=[], function_kwargs={'optimizer': 'adam', 'loss': tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), 'metrics': ['accuracy']})
start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT()
model.summary()
after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='obj.summary()', method_object=model, object_signature=None, function_args=[], function_kwargs={})
epochs = 15
start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT()
history = model.fit(train_ds, validation_data=val_ds, epochs=epochs)
after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='obj.fit(*args, **kwargs)', method_object=model, object_signature=None, function_args=[train_ds], function_kwargs={'validation_data': val_ds, 'epochs': epochs})
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
start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT()
sunflower_path = tf.keras.utils.get_file('Red_sunflower', origin=sunflower_url)
after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tf.keras.utils.get_file(*args, **kwargs)', method_object=None, object_signature=None, function_args=['Red_sunflower'], function_kwargs={'origin': sunflower_url})
start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT()
img = tf.keras.utils.load_img(sunflower_path, target_size=(img_height, img_width))
after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tf.keras.utils.load_img(*args, **kwargs)', method_object=None, object_signature=None, function_args=[sunflower_path], function_kwargs={'target_size': (img_height, img_width)})
start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT()
img_array = tf.keras.utils.img_to_array(img)
after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tf.keras.utils.img_to_array(*args)', method_object=None, object_signature=None, function_args=[img], function_kwargs={})
start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT()
img_array = tf.expand_dims(img_array, 0)
after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tf.expand_dims(*args)', method_object=None, object_signature=None, function_args=[img_array, 0], function_kwargs={})
start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT()
predictions = model.predict(img_array)
after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='obj.predict(*args)', method_object=model, object_signature=None, function_args=[img_array], function_kwargs={})
start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT()
score = tf.nn.softmax(predictions[0])
after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tf.nn.softmax(*args)', method_object=None, object_signature=None, function_args=[predictions[0]], function_kwargs={})
print('This image most likely belongs to {} with a {:.2f} percent confidence.'.format(class_names[np.argmax(score)], 100 * np.max(score)))
start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT()
converter = tf.lite.TFLiteConverter.from_keras_model(model)
after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tf.lite.TFLiteConverter.from_keras_model(*args)', method_object=None, object_signature=None, function_args=[model], function_kwargs={})
start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT()
tflite_model = converter.convert()
after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='obj.convert()', method_object=converter, object_signature=None, function_args=[], function_kwargs={})
with open('model.tflite', 'wb') as f:
    f.write(tflite_model)
TF_MODEL_FILE_PATH = 'model.tflite'
start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT()
interpreter = tf.lite.Interpreter(model_path=TF_MODEL_FILE_PATH)
after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tf.lite.Interpreter(**kwargs)', method_object=None, object_signature=None, function_args=[], function_kwargs={'model_path': TF_MODEL_FILE_PATH})
start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT()
interpreter.get_signature_list()
after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='obj.get_signature_list()', method_object=interpreter, object_signature=None, function_args=[], function_kwargs={})
start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT()
classify_lite = interpreter.get_signature_runner('serving_default')
after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='obj.get_signature_runner(*args)', method_object=interpreter, object_signature=None, function_args=['serving_default'], function_kwargs={})
classify_lite
predictions_lite = classify_lite(sequential_1_input=img_array)['outputs']
start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT()
score_lite = tf.nn.softmax(predictions_lite)
after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tf.nn.softmax(*args)', method_object=None, object_signature=None, function_args=[predictions_lite], function_kwargs={})
print('This image most likely belongs to {} with a {:.2f} percent confidence.'.format(class_names[np.argmax(score_lite)], 100 * np.max(score_lite)))
print(np.max(np.abs(predictions - predictions_lite)))
