import numpy as np
import time
import PIL.Image as Image
import matplotlib.pylab as plt
import tensorflow as tf
import tensorflow_hub as hub
import datetime
import os
from pathlib import Path
import dill as pickle
from tool.client.client_config import EXPERIMENT_DIR, EXPERIMENT_TAG, MAX_WAIT_S, WAIT_AFTER_RUN_S
from tool.server.send_request import send_request
from tool.server.function_details import FunctionDetails
current_path = os.path.abspath(__file__)
(immediate_folder, file_name) = os.path.split(current_path)
immediate_folder = os.path.basename(immediate_folder)
experiment_file_name = os.path.splitext(file_name)[0]
EXPERIMENT_FILE_PATH = EXPERIMENT_DIR / immediate_folder / EXPERIMENT_TAG / (experiment_file_name + '-energy.json')

def custom_method(func, imports: str, function_to_run: str, method_object=None, function_args: list=None, function_kwargs: dict=None, max_wait_secs=MAX_WAIT_S, custom_class=None, wait_after_run_secs=WAIT_AFTER_RUN_S):
    result = send_request(imports=imports, function_to_run=function_to_run, function_args=function_args, function_kwargs=function_kwargs, max_wait_secs=max_wait_secs, method_object=method_object, custom_class=custom_class, experiment_file_path=EXPERIMENT_FILE_PATH)
    return func
if __name__ == '__main__':
    print(EXPERIMENT_FILE_PATH)
mobilenet_v2 = 'https://tfhub.dev/google/tf2-preview/mobilenet_v2/classification/4'
inception_v3 = 'https://tfhub.dev/google/imagenet/inception_v3/classification/5'
classifier_model = mobilenet_v2
IMAGE_SHAPE = (224, 224)
classifier = custom_method(
tf.keras.Sequential([hub.KerasLayer(classifier_model, input_shape=IMAGE_SHAPE + (3,))]), imports='import numpy as np;import tensorflow_hub as hub;import PIL.Image as Image;import tensorflow as tf;import time;import datetime;import matplotlib.pylab as plt', function_to_run='tf.keras.Sequential(*args)', method_object=None, function_args=[eval('[\n    hub.KerasLayer(classifier_model, input_shape=IMAGE_SHAPE+(3,))\n]')], function_kwargs={}, max_wait_secs=0)
grace_hopper = custom_method(
tf.keras.utils.get_file('image.jpg', 'https://storage.googleapis.com/download.tensorflow.org/example_images/grace_hopper.jpg'), imports='import numpy as np;import tensorflow_hub as hub;import PIL.Image as Image;import tensorflow as tf;import time;import datetime;import matplotlib.pylab as plt', function_to_run='tf.keras.utils.get_file(*args)', method_object=None, function_args=[eval("'image.jpg'"), eval("'https://storage.googleapis.com/download.tensorflow.org/example_images/grace_hopper.jpg'")], function_kwargs={}, max_wait_secs=0)
grace_hopper = custom_method(
Image.open(grace_hopper).resize(IMAGE_SHAPE), imports='import numpy as np;import tensorflow_hub as hub;import PIL.Image as Image;import tensorflow as tf;import time;import datetime;import matplotlib.pylab as plt', function_to_run='Image.open(obj).resize(*args)', method_object=eval('grace_hopper'), function_args=[eval('IMAGE_SHAPE')], function_kwargs={}, max_wait_secs=0, custom_class=None)
grace_hopper
grace_hopper = np.array(grace_hopper) / 255.0
grace_hopper.shape
result = custom_method(
classifier.predict(grace_hopper[np.newaxis, ...]), imports='import numpy as np;import tensorflow_hub as hub;import PIL.Image as Image;import tensorflow as tf;import time;import datetime;import matplotlib.pylab as plt', function_to_run='obj.predict(*args)', method_object=eval('classifier'), function_args=[eval('grace_hopper[np.newaxis, ...]')], function_kwargs={}, max_wait_secs=0, custom_class=None)
result.shape
predicted_class = custom_method(
tf.math.argmax(result[0], axis=-1), imports='import numpy as np;import tensorflow_hub as hub;import PIL.Image as Image;import tensorflow as tf;import time;import datetime;import matplotlib.pylab as plt', function_to_run='tf.math.argmax(*args, **kwargs)', method_object=None, function_args=[eval('result[0]')], function_kwargs={'axis': eval('-1')}, max_wait_secs=0)
predicted_class
labels_path = custom_method(
tf.keras.utils.get_file('ImageNetLabels.txt', 'https://storage.googleapis.com/download.tensorflow.org/data/ImageNetLabels.txt'), imports='import numpy as np;import tensorflow_hub as hub;import PIL.Image as Image;import tensorflow as tf;import time;import datetime;import matplotlib.pylab as plt', function_to_run='tf.keras.utils.get_file(*args)', method_object=None, function_args=[eval("'ImageNetLabels.txt'"), eval("'https://storage.googleapis.com/download.tensorflow.org/data/ImageNetLabels.txt'")], function_kwargs={}, max_wait_secs=0)
imagenet_labels = np.array(open(labels_path).read().splitlines())
plt.imshow(grace_hopper)
plt.axis('off')
predicted_class_name = imagenet_labels[predicted_class]
_ = plt.title('Prediction: ' + predicted_class_name.title())
data_root = custom_method(
tf.keras.utils.get_file('flower_photos', 'https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz', untar=True), imports='import numpy as np;import tensorflow_hub as hub;import PIL.Image as Image;import tensorflow as tf;import time;import datetime;import matplotlib.pylab as plt', function_to_run='tf.keras.utils.get_file(*args, **kwargs)', method_object=None, function_args=[eval("'flower_photos'"), eval("'https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz'")], function_kwargs={'untar': eval('True')}, max_wait_secs=0)
batch_size = 32
img_height = 224
img_width = 224
train_ds = custom_method(
tf.keras.utils.image_dataset_from_directory(str(data_root), validation_split=0.2, subset='training', seed=123, image_size=(img_height, img_width), batch_size=batch_size), imports='import numpy as np;import tensorflow_hub as hub;import PIL.Image as Image;import tensorflow as tf;import time;import datetime;import matplotlib.pylab as plt', function_to_run='tf.keras.utils.image_dataset_from_directory(*args, **kwargs)', method_object=None, function_args=[eval('str(data_root)')], function_kwargs={'validation_split': eval('0.2'), 'subset': eval('"training"'), 'seed': eval('123'), 'image_size': eval('(img_height, img_width)'), 'batch_size': eval('batch_size')}, max_wait_secs=0)
val_ds = custom_method(
tf.keras.utils.image_dataset_from_directory(str(data_root), validation_split=0.2, subset='validation', seed=123, image_size=(img_height, img_width), batch_size=batch_size), imports='import numpy as np;import tensorflow_hub as hub;import PIL.Image as Image;import tensorflow as tf;import time;import datetime;import matplotlib.pylab as plt', function_to_run='tf.keras.utils.image_dataset_from_directory(*args, **kwargs)', method_object=None, function_args=[eval('str(data_root)')], function_kwargs={'validation_split': eval('0.2'), 'subset': eval('"validation"'), 'seed': eval('123'), 'image_size': eval('(img_height, img_width)'), 'batch_size': eval('batch_size')}, max_wait_secs=0)
class_names = np.array(train_ds.class_names)
print(class_names)
normalization_layer = custom_method(
tf.keras.layers.Rescaling(1.0 / 255), imports='import numpy as np;import tensorflow_hub as hub;import PIL.Image as Image;import tensorflow as tf;import time;import datetime;import matplotlib.pylab as plt', function_to_run='tf.keras.layers.Rescaling(*args)', method_object=None, function_args=[eval('1./255')], function_kwargs={}, max_wait_secs=0)
train_ds = custom_method(
train_ds.map(lambda x, y: (normalization_layer(x), y)), imports='import numpy as np;import tensorflow_hub as hub;import PIL.Image as Image;import tensorflow as tf;import time;import datetime;import matplotlib.pylab as plt', function_to_run='obj.map(*args)', method_object=eval('train_ds'), function_args=[eval('lambda x, y: (normalization_layer(x), y)')], function_kwargs={}, max_wait_secs=0, custom_class=None)
val_ds = custom_method(
val_ds.map(lambda x, y: (normalization_layer(x), y)), imports='import numpy as np;import tensorflow_hub as hub;import PIL.Image as Image;import tensorflow as tf;import time;import datetime;import matplotlib.pylab as plt', function_to_run='obj.map(*args)', method_object=eval('val_ds'), function_args=[eval('lambda x, y: (normalization_layer(x), y)')], function_kwargs={}, max_wait_secs=0, custom_class=None)
AUTOTUNE = tf.data.AUTOTUNE
train_ds = custom_method(
train_ds.cache().prefetch(buffer_size=AUTOTUNE), imports='import numpy as np;import tensorflow_hub as hub;import PIL.Image as Image;import tensorflow as tf;import time;import datetime;import matplotlib.pylab as plt', function_to_run='obj.cache().prefetch(**kwargs)', method_object=eval('train_ds'), function_args=[], function_kwargs={'buffer_size': eval('AUTOTUNE')}, max_wait_secs=0, custom_class=None)
val_ds = custom_method(
val_ds.cache().prefetch(buffer_size=AUTOTUNE), imports='import numpy as np;import tensorflow_hub as hub;import PIL.Image as Image;import tensorflow as tf;import time;import datetime;import matplotlib.pylab as plt', function_to_run='obj.cache().prefetch(**kwargs)', method_object=eval('val_ds'), function_args=[], function_kwargs={'buffer_size': eval('AUTOTUNE')}, max_wait_secs=0, custom_class=None)
for (image_batch, labels_batch) in train_ds:
    print(image_batch.shape)
    print(labels_batch.shape)
    break
result_batch = custom_method(
classifier.predict(train_ds), imports='import numpy as np;import tensorflow_hub as hub;import PIL.Image as Image;import tensorflow as tf;import time;import datetime;import matplotlib.pylab as plt', function_to_run='obj.predict(*args)', method_object=eval('classifier'), function_args=[eval('train_ds')], function_kwargs={}, max_wait_secs=0, custom_class=None)
predicted_class_names = imagenet_labels[custom_method(
tf.math.argmax(result_batch, axis=-1), imports='import numpy as np;import tensorflow_hub as hub;import PIL.Image as Image;import tensorflow as tf;import time;import datetime;import matplotlib.pylab as plt', function_to_run='tf.math.argmax(*args, **kwargs)', method_object=None, function_args=[eval('result_batch')], function_kwargs={'axis': eval('-1')}, max_wait_secs=0)]
predicted_class_names
plt.figure(figsize=(10, 9))
plt.subplots_adjust(hspace=0.5)
for n in range(30):
    plt.subplot(6, 5, n + 1)
    plt.imshow(image_batch[n])
    plt.title(predicted_class_names[n])
    plt.axis('off')
_ = plt.suptitle('ImageNet predictions')
mobilenet_v2 = 'https://tfhub.dev/google/tf2-preview/mobilenet_v2/feature_vector/4'
inception_v3 = 'https://tfhub.dev/google/tf2-preview/inception_v3/feature_vector/4'
feature_extractor_model = mobilenet_v2
feature_extractor_layer = hub.KerasLayer(feature_extractor_model, input_shape=(224, 224, 3), trainable=False)
feature_batch = feature_extractor_layer(image_batch)
print(feature_batch.shape)
num_classes = len(class_names)
model = custom_method(
tf.keras.Sequential([feature_extractor_layer, tf.keras.layers.Dense(num_classes)]), imports='import numpy as np;import tensorflow_hub as hub;import PIL.Image as Image;import tensorflow as tf;import time;import datetime;import matplotlib.pylab as plt', function_to_run='tf.keras.Sequential(*args)', method_object=None, function_args=[eval('[\n  feature_extractor_layer,\n  tf.keras.layers.Dense(num_classes)\n]')], function_kwargs={}, max_wait_secs=0)
custom_method(
model.summary(), imports='import numpy as np;import tensorflow_hub as hub;import PIL.Image as Image;import tensorflow as tf;import time;import datetime;import matplotlib.pylab as plt', function_to_run='obj.summary()', method_object=eval('model'), function_args=[], function_kwargs={}, max_wait_secs=0, custom_class=None)
predictions = custom_method(
model(image_batch), imports='import numpy as np;import tensorflow_hub as hub;import PIL.Image as Image;import tensorflow as tf;import time;import datetime;import matplotlib.pylab as plt', function_to_run='obj(*args)', method_object=eval('model'), function_args=[eval('image_batch')], function_kwargs={}, max_wait_secs=0, custom_class=None)
predictions.shape
custom_method(
model.compile(optimizer=tf.keras.optimizers.Adam(), loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=['acc']), imports='import numpy as np;import tensorflow_hub as hub;import PIL.Image as Image;import tensorflow as tf;import time;import datetime;import matplotlib.pylab as plt', function_to_run='obj.compile(**kwargs)', method_object=eval('model'), function_args=[], function_kwargs={'optimizer': eval('tf.keras.optimizers.Adam()'), 'loss': eval('tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)'), 'metrics': eval("['acc']")}, max_wait_secs=0, custom_class=None)
log_dir = 'logs/fit/' + datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
tensorboard_callback = custom_method(
tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1), imports='import numpy as np;import tensorflow_hub as hub;import PIL.Image as Image;import tensorflow as tf;import time;import datetime;import matplotlib.pylab as plt', function_to_run='tf.keras.callbacks.TensorBoard(**kwargs)', method_object=None, function_args=[], function_kwargs={'log_dir': eval('log_dir'), 'histogram_freq': eval('1')}, max_wait_secs=0)
NUM_EPOCHS = 10
history = custom_method(
model.fit(train_ds, validation_data=val_ds, epochs=NUM_EPOCHS, callbacks=tensorboard_callback), imports='import numpy as np;import tensorflow_hub as hub;import PIL.Image as Image;import tensorflow as tf;import time;import datetime;import matplotlib.pylab as plt', function_to_run='obj.fit(*args, **kwargs)', method_object=eval('model'), function_args=[eval('train_ds')], function_kwargs={'validation_data': eval('val_ds'), 'epochs': eval('NUM_EPOCHS'), 'callbacks': eval('tensorboard_callback')}, max_wait_secs=0, custom_class=None)
predicted_batch = custom_method(
model.predict(image_batch), imports='import numpy as np;import tensorflow_hub as hub;import PIL.Image as Image;import tensorflow as tf;import time;import datetime;import matplotlib.pylab as plt', function_to_run='obj.predict(*args)', method_object=eval('model'), function_args=[eval('image_batch')], function_kwargs={}, max_wait_secs=0, custom_class=None)
predicted_id = custom_method(
tf.math.argmax(predicted_batch, axis=-1), imports='import numpy as np;import tensorflow_hub as hub;import PIL.Image as Image;import tensorflow as tf;import time;import datetime;import matplotlib.pylab as plt', function_to_run='tf.math.argmax(*args, **kwargs)', method_object=None, function_args=[eval('predicted_batch')], function_kwargs={'axis': eval('-1')}, max_wait_secs=0)
predicted_label_batch = class_names[predicted_id]
print(predicted_label_batch)
plt.figure(figsize=(10, 9))
plt.subplots_adjust(hspace=0.5)
for n in range(30):
    plt.subplot(6, 5, n + 1)
    plt.imshow(image_batch[n])
    plt.title(predicted_label_batch[n].title())
    plt.axis('off')
_ = plt.suptitle('Model predictions')
t = time.time()
export_path = '/tmp/saved_models/{}'.format(int(t))
custom_method(
model.save(export_path), imports='import numpy as np;import tensorflow_hub as hub;import PIL.Image as Image;import tensorflow as tf;import time;import datetime;import matplotlib.pylab as plt', function_to_run='obj.save(*args)', method_object=eval('model'), function_args=[eval('export_path')], function_kwargs={}, max_wait_secs=0, custom_class=None)
export_path
reloaded = custom_method(
tf.keras.models.load_model(export_path), imports='import numpy as np;import tensorflow_hub as hub;import PIL.Image as Image;import tensorflow as tf;import time;import datetime;import matplotlib.pylab as plt', function_to_run='tf.keras.models.load_model(*args)', method_object=None, function_args=[eval('export_path')], function_kwargs={}, max_wait_secs=0)
result_batch = custom_method(
model.predict(image_batch), imports='import numpy as np;import tensorflow_hub as hub;import PIL.Image as Image;import tensorflow as tf;import time;import datetime;import matplotlib.pylab as plt', function_to_run='obj.predict(*args)', method_object=eval('model'), function_args=[eval('image_batch')], function_kwargs={}, max_wait_secs=0, custom_class=None)
reloaded_result_batch = custom_method(
reloaded.predict(image_batch), imports='import numpy as np;import tensorflow_hub as hub;import PIL.Image as Image;import tensorflow as tf;import time;import datetime;import matplotlib.pylab as plt', function_to_run='obj.predict(*args)', method_object=eval('reloaded'), function_args=[eval('image_batch')], function_kwargs={}, max_wait_secs=0, custom_class=None)
abs(reloaded_result_batch - result_batch).max()
reloaded_predicted_id = custom_method(
tf.math.argmax(reloaded_result_batch, axis=-1), imports='import numpy as np;import tensorflow_hub as hub;import PIL.Image as Image;import tensorflow as tf;import time;import datetime;import matplotlib.pylab as plt', function_to_run='tf.math.argmax(*args, **kwargs)', method_object=None, function_args=[eval('reloaded_result_batch')], function_kwargs={'axis': eval('-1')}, max_wait_secs=0)
reloaded_predicted_label_batch = class_names[reloaded_predicted_id]
print(reloaded_predicted_label_batch)
plt.figure(figsize=(10, 9))
plt.subplots_adjust(hspace=0.5)
for n in range(30):
    plt.subplot(6, 5, n + 1)
    plt.imshow(image_batch[n])
    plt.title(reloaded_predicted_label_batch[n].title())
    plt.axis('off')
_ = plt.suptitle('Model predictions')
