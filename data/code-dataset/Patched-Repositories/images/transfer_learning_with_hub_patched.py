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
import sys
import numpy as np
from tool.client.client_config import EXPERIMENT_DIR, MAX_WAIT_S, WAIT_AFTER_RUN_S
from tool.server.send_request import send_request
from tool.server.function_details import FunctionDetails
import json
current_path = os.path.abspath(__file__)
experiment_number = sys.argv[1]
experiment_project = sys.argv[2]
EXPERIMENT_FILE_PATH = EXPERIMENT_DIR / 'method-level' / experiment_project / f'experiment-{experiment_number}.json'
skip_calls_file_path = EXPERIMENT_FILE_PATH.parent / 'skip_calls.json'
if skip_calls_file_path.exists():
    with open(skip_calls_file_path, 'r') as f:
        skip_calls = json.load(f)
else:
    skip_calls = []
    with open(skip_calls_file_path, 'w') as f:
        json.dump(skip_calls, f)

def custom_method(imports: str, function_to_run: str, method_object=None, object_signature=None, function_args: list=None, function_kwargs: dict=None, custom_class=None):
    if skip_calls is not None and any((call['function_to_run'] == function_to_run and np.array_equal(call['function_args'], function_args) and (call['function_kwargs'] == function_kwargs) for call in skip_calls)):
        print('skipping call: ', function_to_run)
        return
    result = send_request(imports=imports, function_to_run=function_to_run, function_args=function_args, function_kwargs=function_kwargs, max_wait_secs=MAX_WAIT_S, wait_after_run_secs=WAIT_AFTER_RUN_S, method_object=method_object, object_signature=object_signature, custom_class=custom_class, experiment_file_path=EXPERIMENT_FILE_PATH)
    if result is not None and isinstance(result, dict) and (len(result) == 1):
        energy_data = next(iter(result.values()))
        if skip_calls is not None and 'start_time_perf' in energy_data['times'] and ('end_time_perf' in energy_data['times']) and ('start_time_nvidia' in energy_data['times']) and ('end_time_nvidia' in energy_data['times']) and (energy_data['times']['start_time_perf'] == energy_data['times']['end_time_perf']) and (energy_data['times']['start_time_nvidia'] == energy_data['times']['end_time_nvidia']):
            call_to_skip = {'function_to_run': function_to_run, 'function_args': function_args, 'function_kwargs': function_kwargs}
            try:
                json.dumps(call_to_skip)
                if call_to_skip not in skip_calls:
                    skip_calls.append(call_to_skip)
                    with open(skip_calls_file_path, 'w') as f:
                        json.dump(skip_calls, f)
                    print('skipping call added, current list is: ', skip_calls)
                else:
                    print('Skipping call already exists.')
            except TypeError:
                print('Ignore: Skipping call is not JSON serializable, skipping append and dump.')
    else:
        print('Invalid dictionary object or does not have one key-value pair.')
mobilenet_v2 = 'https://tfhub.dev/google/tf2-preview/mobilenet_v2/classification/4'
inception_v3 = 'https://tfhub.dev/google/imagenet/inception_v3/classification/5'
classifier_model = mobilenet_v2
IMAGE_SHAPE = (224, 224)
custom_method(imports='import time;import PIL.Image as Image;import tensorflow as tf;import tensorflow_hub as hub;import datetime;import numpy as np;import matplotlib.pylab as plt', function_to_run='tf.keras.Sequential(*args)', method_object=None, object_signature=None, function_args=[eval('[\n    hub.KerasLayer(classifier_model, input_shape=IMAGE_SHAPE+(3,))\n]')], function_kwargs={})
classifier = tf.keras.Sequential([hub.KerasLayer(classifier_model, input_shape=IMAGE_SHAPE + (3,))])
custom_method(imports='import time;import PIL.Image as Image;import tensorflow as tf;import tensorflow_hub as hub;import datetime;import numpy as np;import matplotlib.pylab as plt', function_to_run='tf.keras.utils.get_file(*args)', method_object=None, object_signature=None, function_args=[eval("'image.jpg'"), eval("'https://storage.googleapis.com/download.tensorflow.org/example_images/grace_hopper.jpg'")], function_kwargs={})
grace_hopper = tf.keras.utils.get_file('image.jpg', 'https://storage.googleapis.com/download.tensorflow.org/example_images/grace_hopper.jpg')
grace_hopper = Image.open(grace_hopper).resize(IMAGE_SHAPE)
grace_hopper
grace_hopper = np.array(grace_hopper) / 255.0
grace_hopper.shape
custom_method(imports='import time;import PIL.Image as Image;import tensorflow as tf;import tensorflow_hub as hub;import datetime;import numpy as np;import matplotlib.pylab as plt', function_to_run='obj.predict(*args)', method_object=eval('classifier'), object_signature=None, function_args=[eval('grace_hopper[np.newaxis, ...]')], function_kwargs={}, custom_class=None)
result = classifier.predict(grace_hopper[np.newaxis, ...])
result.shape
custom_method(imports='import time;import PIL.Image as Image;import tensorflow as tf;import tensorflow_hub as hub;import datetime;import numpy as np;import matplotlib.pylab as plt', function_to_run='tf.math.argmax(*args, **kwargs)', method_object=None, object_signature=None, function_args=[eval('result[0]')], function_kwargs={'axis': eval('-1')})
predicted_class = tf.math.argmax(result[0], axis=-1)
predicted_class
custom_method(imports='import time;import PIL.Image as Image;import tensorflow as tf;import tensorflow_hub as hub;import datetime;import numpy as np;import matplotlib.pylab as plt', function_to_run='tf.keras.utils.get_file(*args)', method_object=None, object_signature=None, function_args=[eval("'ImageNetLabels.txt'"), eval("'https://storage.googleapis.com/download.tensorflow.org/data/ImageNetLabels.txt'")], function_kwargs={})
labels_path = tf.keras.utils.get_file('ImageNetLabels.txt', 'https://storage.googleapis.com/download.tensorflow.org/data/ImageNetLabels.txt')
imagenet_labels = np.array(open(labels_path).read().splitlines())
plt.imshow(grace_hopper)
plt.axis('off')
predicted_class_name = imagenet_labels[predicted_class]
_ = plt.title('Prediction: ' + predicted_class_name.title())
custom_method(imports='import time;import PIL.Image as Image;import tensorflow as tf;import tensorflow_hub as hub;import datetime;import numpy as np;import matplotlib.pylab as plt', function_to_run='tf.keras.utils.get_file(*args, **kwargs)', method_object=None, object_signature=None, function_args=[eval("'flower_photos'"), eval("'https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz'")], function_kwargs={'untar': eval('True')})
data_root = tf.keras.utils.get_file('flower_photos', 'https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz', untar=True)
batch_size = 32
img_height = 224
img_width = 224
custom_method(imports='import time;import PIL.Image as Image;import tensorflow as tf;import tensorflow_hub as hub;import datetime;import numpy as np;import matplotlib.pylab as plt', function_to_run='tf.keras.utils.image_dataset_from_directory(*args, **kwargs)', method_object=None, object_signature=None, function_args=[eval('str(data_root)')], function_kwargs={'validation_split': eval('0.2'), 'subset': eval('"training"'), 'seed': eval('123'), 'image_size': eval('(img_height, img_width)'), 'batch_size': eval('batch_size')})
train_ds = tf.keras.utils.image_dataset_from_directory(str(data_root), validation_split=0.2, subset='training', seed=123, image_size=(img_height, img_width), batch_size=batch_size)
custom_method(imports='import time;import PIL.Image as Image;import tensorflow as tf;import tensorflow_hub as hub;import datetime;import numpy as np;import matplotlib.pylab as plt', function_to_run='tf.keras.utils.image_dataset_from_directory(*args, **kwargs)', method_object=None, object_signature=None, function_args=[eval('str(data_root)')], function_kwargs={'validation_split': eval('0.2'), 'subset': eval('"validation"'), 'seed': eval('123'), 'image_size': eval('(img_height, img_width)'), 'batch_size': eval('batch_size')})
val_ds = tf.keras.utils.image_dataset_from_directory(str(data_root), validation_split=0.2, subset='validation', seed=123, image_size=(img_height, img_width), batch_size=batch_size)
class_names = np.array(train_ds.class_names)
print(class_names)
custom_method(imports='import time;import PIL.Image as Image;import tensorflow as tf;import tensorflow_hub as hub;import datetime;import numpy as np;import matplotlib.pylab as plt', function_to_run='tf.keras.layers.Rescaling(*args)', method_object=None, object_signature=None, function_args=[eval('1./255')], function_kwargs={})
normalization_layer = tf.keras.layers.Rescaling(1.0 / 255)
custom_method(imports='import time;import PIL.Image as Image;import tensorflow as tf;import tensorflow_hub as hub;import datetime;import numpy as np;import matplotlib.pylab as plt', function_to_run='obj.map(*args)', method_object=eval('train_ds'), object_signature=None, function_args=[eval('lambda x, y: (normalization_layer(x), y)')], function_kwargs={}, custom_class=None)
train_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
custom_method(imports='import time;import PIL.Image as Image;import tensorflow as tf;import tensorflow_hub as hub;import datetime;import numpy as np;import matplotlib.pylab as plt', function_to_run='obj.map(*args)', method_object=eval('val_ds'), object_signature=None, function_args=[eval('lambda x, y: (normalization_layer(x), y)')], function_kwargs={}, custom_class=None)
val_ds = val_ds.map(lambda x, y: (normalization_layer(x), y))
AUTOTUNE = tf.data.AUTOTUNE
custom_method(imports='import time;import PIL.Image as Image;import tensorflow as tf;import tensorflow_hub as hub;import datetime;import numpy as np;import matplotlib.pylab as plt', function_to_run='obj.cache().prefetch(**kwargs)', method_object=eval('train_ds'), object_signature=None, function_args=[], function_kwargs={'buffer_size': eval('AUTOTUNE')}, custom_class=None)
train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
custom_method(imports='import time;import PIL.Image as Image;import tensorflow as tf;import tensorflow_hub as hub;import datetime;import numpy as np;import matplotlib.pylab as plt', function_to_run='obj.cache().prefetch(**kwargs)', method_object=eval('val_ds'), object_signature=None, function_args=[], function_kwargs={'buffer_size': eval('AUTOTUNE')}, custom_class=None)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)
for (image_batch, labels_batch) in train_ds:
    print(image_batch.shape)
    print(labels_batch.shape)
    break
custom_method(imports='import time;import PIL.Image as Image;import tensorflow as tf;import tensorflow_hub as hub;import datetime;import numpy as np;import matplotlib.pylab as plt', function_to_run='obj.predict(*args)', method_object=eval('classifier'), object_signature=None, function_args=[eval('train_ds')], function_kwargs={}, custom_class=None)
result_batch = classifier.predict(train_ds)
predicted_class_names = imagenet_labels[tf.math.argmax(result_batch, axis=-1)]
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
custom_method(imports='import time;import PIL.Image as Image;import tensorflow as tf;import tensorflow_hub as hub;import datetime;import numpy as np;import matplotlib.pylab as plt', function_to_run='tf.keras.Sequential(*args)', method_object=None, object_signature=None, function_args=[eval('[\n  feature_extractor_layer,\n  tf.keras.layers.Dense(num_classes)\n]')], function_kwargs={})
model = tf.keras.Sequential([feature_extractor_layer, tf.keras.layers.Dense(num_classes)])
custom_method(imports='import time;import PIL.Image as Image;import tensorflow as tf;import tensorflow_hub as hub;import datetime;import numpy as np;import matplotlib.pylab as plt', function_to_run='obj.summary()', method_object=eval('model'), object_signature=None, function_args=[], function_kwargs={}, custom_class=None)
model.summary()
custom_method(imports='import time;import PIL.Image as Image;import tensorflow as tf;import tensorflow_hub as hub;import datetime;import numpy as np;import matplotlib.pylab as plt', function_to_run='obj(*args)', method_object=eval('model'), object_signature=None, function_args=[eval('image_batch')], function_kwargs={}, custom_class=None)
predictions = model(image_batch)
predictions.shape
custom_method(imports='import time;import PIL.Image as Image;import tensorflow as tf;import tensorflow_hub as hub;import datetime;import numpy as np;import matplotlib.pylab as plt', function_to_run='obj.compile(**kwargs)', method_object=eval('model'), object_signature=None, function_args=[], function_kwargs={'optimizer': eval('tf.keras.optimizers.Adam()'), 'loss': eval('tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)'), 'metrics': eval("['acc']")}, custom_class=None)
model.compile(optimizer=tf.keras.optimizers.Adam(), loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=['acc'])
log_dir = 'logs/fit/' + datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
custom_method(imports='import time;import PIL.Image as Image;import tensorflow as tf;import tensorflow_hub as hub;import datetime;import numpy as np;import matplotlib.pylab as plt', function_to_run='tf.keras.callbacks.TensorBoard(**kwargs)', method_object=None, object_signature=None, function_args=[], function_kwargs={'log_dir': eval('log_dir'), 'histogram_freq': eval('1')})
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
NUM_EPOCHS = 10
custom_method(imports='import time;import PIL.Image as Image;import tensorflow as tf;import tensorflow_hub as hub;import datetime;import numpy as np;import matplotlib.pylab as plt', function_to_run='obj.fit(*args, **kwargs)', method_object=eval('model'), object_signature=None, function_args=[eval('train_ds')], function_kwargs={'validation_data': eval('val_ds'), 'epochs': eval('NUM_EPOCHS'), 'callbacks': eval('tensorboard_callback')}, custom_class=None)
history = model.fit(train_ds, validation_data=val_ds, epochs=NUM_EPOCHS, callbacks=tensorboard_callback)
custom_method(imports='import time;import PIL.Image as Image;import tensorflow as tf;import tensorflow_hub as hub;import datetime;import numpy as np;import matplotlib.pylab as plt', function_to_run='obj.predict(*args)', method_object=eval('model'), object_signature=None, function_args=[eval('image_batch')], function_kwargs={}, custom_class=None)
predicted_batch = model.predict(image_batch)
custom_method(imports='import time;import PIL.Image as Image;import tensorflow as tf;import tensorflow_hub as hub;import datetime;import numpy as np;import matplotlib.pylab as plt', function_to_run='tf.math.argmax(*args, **kwargs)', method_object=None, object_signature=None, function_args=[eval('predicted_batch')], function_kwargs={'axis': eval('-1')})
predicted_id = tf.math.argmax(predicted_batch, axis=-1)
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
custom_method(imports='import time;import PIL.Image as Image;import tensorflow as tf;import tensorflow_hub as hub;import datetime;import numpy as np;import matplotlib.pylab as plt', function_to_run='obj.save(*args)', method_object=eval('model'), object_signature=None, function_args=[eval('export_path')], function_kwargs={}, custom_class=None)
model.save(export_path)
export_path
custom_method(imports='import time;import PIL.Image as Image;import tensorflow as tf;import tensorflow_hub as hub;import datetime;import numpy as np;import matplotlib.pylab as plt', function_to_run='tf.keras.models.load_model(*args)', method_object=None, object_signature=None, function_args=[eval('export_path')], function_kwargs={})
reloaded = tf.keras.models.load_model(export_path)
custom_method(imports='import time;import PIL.Image as Image;import tensorflow as tf;import tensorflow_hub as hub;import datetime;import numpy as np;import matplotlib.pylab as plt', function_to_run='obj.predict(*args)', method_object=eval('model'), object_signature=None, function_args=[eval('image_batch')], function_kwargs={}, custom_class=None)
result_batch = model.predict(image_batch)
custom_method(imports='import time;import PIL.Image as Image;import tensorflow as tf;import tensorflow_hub as hub;import datetime;import numpy as np;import matplotlib.pylab as plt', function_to_run='obj.predict(*args)', method_object=eval('reloaded'), object_signature=None, function_args=[eval('image_batch')], function_kwargs={}, custom_class=None)
reloaded_result_batch = reloaded.predict(image_batch)
abs(reloaded_result_batch - result_batch).max()
custom_method(imports='import time;import PIL.Image as Image;import tensorflow as tf;import tensorflow_hub as hub;import datetime;import numpy as np;import matplotlib.pylab as plt', function_to_run='tf.math.argmax(*args, **kwargs)', method_object=None, object_signature=None, function_args=[eval('reloaded_result_batch')], function_kwargs={'axis': eval('-1')})
reloaded_predicted_id = tf.math.argmax(reloaded_result_batch, axis=-1)
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
