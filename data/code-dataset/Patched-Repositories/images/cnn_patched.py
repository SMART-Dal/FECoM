import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt
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
    return func
((train_images, train_labels), (test_images, test_labels)) = custom_method(
datasets.cifar10.load_data(), imports='import matplotlib.pyplot as plt;import tensorflow as tf;from tensorflow.keras import datasets, layers, models', function_to_run='datasets.cifar10.load_data()', method_object=None, object_signature=None, function_args=[], function_kwargs={})
(train_images, test_images) = (train_images / 255.0, test_images / 255.0)
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
plt.figure(figsize=(10, 10))
for i in range(25):
    plt.subplot(5, 5, i + 1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_images[i])
    plt.xlabel(class_names[train_labels[i][0]])
plt.show()
model = custom_method(
models.Sequential(), imports='import matplotlib.pyplot as plt;import tensorflow as tf;from tensorflow.keras import datasets, layers, models', function_to_run='models.Sequential()', method_object=None, object_signature=None, function_args=[], function_kwargs={})
custom_method(
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3))), imports='import matplotlib.pyplot as plt;import tensorflow as tf;from tensorflow.keras import datasets, layers, models', function_to_run='obj.add(*args)', method_object=eval('model'), object_signature=None, function_args=[eval("layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3))")], function_kwargs={}, custom_class=None)
custom_method(
model.add(layers.MaxPooling2D((2, 2))), imports='import matplotlib.pyplot as plt;import tensorflow as tf;from tensorflow.keras import datasets, layers, models', function_to_run='obj.add(*args)', method_object=eval('model'), object_signature=None, function_args=[eval('layers.MaxPooling2D((2, 2))')], function_kwargs={}, custom_class=None)
custom_method(
model.add(layers.Conv2D(64, (3, 3), activation='relu')), imports='import matplotlib.pyplot as plt;import tensorflow as tf;from tensorflow.keras import datasets, layers, models', function_to_run='obj.add(*args)', method_object=eval('model'), object_signature=None, function_args=[eval("layers.Conv2D(64, (3, 3), activation='relu')")], function_kwargs={}, custom_class=None)
custom_method(
model.add(layers.MaxPooling2D((2, 2))), imports='import matplotlib.pyplot as plt;import tensorflow as tf;from tensorflow.keras import datasets, layers, models', function_to_run='obj.add(*args)', method_object=eval('model'), object_signature=None, function_args=[eval('layers.MaxPooling2D((2, 2))')], function_kwargs={}, custom_class=None)
custom_method(
model.add(layers.Conv2D(64, (3, 3), activation='relu')), imports='import matplotlib.pyplot as plt;import tensorflow as tf;from tensorflow.keras import datasets, layers, models', function_to_run='obj.add(*args)', method_object=eval('model'), object_signature=None, function_args=[eval("layers.Conv2D(64, (3, 3), activation='relu')")], function_kwargs={}, custom_class=None)
custom_method(
model.summary(), imports='import matplotlib.pyplot as plt;import tensorflow as tf;from tensorflow.keras import datasets, layers, models', function_to_run='obj.summary()', method_object=eval('model'), object_signature=None, function_args=[], function_kwargs={}, custom_class=None)
custom_method(
model.add(layers.Flatten()), imports='import matplotlib.pyplot as plt;import tensorflow as tf;from tensorflow.keras import datasets, layers, models', function_to_run='obj.add(*args)', method_object=eval('model'), object_signature=None, function_args=[eval('layers.Flatten()')], function_kwargs={}, custom_class=None)
custom_method(
model.add(layers.Dense(64, activation='relu')), imports='import matplotlib.pyplot as plt;import tensorflow as tf;from tensorflow.keras import datasets, layers, models', function_to_run='obj.add(*args)', method_object=eval('model'), object_signature=None, function_args=[eval("layers.Dense(64, activation='relu')")], function_kwargs={}, custom_class=None)
custom_method(
model.add(layers.Dense(10)), imports='import matplotlib.pyplot as plt;import tensorflow as tf;from tensorflow.keras import datasets, layers, models', function_to_run='obj.add(*args)', method_object=eval('model'), object_signature=None, function_args=[eval('layers.Dense(10)')], function_kwargs={}, custom_class=None)
custom_method(
model.summary(), imports='import matplotlib.pyplot as plt;import tensorflow as tf;from tensorflow.keras import datasets, layers, models', function_to_run='obj.summary()', method_object=eval('model'), object_signature=None, function_args=[], function_kwargs={}, custom_class=None)
custom_method(
model.compile(optimizer='adam', loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=['accuracy']), imports='import matplotlib.pyplot as plt;import tensorflow as tf;from tensorflow.keras import datasets, layers, models', function_to_run='obj.compile(**kwargs)', method_object=eval('model'), object_signature=None, function_args=[], function_kwargs={'optimizer': eval("'adam'"), 'loss': eval('tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)'), 'metrics': eval("['accuracy']")}, custom_class=None)
history = custom_method(
model.fit(train_images, train_labels, epochs=10, validation_data=(test_images, test_labels)), imports='import matplotlib.pyplot as plt;import tensorflow as tf;from tensorflow.keras import datasets, layers, models', function_to_run='obj.fit(*args, **kwargs)', method_object=eval('model'), object_signature=None, function_args=[eval('train_images'), eval('train_labels')], function_kwargs={'epochs': eval('10'), 'validation_data': eval('(test_images, test_labels)')}, custom_class=None)
plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label='val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0.5, 1])
plt.legend(loc='lower right')
(test_loss, test_acc) = custom_method(
model.evaluate(test_images, test_labels, verbose=2), imports='import matplotlib.pyplot as plt;import tensorflow as tf;from tensorflow.keras import datasets, layers, models', function_to_run='obj.evaluate(*args, **kwargs)', method_object=eval('model'), object_signature=None, function_args=[eval('test_images'), eval('test_labels')], function_kwargs={'verbose': eval('2')}, custom_class=None)
print(test_acc)
