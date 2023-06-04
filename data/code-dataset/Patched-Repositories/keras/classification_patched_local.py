import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import sys
# NEW
# import os
# from pathlib import Path
# import dill as pickle
from tool.client.client_config import EXPERIMENT_DIR #, MAX_WAIT_S, WAIT_AFTER_RUN_S
# from tool.server.send_request import send_request
# from tool.server.function_details import FunctionDetails
# current_path = os.path.abspath(__file__)
# END NEW
experiment_number = sys.argv[1]
experiment_project = sys.argv[2]
EXPERIMENT_FILE_PATH = EXPERIMENT_DIR / 'local-execution' / experiment_project / f'experiment-{experiment_number}.json'

# NEW
# changed names to make function name clash less likely
from tool.measurement.execution import before_execution as before_execution_INSERTED_INTO_SCRIPT
from tool.measurement.execution import after_execution as after_execution_INSERTED_INTO_SCRIPT
# END NEW

print(tf.__version__)
fashion_mnist = tf.keras.datasets.fashion_mnist
((train_images, train_labels), (test_images, test_labels)) = fashion_mnist.load_data()
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
train_images.shape
len(train_labels)
train_labels
test_images.shape
len(test_labels)
plt.figure()
plt.imshow(train_images[0])
plt.colorbar()
plt.grid(False)
plt.show()
train_images = train_images / 255.0
test_images = test_images / 255.0
plt.figure(figsize=(10, 10))
for i in range(25):
    plt.subplot(5, 5, i + 1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_images[i], cmap=plt.cm.binary)
    plt.xlabel(class_names[train_labels[i]])
plt.show()
# custom_method(imports='import tensorflow as tf;import numpy as np;import matplotlib.pyplot as plt', function_to_run='tf.keras.Sequential(*args)', method_object=None, object_signature=None, function_args=[eval("[\n    tf.keras.layers.Flatten(input_shape=(28, 28)),\n    tf.keras.layers.Dense(128, activation='relu'),\n    tf.keras.layers.Dense(10)\n]")], function_kwargs={})
# NEW
start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT()
model = tf.keras.Sequential([tf.keras.layers.Flatten(input_shape=(28, 28)), tf.keras.layers.Dense(128, activation='relu'), tf.keras.layers.Dense(10)])
after_execution_INSERTED_INTO_SCRIPT(
    start_times=start_times_INSERTED_INTO_SCRIPT,
    experiment_file_path=EXPERIMENT_FILE_PATH,
    # TODO: we don't really need this format anymore for local execution. It would simplify the code
    # to simply have the full function/method signature as function_to_run, here 'tf.keras.Sequential'
    function_to_run='tf.keras.Sequential(*args)',
    function_args=[[tf.keras.layers.Flatten(input_shape=(28, 28)), tf.keras.layers.Dense(128, activation='relu'), tf.keras.layers.Dense(10)]]
)
# END NEW

# NEW
# custom_method(imports='import tensorflow as tf;import numpy as np;import matplotlib.pyplot as plt', function_to_run='obj.compile(**kwargs)', method_object=eval('model'), object_signature=None, function_args=[], function_kwargs={'optimizer': eval("'adam'"), 'loss': eval('tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)'), 'metrics': eval("['accuracy']")}, custom_class=None)
start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT()
model.compile(optimizer='adam', loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=['accuracy'])
after_execution_INSERTED_INTO_SCRIPT(
    start_times=start_times_INSERTED_INTO_SCRIPT,
    experiment_file_path=EXPERIMENT_FILE_PATH,
    function_to_run='obj.compile(**kwargs)',
    method_object=model,
    object_signature='tf.keras.Sequential',
    function_kwargs={'optimizer': 'adam', 'loss': tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), 'metrics': ['accuracy']}
)
# END NEW

# NEW
# custom_method(imports='import tensorflow as tf;import numpy as np;import matplotlib.pyplot as plt', function_to_run='obj.fit(*args, **kwargs)', method_object=eval('model'), object_signature=None, function_args=[eval('train_images'), eval('train_labels')], function_kwargs={'epochs': eval('10')}, custom_class=None)
start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT()
model.fit(train_images, train_labels, epochs=10)
after_execution_INSERTED_INTO_SCRIPT(
    start_times=start_times_INSERTED_INTO_SCRIPT,
    experiment_file_path=EXPERIMENT_FILE_PATH,
    function_to_run='obj.fit(*args, **kwargs)',
    method_object=model,
    object_signature='tf.keras.Sequential',
    function_args=['train_images', 'train_labels'],
    function_kwargs={'epochs': 10}
)
# END NEW

# NEW
# custom_method(imports='import tensorflow as tf;import numpy as np;import matplotlib.pyplot as plt', function_to_run='obj.evaluate(*args, **kwargs)', method_object=eval('model'), object_signature=None, function_args=[eval('test_images'), eval('test_labels')], function_kwargs={'verbose': eval('2')}, custom_class=None)
start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT()
(test_loss, test_acc) = model.evaluate(test_images, test_labels, verbose=2)
after_execution_INSERTED_INTO_SCRIPT(
    start_times=start_times_INSERTED_INTO_SCRIPT,
    experiment_file_path=EXPERIMENT_FILE_PATH,
    function_to_run='obj.evaluate(*args, **kwargs)',
    method_object=model,
    object_signature='tf.keras.Sequential',
    function_args=['test_images', 'test_labels'],
    function_kwargs={'verbose': '2'}
)
# END NEW

print('\nTest accuracy:', test_acc)

# NEW
# custom_method(imports='import tensorflow as tf;import numpy as np;import matplotlib.pyplot as plt', function_to_run='tf.keras.Sequential(*args)', method_object=None, object_signature=None, function_args=[eval('[model, \n                                         tf.keras.layers.Softmax()]')], function_kwargs={})
start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT()
probability_model = tf.keras.Sequential([model, tf.keras.layers.Softmax()])
after_execution_INSERTED_INTO_SCRIPT(
    start_times=start_times_INSERTED_INTO_SCRIPT,
    experiment_file_path=EXPERIMENT_FILE_PATH,
    function_to_run='tf.keras.Sequential(*args)',
    function_args=[[model, tf.keras.layers.Softmax()]]
)
# END NEW

# NEW
# custom_method(imports='import tensorflow as tf;import numpy as np;import matplotlib.pyplot as plt', function_to_run='obj.predict(*args)', method_object=eval('probability_model'), object_signature=None, function_args=[eval('test_images')], function_kwargs={}, custom_class=None)
start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT()
predictions = probability_model.predict(test_images)
after_execution_INSERTED_INTO_SCRIPT(
    start_times=start_times_INSERTED_INTO_SCRIPT,
    experiment_file_path=EXPERIMENT_FILE_PATH,
    function_to_run='obj.predict(*args)',
    method_object=probability_model,
    object_signature='tf.keras.Sequential',
    function_args=[test_images]
)
# END NEW

predictions[0]
np.argmax(predictions[0])
test_labels[0]

def plot_image(i, predictions_array, true_label, img):
    (true_label, img) = (true_label[i], img[i])
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(img, cmap=plt.cm.binary)
    predicted_label = np.argmax(predictions_array)
    if predicted_label == true_label:
        color = 'blue'
    else:
        color = 'red'
    plt.xlabel('{} {:2.0f}% ({})'.format(class_names[predicted_label], 100 * np.max(predictions_array), class_names[true_label]), color=color)

def plot_value_array(i, predictions_array, true_label):
    true_label = true_label[i]
    plt.grid(False)
    plt.xticks(range(10))
    plt.yticks([])
    thisplot = plt.bar(range(10), predictions_array, color='#777777')
    plt.ylim([0, 1])
    predicted_label = np.argmax(predictions_array)
    thisplot[predicted_label].set_color('red')
    thisplot[true_label].set_color('blue')
i = 0
plt.figure(figsize=(6, 3))
plt.subplot(1, 2, 1)
plot_image(i, predictions[i], test_labels, test_images)
plt.subplot(1, 2, 2)
plot_value_array(i, predictions[i], test_labels)
plt.show()
i = 12
plt.figure(figsize=(6, 3))
plt.subplot(1, 2, 1)
plot_image(i, predictions[i], test_labels, test_images)
plt.subplot(1, 2, 2)
plot_value_array(i, predictions[i], test_labels)
plt.show()
num_rows = 5
num_cols = 3
num_images = num_rows * num_cols
plt.figure(figsize=(2 * 2 * num_cols, 2 * num_rows))
for i in range(num_images):
    plt.subplot(num_rows, 2 * num_cols, 2 * i + 1)
    plot_image(i, predictions[i], test_labels, test_images)
    plt.subplot(num_rows, 2 * num_cols, 2 * i + 2)
    plot_value_array(i, predictions[i], test_labels)
plt.tight_layout()
plt.show()
img = test_images[1]
print(img.shape)
img = np.expand_dims(img, 0)
print(img.shape)

# NEW
# custom_method(imports='import tensorflow as tf;import numpy as np;import matplotlib.pyplot as plt', function_to_run='obj.predict(*args)', method_object=eval('probability_model'), object_signature=None, function_args=[eval('img')], function_kwargs={}, custom_class=None)
start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT()
predictions_single = probability_model.predict(img)
after_execution_INSERTED_INTO_SCRIPT(
    start_times=start_times_INSERTED_INTO_SCRIPT,
    experiment_file_path=EXPERIMENT_FILE_PATH,
    function_to_run='obj.predict(*args)',
    method_object=probability_model,
    object_signature='tf.keras.Sequential',
    function_args=[img]
)
# END NEW

print(predictions_single)
plot_value_array(1, predictions_single[0], test_labels)
_ = plt.xticks(range(10), class_names, rotation=45)
plt.show()
np.argmax(predictions_single[0])
