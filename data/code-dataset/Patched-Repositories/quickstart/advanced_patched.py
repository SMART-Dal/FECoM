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

def custom_method(func, imports: str, function_to_run: str, method_object=None, object_signature=None, function_args: list=None, function_kwargs: dict=None, custom_class=None):
    result = send_request(imports=imports, function_to_run=function_to_run, function_args=function_args, function_kwargs=function_kwargs, max_wait_secs=MAX_WAIT_S, wait_after_run_secs=WAIT_AFTER_RUN_S, method_object=method_object, object_signature=object_signature, custom_class=custom_class, experiment_file_path=EXPERIMENT_FILE_PATH)
    return func
print('TensorFlow version:', tf.__version__)
from tensorflow.keras.layers import Dense, Flatten, Conv2D
from tensorflow.keras import Model
mnist = tf.keras.datasets.mnist
((x_train, y_train), (x_test, y_test)) = mnist.load_data()
(x_train, x_test) = (x_train / 255.0, x_test / 255.0)
x_train = custom_method(
x_train[..., tf.newaxis].astype('float32'), imports='import tensorflow as tf;from tensorflow.keras import Model;from tensorflow.keras.layers import Dense, Flatten, Conv2D', function_to_run='x_train[..., tf.newaxis].astype(*args)', method_object=None, object_signature=None, function_args=[eval('"float32"')], function_kwargs={})
x_test = custom_method(
x_test[..., tf.newaxis].astype('float32'), imports='import tensorflow as tf;from tensorflow.keras import Model;from tensorflow.keras.layers import Dense, Flatten, Conv2D', function_to_run='x_test[..., tf.newaxis].astype(*args)', method_object=None, object_signature=None, function_args=[eval('"float32"')], function_kwargs={})
train_ds = custom_method(
tf.data.Dataset.from_tensor_slices((x_train, y_train)).shuffle(10000).batch(32), imports='import tensorflow as tf;from tensorflow.keras import Model;from tensorflow.keras.layers import Dense, Flatten, Conv2D', function_to_run='tf.data.Dataset.from_tensor_slices((x_train, y_train)).shuffle(10000).batch(*args)', method_object=None, object_signature=None, function_args=[eval('32')], function_kwargs={})
test_ds = custom_method(
tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(32), imports='import tensorflow as tf;from tensorflow.keras import Model;from tensorflow.keras.layers import Dense, Flatten, Conv2D', function_to_run='tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(*args)', method_object=None, object_signature=None, function_args=[eval('32')], function_kwargs={})

class MyModel(Model):

    def __init__(self):
        super(MyModel, self).__init__()
        self.conv1 = custom_method(
        Conv2D(32, 3, activation='relu'), imports='import tensorflow as tf;from tensorflow.keras import Model;from tensorflow.keras.layers import Dense, Flatten, Conv2D', function_to_run='Conv2D(*args, **kwargs)', method_object=None, object_signature=None, function_args=[eval('32'), eval('3')], function_kwargs={'activation': eval("'relu'")})
        self.flatten = custom_method(
        Flatten(), imports='import tensorflow as tf;from tensorflow.keras import Model;from tensorflow.keras.layers import Dense, Flatten, Conv2D', function_to_run='Flatten()', method_object=None, object_signature=None, function_args=[], function_kwargs={})
        self.d1 = custom_method(
        Dense(128, activation='relu'), imports='import tensorflow as tf;from tensorflow.keras import Model;from tensorflow.keras.layers import Dense, Flatten, Conv2D', function_to_run='Dense(*args, **kwargs)', method_object=None, object_signature=None, function_args=[eval('128')], function_kwargs={'activation': eval("'relu'")})
        self.d2 = custom_method(
        Dense(10), imports='import tensorflow as tf;from tensorflow.keras import Model;from tensorflow.keras.layers import Dense, Flatten, Conv2D', function_to_run='Dense(*args)', method_object=None, object_signature=None, function_args=[eval('10')], function_kwargs={})

    def call(self, x):
        x = self.conv1(x)
        x = self.flatten(x)
        x = self.d1(x)
        return self.d2(x)
model = MyModel()
loss_object = custom_method(
tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), imports='import tensorflow as tf;from tensorflow.keras import Model;from tensorflow.keras.layers import Dense, Flatten, Conv2D', function_to_run='tf.keras.losses.SparseCategoricalCrossentropy(**kwargs)', method_object=None, object_signature=None, function_args=[], function_kwargs={'from_logits': eval('True')})
optimizer = custom_method(
tf.keras.optimizers.Adam(), imports='import tensorflow as tf;from tensorflow.keras import Model;from tensorflow.keras.layers import Dense, Flatten, Conv2D', function_to_run='tf.keras.optimizers.Adam()', method_object=None, object_signature=None, function_args=[], function_kwargs={})
train_loss = custom_method(
tf.keras.metrics.Mean(name='train_loss'), imports='import tensorflow as tf;from tensorflow.keras import Model;from tensorflow.keras.layers import Dense, Flatten, Conv2D', function_to_run='tf.keras.metrics.Mean(**kwargs)', method_object=None, object_signature=None, function_args=[], function_kwargs={'name': eval("'train_loss'")})
train_accuracy = custom_method(
tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy'), imports='import tensorflow as tf;from tensorflow.keras import Model;from tensorflow.keras.layers import Dense, Flatten, Conv2D', function_to_run='tf.keras.metrics.SparseCategoricalAccuracy(**kwargs)', method_object=None, object_signature=None, function_args=[], function_kwargs={'name': eval("'train_accuracy'")})
test_loss = custom_method(
tf.keras.metrics.Mean(name='test_loss'), imports='import tensorflow as tf;from tensorflow.keras import Model;from tensorflow.keras.layers import Dense, Flatten, Conv2D', function_to_run='tf.keras.metrics.Mean(**kwargs)', method_object=None, object_signature=None, function_args=[], function_kwargs={'name': eval("'test_loss'")})
test_accuracy = custom_method(
tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy'), imports='import tensorflow as tf;from tensorflow.keras import Model;from tensorflow.keras.layers import Dense, Flatten, Conv2D', function_to_run='tf.keras.metrics.SparseCategoricalAccuracy(**kwargs)', method_object=None, object_signature=None, function_args=[], function_kwargs={'name': eval("'test_accuracy'")})

@tf.function
def train_step(images, labels):
    with custom_method(
    tf.GradientTape(), imports='import tensorflow as tf;from tensorflow.keras import Model;from tensorflow.keras.layers import Dense, Flatten, Conv2D', function_to_run='tf.GradientTape()', method_object=None, object_signature=None, function_args=[], function_kwargs={}) as tape:
        predictions = custom_method(
        model(images, training=True), imports='import tensorflow as tf;from tensorflow.keras import Model;from tensorflow.keras.layers import Dense, Flatten, Conv2D', function_to_run='obj(*args, **kwargs)', method_object=eval('model'), object_signature=None, function_args=[eval('images')], function_kwargs={'training': eval('True')}, custom_class="class MyModel(Model):\n  def __init__(self):\n    super(MyModel, self).__init__()\n    self.conv1 = Conv2D(32, 3, activation='relu')\n    self.flatten = Flatten()\n    self.d1 = Dense(128, activation='relu')\n    self.d2 = Dense(10)\n\n  def call(self, x):\n    x = self.conv1(x)\n    x = self.flatten(x)\n    x = self.d1(x)\n    return self.d2(x)")
        loss = custom_method(
        loss_object(labels, predictions), imports='import tensorflow as tf;from tensorflow.keras import Model;from tensorflow.keras.layers import Dense, Flatten, Conv2D', function_to_run='obj(*args)', method_object=eval('loss_object'), object_signature=None, function_args=[eval('labels'), eval('predictions')], function_kwargs={}, custom_class=None)
    gradients = tape.gradient(loss, model.trainable_variables)
    custom_method(
    optimizer.apply_gradients(zip(gradients, model.trainable_variables)), imports='import tensorflow as tf;from tensorflow.keras import Model;from tensorflow.keras.layers import Dense, Flatten, Conv2D', function_to_run='obj.apply_gradients(*args)', method_object=eval('optimizer'), object_signature=None, function_args=[eval('zip(gradients, model.trainable_variables)')], function_kwargs={}, custom_class=None)
    custom_method(
    train_loss(loss), imports='import tensorflow as tf;from tensorflow.keras import Model;from tensorflow.keras.layers import Dense, Flatten, Conv2D', function_to_run='obj(*args)', method_object=eval('train_loss'), object_signature=None, function_args=[eval('loss')], function_kwargs={}, custom_class=None)
    custom_method(
    train_accuracy(labels, predictions), imports='import tensorflow as tf;from tensorflow.keras import Model;from tensorflow.keras.layers import Dense, Flatten, Conv2D', function_to_run='obj(*args)', method_object=eval('train_accuracy'), object_signature=None, function_args=[eval('labels'), eval('predictions')], function_kwargs={}, custom_class=None)

@tf.function
def test_step(images, labels):
    predictions = custom_method(
    model(images, training=False), imports='import tensorflow as tf;from tensorflow.keras import Model;from tensorflow.keras.layers import Dense, Flatten, Conv2D', function_to_run='obj(*args, **kwargs)', method_object=eval('model'), object_signature=None, function_args=[eval('images')], function_kwargs={'training': eval('False')}, custom_class="class MyModel(Model):\n  def __init__(self):\n    super(MyModel, self).__init__()\n    self.conv1 = Conv2D(32, 3, activation='relu')\n    self.flatten = Flatten()\n    self.d1 = Dense(128, activation='relu')\n    self.d2 = Dense(10)\n\n  def call(self, x):\n    x = self.conv1(x)\n    x = self.flatten(x)\n    x = self.d1(x)\n    return self.d2(x)")
    t_loss = custom_method(
    loss_object(labels, predictions), imports='import tensorflow as tf;from tensorflow.keras import Model;from tensorflow.keras.layers import Dense, Flatten, Conv2D', function_to_run='obj(*args)', method_object=eval('loss_object'), object_signature=None, function_args=[eval('labels'), eval('predictions')], function_kwargs={}, custom_class=None)
    custom_method(
    test_loss(t_loss), imports='import tensorflow as tf;from tensorflow.keras import Model;from tensorflow.keras.layers import Dense, Flatten, Conv2D', function_to_run='obj(*args)', method_object=eval('test_loss'), object_signature=None, function_args=[eval('t_loss')], function_kwargs={}, custom_class=None)
    custom_method(
    test_accuracy(labels, predictions), imports='import tensorflow as tf;from tensorflow.keras import Model;from tensorflow.keras.layers import Dense, Flatten, Conv2D', function_to_run='obj(*args)', method_object=eval('test_accuracy'), object_signature=None, function_args=[eval('labels'), eval('predictions')], function_kwargs={}, custom_class=None)
EPOCHS = 5
for epoch in range(EPOCHS):
    custom_method(
    train_loss.reset_states(), imports='import tensorflow as tf;from tensorflow.keras import Model;from tensorflow.keras.layers import Dense, Flatten, Conv2D', function_to_run='obj.reset_states()', method_object=eval('train_loss'), object_signature=None, function_args=[], function_kwargs={}, custom_class=None)
    custom_method(
    train_accuracy.reset_states(), imports='import tensorflow as tf;from tensorflow.keras import Model;from tensorflow.keras.layers import Dense, Flatten, Conv2D', function_to_run='obj.reset_states()', method_object=eval('train_accuracy'), object_signature=None, function_args=[], function_kwargs={}, custom_class=None)
    custom_method(
    test_loss.reset_states(), imports='import tensorflow as tf;from tensorflow.keras import Model;from tensorflow.keras.layers import Dense, Flatten, Conv2D', function_to_run='obj.reset_states()', method_object=eval('test_loss'), object_signature=None, function_args=[], function_kwargs={}, custom_class=None)
    custom_method(
    test_accuracy.reset_states(), imports='import tensorflow as tf;from tensorflow.keras import Model;from tensorflow.keras.layers import Dense, Flatten, Conv2D', function_to_run='obj.reset_states()', method_object=eval('test_accuracy'), object_signature=None, function_args=[], function_kwargs={}, custom_class=None)
    for (images, labels) in train_ds:
        train_step(images, labels)
    for (test_images, test_labels) in test_ds:
        test_step(test_images, test_labels)
    print(f'Epoch {epoch + 1}, Loss: {train_loss.result()}, Accuracy: {train_accuracy.result() * 100}, Test Loss: {test_loss.result()}, Test Accuracy: {test_accuracy.result() * 100}')
