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
print(tf.config.list_physical_devices('GPU'))
layer = custom_method(
tf.keras.layers.Dense(100), imports='import tensorflow as tf', function_to_run='tf.keras.layers.Dense(*args)', method_object=None, object_signature=None, function_args=[eval('100')], function_kwargs={})
layer = custom_method(
tf.keras.layers.Dense(10, input_shape=(None, 5)), imports='import tensorflow as tf', function_to_run='tf.keras.layers.Dense(*args, **kwargs)', method_object=None, object_signature=None, function_args=[eval('10')], function_kwargs={'input_shape': eval('(None, 5)')})
layer(tf.zeros([10, 5]))
layer.variables
(layer.kernel, layer.bias)

class MyDenseLayer(tf.keras.layers.Layer):

    def __init__(self, num_outputs):
        super(MyDenseLayer, self).__init__()
        self.num_outputs = num_outputs

    def build(self, input_shape):
        self.kernel = self.add_weight('kernel', shape=[int(input_shape[-1]), self.num_outputs])

    def call(self, inputs):
        return custom_method(
        tf.matmul(inputs, self.kernel), imports='import tensorflow as tf', function_to_run='tf.matmul(*args)', method_object=None, object_signature=None, function_args=[eval('inputs'), eval('self.kernel')], function_kwargs={})
layer = MyDenseLayer(10)
_ = custom_method(
layer(tf.zeros([10, 5])), imports='import tensorflow as tf', function_to_run='obj(*args)', method_object=eval('layer'), object_signature=None, function_args=[eval('tf.zeros([10, 5])')], function_kwargs={}, custom_class='class MyDenseLayer(tf.keras.layers.Layer):\n  def __init__(self, num_outputs):\n    super(MyDenseLayer, self).__init__()\n    self.num_outputs = num_outputs\n\n  def build(self, input_shape):\n    self.kernel = self.add_weight("kernel",\n                                  shape=[int(input_shape[-1]),\n                                         self.num_outputs])\n\n  def call(self, inputs):\n    return tf.matmul(inputs, self.kernel)')
print([var.name for var in layer.trainable_variables])

class ResnetIdentityBlock(tf.keras.Model):

    def __init__(self, kernel_size, filters):
        super(ResnetIdentityBlock, self).__init__(name='')
        (filters1, filters2, filters3) = filters
        self.conv2a = custom_method(
        tf.keras.layers.Conv2D(filters1, (1, 1)), imports='import tensorflow as tf', function_to_run='tf.keras.layers.Conv2D(*args)', method_object=None, object_signature=None, function_args=[eval('filters1'), eval('(1, 1)')], function_kwargs={})
        self.bn2a = custom_method(
        tf.keras.layers.BatchNormalization(), imports='import tensorflow as tf', function_to_run='tf.keras.layers.BatchNormalization()', method_object=None, object_signature=None, function_args=[], function_kwargs={})
        self.conv2b = custom_method(
        tf.keras.layers.Conv2D(filters2, kernel_size, padding='same'), imports='import tensorflow as tf', function_to_run='tf.keras.layers.Conv2D(*args, **kwargs)', method_object=None, object_signature=None, function_args=[eval('filters2'), eval('kernel_size')], function_kwargs={'padding': eval("'same'")})
        self.bn2b = custom_method(
        tf.keras.layers.BatchNormalization(), imports='import tensorflow as tf', function_to_run='tf.keras.layers.BatchNormalization()', method_object=None, object_signature=None, function_args=[], function_kwargs={})
        self.conv2c = custom_method(
        tf.keras.layers.Conv2D(filters3, (1, 1)), imports='import tensorflow as tf', function_to_run='tf.keras.layers.Conv2D(*args)', method_object=None, object_signature=None, function_args=[eval('filters3'), eval('(1, 1)')], function_kwargs={})
        self.bn2c = custom_method(
        tf.keras.layers.BatchNormalization(), imports='import tensorflow as tf', function_to_run='tf.keras.layers.BatchNormalization()', method_object=None, object_signature=None, function_args=[], function_kwargs={})

    def call(self, input_tensor, training=False):
        x = self.conv2a(input_tensor)
        x = self.bn2a(x, training=training)
        x = custom_method(
        tf.nn.relu(x), imports='import tensorflow as tf', function_to_run='tf.nn.relu(*args)', method_object=None, object_signature=None, function_args=[eval('x')], function_kwargs={})
        x = self.conv2b(x)
        x = self.bn2b(x, training=training)
        x = custom_method(
        tf.nn.relu(x), imports='import tensorflow as tf', function_to_run='tf.nn.relu(*args)', method_object=None, object_signature=None, function_args=[eval('x')], function_kwargs={})
        x = self.conv2c(x)
        x = self.bn2c(x, training=training)
        x += input_tensor
        return custom_method(
        tf.nn.relu(x), imports='import tensorflow as tf', function_to_run='tf.nn.relu(*args)', method_object=None, object_signature=None, function_args=[eval('x')], function_kwargs={})
block = ResnetIdentityBlock(1, [1, 2, 3])
_ = custom_method(
block(tf.zeros([1, 2, 3, 3])), imports='import tensorflow as tf', function_to_run='obj(*args)', method_object=eval('block'), object_signature=None, function_args=[eval('tf.zeros([1, 2, 3, 3])')], function_kwargs={}, custom_class="class ResnetIdentityBlock(tf.keras.Model):\n  def __init__(self, kernel_size, filters):\n    super(ResnetIdentityBlock, self).__init__(name='')\n    filters1, filters2, filters3 = filters\n\n    self.conv2a = tf.keras.layers.Conv2D(filters1, (1, 1))\n    self.bn2a = tf.keras.layers.BatchNormalization()\n\n    self.conv2b = tf.keras.layers.Conv2D(filters2, kernel_size, padding='same')\n    self.bn2b = tf.keras.layers.BatchNormalization()\n\n    self.conv2c = tf.keras.layers.Conv2D(filters3, (1, 1))\n    self.bn2c = tf.keras.layers.BatchNormalization()\n\n  def call(self, input_tensor, training=False):\n    x = self.conv2a(input_tensor)\n    x = self.bn2a(x, training=training)\n    x = tf.nn.relu(x)\n\n    x = self.conv2b(x)\n    x = self.bn2b(x, training=training)\n    x = tf.nn.relu(x)\n\n    x = self.conv2c(x)\n    x = self.bn2c(x, training=training)\n\n    x += input_tensor\n    return tf.nn.relu(x)")
block.layers
len(block.variables)
custom_method(
block.summary(), imports='import tensorflow as tf', function_to_run='obj.summary()', method_object=eval('block'), object_signature=None, function_args=[], function_kwargs={}, custom_class="class ResnetIdentityBlock(tf.keras.Model):\n  def __init__(self, kernel_size, filters):\n    super(ResnetIdentityBlock, self).__init__(name='')\n    filters1, filters2, filters3 = filters\n\n    self.conv2a = tf.keras.layers.Conv2D(filters1, (1, 1))\n    self.bn2a = tf.keras.layers.BatchNormalization()\n\n    self.conv2b = tf.keras.layers.Conv2D(filters2, kernel_size, padding='same')\n    self.bn2b = tf.keras.layers.BatchNormalization()\n\n    self.conv2c = tf.keras.layers.Conv2D(filters3, (1, 1))\n    self.bn2c = tf.keras.layers.BatchNormalization()\n\n  def call(self, input_tensor, training=False):\n    x = self.conv2a(input_tensor)\n    x = self.bn2a(x, training=training)\n    x = tf.nn.relu(x)\n\n    x = self.conv2b(x)\n    x = self.bn2b(x, training=training)\n    x = tf.nn.relu(x)\n\n    x = self.conv2c(x)\n    x = self.bn2c(x, training=training)\n\n    x += input_tensor\n    return tf.nn.relu(x)")
my_seq = custom_method(
tf.keras.Sequential([tf.keras.layers.Conv2D(1, (1, 1), input_shape=(None, None, 3)), tf.keras.layers.BatchNormalization(), tf.keras.layers.Conv2D(2, 1, padding='same'), tf.keras.layers.BatchNormalization(), tf.keras.layers.Conv2D(3, (1, 1)), tf.keras.layers.BatchNormalization()]), imports='import tensorflow as tf', function_to_run='tf.keras.Sequential(*args)', method_object=None, object_signature=None, function_args=[eval("[tf.keras.layers.Conv2D(1, (1, 1),\n                                                    input_shape=(\n                                                        None, None, 3)),\n                             tf.keras.layers.BatchNormalization(),\n                             tf.keras.layers.Conv2D(2, 1,\n                                                    padding='same'),\n                             tf.keras.layers.BatchNormalization(),\n                             tf.keras.layers.Conv2D(3, (1, 1)),\n                             tf.keras.layers.BatchNormalization()]")], function_kwargs={})
my_seq(tf.zeros([1, 2, 3, 3]))
my_seq.summary()
