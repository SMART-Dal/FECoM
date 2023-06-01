import tensorflow as tf
import sys
from tool.client.client_config import EXPERIMENT_DIR
from tool.server.local_execution import before_execution as before_execution_INSERTED_INTO_SCRIPT
from tool.server.local_execution import after_execution as after_execution_INSERTED_INTO_SCRIPT
experiment_number = sys.argv[1]
experiment_project = sys.argv[2]
EXPERIMENT_FILE_PATH = EXPERIMENT_DIR / 'method-level' / experiment_project / f'experiment-{experiment_number}.json'
print(tf.config.list_physical_devices('GPU'))
start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT()
layer = tf.keras.layers.Dense(100)
after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tf.keras.layers.Dense(*args)', method_object=None, object_signature=None, function_args=[100], function_kwargs={})
start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT()
layer = tf.keras.layers.Dense(10, input_shape=(None, 5))
after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tf.keras.layers.Dense(*args, **kwargs)', method_object=None, object_signature=None, function_args=[10], function_kwargs={'input_shape': (None, 5)})
start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT()
layer(tf.zeros([10, 5]))
after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='obj(*args)', method_object=layer, object_signature=None, function_args=[tf.zeros([10, 5])], function_kwargs={})
layer.variables
(layer.kernel, layer.bias)

class MyDenseLayer(tf.keras.layers.Layer):

    def __init__(self, num_outputs):
        super(MyDenseLayer, self).__init__()
        self.num_outputs = num_outputs

    def build(self, input_shape):
        self.kernel = self.add_weight('kernel', shape=[int(input_shape[-1]), self.num_outputs])

    def call(self, inputs):
        return tf.matmul(inputs, self.kernel)
layer = MyDenseLayer(10)
start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT()
_ = layer(tf.zeros([10, 5]))
after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='obj(*args)', method_object='layer', object_signature=None, function_args=[tf.zeros([10, 5])], function_kwargs={})
print([var.name for var in layer.trainable_variables])

class ResnetIdentityBlock(tf.keras.Model):

    def __init__(self, kernel_size, filters):
        super(ResnetIdentityBlock, self).__init__(name='')
        (filters1, filters2, filters3) = filters
        start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT()
        self.conv2a = tf.keras.layers.Conv2D(filters1, (1, 1))
        after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tf.keras.layers.Conv2D(*args)', method_object=None, object_signature=None, function_args=[filters1, (1, 1)], function_kwargs={})
        start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT()
        self.bn2a = tf.keras.layers.BatchNormalization()
        after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tf.keras.layers.BatchNormalization()', method_object=None, object_signature=None, function_args=[], function_kwargs={})
        start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT()
        self.conv2b = tf.keras.layers.Conv2D(filters2, kernel_size, padding='same')
        after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tf.keras.layers.Conv2D(*args, **kwargs)', method_object=None, object_signature=None, function_args=[filters2, kernel_size], function_kwargs={'padding': 'same'})
        start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT()
        self.bn2b = tf.keras.layers.BatchNormalization()
        after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tf.keras.layers.BatchNormalization()', method_object=None, object_signature=None, function_args=[], function_kwargs={})
        start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT()
        self.conv2c = tf.keras.layers.Conv2D(filters3, (1, 1))
        after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tf.keras.layers.Conv2D(*args)', method_object=None, object_signature=None, function_args=[filters3, (1, 1)], function_kwargs={})
        start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT()
        self.bn2c = tf.keras.layers.BatchNormalization()
        after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tf.keras.layers.BatchNormalization()', method_object=None, object_signature=None, function_args=[], function_kwargs={})

    def call(self, input_tensor, training=False):
        x = self.conv2a(input_tensor)
        x = self.bn2a(x, training=training)
        start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT()
        x = tf.nn.relu(x)
        after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tf.nn.relu(*args)', method_object=None, object_signature=None, function_args=[x], function_kwargs={})
        x = self.conv2b(x)
        x = self.bn2b(x, training=training)
        start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT()
        x = tf.nn.relu(x)
        after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tf.nn.relu(*args)', method_object=None, object_signature=None, function_args=[x], function_kwargs={})
        x = self.conv2c(x)
        x = self.bn2c(x, training=training)
        x += input_tensor
        return tf.nn.relu(x)
block = ResnetIdentityBlock(1, [1, 2, 3])
start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT()
_ = block(tf.zeros([1, 2, 3, 3]))
after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='obj(*args)', method_object='block', object_signature=None, function_args=[tf.zeros([1, 2, 3, 3])], function_kwargs={})
block.layers
len(block.variables)
start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT()
block.summary()
after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='obj.summary()', method_object='block', object_signature=None, function_args=[], function_kwargs={})
start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT()
my_seq = tf.keras.Sequential([tf.keras.layers.Conv2D(1, (1, 1), input_shape=(None, None, 3)), tf.keras.layers.BatchNormalization(), tf.keras.layers.Conv2D(2, 1, padding='same'), tf.keras.layers.BatchNormalization(), tf.keras.layers.Conv2D(3, (1, 1)), tf.keras.layers.BatchNormalization()])
after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tf.keras.Sequential(*args)', method_object=None, object_signature=None, function_args=[[tf.keras.layers.Conv2D(1, (1, 1), input_shape=(None, None, 3)), tf.keras.layers.BatchNormalization(), tf.keras.layers.Conv2D(2, 1, padding='same'), tf.keras.layers.BatchNormalization(), tf.keras.layers.Conv2D(3, (1, 1)), tf.keras.layers.BatchNormalization()]], function_kwargs={})
start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT()
my_seq(tf.zeros([1, 2, 3, 3]))
after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='obj(*args)', method_object=my_seq, object_signature=None, function_args=[tf.zeros([1, 2, 3, 3])], function_kwargs={})
start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT()
my_seq.summary()
after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='obj.summary()', method_object=my_seq, object_signature=None, function_args=[], function_kwargs={})
