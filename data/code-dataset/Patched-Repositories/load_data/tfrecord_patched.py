import tensorflow as tf
import numpy as np
import IPython.display as display
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

def _bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy()
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _float_feature(value):
    """Returns a float_list from a float / double."""
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

def _int64_feature(value):
    """Returns an int64_list from a bool / enum / int / uint."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))
print(_bytes_feature(b'test_string'))
print(_bytes_feature(u'test_bytes'.encode('utf-8')))
print(_float_feature(np.exp(1)))
print(_int64_feature(True))
print(_int64_feature(1))
feature = _float_feature(np.exp(1))
feature.SerializeToString()
n_observations = int(10000.0)
feature0 = np.random.choice([False, True], n_observations)
feature1 = np.random.randint(0, 5, n_observations)
strings = np.array([b'cat', b'dog', b'chicken', b'horse', b'goat'])
feature2 = strings[feature1]
feature3 = np.random.randn(n_observations)

def serialize_example(feature0, feature1, feature2, feature3):
    """
  Creates a tf.train.Example message ready to be written to a file.
  """
    feature = {'feature0': _int64_feature(feature0), 'feature1': _int64_feature(feature1), 'feature2': _bytes_feature(feature2), 'feature3': _float_feature(feature3)}
    custom_method(imports='import numpy as np;import tensorflow as tf;import IPython.display as display', function_to_run='tf.train.Example(**kwargs)', method_object=None, object_signature=None, function_args=[], function_kwargs={'features': eval('tf.train.Features(feature=feature)')})
    example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
    return example_proto.SerializeToString()
example_observation = []
serialized_example = serialize_example(False, 4, b'goat', 0.9876)
serialized_example
custom_method(imports='import numpy as np;import tensorflow as tf;import IPython.display as display', function_to_run='tf.train.Example.FromString(*args)', method_object=None, object_signature=None, function_args=[eval('serialized_example')], function_kwargs={})
example_proto = tf.train.Example.FromString(serialized_example)
example_proto
custom_method(imports='import numpy as np;import tensorflow as tf;import IPython.display as display', function_to_run='tf.data.Dataset.from_tensor_slices(*args)', method_object=None, object_signature=None, function_args=[eval('feature1')], function_kwargs={})
tf.data.Dataset.from_tensor_slices(feature1)
custom_method(imports='import numpy as np;import tensorflow as tf;import IPython.display as display', function_to_run='tf.data.Dataset.from_tensor_slices(*args)', method_object=None, object_signature=None, function_args=[eval('(feature0, feature1, feature2, feature3)')], function_kwargs={})
features_dataset = tf.data.Dataset.from_tensor_slices((feature0, feature1, feature2, feature3))
features_dataset
for (f0, f1, f2, f3) in features_dataset.take(1):
    print(f0)
    print(f1)
    print(f2)
    print(f3)

def tf_serialize_example(f0, f1, f2, f3):
    custom_method(imports='import numpy as np;import tensorflow as tf;import IPython.display as display', function_to_run='tf.py_function(*args)', method_object=None, object_signature=None, function_args=[eval('serialize_example'), eval('(f0, f1, f2, f3)'), eval('tf.string')], function_kwargs={})
    tf_string = tf.py_function(serialize_example, (f0, f1, f2, f3), tf.string)
    return tf.reshape(tf_string, ())
tf_serialize_example(f0, f1, f2, f3)
custom_method(imports='import numpy as np;import tensorflow as tf;import IPython.display as display', function_to_run='obj.map(*args)', method_object=eval('features_dataset'), object_signature=None, function_args=[eval('tf_serialize_example')], function_kwargs={}, custom_class=None)
serialized_features_dataset = features_dataset.map(tf_serialize_example)
serialized_features_dataset

def generator():
    for features in features_dataset:
        yield serialize_example(*features)
custom_method(imports='import numpy as np;import tensorflow as tf;import IPython.display as display', function_to_run='tf.data.Dataset.from_generator(*args, **kwargs)', method_object=None, object_signature=None, function_args=[eval('generator')], function_kwargs={'output_types': eval('tf.string'), 'output_shapes': eval('()')})
serialized_features_dataset = tf.data.Dataset.from_generator(generator, output_types=tf.string, output_shapes=())
serialized_features_dataset
filename = 'test.tfrecord'
custom_method(imports='import numpy as np;import tensorflow as tf;import IPython.display as display', function_to_run='tf.data.experimental.TFRecordWriter(*args)', method_object=None, object_signature=None, function_args=[eval('filename')], function_kwargs={})
writer = tf.data.experimental.TFRecordWriter(filename)
custom_method(imports='import numpy as np;import tensorflow as tf;import IPython.display as display', function_to_run='obj.write(*args)', method_object=eval('writer'), object_signature=None, function_args=[eval('serialized_features_dataset')], function_kwargs={}, custom_class=None)
writer.write(serialized_features_dataset)
filenames = [filename]
custom_method(imports='import numpy as np;import tensorflow as tf;import IPython.display as display', function_to_run='tf.data.TFRecordDataset(*args)', method_object=None, object_signature=None, function_args=[eval('filenames')], function_kwargs={})
raw_dataset = tf.data.TFRecordDataset(filenames)
raw_dataset
for raw_record in raw_dataset.take(10):
    print(repr(raw_record))
feature_description = {'feature0': tf.io.FixedLenFeature([], tf.int64, default_value=0), 'feature1': tf.io.FixedLenFeature([], tf.int64, default_value=0), 'feature2': tf.io.FixedLenFeature([], tf.string, default_value=''), 'feature3': tf.io.FixedLenFeature([], tf.float32, default_value=0.0)}

def _parse_function(example_proto):
    return tf.io.parse_single_example(example_proto, feature_description)
custom_method(imports='import numpy as np;import tensorflow as tf;import IPython.display as display', function_to_run='obj.map(*args)', method_object=eval('raw_dataset'), object_signature=None, function_args=[eval('_parse_function')], function_kwargs={}, custom_class=None)
parsed_dataset = raw_dataset.map(_parse_function)
parsed_dataset
for parsed_record in parsed_dataset.take(10):
    print(repr(parsed_record))
with tf.io.TFRecordWriter(filename) as writer:
    for i in range(n_observations):
        example = serialize_example(feature0[i], feature1[i], feature2[i], feature3[i])
        custom_method(imports='import numpy as np;import tensorflow as tf;import IPython.display as display', function_to_run='obj.write(*args)', method_object=eval('writer'), object_signature=None, function_args=[eval('example')], function_kwargs={}, custom_class=None)
        writer.write(example)
filenames = [filename]
custom_method(imports='import numpy as np;import tensorflow as tf;import IPython.display as display', function_to_run='tf.data.TFRecordDataset(*args)', method_object=None, object_signature=None, function_args=[eval('filenames')], function_kwargs={})
raw_dataset = tf.data.TFRecordDataset(filenames)
raw_dataset
for raw_record in raw_dataset.take(1):
    custom_method(imports='import numpy as np;import tensorflow as tf;import IPython.display as display', function_to_run='tf.train.Example()', method_object=None, object_signature=None, function_args=[], function_kwargs={})
    example = tf.train.Example()
    custom_method(imports='import numpy as np;import tensorflow as tf;import IPython.display as display', function_to_run='obj.ParseFromString(*args)', method_object=eval('example'), object_signature=None, function_args=[eval('raw_record.numpy()')], function_kwargs={}, custom_class=None)
    example.ParseFromString(raw_record.numpy())
    print(example)
result = {}
for (key, feature) in example.features.feature.items():
    kind = feature.WhichOneof('kind')
    result[key] = np.array(getattr(feature, kind).value)
result
custom_method(imports='import numpy as np;import tensorflow as tf;import IPython.display as display', function_to_run='tf.keras.utils.get_file(*args)', method_object=None, object_signature=None, function_args=[eval("'320px-Felis_catus-cat_on_snow.jpg'"), eval("'https://storage.googleapis.com/download.tensorflow.org/example_images/320px-Felis_catus-cat_on_snow.jpg'")], function_kwargs={})
cat_in_snow = tf.keras.utils.get_file('320px-Felis_catus-cat_on_snow.jpg', 'https://storage.googleapis.com/download.tensorflow.org/example_images/320px-Felis_catus-cat_on_snow.jpg')
custom_method(imports='import numpy as np;import tensorflow as tf;import IPython.display as display', function_to_run='tf.keras.utils.get_file(*args)', method_object=None, object_signature=None, function_args=[eval("'194px-New_East_River_Bridge_from_Brooklyn_det.4a09796u.jpg'"), eval("'https://storage.googleapis.com/download.tensorflow.org/example_images/194px-New_East_River_Bridge_from_Brooklyn_det.4a09796u.jpg'")], function_kwargs={})
williamsburg_bridge = tf.keras.utils.get_file('194px-New_East_River_Bridge_from_Brooklyn_det.4a09796u.jpg', 'https://storage.googleapis.com/download.tensorflow.org/example_images/194px-New_East_River_Bridge_from_Brooklyn_det.4a09796u.jpg')
display.display(display.Image(filename=cat_in_snow))
display.display(display.HTML('Image cc-by: <a "href=https://commons.wikimedia.org/wiki/File:Felis_catus-cat_on_snow.jpg">Von.grzanka</a>'))
display.display(display.Image(filename=williamsburg_bridge))
display.display(display.HTML('<a "href=https://commons.wikimedia.org/wiki/File:New_East_River_Bridge_from_Brooklyn_det.4a09796u.jpg">From Wikimedia</a>'))
image_labels = {cat_in_snow: 0, williamsburg_bridge: 1}
image_string = open(cat_in_snow, 'rb').read()
label = image_labels[cat_in_snow]

def image_example(image_string, label):
    image_shape = tf.io.decode_jpeg(image_string).shape
    feature = {'height': _int64_feature(image_shape[0]), 'width': _int64_feature(image_shape[1]), 'depth': _int64_feature(image_shape[2]), 'label': _int64_feature(label), 'image_raw': _bytes_feature(image_string)}
    return tf.train.Example(features=tf.train.Features(feature=feature))
for line in str(image_example(image_string, label)).split('\n')[:15]:
    print(line)
print('...')
record_file = 'images.tfrecords'
with tf.io.TFRecordWriter(record_file) as writer:
    for (filename, label) in image_labels.items():
        image_string = open(filename, 'rb').read()
        tf_example = image_example(image_string, label)
        custom_method(imports='import numpy as np;import tensorflow as tf;import IPython.display as display', function_to_run='obj.write(*args)', method_object=eval('writer'), object_signature=None, function_args=[eval('tf_example.SerializeToString()')], function_kwargs={}, custom_class=None)
        writer.write(tf_example.SerializeToString())
custom_method(imports='import numpy as np;import tensorflow as tf;import IPython.display as display', function_to_run='tf.data.TFRecordDataset(*args)', method_object=None, object_signature=None, function_args=[eval("'images.tfrecords'")], function_kwargs={})
raw_image_dataset = tf.data.TFRecordDataset('images.tfrecords')
image_feature_description = {'height': tf.io.FixedLenFeature([], tf.int64), 'width': tf.io.FixedLenFeature([], tf.int64), 'depth': tf.io.FixedLenFeature([], tf.int64), 'label': tf.io.FixedLenFeature([], tf.int64), 'image_raw': tf.io.FixedLenFeature([], tf.string)}

def _parse_image_function(example_proto):
    return tf.io.parse_single_example(example_proto, image_feature_description)
custom_method(imports='import numpy as np;import tensorflow as tf;import IPython.display as display', function_to_run='obj.map(*args)', method_object=eval('raw_image_dataset'), object_signature=None, function_args=[eval('_parse_image_function')], function_kwargs={}, custom_class=None)
parsed_image_dataset = raw_image_dataset.map(_parse_image_function)
parsed_image_dataset
for image_features in parsed_image_dataset:
    image_raw = image_features['image_raw'].numpy()
    display.display(display.Image(data=image_raw))
