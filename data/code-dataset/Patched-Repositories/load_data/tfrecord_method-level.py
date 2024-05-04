import tensorflow as tf
import numpy as np
import IPython.display as display
import sys
from fecom.patching.patching_config import EXPERIMENT_DIR
from fecom.measurement.execution import before_execution as before_execution_INSERTED_INTO_SCRIPT
from fecom.measurement.execution import after_execution as after_execution_INSERTED_INTO_SCRIPT
from fecom.experiment.experiment_kinds import ExperimentKinds
experiment_number = sys.argv[1]
experiment_project = sys.argv[2]
EXPERIMENT_FILE_PATH = EXPERIMENT_DIR / ExperimentKinds.METHOD_LEVEL.value / experiment_project / f'experiment-{experiment_number}.json'

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
    start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT(experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tensorflow.train.Example()')
    example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
    after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tensorflow.train.Example()', method_object=None, function_args=None, function_kwargs={'features': tf.train.Features(feature=feature)})
    return example_proto.SerializeToString()
example_observation = []
serialized_example = serialize_example(False, 4, b'goat', 0.9876)
serialized_example
start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT(experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tensorflow.train.Example.FromString()')
example_proto = tf.train.Example.FromString(serialized_example)
after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tensorflow.train.Example.FromString()', method_object=None, function_args=[serialized_example], function_kwargs=None)
example_proto
start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT(experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tensorflow.data.Dataset.from_tensor_slices()')
tf.data.Dataset.from_tensor_slices(feature1)
after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tensorflow.data.Dataset.from_tensor_slices()', method_object=None, function_args=[feature1], function_kwargs=None)
start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT(experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tensorflow.data.Dataset.from_tensor_slices()')
features_dataset = tf.data.Dataset.from_tensor_slices((feature0, feature1, feature2, feature3))
after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tensorflow.data.Dataset.from_tensor_slices()', method_object=None, function_args=[(feature0, feature1, feature2, feature3)], function_kwargs=None)
features_dataset
for (f0, f1, f2, f3) in features_dataset.take(1):
    print(f0)
    print(f1)
    print(f2)
    print(f3)

def tf_serialize_example(f0, f1, f2, f3):
    start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT(experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tensorflow.py_function()')
    tf_string = tf.py_function(serialize_example, (f0, f1, f2, f3), tf.string)
    after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tensorflow.py_function()', method_object=None, function_args=[serialize_example, (f0, f1, f2, f3), tf.string], function_kwargs=None)
    return tf.reshape(tf_string, ())
tf_serialize_example(f0, f1, f2, f3)
start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT(experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tensorflow.data.Dataset.from_tensor_slices.map()')
serialized_features_dataset = features_dataset.map(tf_serialize_example)
after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tensorflow.data.Dataset.from_tensor_slices.map()', method_object=features_dataset, function_args=[tf_serialize_example], function_kwargs=None)
serialized_features_dataset

def generator():
    for features in features_dataset:
        yield serialize_example(*features)
start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT(experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tensorflow.data.Dataset.from_generator()')
serialized_features_dataset = tf.data.Dataset.from_generator(generator, output_types=tf.string, output_shapes=())
after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tensorflow.data.Dataset.from_generator()', method_object=None, function_args=[generator], function_kwargs={'output_types': tf.string, 'output_shapes': ()})
serialized_features_dataset
filename = 'test.tfrecord'
start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT(experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tensorflow.data.experimental.TFRecordWriter()')
writer = tf.data.experimental.TFRecordWriter(filename)
after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tensorflow.data.experimental.TFRecordWriter()', method_object=None, function_args=[filename], function_kwargs=None)
start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT(experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tensorflow.data.experimental.TFRecordWriter.write()')
writer.write(serialized_features_dataset)
after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tensorflow.data.experimental.TFRecordWriter.write()', method_object=writer, function_args=[serialized_features_dataset], function_kwargs=None)
filenames = [filename]
start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT(experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tensorflow.data.TFRecordDataset()')
raw_dataset = tf.data.TFRecordDataset(filenames)
after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tensorflow.data.TFRecordDataset()', method_object=None, function_args=[filenames], function_kwargs=None)
raw_dataset
for raw_record in raw_dataset.take(10):
    print(repr(raw_record))
feature_description = {'feature0': tf.io.FixedLenFeature([], tf.int64, default_value=0), 'feature1': tf.io.FixedLenFeature([], tf.int64, default_value=0), 'feature2': tf.io.FixedLenFeature([], tf.string, default_value=''), 'feature3': tf.io.FixedLenFeature([], tf.float32, default_value=0.0)}

def _parse_function(example_proto):
    return tf.io.parse_single_example(example_proto, feature_description)
start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT(experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tensorflow.data.TFRecordDataset.map()')
parsed_dataset = raw_dataset.map(_parse_function)
after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tensorflow.data.TFRecordDataset.map()', method_object=raw_dataset, function_args=[_parse_function], function_kwargs=None)
parsed_dataset
for parsed_record in parsed_dataset.take(10):
    print(repr(parsed_record))
with tf.io.TFRecordWriter(filename) as writer:
    for i in range(n_observations):
        example = serialize_example(feature0[i], feature1[i], feature2[i], feature3[i])
        start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT(experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tensorflow.data.experimental.TFRecordWriter.write()')
        writer.write(example)
        after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tensorflow.data.experimental.TFRecordWriter.write()', method_object=writer, function_args=[example], function_kwargs=None)
filenames = [filename]
start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT(experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tensorflow.data.TFRecordDataset()')
raw_dataset = tf.data.TFRecordDataset(filenames)
after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tensorflow.data.TFRecordDataset()', method_object=None, function_args=[filenames], function_kwargs=None)
raw_dataset
for raw_record in raw_dataset.take(1):
    start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT(experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tensorflow.train.Example()')
    example = tf.train.Example()
    after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tensorflow.train.Example()', method_object=None, function_args=None, function_kwargs=None)
    start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT(experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tensorflow.train.Example.ParseFromString()')
    example.ParseFromString(raw_record.numpy())
    after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tensorflow.train.Example.ParseFromString()', method_object=example, function_args=[raw_record.numpy()], function_kwargs=None)
    print(example)
result = {}
for (key, feature) in example.features.feature.items():
    kind = feature.WhichOneof('kind')
    result[key] = np.array(getattr(feature, kind).value)
result
start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT(experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tensorflow.keras.utils.get_file()')
cat_in_snow = tf.keras.utils.get_file('320px-Felis_catus-cat_on_snow.jpg', 'https://storage.googleapis.com/download.tensorflow.org/example_images/320px-Felis_catus-cat_on_snow.jpg')
after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tensorflow.keras.utils.get_file()', method_object=None, function_args=['320px-Felis_catus-cat_on_snow.jpg', 'https://storage.googleapis.com/download.tensorflow.org/example_images/320px-Felis_catus-cat_on_snow.jpg'], function_kwargs=None)
start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT(experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tensorflow.keras.utils.get_file()')
williamsburg_bridge = tf.keras.utils.get_file('194px-New_East_River_Bridge_from_Brooklyn_det.4a09796u.jpg', 'https://storage.googleapis.com/download.tensorflow.org/example_images/194px-New_East_River_Bridge_from_Brooklyn_det.4a09796u.jpg')
after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tensorflow.keras.utils.get_file()', method_object=None, function_args=['194px-New_East_River_Bridge_from_Brooklyn_det.4a09796u.jpg', 'https://storage.googleapis.com/download.tensorflow.org/example_images/194px-New_East_River_Bridge_from_Brooklyn_det.4a09796u.jpg'], function_kwargs=None)
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
        start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT(experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tensorflow.data.experimental.TFRecordWriter.write()')
        writer.write(tf_example.SerializeToString())
        after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tensorflow.data.experimental.TFRecordWriter.write()', method_object=writer, function_args=[tf_example.SerializeToString()], function_kwargs=None)
start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT(experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tensorflow.data.TFRecordDataset()')
raw_image_dataset = tf.data.TFRecordDataset('images.tfrecords')
after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tensorflow.data.TFRecordDataset()', method_object=None, function_args=['images.tfrecords'], function_kwargs=None)
image_feature_description = {'height': tf.io.FixedLenFeature([], tf.int64), 'width': tf.io.FixedLenFeature([], tf.int64), 'depth': tf.io.FixedLenFeature([], tf.int64), 'label': tf.io.FixedLenFeature([], tf.int64), 'image_raw': tf.io.FixedLenFeature([], tf.string)}

def _parse_image_function(example_proto):
    return tf.io.parse_single_example(example_proto, image_feature_description)
start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT(experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tensorflow.data.TFRecordDataset.map()')
parsed_image_dataset = raw_image_dataset.map(_parse_image_function)
after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tensorflow.data.TFRecordDataset.map()', method_object=raw_image_dataset, function_args=[_parse_image_function], function_kwargs=None)
parsed_image_dataset
for image_features in parsed_image_dataset:
    image_raw = image_features['image_raw'].numpy()
    display.display(display.Image(data=image_raw))
