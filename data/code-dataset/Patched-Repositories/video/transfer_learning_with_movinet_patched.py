import tqdm
import random
import pathlib
import itertools
import collections
import cv2
import numpy as np
import remotezip as rz
import seaborn as sns
import matplotlib.pyplot as plt
import keras
import tensorflow as tf
import tensorflow_hub as hub
from tensorflow.keras import layers
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from official.projects.movinet.modeling import movinet
from official.projects.movinet.modeling import movinet_model
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

def list_files_per_class(zip_url):
    """
    List the files in each class of the dataset given the zip URL.

    Args:
      zip_url: URL from which the files can be unzipped. 

    Return:
      files: List of files in each of the classes.
  """
    files = []
    with rz.RemoteZip(URL) as zip:
        for zip_info in zip.infolist():
            files.append(zip_info.filename)
    return files

def get_class(fname):
    """
    Retrieve the name of the class given a filename.

    Args:
      fname: Name of the file in the UCF101 dataset.

    Return:
      Class that the file belongs to.
  """
    return fname.split('_')[-3]

def get_files_per_class(files):
    """
    Retrieve the files that belong to each class. 

    Args:
      files: List of files in the dataset.

    Return:
      Dictionary of class names (key) and files (values).
  """
    files_for_class = collections.defaultdict(list)
    for fname in files:
        class_name = get_class(fname)
        files_for_class[class_name].append(fname)
    return files_for_class

def download_from_zip(zip_url, to_dir, file_names):
    """
    Download the contents of the zip file from the zip URL.

    Args:
      zip_url: Zip URL containing data.
      to_dir: Directory to download data to.
      file_names: Names of files to download.
  """
    with rz.RemoteZip(zip_url) as zip:
        for fn in tqdm.tqdm(file_names):
            class_name = get_class(fn)
            zip.extract(fn, str(to_dir / class_name))
            unzipped_file = to_dir / class_name / fn
            fn = pathlib.Path(fn).parts[-1]
            output_file = to_dir / class_name / fn
            unzipped_file.rename(output_file)

def split_class_lists(files_for_class, count):
    """
    Returns the list of files belonging to a subset of data as well as the remainder of
    files that need to be downloaded.

    Args:
      files_for_class: Files belonging to a particular class of data.
      count: Number of files to download.

    Return:
      split_files: Files belonging to the subset of data.
      remainder: Dictionary of the remainder of files that need to be downloaded.
  """
    split_files = []
    remainder = {}
    for cls in files_for_class:
        split_files.extend(files_for_class[cls][:count])
        remainder[cls] = files_for_class[cls][count:]
    return (split_files, remainder)

def download_ufc_101_subset(zip_url, num_classes, splits, download_dir):
    """
    Download a subset of the UFC101 dataset and split them into various parts, such as
    training, validation, and test. 

    Args:
      zip_url: Zip URL containing data.
      num_classes: Number of labels.
      splits: Dictionary specifying the training, validation, test, etc. (key) division of data 
              (value is number of files per split).
      download_dir: Directory to download data to.

    Return:
      dir: Posix path of the resulting directories containing the splits of data.
  """
    files = list_files_per_class(zip_url)
    for f in files:
        tokens = f.split('/')
        if len(tokens) <= 2:
            files.remove(f)
    files_for_class = get_files_per_class(files)
    classes = list(files_for_class.keys())[:num_classes]
    for cls in classes:
        new_files_for_class = files_for_class[cls]
        random.shuffle(new_files_for_class)
        files_for_class[cls] = new_files_for_class
    files_for_class = {x: files_for_class[x] for x in list(files_for_class)[:num_classes]}
    dirs = {}
    for (split_name, split_count) in splits.items():
        print(split_name, ':')
        split_dir = download_dir / split_name
        (split_files, files_for_class) = split_class_lists(files_for_class, split_count)
        download_from_zip(zip_url, split_dir, split_files)
        dirs[split_name] = split_dir
    return dirs

def format_frames(frame, output_size):
    """
    Pad and resize an image from a video.

    Args:
      frame: Image that needs to resized and padded. 
      output_size: Pixel size of the output frame image.

    Return:
      Formatted frame with padding of specified output size.
  """
    custom_method(imports='import numpy as np;import itertools;import tensorflow_hub as hub;import random;import matplotlib.pyplot as plt;import collections;from official.projects.movinet.modeling import movinet_model;import cv2;from tensorflow.keras.losses import SparseCategoricalCrossentropy;from tensorflow.keras import layers;from official.projects.movinet.modeling import movinet;import keras;import tqdm;import pathlib;import remotezip as rz;import tensorflow as tf;from tensorflow.keras.optimizers import Adam;import seaborn as sns', function_to_run='tf.image.convert_image_dtype(*args)', method_object=None, object_signature=None, function_args=[eval('frame'), eval('tf.float32')], function_kwargs={})
    frame = tf.image.convert_image_dtype(frame, tf.float32)
    custom_method(imports='import numpy as np;import itertools;import tensorflow_hub as hub;import random;import matplotlib.pyplot as plt;import collections;from official.projects.movinet.modeling import movinet_model;import cv2;from tensorflow.keras.losses import SparseCategoricalCrossentropy;from tensorflow.keras import layers;from official.projects.movinet.modeling import movinet;import keras;import tqdm;import pathlib;import remotezip as rz;import tensorflow as tf;from tensorflow.keras.optimizers import Adam;import seaborn as sns', function_to_run='tf.image.resize_with_pad(*args)', method_object=None, object_signature=None, function_args=[eval('frame'), eval('*output_size')], function_kwargs={})
    frame = tf.image.resize_with_pad(frame, *output_size)
    return frame

def frames_from_video_file(video_path, n_frames, output_size=(224, 224), frame_step=15):
    """
    Creates frames from each video file present for each category.

    Args:
      video_path: File path to the video.
      n_frames: Number of frames to be created per video file.
      output_size: Pixel size of the output frame image.

    Return:
      An NumPy array of frames in the shape of (n_frames, height, width, channels).
  """
    result = []
    src = cv2.VideoCapture(str(video_path))
    video_length = src.get(cv2.CAP_PROP_FRAME_COUNT)
    need_length = 1 + (n_frames - 1) * frame_step
    if need_length > video_length:
        start = 0
    else:
        max_start = video_length - need_length
        start = random.randint(0, max_start + 1)
    src.set(cv2.CAP_PROP_POS_FRAMES, start)
    (ret, frame) = src.read()
    result.append(format_frames(frame, output_size))
    for _ in range(n_frames - 1):
        for _ in range(frame_step):
            (ret, frame) = src.read()
        if ret:
            frame = format_frames(frame, output_size)
            result.append(frame)
        else:
            result.append(np.zeros_like(result[0]))
    src.release()
    result = np.array(result)[..., [2, 1, 0]]
    return result

class FrameGenerator:

    def __init__(self, path, n_frames, training=False):
        """ Returns a set of frames with their associated label. 

      Args:
        path: Video file paths.
        n_frames: Number of frames. 
        training: Boolean to determine if training dataset is being created.
    """
        self.path = path
        self.n_frames = n_frames
        self.training = training
        self.class_names = sorted(set((p.name for p in self.path.iterdir() if p.is_dir())))
        self.class_ids_for_name = dict(((name, idx) for (idx, name) in enumerate(self.class_names)))

    def get_files_and_class_names(self):
        video_paths = list(self.path.glob('*/*.avi'))
        classes = [p.parent.name for p in video_paths]
        return (video_paths, classes)

    def __call__(self):
        (video_paths, classes) = self.get_files_and_class_names()
        pairs = list(zip(video_paths, classes))
        if self.training:
            random.shuffle(pairs)
        for (path, name) in pairs:
            video_frames = frames_from_video_file(path, self.n_frames)
            label = self.class_ids_for_name[name]
            yield (video_frames, label)
URL = 'https://storage.googleapis.com/thumos14_files/UCF101_videos.zip'
download_dir = pathlib.Path('./UCF101_subset/')
subset_paths = download_ufc_101_subset(URL, num_classes=10, splits={'train': 30, 'test': 20}, download_dir=download_dir)
batch_size = 8
num_frames = 8
output_signature = (tf.TensorSpec(shape=(None, None, None, 3), dtype=tf.float32), tf.TensorSpec(shape=(), dtype=tf.int16))
custom_method(imports='import numpy as np;import itertools;import tensorflow_hub as hub;import random;import matplotlib.pyplot as plt;import collections;from official.projects.movinet.modeling import movinet_model;import cv2;from tensorflow.keras.losses import SparseCategoricalCrossentropy;from tensorflow.keras import layers;from official.projects.movinet.modeling import movinet;import keras;import tqdm;import pathlib;import remotezip as rz;import tensorflow as tf;from tensorflow.keras.optimizers import Adam;import seaborn as sns', function_to_run='tf.data.Dataset.from_generator(*args, **kwargs)', method_object=None, object_signature=None, function_args=[eval("FrameGenerator(subset_paths['train'], num_frames, training = True)")], function_kwargs={'output_signature': eval('output_signature')})
train_ds = tf.data.Dataset.from_generator(FrameGenerator(subset_paths['train'], num_frames, training=True), output_signature=output_signature)
custom_method(imports='import numpy as np;import itertools;import tensorflow_hub as hub;import random;import matplotlib.pyplot as plt;import collections;from official.projects.movinet.modeling import movinet_model;import cv2;from tensorflow.keras.losses import SparseCategoricalCrossentropy;from tensorflow.keras import layers;from official.projects.movinet.modeling import movinet;import keras;import tqdm;import pathlib;import remotezip as rz;import tensorflow as tf;from tensorflow.keras.optimizers import Adam;import seaborn as sns', function_to_run='obj.batch(*args)', method_object=eval('train_ds'), object_signature=None, function_args=[eval('batch_size')], function_kwargs={}, custom_class=None)
train_ds = train_ds.batch(batch_size)
custom_method(imports='import numpy as np;import itertools;import tensorflow_hub as hub;import random;import matplotlib.pyplot as plt;import collections;from official.projects.movinet.modeling import movinet_model;import cv2;from tensorflow.keras.losses import SparseCategoricalCrossentropy;from tensorflow.keras import layers;from official.projects.movinet.modeling import movinet;import keras;import tqdm;import pathlib;import remotezip as rz;import tensorflow as tf;from tensorflow.keras.optimizers import Adam;import seaborn as sns', function_to_run='tf.data.Dataset.from_generator(*args, **kwargs)', method_object=None, object_signature=None, function_args=[eval("FrameGenerator(subset_paths['test'], num_frames)")], function_kwargs={'output_signature': eval('output_signature')})
test_ds = tf.data.Dataset.from_generator(FrameGenerator(subset_paths['test'], num_frames), output_signature=output_signature)
custom_method(imports='import numpy as np;import itertools;import tensorflow_hub as hub;import random;import matplotlib.pyplot as plt;import collections;from official.projects.movinet.modeling import movinet_model;import cv2;from tensorflow.keras.losses import SparseCategoricalCrossentropy;from tensorflow.keras import layers;from official.projects.movinet.modeling import movinet;import keras;import tqdm;import pathlib;import remotezip as rz;import tensorflow as tf;from tensorflow.keras.optimizers import Adam;import seaborn as sns', function_to_run='obj.batch(*args)', method_object=eval('test_ds'), object_signature=None, function_args=[eval('batch_size')], function_kwargs={}, custom_class=None)
test_ds = test_ds.batch(batch_size)
for (frames, labels) in train_ds.take(10):
    print(labels)
print(f'Shape: {frames.shape}')
print(f'Label: {labels.shape}')
custom_method(imports='import numpy as np;import itertools;import tensorflow_hub as hub;import random;import matplotlib.pyplot as plt;import collections;from official.projects.movinet.modeling import movinet_model;import cv2;from tensorflow.keras.losses import SparseCategoricalCrossentropy;from tensorflow.keras import layers;from official.projects.movinet.modeling import movinet;import keras;import tqdm;import pathlib;import remotezip as rz;import tensorflow as tf;from tensorflow.keras.optimizers import Adam;import seaborn as sns', function_to_run='layers.GRU(**kwargs)', method_object=None, object_signature=None, function_args=[], function_kwargs={'units': eval('4'), 'return_sequences': eval('True'), 'return_state': eval('True')})
gru = layers.GRU(units=4, return_sequences=True, return_state=True)
custom_method(imports='import numpy as np;import itertools;import tensorflow_hub as hub;import random;import matplotlib.pyplot as plt;import collections;from official.projects.movinet.modeling import movinet_model;import cv2;from tensorflow.keras.losses import SparseCategoricalCrossentropy;from tensorflow.keras import layers;from official.projects.movinet.modeling import movinet;import keras;import tqdm;import pathlib;import remotezip as rz;import tensorflow as tf;from tensorflow.keras.optimizers import Adam;import seaborn as sns', function_to_run='tf.random.normal(**kwargs)', method_object=None, object_signature=None, function_args=[], function_kwargs={'shape': eval('[1, 10, 8]')})
inputs = tf.random.normal(shape=[1, 10, 8])
custom_method(imports='import numpy as np;import itertools;import tensorflow_hub as hub;import random;import matplotlib.pyplot as plt;import collections;from official.projects.movinet.modeling import movinet_model;import cv2;from tensorflow.keras.losses import SparseCategoricalCrossentropy;from tensorflow.keras import layers;from official.projects.movinet.modeling import movinet;import keras;import tqdm;import pathlib;import remotezip as rz;import tensorflow as tf;from tensorflow.keras.optimizers import Adam;import seaborn as sns', function_to_run='obj(*args)', method_object=eval('gru'), object_signature=None, function_args=[eval('inputs')], function_kwargs={}, custom_class=None)
(result, state) = gru(inputs)
custom_method(imports='import numpy as np;import itertools;import tensorflow_hub as hub;import random;import matplotlib.pyplot as plt;import collections;from official.projects.movinet.modeling import movinet_model;import cv2;from tensorflow.keras.losses import SparseCategoricalCrossentropy;from tensorflow.keras import layers;from official.projects.movinet.modeling import movinet;import keras;import tqdm;import pathlib;import remotezip as rz;import tensorflow as tf;from tensorflow.keras.optimizers import Adam;import seaborn as sns', function_to_run='obj(*args)', method_object=eval('gru'), object_signature=None, function_args=[eval('inputs[:, :5, :]')], function_kwargs={}, custom_class=None)
(first_half, state) = gru(inputs[:, :5, :])
custom_method(imports='import numpy as np;import itertools;import tensorflow_hub as hub;import random;import matplotlib.pyplot as plt;import collections;from official.projects.movinet.modeling import movinet_model;import cv2;from tensorflow.keras.losses import SparseCategoricalCrossentropy;from tensorflow.keras import layers;from official.projects.movinet.modeling import movinet;import keras;import tqdm;import pathlib;import remotezip as rz;import tensorflow as tf;from tensorflow.keras.optimizers import Adam;import seaborn as sns', function_to_run='obj(*args, **kwargs)', method_object=eval('gru'), object_signature=None, function_args=[eval('inputs[:,5:, :]')], function_kwargs={'initial_state': eval('state')}, custom_class=None)
(second_half, _) = gru(inputs[:, 5:, :], initial_state=state)
print(np.allclose(result[:, :5, :], first_half))
print(np.allclose(result[:, 5:, :], second_half))
model_id = 'a0'
resolution = 224
custom_method(imports='import numpy as np;import itertools;import tensorflow_hub as hub;import random;import matplotlib.pyplot as plt;import collections;from official.projects.movinet.modeling import movinet_model;import cv2;from tensorflow.keras.losses import SparseCategoricalCrossentropy;from tensorflow.keras import layers;from official.projects.movinet.modeling import movinet;import keras;import tqdm;import pathlib;import remotezip as rz;import tensorflow as tf;from tensorflow.keras.optimizers import Adam;import seaborn as sns', function_to_run='tf.keras.backend.clear_session()', method_object=None, object_signature=None, function_args=[], function_kwargs={})
tf.keras.backend.clear_session()
backbone = movinet.Movinet(model_id=model_id)
backbone.trainable = False
model = movinet_model.MovinetClassifier(backbone=backbone, num_classes=600)
model.build([None, None, None, None, 3])
checkpoint_dir = f'movinet_{model_id}_base'
custom_method(imports='import numpy as np;import itertools;import tensorflow_hub as hub;import random;import matplotlib.pyplot as plt;import collections;from official.projects.movinet.modeling import movinet_model;import cv2;from tensorflow.keras.losses import SparseCategoricalCrossentropy;from tensorflow.keras import layers;from official.projects.movinet.modeling import movinet;import keras;import tqdm;import pathlib;import remotezip as rz;import tensorflow as tf;from tensorflow.keras.optimizers import Adam;import seaborn as sns', function_to_run='tf.train.latest_checkpoint(*args)', method_object=None, object_signature=None, function_args=[eval('checkpoint_dir')], function_kwargs={})
checkpoint_path = tf.train.latest_checkpoint(checkpoint_dir)
custom_method(imports='import numpy as np;import itertools;import tensorflow_hub as hub;import random;import matplotlib.pyplot as plt;import collections;from official.projects.movinet.modeling import movinet_model;import cv2;from tensorflow.keras.losses import SparseCategoricalCrossentropy;from tensorflow.keras import layers;from official.projects.movinet.modeling import movinet;import keras;import tqdm;import pathlib;import remotezip as rz;import tensorflow as tf;from tensorflow.keras.optimizers import Adam;import seaborn as sns', function_to_run='tf.train.Checkpoint(**kwargs)', method_object=None, object_signature=None, function_args=[], function_kwargs={'model': eval('model')})
checkpoint = tf.train.Checkpoint(model=model)
custom_method(imports='import numpy as np;import itertools;import tensorflow_hub as hub;import random;import matplotlib.pyplot as plt;import collections;from official.projects.movinet.modeling import movinet_model;import cv2;from tensorflow.keras.losses import SparseCategoricalCrossentropy;from tensorflow.keras import layers;from official.projects.movinet.modeling import movinet;import keras;import tqdm;import pathlib;import remotezip as rz;import tensorflow as tf;from tensorflow.keras.optimizers import Adam;import seaborn as sns', function_to_run='obj.restore(*args)', method_object=eval('checkpoint'), object_signature=None, function_args=[eval('checkpoint_path')], function_kwargs={}, custom_class=None)
status = checkpoint.restore(checkpoint_path)
status.assert_existing_objects_matched()

def build_classifier(batch_size, num_frames, resolution, backbone, num_classes):
    """Builds a classifier on top of a backbone model."""
    model = movinet_model.MovinetClassifier(backbone=backbone, num_classes=num_classes)
    model.build([batch_size, num_frames, resolution, resolution, 3])
    return model
model = build_classifier(batch_size, num_frames, resolution, backbone, 10)
num_epochs = 2
custom_method(imports='import numpy as np;import itertools;import tensorflow_hub as hub;import random;import matplotlib.pyplot as plt;import collections;from official.projects.movinet.modeling import movinet_model;import cv2;from tensorflow.keras.losses import SparseCategoricalCrossentropy;from tensorflow.keras import layers;from official.projects.movinet.modeling import movinet;import keras;import tqdm;import pathlib;import remotezip as rz;import tensorflow as tf;from tensorflow.keras.optimizers import Adam;import seaborn as sns', function_to_run='tf.keras.losses.SparseCategoricalCrossentropy(**kwargs)', method_object=None, object_signature=None, function_args=[], function_kwargs={'from_logits': eval('True')})
loss_obj = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
custom_method(imports='import numpy as np;import itertools;import tensorflow_hub as hub;import random;import matplotlib.pyplot as plt;import collections;from official.projects.movinet.modeling import movinet_model;import cv2;from tensorflow.keras.losses import SparseCategoricalCrossentropy;from tensorflow.keras import layers;from official.projects.movinet.modeling import movinet;import keras;import tqdm;import pathlib;import remotezip as rz;import tensorflow as tf;from tensorflow.keras.optimizers import Adam;import seaborn as sns', function_to_run='tf.keras.optimizers.Adam(**kwargs)', method_object=None, object_signature=None, function_args=[], function_kwargs={'learning_rate': eval('0.001')})
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
model.compile(loss=loss_obj, optimizer=optimizer, metrics=['accuracy'])
results = model.fit(train_ds, validation_data=test_ds, epochs=num_epochs, validation_freq=1, verbose=1)
model.evaluate(test_ds, return_dict=True)

def get_actual_predicted_labels(dataset):
    """
    Create a list of actual ground truth values and the predictions from the model.

    Args:
      dataset: An iterable data structure, such as a TensorFlow Dataset, with features and labels.

    Return:
      Ground truth and predicted values for a particular dataset.
  """
    actual = [labels for (_, labels) in dataset.unbatch()]
    predicted = model.predict(dataset)
    custom_method(imports='import numpy as np;import itertools;import tensorflow_hub as hub;import random;import matplotlib.pyplot as plt;import collections;from official.projects.movinet.modeling import movinet_model;import cv2;from tensorflow.keras.losses import SparseCategoricalCrossentropy;from tensorflow.keras import layers;from official.projects.movinet.modeling import movinet;import keras;import tqdm;import pathlib;import remotezip as rz;import tensorflow as tf;from tensorflow.keras.optimizers import Adam;import seaborn as sns', function_to_run='tf.stack(*args, **kwargs)', method_object=None, object_signature=None, function_args=[eval('actual')], function_kwargs={'axis': eval('0')})
    actual = tf.stack(actual, axis=0)
    custom_method(imports='import numpy as np;import itertools;import tensorflow_hub as hub;import random;import matplotlib.pyplot as plt;import collections;from official.projects.movinet.modeling import movinet_model;import cv2;from tensorflow.keras.losses import SparseCategoricalCrossentropy;from tensorflow.keras import layers;from official.projects.movinet.modeling import movinet;import keras;import tqdm;import pathlib;import remotezip as rz;import tensorflow as tf;from tensorflow.keras.optimizers import Adam;import seaborn as sns', function_to_run='tf.concat(*args, **kwargs)', method_object=None, object_signature=None, function_args=[eval('predicted')], function_kwargs={'axis': eval('0')})
    predicted = tf.concat(predicted, axis=0)
    custom_method(imports='import numpy as np;import itertools;import tensorflow_hub as hub;import random;import matplotlib.pyplot as plt;import collections;from official.projects.movinet.modeling import movinet_model;import cv2;from tensorflow.keras.losses import SparseCategoricalCrossentropy;from tensorflow.keras import layers;from official.projects.movinet.modeling import movinet;import keras;import tqdm;import pathlib;import remotezip as rz;import tensorflow as tf;from tensorflow.keras.optimizers import Adam;import seaborn as sns', function_to_run='tf.argmax(*args, **kwargs)', method_object=None, object_signature=None, function_args=[eval('predicted')], function_kwargs={'axis': eval('1')})
    predicted = tf.argmax(predicted, axis=1)
    return (actual, predicted)

def plot_confusion_matrix(actual, predicted, labels, ds_type):
    custom_method(imports='import numpy as np;import itertools;import tensorflow_hub as hub;import random;import matplotlib.pyplot as plt;import collections;from official.projects.movinet.modeling import movinet_model;import cv2;from tensorflow.keras.losses import SparseCategoricalCrossentropy;from tensorflow.keras import layers;from official.projects.movinet.modeling import movinet;import keras;import tqdm;import pathlib;import remotezip as rz;import tensorflow as tf;from tensorflow.keras.optimizers import Adam;import seaborn as sns', function_to_run='tf.math.confusion_matrix(*args)', method_object=None, object_signature=None, function_args=[eval('actual'), eval('predicted')], function_kwargs={})
    cm = tf.math.confusion_matrix(actual, predicted)
    ax = sns.heatmap(cm, annot=True, fmt='g')
    sns.set(rc={'figure.figsize': (12, 12)})
    sns.set(font_scale=1.4)
    ax.set_title('Confusion matrix of action recognition for ' + ds_type)
    ax.set_xlabel('Predicted Action')
    ax.set_ylabel('Actual Action')
    plt.xticks(rotation=90)
    plt.yticks(rotation=0)
    ax.xaxis.set_ticklabels(labels)
    ax.yaxis.set_ticklabels(labels)
fg = FrameGenerator(subset_paths['train'], num_frames, training=True)
label_names = list(fg.class_ids_for_name.keys())
(actual, predicted) = get_actual_predicted_labels(test_ds)
plot_confusion_matrix(actual, predicted, label_names, 'test')
