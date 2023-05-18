import os
import pathlib
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import models
from IPython import display
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
seed = 42
custom_method(imports='import numpy as np;import seaborn as sns;from tensorflow.keras import models;import os;import tensorflow as tf;from IPython import display;from tensorflow.keras import layers;import matplotlib.pyplot as plt;import pathlib', function_to_run='tf.random.set_seed(*args)', method_object=None, object_signature=None, function_args=[eval('seed')], function_kwargs={})
tf.random.set_seed(seed)
np.random.seed(seed)
DATASET_PATH = 'data/mini_speech_commands'
data_dir = pathlib.Path(DATASET_PATH)
if not data_dir.exists():
    custom_method(imports='import numpy as np;import seaborn as sns;from tensorflow.keras import models;import os;import tensorflow as tf;from IPython import display;from tensorflow.keras import layers;import matplotlib.pyplot as plt;import pathlib', function_to_run='tf.keras.utils.get_file(*args, **kwargs)', method_object=None, object_signature=None, function_args=[eval("'mini_speech_commands.zip'")], function_kwargs={'origin': eval('"http://storage.googleapis.com/download.tensorflow.org/data/mini_speech_commands.zip"'), 'extract': eval('True'), 'cache_dir': eval("'.'"), 'cache_subdir': eval("'data'")})
    tf.keras.utils.get_file('mini_speech_commands.zip', origin='http://storage.googleapis.com/download.tensorflow.org/data/mini_speech_commands.zip', extract=True, cache_dir='.', cache_subdir='data')
commands = np.array(tf.io.gfile.listdir(str(data_dir)))
commands = commands[(commands != 'README.md') & (commands != '.DS_Store')]
print('Commands:', commands)
custom_method(imports='import numpy as np;import seaborn as sns;from tensorflow.keras import models;import os;import tensorflow as tf;from IPython import display;from tensorflow.keras import layers;import matplotlib.pyplot as plt;import pathlib', function_to_run='tf.keras.utils.audio_dataset_from_directory(**kwargs)', method_object=None, object_signature=None, function_args=[], function_kwargs={'directory': eval('data_dir'), 'batch_size': eval('64'), 'validation_split': eval('0.2'), 'seed': eval('0'), 'output_sequence_length': eval('16000'), 'subset': eval("'both'")})
(train_ds, val_ds) = tf.keras.utils.audio_dataset_from_directory(directory=data_dir, batch_size=64, validation_split=0.2, seed=0, output_sequence_length=16000, subset='both')
label_names = np.array(train_ds.class_names)
print()
print('label names:', label_names)
train_ds.element_spec

def squeeze(audio, labels):
    custom_method(imports='import numpy as np;import seaborn as sns;from tensorflow.keras import models;import os;import tensorflow as tf;from IPython import display;from tensorflow.keras import layers;import matplotlib.pyplot as plt;import pathlib', function_to_run='tf.squeeze(*args, **kwargs)', method_object=None, object_signature=None, function_args=[eval('audio')], function_kwargs={'axis': eval('-1')})
    audio = tf.squeeze(audio, axis=-1)
    return (audio, labels)
train_ds = train_ds.map(squeeze, tf.data.AUTOTUNE)
val_ds = val_ds.map(squeeze, tf.data.AUTOTUNE)
test_ds = val_ds.shard(num_shards=2, index=0)
val_ds = val_ds.shard(num_shards=2, index=1)
for (example_audio, example_labels) in train_ds.take(1):
    print(example_audio.shape)
    print(example_labels.shape)
label_names[[1, 1, 3, 0]]
plt.figure(figsize=(16, 10))
rows = 3
cols = 3
n = rows * cols
for i in range(n):
    plt.subplot(rows, cols, i + 1)
    audio_signal = example_audio[i]
    plt.plot(audio_signal)
    plt.title(label_names[example_labels[i]])
    plt.yticks(np.arange(-1.2, 1.2, 0.2))
    plt.ylim([-1.1, 1.1])

def get_spectrogram(waveform):
    custom_method(imports='import numpy as np;import seaborn as sns;from tensorflow.keras import models;import os;import tensorflow as tf;from IPython import display;from tensorflow.keras import layers;import matplotlib.pyplot as plt;import pathlib', function_to_run='tf.signal.stft(*args, **kwargs)', method_object=None, object_signature=None, function_args=[eval('waveform')], function_kwargs={'frame_length': eval('255'), 'frame_step': eval('128')})
    spectrogram = tf.signal.stft(waveform, frame_length=255, frame_step=128)
    custom_method(imports='import numpy as np;import seaborn as sns;from tensorflow.keras import models;import os;import tensorflow as tf;from IPython import display;from tensorflow.keras import layers;import matplotlib.pyplot as plt;import pathlib', function_to_run='tf.abs(*args)', method_object=None, object_signature=None, function_args=[eval('spectrogram')], function_kwargs={})
    spectrogram = tf.abs(spectrogram)
    spectrogram = spectrogram[..., tf.newaxis]
    return spectrogram
for i in range(3):
    label = label_names[example_labels[i]]
    waveform = example_audio[i]
    spectrogram = get_spectrogram(waveform)
    print('Label:', label)
    print('Waveform shape:', waveform.shape)
    print('Spectrogram shape:', spectrogram.shape)
    print('Audio playback')
    display.display(display.Audio(waveform, rate=16000))

def plot_spectrogram(spectrogram, ax):
    if len(spectrogram.shape) > 2:
        assert len(spectrogram.shape) == 3
        spectrogram = np.squeeze(spectrogram, axis=-1)
    log_spec = np.log(spectrogram.T + np.finfo(float).eps)
    height = log_spec.shape[0]
    width = log_spec.shape[1]
    X = np.linspace(0, np.size(spectrogram), num=width, dtype=int)
    Y = range(height)
    ax.pcolormesh(X, Y, log_spec)
(fig, axes) = plt.subplots(2, figsize=(12, 8))
timescale = np.arange(waveform.shape[0])
axes[0].plot(timescale, waveform.numpy())
axes[0].set_title('Waveform')
axes[0].set_xlim([0, 16000])
plot_spectrogram(spectrogram.numpy(), axes[1])
axes[1].set_title('Spectrogram')
plt.suptitle(label.title())
plt.show()

def make_spec_ds(ds):
    return ds.map(map_func=lambda audio, label: (get_spectrogram(audio), label), num_parallel_calls=tf.data.AUTOTUNE)
train_spectrogram_ds = make_spec_ds(train_ds)
val_spectrogram_ds = make_spec_ds(val_ds)
test_spectrogram_ds = make_spec_ds(test_ds)
for (example_spectrograms, example_spect_labels) in train_spectrogram_ds.take(1):
    break
rows = 3
cols = 3
n = rows * cols
(fig, axes) = plt.subplots(rows, cols, figsize=(16, 9))
for i in range(n):
    r = i // cols
    c = i % cols
    ax = axes[r][c]
    plot_spectrogram(example_spectrograms[i].numpy(), ax)
    ax.set_title(label_names[example_spect_labels[i].numpy()])
plt.show()
train_spectrogram_ds = train_spectrogram_ds.cache().shuffle(10000).prefetch(tf.data.AUTOTUNE)
val_spectrogram_ds = val_spectrogram_ds.cache().prefetch(tf.data.AUTOTUNE)
test_spectrogram_ds = test_spectrogram_ds.cache().prefetch(tf.data.AUTOTUNE)
input_shape = example_spectrograms.shape[1:]
print('Input shape:', input_shape)
num_labels = len(label_names)
custom_method(imports='import numpy as np;import seaborn as sns;from tensorflow.keras import models;import os;import tensorflow as tf;from IPython import display;from tensorflow.keras import layers;import matplotlib.pyplot as plt;import pathlib', function_to_run='layers.Normalization()', method_object=None, object_signature=None, function_args=[], function_kwargs={})
norm_layer = layers.Normalization()
custom_method(imports='import numpy as np;import seaborn as sns;from tensorflow.keras import models;import os;import tensorflow as tf;from IPython import display;from tensorflow.keras import layers;import matplotlib.pyplot as plt;import pathlib', function_to_run='obj.adapt(**kwargs)', method_object=eval('norm_layer'), object_signature=None, function_args=[], function_kwargs={'data': eval('train_spectrogram_ds.map(map_func=lambda spec, label: spec)')}, custom_class=None)
norm_layer.adapt(data=train_spectrogram_ds.map(map_func=lambda spec, label: spec))
custom_method(imports='import numpy as np;import seaborn as sns;from tensorflow.keras import models;import os;import tensorflow as tf;from IPython import display;from tensorflow.keras import layers;import matplotlib.pyplot as plt;import pathlib', function_to_run='models.Sequential(*args)', method_object=None, object_signature=None, function_args=[eval("[\n    layers.Input(shape=input_shape),\n    layers.Resizing(32, 32),\n    norm_layer,\n    layers.Conv2D(32, 3, activation='relu'),\n    layers.Conv2D(64, 3, activation='relu'),\n    layers.MaxPooling2D(),\n    layers.Dropout(0.25),\n    layers.Flatten(),\n    layers.Dense(128, activation='relu'),\n    layers.Dropout(0.5),\n    layers.Dense(num_labels),\n]")], function_kwargs={})
model = models.Sequential([layers.Input(shape=input_shape), layers.Resizing(32, 32), norm_layer, layers.Conv2D(32, 3, activation='relu'), layers.Conv2D(64, 3, activation='relu'), layers.MaxPooling2D(), layers.Dropout(0.25), layers.Flatten(), layers.Dense(128, activation='relu'), layers.Dropout(0.5), layers.Dense(num_labels)])
custom_method(imports='import numpy as np;import seaborn as sns;from tensorflow.keras import models;import os;import tensorflow as tf;from IPython import display;from tensorflow.keras import layers;import matplotlib.pyplot as plt;import pathlib', function_to_run='obj.summary()', method_object=eval('model'), object_signature=None, function_args=[], function_kwargs={}, custom_class=None)
model.summary()
custom_method(imports='import numpy as np;import seaborn as sns;from tensorflow.keras import models;import os;import tensorflow as tf;from IPython import display;from tensorflow.keras import layers;import matplotlib.pyplot as plt;import pathlib', function_to_run='obj.compile(**kwargs)', method_object=eval('model'), object_signature=None, function_args=[], function_kwargs={'optimizer': eval('tf.keras.optimizers.Adam()'), 'loss': eval('tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)'), 'metrics': eval("['accuracy']")}, custom_class=None)
model.compile(optimizer=tf.keras.optimizers.Adam(), loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=['accuracy'])
EPOCHS = 10
custom_method(imports='import numpy as np;import seaborn as sns;from tensorflow.keras import models;import os;import tensorflow as tf;from IPython import display;from tensorflow.keras import layers;import matplotlib.pyplot as plt;import pathlib', function_to_run='obj.fit(*args, **kwargs)', method_object=eval('model'), object_signature=None, function_args=[eval('train_spectrogram_ds')], function_kwargs={'validation_data': eval('val_spectrogram_ds'), 'epochs': eval('EPOCHS'), 'callbacks': eval('tf.keras.callbacks.EarlyStopping(verbose=1, patience=2)')}, custom_class=None)
history = model.fit(train_spectrogram_ds, validation_data=val_spectrogram_ds, epochs=EPOCHS, callbacks=tf.keras.callbacks.EarlyStopping(verbose=1, patience=2))
metrics = history.history
plt.figure(figsize=(16, 6))
plt.subplot(1, 2, 1)
plt.plot(history.epoch, metrics['loss'], metrics['val_loss'])
plt.legend(['loss', 'val_loss'])
plt.ylim([0, max(plt.ylim())])
plt.xlabel('Epoch')
plt.ylabel('Loss [CrossEntropy]')
plt.subplot(1, 2, 2)
plt.plot(history.epoch, 100 * np.array(metrics['accuracy']), 100 * np.array(metrics['val_accuracy']))
plt.legend(['accuracy', 'val_accuracy'])
plt.ylim([0, 100])
plt.xlabel('Epoch')
plt.ylabel('Accuracy [%]')
custom_method(imports='import numpy as np;import seaborn as sns;from tensorflow.keras import models;import os;import tensorflow as tf;from IPython import display;from tensorflow.keras import layers;import matplotlib.pyplot as plt;import pathlib', function_to_run='obj.evaluate(*args, **kwargs)', method_object=eval('model'), object_signature=None, function_args=[eval('test_spectrogram_ds')], function_kwargs={'return_dict': eval('True')}, custom_class=None)
model.evaluate(test_spectrogram_ds, return_dict=True)
custom_method(imports='import numpy as np;import seaborn as sns;from tensorflow.keras import models;import os;import tensorflow as tf;from IPython import display;from tensorflow.keras import layers;import matplotlib.pyplot as plt;import pathlib', function_to_run='obj.predict(*args)', method_object=eval('model'), object_signature=None, function_args=[eval('test_spectrogram_ds')], function_kwargs={}, custom_class=None)
y_pred = model.predict(test_spectrogram_ds)
custom_method(imports='import numpy as np;import seaborn as sns;from tensorflow.keras import models;import os;import tensorflow as tf;from IPython import display;from tensorflow.keras import layers;import matplotlib.pyplot as plt;import pathlib', function_to_run='tf.argmax(*args, **kwargs)', method_object=None, object_signature=None, function_args=[eval('y_pred')], function_kwargs={'axis': eval('1')})
y_pred = tf.argmax(y_pred, axis=1)
custom_method(imports='import numpy as np;import seaborn as sns;from tensorflow.keras import models;import os;import tensorflow as tf;from IPython import display;from tensorflow.keras import layers;import matplotlib.pyplot as plt;import pathlib', function_to_run='tf.concat(*args, **kwargs)', method_object=None, object_signature=None, function_args=[eval('list(test_spectrogram_ds.map(lambda s,lab: lab))')], function_kwargs={'axis': eval('0')})
y_true = tf.concat(list(test_spectrogram_ds.map(lambda s, lab: lab)), axis=0)
custom_method(imports='import numpy as np;import seaborn as sns;from tensorflow.keras import models;import os;import tensorflow as tf;from IPython import display;from tensorflow.keras import layers;import matplotlib.pyplot as plt;import pathlib', function_to_run='tf.math.confusion_matrix(*args)', method_object=None, object_signature=None, function_args=[eval('y_true'), eval('y_pred')], function_kwargs={})
confusion_mtx = tf.math.confusion_matrix(y_true, y_pred)
plt.figure(figsize=(10, 8))
sns.heatmap(confusion_mtx, xticklabels=label_names, yticklabels=label_names, annot=True, fmt='g')
plt.xlabel('Prediction')
plt.ylabel('Label')
plt.show()
x = data_dir / 'no/01bb6a2a_nohash_0.wav'
custom_method(imports='import numpy as np;import seaborn as sns;from tensorflow.keras import models;import os;import tensorflow as tf;from IPython import display;from tensorflow.keras import layers;import matplotlib.pyplot as plt;import pathlib', function_to_run='tf.io.read_file(*args)', method_object=None, object_signature=None, function_args=[eval('str(x)')], function_kwargs={})
x = tf.io.read_file(str(x))
custom_method(imports='import numpy as np;import seaborn as sns;from tensorflow.keras import models;import os;import tensorflow as tf;from IPython import display;from tensorflow.keras import layers;import matplotlib.pyplot as plt;import pathlib', function_to_run='tf.audio.decode_wav(*args, **kwargs)', method_object=None, object_signature=None, function_args=[eval('x')], function_kwargs={'desired_channels': eval('1'), 'desired_samples': eval('16000')})
(x, sample_rate) = tf.audio.decode_wav(x, desired_channels=1, desired_samples=16000)
custom_method(imports='import numpy as np;import seaborn as sns;from tensorflow.keras import models;import os;import tensorflow as tf;from IPython import display;from tensorflow.keras import layers;import matplotlib.pyplot as plt;import pathlib', function_to_run='tf.squeeze(*args, **kwargs)', method_object=None, object_signature=None, function_args=[eval('x')], function_kwargs={'axis': eval('-1')})
x = tf.squeeze(x, axis=-1)
waveform = x
x = get_spectrogram(x)
x = x[tf.newaxis, ...]
custom_method(imports='import numpy as np;import seaborn as sns;from tensorflow.keras import models;import os;import tensorflow as tf;from IPython import display;from tensorflow.keras import layers;import matplotlib.pyplot as plt;import pathlib', function_to_run='obj(*args)', method_object=eval('model'), object_signature=None, function_args=[eval('x')], function_kwargs={}, custom_class=None)
prediction = model(x)
x_labels = ['no', 'yes', 'down', 'go', 'left', 'up', 'right', 'stop']
plt.bar(x_labels, tf.nn.softmax(prediction[0]))
plt.title('No')
plt.show()
display.display(display.Audio(waveform, rate=16000))

class ExportModel(tf.Module):

    def __init__(self, model):
        self.model = model
        self.__call__.get_concrete_function(x=tf.TensorSpec(shape=(), dtype=tf.string))
        self.__call__.get_concrete_function(x=tf.TensorSpec(shape=[None, 16000], dtype=tf.float32))

    @tf.function
    def __call__(self, x):
        if x.dtype == tf.string:
            custom_method(imports='import numpy as np;import seaborn as sns;from tensorflow.keras import models;import os;import tensorflow as tf;from IPython import display;from tensorflow.keras import layers;import matplotlib.pyplot as plt;import pathlib', function_to_run='tf.io.read_file(*args)', method_object=None, object_signature=None, function_args=[eval('x')], function_kwargs={})
            x = tf.io.read_file(x)
            custom_method(imports='import numpy as np;import seaborn as sns;from tensorflow.keras import models;import os;import tensorflow as tf;from IPython import display;from tensorflow.keras import layers;import matplotlib.pyplot as plt;import pathlib', function_to_run='tf.audio.decode_wav(*args, **kwargs)', method_object=None, object_signature=None, function_args=[eval('x')], function_kwargs={'desired_channels': eval('1'), 'desired_samples': eval('16000')})
            (x, _) = tf.audio.decode_wav(x, desired_channels=1, desired_samples=16000)
            custom_method(imports='import numpy as np;import seaborn as sns;from tensorflow.keras import models;import os;import tensorflow as tf;from IPython import display;from tensorflow.keras import layers;import matplotlib.pyplot as plt;import pathlib', function_to_run='tf.squeeze(*args, **kwargs)', method_object=None, object_signature=None, function_args=[eval('x')], function_kwargs={'axis': eval('-1')})
            x = tf.squeeze(x, axis=-1)
            x = x[tf.newaxis, :]
        x = get_spectrogram(x)
        result = self.model(x, training=False)
        custom_method(imports='import numpy as np;import seaborn as sns;from tensorflow.keras import models;import os;import tensorflow as tf;from IPython import display;from tensorflow.keras import layers;import matplotlib.pyplot as plt;import pathlib', function_to_run='tf.argmax(*args, **kwargs)', method_object=None, object_signature=None, function_args=[eval('result')], function_kwargs={'axis': eval('-1')})
        class_ids = tf.argmax(result, axis=-1)
        custom_method(imports='import numpy as np;import seaborn as sns;from tensorflow.keras import models;import os;import tensorflow as tf;from IPython import display;from tensorflow.keras import layers;import matplotlib.pyplot as plt;import pathlib', function_to_run='tf.gather(*args)', method_object=None, object_signature=None, function_args=[eval('label_names'), eval('class_ids')], function_kwargs={})
        class_names = tf.gather(label_names, class_ids)
        return {'predictions': result, 'class_ids': class_ids, 'class_names': class_names}
export = ExportModel(model)
custom_method(imports='import numpy as np;import seaborn as sns;from tensorflow.keras import models;import os;import tensorflow as tf;from IPython import display;from tensorflow.keras import layers;import matplotlib.pyplot as plt;import pathlib', function_to_run='obj(*args)', method_object=eval('export'), object_signature='ExportModel(model)', function_args=[eval("tf.constant(str(data_dir/'no/01bb6a2a_nohash_0.wav'))")], function_kwargs={}, custom_class="class ExportModel(tf.Module):\n  def __init__(self, model):\n    self.model = model\n\n    self.__call__.get_concrete_function(\n        x=tf.TensorSpec(shape=(), dtype=tf.string))\n    self.__call__.get_concrete_function(\n       x=tf.TensorSpec(shape=[None, 16000], dtype=tf.float32))\n\n\n  @tf.function\n  def __call__(self, x):\n    if x.dtype == tf.string:\n      x = tf.io.read_file(x)\n      x, _ = tf.audio.decode_wav(x, desired_channels=1, desired_samples=16000,)\n      x = tf.squeeze(x, axis=-1)\n      x = x[tf.newaxis, :]\n    \n    x = get_spectrogram(x)  \n    result = self.model(x, training=False)\n    \n    class_ids = tf.argmax(result, axis=-1)\n    class_names = tf.gather(label_names, class_ids)\n    return {'predictions':result,\n            'class_ids': class_ids,\n            'class_names': class_names}")
export(tf.constant(str(data_dir / 'no/01bb6a2a_nohash_0.wav')))
custom_method(imports='import numpy as np;import seaborn as sns;from tensorflow.keras import models;import os;import tensorflow as tf;from IPython import display;from tensorflow.keras import layers;import matplotlib.pyplot as plt;import pathlib', function_to_run='tf.saved_model.save(*args)', method_object=None, object_signature=None, function_args=[eval('export'), eval('"saved"')], function_kwargs={})
tf.saved_model.save(export, 'saved')
custom_method(imports='import numpy as np;import seaborn as sns;from tensorflow.keras import models;import os;import tensorflow as tf;from IPython import display;from tensorflow.keras import layers;import matplotlib.pyplot as plt;import pathlib', function_to_run='tf.saved_model.load(*args)', method_object=None, object_signature=None, function_args=[eval('"saved"')], function_kwargs={})
imported = tf.saved_model.load('saved')
custom_method(imports='import numpy as np;import seaborn as sns;from tensorflow.keras import models;import os;import tensorflow as tf;from IPython import display;from tensorflow.keras import layers;import matplotlib.pyplot as plt;import pathlib', function_to_run='obj(*args)', method_object=eval('imported'), object_signature=None, function_args=[eval('waveform[tf.newaxis, :]')], function_kwargs={}, custom_class=None)
imported(waveform[tf.newaxis, :])
