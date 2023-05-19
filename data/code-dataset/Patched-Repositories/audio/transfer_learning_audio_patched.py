import os
from IPython import display
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_io as tfio
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
yamnet_model_handle = 'https://tfhub.dev/google/yamnet/1'
yamnet_model = hub.load(yamnet_model_handle)
custom_method(imports='import matplotlib.pyplot as plt;import os;from IPython import display;import pandas as pd;import tensorflow as tf;import tensorflow_hub as hub;import tensorflow_io as tfio;import numpy as np', function_to_run='tf.keras.utils.get_file(*args, **kwargs)', method_object=None, object_signature=None, function_args=[eval("'miaow_16k.wav'"), eval("'https://storage.googleapis.com/audioset/miaow_16k.wav'")], function_kwargs={'cache_dir': eval("'./'"), 'cache_subdir': eval("'test_data'")})
testing_wav_file_name = tf.keras.utils.get_file('miaow_16k.wav', 'https://storage.googleapis.com/audioset/miaow_16k.wav', cache_dir='./', cache_subdir='test_data')
print(testing_wav_file_name)

@tf.function
def load_wav_16k_mono(filename):
    """ Load a WAV file, convert it to a float tensor, resample to 16 kHz single-channel audio. """
    custom_method(imports='import matplotlib.pyplot as plt;import os;from IPython import display;import pandas as pd;import tensorflow as tf;import tensorflow_hub as hub;import tensorflow_io as tfio;import numpy as np', function_to_run='tf.io.read_file(*args)', method_object=None, object_signature=None, function_args=[eval('filename')], function_kwargs={})
    file_contents = tf.io.read_file(filename)
    custom_method(imports='import matplotlib.pyplot as plt;import os;from IPython import display;import pandas as pd;import tensorflow as tf;import tensorflow_hub as hub;import tensorflow_io as tfio;import numpy as np', function_to_run='tf.audio.decode_wav(*args, **kwargs)', method_object=None, object_signature=None, function_args=[eval('file_contents')], function_kwargs={'desired_channels': eval('1')})
    (wav, sample_rate) = tf.audio.decode_wav(file_contents, desired_channels=1)
    custom_method(imports='import matplotlib.pyplot as plt;import os;from IPython import display;import pandas as pd;import tensorflow as tf;import tensorflow_hub as hub;import tensorflow_io as tfio;import numpy as np', function_to_run='tf.squeeze(*args, **kwargs)', method_object=None, object_signature=None, function_args=[eval('wav')], function_kwargs={'axis': eval('-1')})
    wav = tf.squeeze(wav, axis=-1)
    custom_method(imports='import matplotlib.pyplot as plt;import os;from IPython import display;import pandas as pd;import tensorflow as tf;import tensorflow_hub as hub;import tensorflow_io as tfio;import numpy as np', function_to_run='tf.cast(*args, **kwargs)', method_object=None, object_signature=None, function_args=[eval('sample_rate')], function_kwargs={'dtype': eval('tf.int64')})
    sample_rate = tf.cast(sample_rate, dtype=tf.int64)
    wav = tfio.audio.resample(wav, rate_in=sample_rate, rate_out=16000)
    return wav
testing_wav_data = load_wav_16k_mono(testing_wav_file_name)
_ = plt.plot(testing_wav_data)
display.Audio(testing_wav_data, rate=16000)
class_map_path = yamnet_model.class_map_path().numpy().decode('utf-8')
class_names = list(pd.read_csv(class_map_path)['display_name'])
for name in class_names[:20]:
    print(name)
print('...')
(scores, embeddings, spectrogram) = yamnet_model(testing_wav_data)
custom_method(imports='import matplotlib.pyplot as plt;import os;from IPython import display;import pandas as pd;import tensorflow as tf;import tensorflow_hub as hub;import tensorflow_io as tfio;import numpy as np', function_to_run='tf.reduce_mean(*args, **kwargs)', method_object=None, object_signature=None, function_args=[eval('scores')], function_kwargs={'axis': eval('0')})
class_scores = tf.reduce_mean(scores, axis=0)
custom_method(imports='import matplotlib.pyplot as plt;import os;from IPython import display;import pandas as pd;import tensorflow as tf;import tensorflow_hub as hub;import tensorflow_io as tfio;import numpy as np', function_to_run='tf.math.argmax(*args)', method_object=None, object_signature=None, function_args=[eval('class_scores')], function_kwargs={})
top_class = tf.math.argmax(class_scores)
inferred_class = class_names[top_class]
print(f'The main sound is: {inferred_class}')
print(f'The embeddings shape: {embeddings.shape}')
custom_method(imports='import matplotlib.pyplot as plt;import os;from IPython import display;import pandas as pd;import tensorflow as tf;import tensorflow_hub as hub;import tensorflow_io as tfio;import numpy as np', function_to_run='tf.keras.utils.get_file(*args, **kwargs)', method_object=None, object_signature=None, function_args=[eval("'esc-50.zip'"), eval("'https://github.com/karoldvl/ESC-50/archive/master.zip'")], function_kwargs={'cache_dir': eval("'./'"), 'cache_subdir': eval("'datasets'"), 'extract': eval('True')})
_ = tf.keras.utils.get_file('esc-50.zip', 'https://github.com/karoldvl/ESC-50/archive/master.zip', cache_dir='./', cache_subdir='datasets', extract=True)
esc50_csv = './datasets/ESC-50-master/meta/esc50.csv'
base_data_path = './datasets/ESC-50-master/audio/'
pd_data = pd.read_csv(esc50_csv)
pd_data.head()
my_classes = ['dog', 'cat']
map_class_to_id = {'dog': 0, 'cat': 1}
filtered_pd = pd_data[pd_data.category.isin(my_classes)]
class_id = filtered_pd['category'].apply(lambda name: map_class_to_id[name])
filtered_pd = filtered_pd.assign(target=class_id)
full_path = filtered_pd['filename'].apply(lambda row: os.path.join(base_data_path, row))
filtered_pd = filtered_pd.assign(filename=full_path)
filtered_pd.head(10)
filenames = filtered_pd['filename']
targets = filtered_pd['target']
folds = filtered_pd['fold']
custom_method(imports='import matplotlib.pyplot as plt;import os;from IPython import display;import pandas as pd;import tensorflow as tf;import tensorflow_hub as hub;import tensorflow_io as tfio;import numpy as np', function_to_run='tf.data.Dataset.from_tensor_slices(*args)', method_object=None, object_signature=None, function_args=[eval('(filenames, targets, folds)')], function_kwargs={})
main_ds = tf.data.Dataset.from_tensor_slices((filenames, targets, folds))
main_ds.element_spec

def load_wav_for_map(filename, label, fold):
    return (load_wav_16k_mono(filename), label, fold)
custom_method(imports='import matplotlib.pyplot as plt;import os;from IPython import display;import pandas as pd;import tensorflow as tf;import tensorflow_hub as hub;import tensorflow_io as tfio;import numpy as np', function_to_run='obj.map(*args)', method_object=eval('main_ds'), object_signature=None, function_args=[eval('load_wav_for_map')], function_kwargs={}, custom_class=None)
main_ds = main_ds.map(load_wav_for_map)
main_ds.element_spec

def extract_embedding(wav_data, label, fold):
    """ run YAMNet to extract embedding from the wav data """
    (scores, embeddings, spectrogram) = yamnet_model(wav_data)
    num_embeddings = tf.shape(embeddings)[0]
    return (embeddings, tf.repeat(label, num_embeddings), tf.repeat(fold, num_embeddings))
custom_method(imports='import matplotlib.pyplot as plt;import os;from IPython import display;import pandas as pd;import tensorflow as tf;import tensorflow_hub as hub;import tensorflow_io as tfio;import numpy as np', function_to_run='obj.map(extract_embedding).unbatch()', method_object=eval('main_ds'), object_signature=None, function_args=[], function_kwargs={}, custom_class=None)
main_ds = main_ds.map(extract_embedding).unbatch()
main_ds.element_spec
custom_method(imports='import matplotlib.pyplot as plt;import os;from IPython import display;import pandas as pd;import tensorflow as tf;import tensorflow_hub as hub;import tensorflow_io as tfio;import numpy as np', function_to_run='obj.cache()', method_object=eval('main_ds'), object_signature=None, function_args=[], function_kwargs={}, custom_class=None)
cached_ds = main_ds.cache()
train_ds = cached_ds.filter(lambda embedding, label, fold: fold < 4)
val_ds = cached_ds.filter(lambda embedding, label, fold: fold == 4)
test_ds = cached_ds.filter(lambda embedding, label, fold: fold == 5)
remove_fold_column = lambda embedding, label, fold: (embedding, label)
train_ds = train_ds.map(remove_fold_column)
val_ds = val_ds.map(remove_fold_column)
test_ds = test_ds.map(remove_fold_column)
train_ds = train_ds.cache().shuffle(1000).batch(32).prefetch(tf.data.AUTOTUNE)
val_ds = val_ds.cache().batch(32).prefetch(tf.data.AUTOTUNE)
test_ds = test_ds.cache().batch(32).prefetch(tf.data.AUTOTUNE)
custom_method(imports='import matplotlib.pyplot as plt;import os;from IPython import display;import pandas as pd;import tensorflow as tf;import tensorflow_hub as hub;import tensorflow_io as tfio;import numpy as np', function_to_run='tf.keras.Sequential(*args, **kwargs)', method_object=None, object_signature=None, function_args=[eval("[\n    tf.keras.layers.Input(shape=(1024), dtype=tf.float32,\n                          name='input_embedding'),\n    tf.keras.layers.Dense(512, activation='relu'),\n    tf.keras.layers.Dense(len(my_classes))\n]")], function_kwargs={'name': eval("'my_model'")})
my_model = tf.keras.Sequential([tf.keras.layers.Input(shape=1024, dtype=tf.float32, name='input_embedding'), tf.keras.layers.Dense(512, activation='relu'), tf.keras.layers.Dense(len(my_classes))], name='my_model')
custom_method(imports='import matplotlib.pyplot as plt;import os;from IPython import display;import pandas as pd;import tensorflow as tf;import tensorflow_hub as hub;import tensorflow_io as tfio;import numpy as np', function_to_run='obj.summary()', method_object=eval('my_model'), object_signature=None, function_args=[], function_kwargs={}, custom_class=None)
my_model.summary()
custom_method(imports='import matplotlib.pyplot as plt;import os;from IPython import display;import pandas as pd;import tensorflow as tf;import tensorflow_hub as hub;import tensorflow_io as tfio;import numpy as np', function_to_run='obj.compile(**kwargs)', method_object=eval('my_model'), object_signature=None, function_args=[], function_kwargs={'loss': eval('tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)'), 'optimizer': eval('"adam"'), 'metrics': eval("['accuracy']")}, custom_class=None)
my_model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), optimizer='adam', metrics=['accuracy'])
custom_method(imports='import matplotlib.pyplot as plt;import os;from IPython import display;import pandas as pd;import tensorflow as tf;import tensorflow_hub as hub;import tensorflow_io as tfio;import numpy as np', function_to_run='tf.keras.callbacks.EarlyStopping(**kwargs)', method_object=None, object_signature=None, function_args=[], function_kwargs={'monitor': eval("'loss'"), 'patience': eval('3'), 'restore_best_weights': eval('True')})
callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=3, restore_best_weights=True)
custom_method(imports='import matplotlib.pyplot as plt;import os;from IPython import display;import pandas as pd;import tensorflow as tf;import tensorflow_hub as hub;import tensorflow_io as tfio;import numpy as np', function_to_run='obj.fit(*args, **kwargs)', method_object=eval('my_model'), object_signature=None, function_args=[eval('train_ds')], function_kwargs={'epochs': eval('20'), 'validation_data': eval('val_ds'), 'callbacks': eval('callback')}, custom_class=None)
history = my_model.fit(train_ds, epochs=20, validation_data=val_ds, callbacks=callback)
custom_method(imports='import matplotlib.pyplot as plt;import os;from IPython import display;import pandas as pd;import tensorflow as tf;import tensorflow_hub as hub;import tensorflow_io as tfio;import numpy as np', function_to_run='obj.evaluate(*args)', method_object=eval('my_model'), object_signature=None, function_args=[eval('test_ds')], function_kwargs={}, custom_class=None)
(loss, accuracy) = my_model.evaluate(test_ds)
print('Loss: ', loss)
print('Accuracy: ', accuracy)
(scores, embeddings, spectrogram) = yamnet_model(testing_wav_data)
custom_method(imports='import matplotlib.pyplot as plt;import os;from IPython import display;import pandas as pd;import tensorflow as tf;import tensorflow_hub as hub;import tensorflow_io as tfio;import numpy as np', function_to_run='obj(embeddings).numpy()', method_object=eval('my_model'), object_signature=None, function_args=[], function_kwargs={}, custom_class=None)
result = my_model(embeddings).numpy()
inferred_class = my_classes[result.mean(axis=0).argmax()]
print(f'The main sound is: {inferred_class}')

class ReduceMeanLayer(tf.keras.layers.Layer):

    def __init__(self, axis=0, **kwargs):
        super(ReduceMeanLayer, self).__init__(**kwargs)
        self.axis = axis

    def call(self, input):
        return tf.math.reduce_mean(input, axis=self.axis)
saved_model_path = './dogs_and_cats_yamnet'
custom_method(imports='import matplotlib.pyplot as plt;import os;from IPython import display;import pandas as pd;import tensorflow as tf;import tensorflow_hub as hub;import tensorflow_io as tfio;import numpy as np', function_to_run='tf.keras.layers.Input(**kwargs)', method_object=None, object_signature=None, function_args=[], function_kwargs={'shape': eval('()'), 'dtype': eval('tf.float32'), 'name': eval("'audio'")})
input_segment = tf.keras.layers.Input(shape=(), dtype=tf.float32, name='audio')
embedding_extraction_layer = hub.KerasLayer(yamnet_model_handle, trainable=False, name='yamnet')
(_, embeddings_output, _) = embedding_extraction_layer(input_segment)
custom_method(imports='import matplotlib.pyplot as plt;import os;from IPython import display;import pandas as pd;import tensorflow as tf;import tensorflow_hub as hub;import tensorflow_io as tfio;import numpy as np', function_to_run='obj(*args)', method_object=eval('my_model'), object_signature=None, function_args=[eval('embeddings_output')], function_kwargs={}, custom_class=None)
serving_outputs = my_model(embeddings_output)
serving_outputs = ReduceMeanLayer(axis=0, name='classifier')(serving_outputs)
custom_method(imports='import matplotlib.pyplot as plt;import os;from IPython import display;import pandas as pd;import tensorflow as tf;import tensorflow_hub as hub;import tensorflow_io as tfio;import numpy as np', function_to_run='tf.keras.Model(*args)', method_object=None, object_signature=None, function_args=[eval('input_segment'), eval('serving_outputs')], function_kwargs={})
serving_model = tf.keras.Model(input_segment, serving_outputs)
custom_method(imports='import matplotlib.pyplot as plt;import os;from IPython import display;import pandas as pd;import tensorflow as tf;import tensorflow_hub as hub;import tensorflow_io as tfio;import numpy as np', function_to_run='obj.save(*args, **kwargs)', method_object=eval('serving_model'), object_signature=None, function_args=[eval('saved_model_path')], function_kwargs={'include_optimizer': eval('False')}, custom_class=None)
serving_model.save(saved_model_path, include_optimizer=False)
custom_method(imports='import matplotlib.pyplot as plt;import os;from IPython import display;import pandas as pd;import tensorflow as tf;import tensorflow_hub as hub;import tensorflow_io as tfio;import numpy as np', function_to_run='tf.keras.utils.plot_model(*args)', method_object=None, object_signature=None, function_args=[eval('serving_model')], function_kwargs={})
tf.keras.utils.plot_model(serving_model)
custom_method(imports='import matplotlib.pyplot as plt;import os;from IPython import display;import pandas as pd;import tensorflow as tf;import tensorflow_hub as hub;import tensorflow_io as tfio;import numpy as np', function_to_run='tf.saved_model.load(*args)', method_object=None, object_signature=None, function_args=[eval('saved_model_path')], function_kwargs={})
reloaded_model = tf.saved_model.load(saved_model_path)
custom_method(imports='import matplotlib.pyplot as plt;import os;from IPython import display;import pandas as pd;import tensorflow as tf;import tensorflow_hub as hub;import tensorflow_io as tfio;import numpy as np', function_to_run='obj(*args)', method_object=eval('reloaded_model'), object_signature=None, function_args=[eval('testing_wav_data')], function_kwargs={}, custom_class=None)
reloaded_results = reloaded_model(testing_wav_data)
cat_or_dog = my_classes[tf.math.argmax(reloaded_results)]
print(f'The main sound is: {cat_or_dog}')
custom_method(imports='import matplotlib.pyplot as plt;import os;from IPython import display;import pandas as pd;import tensorflow as tf;import tensorflow_hub as hub;import tensorflow_io as tfio;import numpy as np', function_to_run="obj.signatures['serving_default'](*args)", method_object=eval('reloaded_model'), object_signature=None, function_args=[eval('testing_wav_data')], function_kwargs={}, custom_class=None)
serving_results = reloaded_model.signatures['serving_default'](testing_wav_data)
cat_or_dog = my_classes[tf.math.argmax(serving_results['classifier'])]
print(f'The main sound is: {cat_or_dog}')
test_pd = filtered_pd.loc[filtered_pd['fold'] == 5]
row = test_pd.sample(1)
filename = row['filename'].item()
print(filename)
waveform = load_wav_16k_mono(filename)
print(f'Waveform values: {waveform}')
_ = plt.plot(waveform)
display.Audio(waveform, rate=16000)
(scores, embeddings, spectrogram) = yamnet_model(waveform)
custom_method(imports='import matplotlib.pyplot as plt;import os;from IPython import display;import pandas as pd;import tensorflow as tf;import tensorflow_hub as hub;import tensorflow_io as tfio;import numpy as np', function_to_run='tf.reduce_mean(*args, **kwargs)', method_object=None, object_signature=None, function_args=[eval('scores')], function_kwargs={'axis': eval('0')})
class_scores = tf.reduce_mean(scores, axis=0)
custom_method(imports='import matplotlib.pyplot as plt;import os;from IPython import display;import pandas as pd;import tensorflow as tf;import tensorflow_hub as hub;import tensorflow_io as tfio;import numpy as np', function_to_run='tf.math.argmax(*args)', method_object=None, object_signature=None, function_args=[eval('class_scores')], function_kwargs={})
top_class = tf.math.argmax(class_scores)
inferred_class = class_names[top_class]
top_score = class_scores[top_class]
print(f'[YAMNet] The main sound is: {inferred_class} ({top_score})')
custom_method(imports='import matplotlib.pyplot as plt;import os;from IPython import display;import pandas as pd;import tensorflow as tf;import tensorflow_hub as hub;import tensorflow_io as tfio;import numpy as np', function_to_run='obj(*args)', method_object=eval('reloaded_model'), object_signature=None, function_args=[eval('waveform')], function_kwargs={}, custom_class=None)
reloaded_results = reloaded_model(waveform)
custom_method(imports='import matplotlib.pyplot as plt;import os;from IPython import display;import pandas as pd;import tensorflow as tf;import tensorflow_hub as hub;import tensorflow_io as tfio;import numpy as np', function_to_run='tf.math.argmax(*args)', method_object=None, object_signature=None, function_args=[eval('reloaded_results')], function_kwargs={})
your_top_class = tf.math.argmax(reloaded_results)
your_inferred_class = my_classes[your_top_class]
custom_method(imports='import matplotlib.pyplot as plt;import os;from IPython import display;import pandas as pd;import tensorflow as tf;import tensorflow_hub as hub;import tensorflow_io as tfio;import numpy as np', function_to_run='tf.nn.softmax(*args, **kwargs)', method_object=None, object_signature=None, function_args=[eval('reloaded_results')], function_kwargs={'axis': eval('-1')})
class_probabilities = tf.nn.softmax(reloaded_results, axis=-1)
your_top_score = class_probabilities[your_top_class]
print(f'[Your model] The main sound is: {your_inferred_class} ({your_top_score})')
