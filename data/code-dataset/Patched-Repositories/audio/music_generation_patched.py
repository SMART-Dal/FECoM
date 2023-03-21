import collections
import datetime
import fluidsynth
import glob
import numpy as np
import pathlib
import pandas as pd
import pretty_midi
import seaborn as sns
import tensorflow as tf
from IPython import display
from matplotlib import pyplot as plt
from typing import Dict, List, Optional, Sequence, Tuple
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

def custom_method(func, imports: str, function_to_run: str, method_object=None, object_signature=None, function_args: list=None, function_kwargs: dict=None, max_wait_secs=MAX_WAIT_S, custom_class=None, wait_after_run_secs=WAIT_AFTER_RUN_S):
    result = send_request(imports=imports, function_to_run=function_to_run, function_args=function_args, function_kwargs=function_kwargs, max_wait_secs=max_wait_secs, wait_after_run_secs=wait_after_run_secs, method_object=method_object, object_signature=object_signature, custom_class=custom_class, experiment_file_path=EXPERIMENT_FILE_PATH)
    return func
seed = 42
custom_method(
tf.random.set_seed(seed), imports='import pandas as pd;from matplotlib import pyplot as plt;import datetime;import glob;import numpy as np;import pretty_midi;from typing import Dict, List, Optional, Sequence, Tuple;import collections;import fluidsynth;import pathlib;import tensorflow as tf;from IPython import display;import seaborn as sns', function_to_run='tf.random.set_seed(*args)', method_object=None, object_signature=None, function_args=[eval('seed')], function_kwargs={}, max_wait_secs=0)
np.random.seed(seed)
_SAMPLING_RATE = 16000
data_dir = pathlib.Path('data/maestro-v2.0.0')
if not data_dir.exists():
    custom_method(
    tf.keras.utils.get_file('maestro-v2.0.0-midi.zip', origin='https://storage.googleapis.com/magentadata/datasets/maestro/v2.0.0/maestro-v2.0.0-midi.zip', extract=True, cache_dir='.', cache_subdir='data'), imports='import pandas as pd;from matplotlib import pyplot as plt;import datetime;import glob;import numpy as np;import pretty_midi;from typing import Dict, List, Optional, Sequence, Tuple;import collections;import fluidsynth;import pathlib;import tensorflow as tf;from IPython import display;import seaborn as sns', function_to_run='tf.keras.utils.get_file(*args, **kwargs)', method_object=None, object_signature=None, function_args=[eval("'maestro-v2.0.0-midi.zip'")], function_kwargs={'origin': eval("'https://storage.googleapis.com/magentadata/datasets/maestro/v2.0.0/maestro-v2.0.0-midi.zip'"), 'extract': eval('True'), 'cache_dir': eval("'.'"), 'cache_subdir': eval("'data'")}, max_wait_secs=0)
filenames = glob.glob(str(data_dir / '**/*.mid*'))
print('Number of files:', len(filenames))
sample_file = filenames[1]
print(sample_file)
pm = pretty_midi.PrettyMIDI(sample_file)

def display_audio(pm: pretty_midi.PrettyMIDI, seconds=30):
    waveform = pm.fluidsynth(fs=_SAMPLING_RATE)
    waveform_short = waveform[:seconds * _SAMPLING_RATE]
    return display.Audio(waveform_short, rate=_SAMPLING_RATE)
display_audio(pm)
print('Number of instruments:', len(pm.instruments))
instrument = pm.instruments[0]
instrument_name = pretty_midi.program_to_instrument_name(instrument.program)
print('Instrument name:', instrument_name)
for (i, note) in enumerate(instrument.notes[:10]):
    note_name = pretty_midi.note_number_to_name(note.pitch)
    duration = note.end - note.start
    print(f'{i}: pitch={note.pitch}, note_name={note_name}, duration={duration:.4f}')

def midi_to_notes(midi_file: str) -> pd.DataFrame:
    pm = pretty_midi.PrettyMIDI(midi_file)
    instrument = pm.instruments[0]
    notes = collections.defaultdict(list)
    sorted_notes = sorted(instrument.notes, key=lambda note: note.start)
    prev_start = sorted_notes[0].start
    for note in sorted_notes:
        start = note.start
        end = note.end
        notes['pitch'].append(note.pitch)
        notes['start'].append(start)
        notes['end'].append(end)
        notes['step'].append(start - prev_start)
        notes['duration'].append(end - start)
        prev_start = start
    return pd.DataFrame({name: np.array(value) for (name, value) in notes.items()})
raw_notes = midi_to_notes(sample_file)
raw_notes.head()
get_note_names = np.vectorize(pretty_midi.note_number_to_name)
sample_note_names = get_note_names(raw_notes['pitch'])
sample_note_names[:10]

def plot_piano_roll(notes: pd.DataFrame, count: Optional[int]=None):
    if count:
        title = f'First {count} notes'
    else:
        title = f'Whole track'
        count = len(notes['pitch'])
    plt.figure(figsize=(20, 4))
    plot_pitch = np.stack([notes['pitch'], notes['pitch']], axis=0)
    plot_start_stop = np.stack([notes['start'], notes['end']], axis=0)
    plt.plot(plot_start_stop[:, :count], plot_pitch[:, :count], color='b', marker='.')
    plt.xlabel('Time [s]')
    plt.ylabel('Pitch')
    _ = plt.title(title)
plot_piano_roll(raw_notes, count=100)
plot_piano_roll(raw_notes)

def plot_distributions(notes: pd.DataFrame, drop_percentile=2.5):
    plt.figure(figsize=[15, 5])
    plt.subplot(1, 3, 1)
    sns.histplot(notes, x='pitch', bins=20)
    plt.subplot(1, 3, 2)
    max_step = np.percentile(notes['step'], 100 - drop_percentile)
    sns.histplot(notes, x='step', bins=np.linspace(0, max_step, 21))
    plt.subplot(1, 3, 3)
    max_duration = np.percentile(notes['duration'], 100 - drop_percentile)
    sns.histplot(notes, x='duration', bins=np.linspace(0, max_duration, 21))
plot_distributions(raw_notes)

def notes_to_midi(notes: pd.DataFrame, out_file: str, instrument_name: str, velocity: int=100) -> pretty_midi.PrettyMIDI:
    pm = pretty_midi.PrettyMIDI()
    instrument = pretty_midi.Instrument(program=pretty_midi.instrument_name_to_program(instrument_name))
    prev_start = 0
    for (i, note) in notes.iterrows():
        start = float(prev_start + note['step'])
        end = float(start + note['duration'])
        note = pretty_midi.Note(velocity=velocity, pitch=int(note['pitch']), start=start, end=end)
        instrument.notes.append(note)
        prev_start = start
    pm.instruments.append(instrument)
    pm.write(out_file)
    return pm
example_file = 'example.midi'
example_pm = notes_to_midi(raw_notes, out_file=example_file, instrument_name=instrument_name)
display_audio(example_pm)
num_files = 5
all_notes = []
for f in filenames[:num_files]:
    notes = midi_to_notes(f)
    all_notes.append(notes)
all_notes = pd.concat(all_notes)
n_notes = len(all_notes)
print('Number of notes parsed:', n_notes)
key_order = ['pitch', 'step', 'duration']
train_notes = np.stack([all_notes[key] for key in key_order], axis=1)
notes_ds = custom_method(
tf.data.Dataset.from_tensor_slices(train_notes), imports='import pandas as pd;from matplotlib import pyplot as plt;import datetime;import glob;import numpy as np;import pretty_midi;from typing import Dict, List, Optional, Sequence, Tuple;import collections;import fluidsynth;import pathlib;import tensorflow as tf;from IPython import display;import seaborn as sns', function_to_run='tf.data.Dataset.from_tensor_slices(*args)', method_object=None, object_signature=None, function_args=[eval('train_notes')], function_kwargs={}, max_wait_secs=0)
notes_ds.element_spec

def create_sequences(dataset: tf.data.Dataset, seq_length: int, vocab_size=128) -> tf.data.Dataset:
    """Returns TF Dataset of sequence and label examples."""
    seq_length = seq_length + 1
    windows = dataset.window(seq_length, shift=1, stride=1, drop_remainder=True)
    flatten = lambda x: custom_method(
    x.batch(seq_length, drop_remainder=True), imports='import pandas as pd;from matplotlib import pyplot as plt;import datetime;import glob;import numpy as np;import pretty_midi;from typing import Dict, List, Optional, Sequence, Tuple;import collections;import fluidsynth;import pathlib;import tensorflow as tf;from IPython import display;import seaborn as sns', function_to_run='obj.batch(*args, **kwargs)', method_object=eval('x'), object_signature='tf.keras.layers.LSTM(128)', function_args=[eval('seq_length')], function_kwargs={'drop_remainder': eval('True')}, max_wait_secs=0, custom_class=None)
    sequences = windows.flat_map(flatten)

    def scale_pitch(x):
        x = x / [vocab_size, 1.0, 1.0]
        return x

    def split_labels(sequences):
        inputs = sequences[:-1]
        labels_dense = sequences[-1]
        labels = {key: labels_dense[i] for (i, key) in enumerate(key_order)}
        return (custom_method(
        scale_pitch(inputs), imports='import pandas as pd;from matplotlib import pyplot as plt;import datetime;import glob;import numpy as np;import pretty_midi;from typing import Dict, List, Optional, Sequence, Tuple;import collections;import fluidsynth;import pathlib;import tensorflow as tf;from IPython import display;import seaborn as sns', function_to_run='obj(*args)', method_object=eval('scale_pitch'), object_signature='tf.random.categorical', function_args=[eval('inputs')], function_kwargs={}, max_wait_secs=0, custom_class=None), labels)
    return sequences.map(split_labels, num_parallel_calls=tf.data.AUTOTUNE)
seq_length = 25
vocab_size = 128
seq_ds = create_sequences(notes_ds, seq_length, vocab_size)
seq_ds.element_spec
for (seq, target) in seq_ds.take(1):
    print('sequence shape:', seq.shape)
    print('sequence elements (first 10):', seq[0:10])
    print()
    print('target:', target)
batch_size = 64
buffer_size = n_notes - seq_length
train_ds = seq_ds.shuffle(buffer_size).batch(batch_size, drop_remainder=True).cache().prefetch(tf.data.experimental.AUTOTUNE)
train_ds.element_spec

def mse_with_positive_pressure(y_true: tf.Tensor, y_pred: tf.Tensor):
    mse = (y_true - y_pred) ** 2
    positive_pressure = 10 * custom_method(
    tf.maximum(-y_pred, 0.0), imports='import pandas as pd;from matplotlib import pyplot as plt;import datetime;import glob;import numpy as np;import pretty_midi;from typing import Dict, List, Optional, Sequence, Tuple;import collections;import fluidsynth;import pathlib;import tensorflow as tf;from IPython import display;import seaborn as sns', function_to_run='tf.maximum(*args)', method_object=None, object_signature=None, function_args=[eval('-y_pred'), eval('0.0')], function_kwargs={}, max_wait_secs=0)
    return custom_method(
    tf.reduce_mean(mse + positive_pressure), imports='import pandas as pd;from matplotlib import pyplot as plt;import datetime;import glob;import numpy as np;import pretty_midi;from typing import Dict, List, Optional, Sequence, Tuple;import collections;import fluidsynth;import pathlib;import tensorflow as tf;from IPython import display;import seaborn as sns', function_to_run='tf.reduce_mean(*args)', method_object=None, object_signature=None, function_args=[eval('mse + positive_pressure')], function_kwargs={}, max_wait_secs=0)
input_shape = (seq_length, 3)
learning_rate = 0.005
inputs = custom_method(
tf.keras.Input(input_shape), imports='import pandas as pd;from matplotlib import pyplot as plt;import datetime;import glob;import numpy as np;import pretty_midi;from typing import Dict, List, Optional, Sequence, Tuple;import collections;import fluidsynth;import pathlib;import tensorflow as tf;from IPython import display;import seaborn as sns', function_to_run='tf.keras.Input(*args)', method_object=None, object_signature=None, function_args=[eval('input_shape')], function_kwargs={}, max_wait_secs=0)
x = custom_method(
tf.keras.layers.LSTM(128)(inputs), imports='import pandas as pd;from matplotlib import pyplot as plt;import datetime;import glob;import numpy as np;import pretty_midi;from typing import Dict, List, Optional, Sequence, Tuple;import collections;import fluidsynth;import pathlib;import tensorflow as tf;from IPython import display;import seaborn as sns', function_to_run='tf.keras.layers.LSTM(128)(*args)', method_object=None, object_signature=None, function_args=[eval('inputs')], function_kwargs={}, max_wait_secs=0)
outputs = {'pitch': custom_method(
tf.keras.layers.Dense(128, name='pitch')(x), imports='import pandas as pd;from matplotlib import pyplot as plt;import datetime;import glob;import numpy as np;import pretty_midi;from typing import Dict, List, Optional, Sequence, Tuple;import collections;import fluidsynth;import pathlib;import tensorflow as tf;from IPython import display;import seaborn as sns', function_to_run="tf.keras.layers.Dense(128, name='pitch')(*args)", method_object=None, object_signature=None, function_args=[eval('x')], function_kwargs={}, max_wait_secs=0), 'step': custom_method(
tf.keras.layers.Dense(1, name='step')(x), imports='import pandas as pd;from matplotlib import pyplot as plt;import datetime;import glob;import numpy as np;import pretty_midi;from typing import Dict, List, Optional, Sequence, Tuple;import collections;import fluidsynth;import pathlib;import tensorflow as tf;from IPython import display;import seaborn as sns', function_to_run="tf.keras.layers.Dense(1, name='step')(*args)", method_object=None, object_signature=None, function_args=[eval('x')], function_kwargs={}, max_wait_secs=0), 'duration': custom_method(
tf.keras.layers.Dense(1, name='duration')(x), imports='import pandas as pd;from matplotlib import pyplot as plt;import datetime;import glob;import numpy as np;import pretty_midi;from typing import Dict, List, Optional, Sequence, Tuple;import collections;import fluidsynth;import pathlib;import tensorflow as tf;from IPython import display;import seaborn as sns', function_to_run="tf.keras.layers.Dense(1, name='duration')(*args)", method_object=None, object_signature=None, function_args=[eval('x')], function_kwargs={}, max_wait_secs=0)}
model = custom_method(
tf.keras.Model(inputs, outputs), imports='import pandas as pd;from matplotlib import pyplot as plt;import datetime;import glob;import numpy as np;import pretty_midi;from typing import Dict, List, Optional, Sequence, Tuple;import collections;import fluidsynth;import pathlib;import tensorflow as tf;from IPython import display;import seaborn as sns', function_to_run='tf.keras.Model(*args)', method_object=None, object_signature=None, function_args=[eval('inputs'), eval('outputs')], function_kwargs={}, max_wait_secs=0)
loss = {'pitch': custom_method(
tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), imports='import pandas as pd;from matplotlib import pyplot as plt;import datetime;import glob;import numpy as np;import pretty_midi;from typing import Dict, List, Optional, Sequence, Tuple;import collections;import fluidsynth;import pathlib;import tensorflow as tf;from IPython import display;import seaborn as sns', function_to_run='tf.keras.losses.SparseCategoricalCrossentropy(**kwargs)', method_object=None, object_signature=None, function_args=[], function_kwargs={'from_logits': eval('True')}, max_wait_secs=0), 'step': mse_with_positive_pressure, 'duration': mse_with_positive_pressure}
optimizer = custom_method(
tf.keras.optimizers.Adam(learning_rate=learning_rate), imports='import pandas as pd;from matplotlib import pyplot as plt;import datetime;import glob;import numpy as np;import pretty_midi;from typing import Dict, List, Optional, Sequence, Tuple;import collections;import fluidsynth;import pathlib;import tensorflow as tf;from IPython import display;import seaborn as sns', function_to_run='tf.keras.optimizers.Adam(**kwargs)', method_object=None, object_signature=None, function_args=[], function_kwargs={'learning_rate': eval('learning_rate')}, max_wait_secs=0)
custom_method(
model.compile(loss=loss, optimizer=optimizer), imports='import pandas as pd;from matplotlib import pyplot as plt;import datetime;import glob;import numpy as np;import pretty_midi;from typing import Dict, List, Optional, Sequence, Tuple;import collections;import fluidsynth;import pathlib;import tensorflow as tf;from IPython import display;import seaborn as sns', function_to_run='obj.compile(**kwargs)', method_object=eval('model'), object_signature='tf.keras.Model', function_args=[], function_kwargs={'loss': eval('loss'), 'optimizer': eval('optimizer')}, max_wait_secs=0, custom_class=None)
custom_method(
model.summary(), imports='import pandas as pd;from matplotlib import pyplot as plt;import datetime;import glob;import numpy as np;import pretty_midi;from typing import Dict, List, Optional, Sequence, Tuple;import collections;import fluidsynth;import pathlib;import tensorflow as tf;from IPython import display;import seaborn as sns', function_to_run='obj.summary()', method_object=eval('model'), object_signature='tf.keras.Model', function_args=[], function_kwargs={}, max_wait_secs=0, custom_class=None)
losses = custom_method(
model.evaluate(train_ds, return_dict=True), imports='import pandas as pd;from matplotlib import pyplot as plt;import datetime;import glob;import numpy as np;import pretty_midi;from typing import Dict, List, Optional, Sequence, Tuple;import collections;import fluidsynth;import pathlib;import tensorflow as tf;from IPython import display;import seaborn as sns', function_to_run='obj.evaluate(*args, **kwargs)', method_object=eval('model'), object_signature='tf.keras.Model', function_args=[eval('train_ds')], function_kwargs={'return_dict': eval('True')}, max_wait_secs=0, custom_class=None)
losses
custom_method(
model.compile(loss=loss, loss_weights={'pitch': 0.05, 'step': 1.0, 'duration': 1.0}, optimizer=optimizer), imports='import pandas as pd;from matplotlib import pyplot as plt;import datetime;import glob;import numpy as np;import pretty_midi;from typing import Dict, List, Optional, Sequence, Tuple;import collections;import fluidsynth;import pathlib;import tensorflow as tf;from IPython import display;import seaborn as sns', function_to_run='obj.compile(**kwargs)', method_object=eval('model'), object_signature='tf.keras.Model', function_args=[], function_kwargs={'loss': eval('loss'), 'loss_weights': eval("{\n        'pitch': 0.05,\n        'step': 1.0,\n        'duration':1.0,\n    }"), 'optimizer': eval('optimizer')}, max_wait_secs=0, custom_class=None)
custom_method(
model.evaluate(train_ds, return_dict=True), imports='import pandas as pd;from matplotlib import pyplot as plt;import datetime;import glob;import numpy as np;import pretty_midi;from typing import Dict, List, Optional, Sequence, Tuple;import collections;import fluidsynth;import pathlib;import tensorflow as tf;from IPython import display;import seaborn as sns', function_to_run='obj.evaluate(*args, **kwargs)', method_object=eval('model'), object_signature='tf.keras.Model', function_args=[eval('train_ds')], function_kwargs={'return_dict': eval('True')}, max_wait_secs=0, custom_class=None)
callbacks = [custom_method(
tf.keras.callbacks.ModelCheckpoint(filepath='./training_checkpoints/ckpt_{epoch}', save_weights_only=True), imports='import pandas as pd;from matplotlib import pyplot as plt;import datetime;import glob;import numpy as np;import pretty_midi;from typing import Dict, List, Optional, Sequence, Tuple;import collections;import fluidsynth;import pathlib;import tensorflow as tf;from IPython import display;import seaborn as sns', function_to_run='tf.keras.callbacks.ModelCheckpoint(**kwargs)', method_object=None, object_signature=None, function_args=[], function_kwargs={'filepath': eval("'./training_checkpoints/ckpt_{epoch}'"), 'save_weights_only': eval('True')}, max_wait_secs=0), custom_method(
tf.keras.callbacks.EarlyStopping(monitor='loss', patience=5, verbose=1, restore_best_weights=True), imports='import pandas as pd;from matplotlib import pyplot as plt;import datetime;import glob;import numpy as np;import pretty_midi;from typing import Dict, List, Optional, Sequence, Tuple;import collections;import fluidsynth;import pathlib;import tensorflow as tf;from IPython import display;import seaborn as sns', function_to_run='tf.keras.callbacks.EarlyStopping(**kwargs)', method_object=None, object_signature=None, function_args=[], function_kwargs={'monitor': eval("'loss'"), 'patience': eval('5'), 'verbose': eval('1'), 'restore_best_weights': eval('True')}, max_wait_secs=0)]
epochs = 50
history = custom_method(
model.fit(train_ds, epochs=epochs, callbacks=callbacks), imports='import pandas as pd;from matplotlib import pyplot as plt;import datetime;import glob;import numpy as np;import pretty_midi;from typing import Dict, List, Optional, Sequence, Tuple;import collections;import fluidsynth;import pathlib;import tensorflow as tf;from IPython import display;import seaborn as sns', function_to_run='obj.fit(*args, **kwargs)', method_object=eval('model'), object_signature='tf.keras.Model', function_args=[eval('train_ds')], function_kwargs={'epochs': eval('epochs'), 'callbacks': eval('callbacks')}, max_wait_secs=0, custom_class=None)
plt.plot(history.epoch, history.history['loss'], label='total loss')
plt.show()

def predict_next_note(notes: np.ndarray, keras_model: tf.keras.Model, temperature: float=1.0) -> int:
    """Generates a note IDs using a trained sequence model."""
    assert temperature > 0
    inputs = custom_method(
    tf.expand_dims(notes, 0), imports='import pandas as pd;from matplotlib import pyplot as plt;import datetime;import glob;import numpy as np;import pretty_midi;from typing import Dict, List, Optional, Sequence, Tuple;import collections;import fluidsynth;import pathlib;import tensorflow as tf;from IPython import display;import seaborn as sns', function_to_run='tf.expand_dims(*args)', method_object=None, object_signature=None, function_args=[eval('notes'), eval('0')], function_kwargs={}, max_wait_secs=0)
    predictions = custom_method(
    model.predict(inputs), imports='import pandas as pd;from matplotlib import pyplot as plt;import datetime;import glob;import numpy as np;import pretty_midi;from typing import Dict, List, Optional, Sequence, Tuple;import collections;import fluidsynth;import pathlib;import tensorflow as tf;from IPython import display;import seaborn as sns', function_to_run='obj.predict(*args)', method_object=eval('model'), object_signature='tf.keras.Model', function_args=[eval('inputs')], function_kwargs={}, max_wait_secs=0, custom_class=None)
    pitch_logits = predictions['pitch']
    step = predictions['step']
    duration = predictions['duration']
    pitch_logits /= temperature
    pitch = custom_method(
    tf.random.categorical(pitch_logits, num_samples=1), imports='import pandas as pd;from matplotlib import pyplot as plt;import datetime;import glob;import numpy as np;import pretty_midi;from typing import Dict, List, Optional, Sequence, Tuple;import collections;import fluidsynth;import pathlib;import tensorflow as tf;from IPython import display;import seaborn as sns', function_to_run='tf.random.categorical(*args, **kwargs)', method_object=None, object_signature=None, function_args=[eval('pitch_logits')], function_kwargs={'num_samples': eval('1')}, max_wait_secs=0)
    pitch = custom_method(
    tf.squeeze(pitch, axis=-1), imports='import pandas as pd;from matplotlib import pyplot as plt;import datetime;import glob;import numpy as np;import pretty_midi;from typing import Dict, List, Optional, Sequence, Tuple;import collections;import fluidsynth;import pathlib;import tensorflow as tf;from IPython import display;import seaborn as sns', function_to_run='tf.squeeze(*args, **kwargs)', method_object=None, object_signature=None, function_args=[eval('pitch')], function_kwargs={'axis': eval('-1')}, max_wait_secs=0)
    duration = custom_method(
    tf.squeeze(duration, axis=-1), imports='import pandas as pd;from matplotlib import pyplot as plt;import datetime;import glob;import numpy as np;import pretty_midi;from typing import Dict, List, Optional, Sequence, Tuple;import collections;import fluidsynth;import pathlib;import tensorflow as tf;from IPython import display;import seaborn as sns', function_to_run='tf.squeeze(*args, **kwargs)', method_object=None, object_signature=None, function_args=[eval('duration')], function_kwargs={'axis': eval('-1')}, max_wait_secs=0)
    step = custom_method(
    tf.squeeze(step, axis=-1), imports='import pandas as pd;from matplotlib import pyplot as plt;import datetime;import glob;import numpy as np;import pretty_midi;from typing import Dict, List, Optional, Sequence, Tuple;import collections;import fluidsynth;import pathlib;import tensorflow as tf;from IPython import display;import seaborn as sns', function_to_run='tf.squeeze(*args, **kwargs)', method_object=None, object_signature=None, function_args=[eval('step')], function_kwargs={'axis': eval('-1')}, max_wait_secs=0)
    step = custom_method(
    tf.maximum(0, step), imports='import pandas as pd;from matplotlib import pyplot as plt;import datetime;import glob;import numpy as np;import pretty_midi;from typing import Dict, List, Optional, Sequence, Tuple;import collections;import fluidsynth;import pathlib;import tensorflow as tf;from IPython import display;import seaborn as sns', function_to_run='tf.maximum(*args)', method_object=None, object_signature=None, function_args=[eval('0'), eval('step')], function_kwargs={}, max_wait_secs=0)
    duration = custom_method(
    tf.maximum(0, duration), imports='import pandas as pd;from matplotlib import pyplot as plt;import datetime;import glob;import numpy as np;import pretty_midi;from typing import Dict, List, Optional, Sequence, Tuple;import collections;import fluidsynth;import pathlib;import tensorflow as tf;from IPython import display;import seaborn as sns', function_to_run='tf.maximum(*args)', method_object=None, object_signature=None, function_args=[eval('0'), eval('duration')], function_kwargs={}, max_wait_secs=0)
    return (int(pitch), float(step), float(duration))
temperature = 2.0
num_predictions = 120
sample_notes = np.stack([raw_notes[key] for key in key_order], axis=1)
input_notes = sample_notes[:seq_length] / np.array([vocab_size, 1, 1])
generated_notes = []
prev_start = 0
for _ in range(num_predictions):
    (pitch, step, duration) = custom_method(
    predict_next_note(input_notes, model, temperature), imports='import pandas as pd;from matplotlib import pyplot as plt;import datetime;import glob;import numpy as np;import pretty_midi;from typing import Dict, List, Optional, Sequence, Tuple;import collections;import fluidsynth;import pathlib;import tensorflow as tf;from IPython import display;import seaborn as sns', function_to_run='obj(*args)', method_object=eval('predict_next_note'), object_signature='tf.keras.layers.LSTM(128)', function_args=[eval('input_notes'), eval('model'), eval('temperature')], function_kwargs={}, max_wait_secs=0, custom_class=None)
    start = prev_start + step
    end = start + duration
    input_note = (pitch, step, duration)
    generated_notes.append((*input_note, start, end))
    input_notes = np.delete(input_notes, 0, axis=0)
    input_notes = np.append(input_notes, np.expand_dims(input_note, 0), axis=0)
    prev_start = start
generated_notes = pd.DataFrame(generated_notes, columns=(*key_order, 'start', 'end'))
generated_notes.head(10)
out_file = 'output.mid'
out_pm = notes_to_midi(generated_notes, out_file=out_file, instrument_name=instrument_name)
display_audio(out_pm)
plot_piano_roll(generated_notes)
plot_distributions(generated_notes)
