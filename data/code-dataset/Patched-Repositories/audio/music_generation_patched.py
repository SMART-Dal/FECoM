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

def custom_method(func, imports: str, function_to_run: str, method_object=None, object_signature=None, function_args: list=None, function_kwargs: dict=None, custom_class=None):
    result = send_request(imports=imports, function_to_run=function_to_run, function_args=function_args, function_kwargs=function_kwargs, max_wait_secs=MAX_WAIT_S, wait_after_run_secs=WAIT_AFTER_RUN_S, method_object=method_object, object_signature=object_signature, custom_class=custom_class, experiment_file_path=EXPERIMENT_FILE_PATH)
    return func
seed = 42
custom_method(
tf.random.set_seed(seed), imports='import numpy as np;import fluidsynth;from typing import Dict, List, Optional, Sequence, Tuple;import collections;import pandas as pd;import tensorflow as tf;from matplotlib import pyplot as plt;import datetime;import glob;import seaborn as sns;from IPython import display;import pretty_midi;import pathlib', function_to_run='tf.random.set_seed(*args)', method_object=None, object_signature=None, function_args=[eval('seed')], function_kwargs={})
np.random.seed(seed)
_SAMPLING_RATE = 16000
data_dir = pathlib.Path('data/maestro-v2.0.0')
if not data_dir.exists():
    custom_method(
    tf.keras.utils.get_file('maestro-v2.0.0-midi.zip', origin='https://storage.googleapis.com/magentadata/datasets/maestro/v2.0.0/maestro-v2.0.0-midi.zip', extract=True, cache_dir='.', cache_subdir='data'), imports='import numpy as np;import fluidsynth;from typing import Dict, List, Optional, Sequence, Tuple;import collections;import pandas as pd;import tensorflow as tf;from matplotlib import pyplot as plt;import datetime;import glob;import seaborn as sns;from IPython import display;import pretty_midi;import pathlib', function_to_run='tf.keras.utils.get_file(*args, **kwargs)', method_object=None, object_signature=None, function_args=[eval("'maestro-v2.0.0-midi.zip'")], function_kwargs={'origin': eval("'https://storage.googleapis.com/magentadata/datasets/maestro/v2.0.0/maestro-v2.0.0-midi.zip'"), 'extract': eval('True'), 'cache_dir': eval("'.'"), 'cache_subdir': eval("'data'")})
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
tf.data.Dataset.from_tensor_slices(train_notes), imports='import numpy as np;import fluidsynth;from typing import Dict, List, Optional, Sequence, Tuple;import collections;import pandas as pd;import tensorflow as tf;from matplotlib import pyplot as plt;import datetime;import glob;import seaborn as sns;from IPython import display;import pretty_midi;import pathlib', function_to_run='tf.data.Dataset.from_tensor_slices(*args)', method_object=None, object_signature=None, function_args=[eval('train_notes')], function_kwargs={})
notes_ds.element_spec

def create_sequences(dataset: tf.data.Dataset, seq_length: int, vocab_size=128) -> tf.data.Dataset:
    """Returns TF Dataset of sequence and label examples."""
    seq_length = seq_length + 1
    windows = dataset.window(seq_length, shift=1, stride=1, drop_remainder=True)
    flatten = lambda x: x.batch(seq_length, drop_remainder=True)
    sequences = windows.flat_map(flatten)

    def scale_pitch(x):
        x = x / [vocab_size, 1.0, 1.0]
        return x

    def split_labels(sequences):
        inputs = sequences[:-1]
        labels_dense = sequences[-1]
        labels = {key: labels_dense[i] for (i, key) in enumerate(key_order)}
        return (scale_pitch(inputs), labels)
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
    positive_pressure = 10 * tf.maximum(-y_pred, 0.0)
    return custom_method(
    tf.reduce_mean(mse + positive_pressure), imports='import numpy as np;import fluidsynth;from typing import Dict, List, Optional, Sequence, Tuple;import collections;import pandas as pd;import tensorflow as tf;from matplotlib import pyplot as plt;import datetime;import glob;import seaborn as sns;from IPython import display;import pretty_midi;import pathlib', function_to_run='tf.reduce_mean(*args)', method_object=None, object_signature=None, function_args=[eval('mse + positive_pressure')], function_kwargs={})
input_shape = (seq_length, 3)
learning_rate = 0.005
inputs = custom_method(
tf.keras.Input(input_shape), imports='import numpy as np;import fluidsynth;from typing import Dict, List, Optional, Sequence, Tuple;import collections;import pandas as pd;import tensorflow as tf;from matplotlib import pyplot as plt;import datetime;import glob;import seaborn as sns;from IPython import display;import pretty_midi;import pathlib', function_to_run='tf.keras.Input(*args)', method_object=None, object_signature=None, function_args=[eval('input_shape')], function_kwargs={})
x = custom_method(
tf.keras.layers.LSTM(128)(inputs), imports='import numpy as np;import fluidsynth;from typing import Dict, List, Optional, Sequence, Tuple;import collections;import pandas as pd;import tensorflow as tf;from matplotlib import pyplot as plt;import datetime;import glob;import seaborn as sns;from IPython import display;import pretty_midi;import pathlib', function_to_run='tf.keras.layers.LSTM(128)(*args)', method_object=None, object_signature=None, function_args=[eval('inputs')], function_kwargs={})
outputs = {'pitch': tf.keras.layers.Dense(128, name='pitch')(x), 'step': tf.keras.layers.Dense(1, name='step')(x), 'duration': tf.keras.layers.Dense(1, name='duration')(x)}
model = custom_method(
tf.keras.Model(inputs, outputs), imports='import numpy as np;import fluidsynth;from typing import Dict, List, Optional, Sequence, Tuple;import collections;import pandas as pd;import tensorflow as tf;from matplotlib import pyplot as plt;import datetime;import glob;import seaborn as sns;from IPython import display;import pretty_midi;import pathlib', function_to_run='tf.keras.Model(*args)', method_object=None, object_signature=None, function_args=[eval('inputs'), eval('outputs')], function_kwargs={})
loss = {'pitch': tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), 'step': mse_with_positive_pressure, 'duration': mse_with_positive_pressure}
optimizer = custom_method(
tf.keras.optimizers.Adam(learning_rate=learning_rate), imports='import numpy as np;import fluidsynth;from typing import Dict, List, Optional, Sequence, Tuple;import collections;import pandas as pd;import tensorflow as tf;from matplotlib import pyplot as plt;import datetime;import glob;import seaborn as sns;from IPython import display;import pretty_midi;import pathlib', function_to_run='tf.keras.optimizers.Adam(**kwargs)', method_object=None, object_signature=None, function_args=[], function_kwargs={'learning_rate': eval('learning_rate')})
model.compile(loss=loss, optimizer=optimizer)
model.summary()
losses = model.evaluate(train_ds, return_dict=True)
losses
model.compile(loss=loss, loss_weights={'pitch': 0.05, 'step': 1.0, 'duration': 1.0}, optimizer=optimizer)
model.evaluate(train_ds, return_dict=True)
callbacks = [tf.keras.callbacks.ModelCheckpoint(filepath='./training_checkpoints/ckpt_{epoch}', save_weights_only=True), tf.keras.callbacks.EarlyStopping(monitor='loss', patience=5, verbose=1, restore_best_weights=True)]
epochs = 50
history = model.fit(train_ds, epochs=epochs, callbacks=callbacks)
plt.plot(history.epoch, history.history['loss'], label='total loss')
plt.show()

def predict_next_note(notes: np.ndarray, keras_model: tf.keras.Model, temperature: float=1.0) -> int:
    """Generates a note IDs using a trained sequence model."""
    assert temperature > 0
    inputs = custom_method(
    tf.expand_dims(notes, 0), imports='import numpy as np;import fluidsynth;from typing import Dict, List, Optional, Sequence, Tuple;import collections;import pandas as pd;import tensorflow as tf;from matplotlib import pyplot as plt;import datetime;import glob;import seaborn as sns;from IPython import display;import pretty_midi;import pathlib', function_to_run='tf.expand_dims(*args)', method_object=None, object_signature=None, function_args=[eval('notes'), eval('0')], function_kwargs={})
    predictions = model.predict(inputs)
    pitch_logits = predictions['pitch']
    step = predictions['step']
    duration = predictions['duration']
    pitch_logits /= temperature
    pitch = custom_method(
    tf.random.categorical(pitch_logits, num_samples=1), imports='import numpy as np;import fluidsynth;from typing import Dict, List, Optional, Sequence, Tuple;import collections;import pandas as pd;import tensorflow as tf;from matplotlib import pyplot as plt;import datetime;import glob;import seaborn as sns;from IPython import display;import pretty_midi;import pathlib', function_to_run='tf.random.categorical(*args, **kwargs)', method_object=None, object_signature=None, function_args=[eval('pitch_logits')], function_kwargs={'num_samples': eval('1')})
    pitch = custom_method(
    tf.squeeze(pitch, axis=-1), imports='import numpy as np;import fluidsynth;from typing import Dict, List, Optional, Sequence, Tuple;import collections;import pandas as pd;import tensorflow as tf;from matplotlib import pyplot as plt;import datetime;import glob;import seaborn as sns;from IPython import display;import pretty_midi;import pathlib', function_to_run='tf.squeeze(*args, **kwargs)', method_object=None, object_signature=None, function_args=[eval('pitch')], function_kwargs={'axis': eval('-1')})
    duration = custom_method(
    tf.squeeze(duration, axis=-1), imports='import numpy as np;import fluidsynth;from typing import Dict, List, Optional, Sequence, Tuple;import collections;import pandas as pd;import tensorflow as tf;from matplotlib import pyplot as plt;import datetime;import glob;import seaborn as sns;from IPython import display;import pretty_midi;import pathlib', function_to_run='tf.squeeze(*args, **kwargs)', method_object=None, object_signature=None, function_args=[eval('duration')], function_kwargs={'axis': eval('-1')})
    step = custom_method(
    tf.squeeze(step, axis=-1), imports='import numpy as np;import fluidsynth;from typing import Dict, List, Optional, Sequence, Tuple;import collections;import pandas as pd;import tensorflow as tf;from matplotlib import pyplot as plt;import datetime;import glob;import seaborn as sns;from IPython import display;import pretty_midi;import pathlib', function_to_run='tf.squeeze(*args, **kwargs)', method_object=None, object_signature=None, function_args=[eval('step')], function_kwargs={'axis': eval('-1')})
    step = custom_method(
    tf.maximum(0, step), imports='import numpy as np;import fluidsynth;from typing import Dict, List, Optional, Sequence, Tuple;import collections;import pandas as pd;import tensorflow as tf;from matplotlib import pyplot as plt;import datetime;import glob;import seaborn as sns;from IPython import display;import pretty_midi;import pathlib', function_to_run='tf.maximum(*args)', method_object=None, object_signature=None, function_args=[eval('0'), eval('step')], function_kwargs={})
    duration = custom_method(
    tf.maximum(0, duration), imports='import numpy as np;import fluidsynth;from typing import Dict, List, Optional, Sequence, Tuple;import collections;import pandas as pd;import tensorflow as tf;from matplotlib import pyplot as plt;import datetime;import glob;import seaborn as sns;from IPython import display;import pretty_midi;import pathlib', function_to_run='tf.maximum(*args)', method_object=None, object_signature=None, function_args=[eval('0'), eval('duration')], function_kwargs={})
    return (int(pitch), float(step), float(duration))
temperature = 2.0
num_predictions = 120
sample_notes = np.stack([raw_notes[key] for key in key_order], axis=1)
input_notes = sample_notes[:seq_length] / np.array([vocab_size, 1, 1])
generated_notes = []
prev_start = 0
for _ in range(num_predictions):
    (pitch, step, duration) = predict_next_note(input_notes, model, temperature)
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
