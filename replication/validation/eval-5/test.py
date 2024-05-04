import unittest
import tensorflow as tf
from tensorflow.keras import datasets, layers, regularizers
import shutil
import tempfile
from tensorflow_docs import modeling, plots
import tensorflow_docs as tfdocs

# Define the value of FEATURES here, same as in the main code
FEATURES = 28
gz = tf.keras.utils.get_file('HIGGS.csv.gz', 'http://mlphysics.ics.uci.edu/data/higgs/HIGGS.csv.gz')
FEATURES = 28
ds = tf.data.experimental.CsvDataset(gz,[float(),]*(FEATURES+1), compression_type="GZIP")
def pack_row(*row):
  label = row[0]
  features = tf.stack(row[1:],1)
  return features, label
packed_ds = ds.batch(10000).map(pack_row).unbatch()
N_VALIDATION = int(1e3)
N_TRAIN = int(1e4)
BUFFER_SIZE = int(1e4)
BATCH_SIZE = 500
STEPS_PER_EPOCH = N_TRAIN//BATCH_SIZE
validate_ds = packed_ds.take(N_VALIDATION).cache()
train_ds = packed_ds.skip(N_VALIDATION).take(N_TRAIN).cache()
train_ds
validate_ds = validate_ds.batch(BATCH_SIZE)
train_ds = train_ds.shuffle(BUFFER_SIZE).repeat().batch(BATCH_SIZE)
lr_schedule = tf.keras.optimizers.schedules.InverseTimeDecay(
  0.001,
  decay_steps=STEPS_PER_EPOCH*1000,
  decay_rate=1,
  staircase=False)

# Define the log directory for TensorBoard
logdir = tempfile.mkdtemp()

def get_callbacks(name):
  return [
    tfdocs.modeling.EpochDots(),
    tf.keras.callbacks.EarlyStopping(monitor='val_binary_crossentropy', patience=200)
  ]

def get_optimizer():
  return tf.keras.optimizers.Adam(lr_schedule)

def compile_and_fit(model, name, optimizer=None, max_epochs=1000):
  if optimizer is None:
    optimizer = get_optimizer()
  model.compile(optimizer=optimizer,
                loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
                metrics=[
                  tf.keras.metrics.BinaryCrossentropy(
                      from_logits=True, name='binary_crossentropy'),
                  'accuracy'])

  model.summary()

  history = model.fit(
    train_ds,
    steps_per_epoch = STEPS_PER_EPOCH,
    epochs=max_epochs,
    validation_data=validate_ds,
    callbacks=get_callbacks(name),
    verbose=0)
  return history

class TestRegularizedModels(unittest.TestCase):
    def setUp(self):
        # Load the dataset and preprocess it
        (train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()

        # Preprocess the data
        train_images, test_images = train_images / 255.0, test_images / 255.0

        self.train_images = train_images
        self.train_labels = train_labels
        self.test_images = test_images
        self.test_labels = test_labels

    def test_tiny_model(self):
        tiny_model = tf.keras.Sequential([
            layers.Dense(16, activation='elu', input_shape=(FEATURES,)),
            layers.Dense(1)
        ])
        size_histories = {}
        size_histories['Tiny'] = compile_and_fit(tiny_model, 'sizes/Tiny')

        # Check if training was successful and model is not overfitting
        self.assertLess(size_histories['Tiny'].history['loss'][-1], 0.8)
        self.assertLess(size_histories['Tiny'].history['val_loss'][-1], 0.8)

        # Check if model accuracy is within acceptable range
        self.assertGreater(size_histories['Tiny'].history['accuracy'][-1], 0.5)
        self.assertGreater(size_histories['Tiny'].history['val_accuracy'][-1], 0.5)

    def test_small_model(self):
        small_model = tf.keras.Sequential([
            layers.Dense(16, activation='elu', input_shape=(FEATURES,)),
            layers.Dense(16, activation='elu'),
            layers.Dense(1)
        ])
        size_histories = {}
        size_histories['Small'] = compile_and_fit(small_model, 'sizes/Small')

        # Check if training was successful and model is not overfitting
        self.assertLess(size_histories['Small'].history['loss'][-1], 0.8)
        self.assertLess(size_histories['Small'].history['val_loss'][-1], 0.8)

        # Check if model accuracy is within acceptable range
        self.assertGreater(size_histories['Small'].history['accuracy'][-1], 0.5)
        self.assertGreater(size_histories['Small'].history['val_accuracy'][-1], 0.5)

    def test_medium_model(self):
        medium_model = tf.keras.Sequential([
            layers.Dense(64, activation='elu', input_shape=(FEATURES,)),
            layers.Dense(64, activation='elu'),
            layers.Dense(64, activation='elu'),
            layers.Dense(1)
        ])
        size_histories = {}
        size_histories['Medium'] = compile_and_fit(medium_model, "sizes/Medium")

        # Check if training was successful and model is not overfitting
        self.assertLess(size_histories['Medium'].history['loss'][-1], 0.8)
        self.assertLess(size_histories['Medium'].history['val_loss'][-1], 0.8)

        # Check if model accuracy is within acceptable range
        self.assertGreater(size_histories['Medium'].history['accuracy'][-1], 0.5)
        self.assertGreater(size_histories['Medium'].history['val_accuracy'][-1], 0.5)


if __name__ == '__main__':
    unittest.main()
