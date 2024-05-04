import unittest
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from IPython import display
import pathlib

class TestYourFunctions(unittest.TestCase):
    def setUp(self):
        seed = 42
        tf.random.set_seed(seed)
        np.random.seed(seed)

    def squeeze(self, audio, labels):
        audio = tf.squeeze(audio, axis=-1)
        return audio, labels

    def get_spectrogram(self, waveform):
        spectrogram = tf.signal.stft(
            waveform, frame_length=255, frame_step=128)
        spectrogram = tf.abs(spectrogram)
        spectrogram = spectrogram[..., tf.newaxis]
        return spectrogram

    def make_spec_ds(self, ds):
        return ds.map(
            map_func=lambda audio, label: (self.get_spectrogram(audio), label),
            num_parallel_calls=tf.data.AUTOTUNE)

    def create_sample_model(self, num_labels):
        model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(124, 129, 1)),
            tf.keras.layers.Conv2D(32, 3, activation='relu'),
            tf.keras.layers.MaxPooling2D(),
            tf.keras.layers.Dropout(0.25),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(num_labels),
        ])
        return model

    def train_model(self, model, train_ds, val_ds):
        model.compile(
            optimizer=tf.keras.optimizers.Adam(),
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=['accuracy'],
        )

        history = model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=10,
            callbacks=tf.keras.callbacks.EarlyStopping(verbose=1, patience=2),
        )
        return history

    def evaluate_model(self, model, test_ds):
        evaluation_result = model.evaluate(test_ds, return_dict=True)
        return evaluation_result

if __name__ == '__main__':
    unittest.main()
