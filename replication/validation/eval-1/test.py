import os
import unittest
import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_io as tfio
import pandas as pd

class ReduceMeanLayer(tf.keras.layers.Layer):
    def __init__(self, axis=0, **kwargs):
        super(ReduceMeanLayer, self).__init__(**kwargs)
        self.axis = axis

    def call(self, input):
        return tf.math.reduce_mean(input, axis=self.axis)

def load_wav_16k_mono(filename):
    """ Load a WAV file, convert it to a float tensor, resample to 16 kHz single-channel audio. """
    file_contents = tf.io.read_file(filename)
    wav, sample_rate = tf.audio.decode_wav(
        file_contents,
        desired_channels=1)
    wav = tf.squeeze(wav, axis=-1)
    sample_rate = tf.cast(sample_rate, dtype=tf.int64)
    wav = tfio.audio.resample(wav, rate_in=sample_rate, rate_out=16000)
    return wav

def create_model(num_classes):
    return tf.keras.Sequential([
        tf.keras.layers.Input(shape=(1024), dtype=tf.float32, name='input_embedding'),
        tf.keras.layers.Dense(512, activation='relu'),
        tf.keras.layers.Dense(num_classes)
    ], name='my_model')

class TestYamnetModel(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.testing_wav_file_name = tf.keras.utils.get_file('miaow_16k.wav',
                                                           'https://storage.googleapis.com/audioset/miaow_16k.wav',
                                                           cache_dir='./',
                                                           cache_subdir='test_data')

        cls.yamnet_model_handle = 'https://tfhub.dev/google/yamnet/1'
        cls.yamnet_model = hub.load(cls.yamnet_model_handle)

        cls.testing_wav_data = load_wav_16k_mono(cls.testing_wav_file_name)

    def test_load_wav_16k_mono(self):
        wav_data = load_wav_16k_mono(self.testing_wav_file_name)
        self.assertTrue(isinstance(wav_data, tf.Tensor))
        self.assertEqual(wav_data.shape, (107698,))

    def test_reduce_mean_layer(self):
        layer = ReduceMeanLayer(axis=0)
        input_data = tf.constant([[1, 2], [3, 4]], dtype=tf.float32)
        expected_output = tf.constant([2, 3], dtype=tf.float32)
        output = layer(input_data)
        tf.debugging.assert_near(output, expected_output)

    def test_create_model(self):
        num_classes = 2  # 'dog' and 'cat'
        model = create_model(num_classes)
        self.assertIsInstance(model, tf.keras.Model)


if __name__ == '__main__':
    unittest.main()
