import unittest
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.model_selection import train_test_split

# Define the AnomalyDetector model as shown in the code snippet

class AnomalyDetector(tf.keras.Model):
    def __init__(self):
        super(AnomalyDetector, self).__init__()
        self.encoder = tf.keras.Sequential([
            tf.keras.layers.Dense(32, activation="relu"),
            tf.keras.layers.Dense(16, activation="relu"),
            tf.keras.layers.Dense(8, activation="relu")])

        self.decoder = tf.keras.Sequential([
            tf.keras.layers.Dense(16, activation="relu"),
            tf.keras.layers.Dense(32, activation="relu"),
            tf.keras.layers.Dense(140, activation="sigmoid")])

    def call(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


class TestAnomalyDetector(unittest.TestCase):
    def setUp(self):
        dataframe = pd.read_csv('http://storage.googleapis.com/download.tensorflow.org/data/ecg.csv', header=None)
        raw_data = dataframe.values
        labels = raw_data[:, -1]
        data = raw_data[:, 0:-1]
        train_data, test_data, train_labels, test_labels = train_test_split(
            data, labels, test_size=0.2, random_state=21
        )
        min_val = tf.reduce_min(train_data)
        max_val = tf.reduce_max(train_data)

        train_data = (train_data - min_val) / (max_val - min_val)
        test_data = (test_data - min_val) / (max_val - min_val)

        train_data = tf.cast(train_data, tf.float32)
        test_data = tf.cast(test_data, tf.float32)
        train_labels = train_labels.astype(bool)
        test_labels = test_labels.astype(bool)

        self.normal_train_data = train_data[train_labels]
        self.normal_test_data = test_data[test_labels]

        self.anomalous_train_data = train_data[~train_labels]
        self.anomalous_test_data = test_data[~test_labels]

        self.autoencoder = AnomalyDetector()
        self.autoencoder.compile(optimizer='adam', loss='mae')
        self.autoencoder.fit(self.normal_train_data, self.normal_train_data,
                             epochs=20,
                             batch_size=512,
                             validation_data=(self.normal_test_data, self.normal_test_data),
                             shuffle=True)

    def test_anomaly_detection(self):
        reconstructions = self.autoencoder.predict(self.anomalous_test_data)
        test_loss = tf.keras.losses.mae(reconstructions, self.anomalous_test_data)
        threshold = np.mean(test_loss) + np.std(test_loss)
        preds = tf.math.less(test_loss, threshold)
        # Convert continuous values to binary based on the threshold
        binary_preds = preds.numpy().astype(int)
        binary_labels = np.ones_like(binary_preds)

        self.print_stats(binary_preds, binary_labels)

    def print_stats(self, predictions, labels):
        accuracy = accuracy_score(labels, predictions)
        precision = precision_score(labels, predictions)
        recall = recall_score(labels, predictions)
        print(f"Accuracy: {accuracy}")
        print(f"Precision: {precision}")
        print(f"Recall: {recall}")

if __name__ == '__main__':
    unittest.main()