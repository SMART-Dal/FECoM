import unittest
import tensorflow as tf
from tensorflow.keras import datasets, layers, models
from io import StringIO
import sys

class TestCIFAR10Model(unittest.TestCase):
    def setUp(self):
        # Load the CIFAR-10 dataset and preprocess it
        (train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()
        self.train_images, self.test_images = train_images / 255.0, test_images / 255.0
        self.train_labels, self.test_labels = train_labels, test_labels

    def test_model_architecture(self):
        # Test the model architecture
        model = models.Sequential()
        model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Conv2D(64, (3, 3), activation='relu'))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Conv2D(64, (3, 3), activation='relu'))

        # Check the number of layers
        self.assertEqual(len(model.layers), 5)

        # Check the total number of parameters
        total_params = model.count_params()
        expected_total_params = 56320
        self.assertEqual(total_params, expected_total_params)

        # Check the number of trainable parameters
        trainable_params = sum([tf.keras.backend.count_params(w) for w in model.trainable_weights])
        expected_trainable_params = 56320
        self.assertEqual(trainable_params, expected_trainable_params)

        # Check the number of non-trainable parameters
        non_trainable_params = sum([tf.keras.backend.count_params(w) for w in model.non_trainable_weights])
        expected_non_trainable_params = 0
        self.assertEqual(non_trainable_params, expected_non_trainable_params)

    def test_model_training(self):
        # Test the model training
        model = models.Sequential([
            layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.Flatten(),
            layers.Dense(64, activation='relu'),
            layers.Dense(10)
        ])

        model.compile(optimizer='adam',
                      loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                      metrics=['accuracy'])

        # Train the model and check if the training loss decreases
        history = model.fit(self.train_images, self.train_labels, epochs=2, 
                            validation_data=(self.test_images, self.test_labels))
        self.assertLess(history.history['loss'][-1], history.history['loss'][0])

    def test_model_evaluation(self):
        # Test the model evaluation
        model = models.Sequential([
            layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.Flatten(),
            layers.Dense(64, activation='relu'),
            layers.Dense(10)
        ])

        model.compile(optimizer='adam',
                      loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                      metrics=['accuracy'])

        # Train the model
        model.fit(self.train_images, self.train_labels, epochs=2)

        # Evaluate the model on test data and check accuracy
        test_loss, test_acc = model.evaluate(self.test_images, self.test_labels, verbose=0)
        self.assertGreater(test_acc, 0.5)

if __name__ == '__main__':
    unittest.main()
