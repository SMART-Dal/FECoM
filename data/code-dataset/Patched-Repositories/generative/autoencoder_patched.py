import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, losses
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.models import Model
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
custom_method(imports='import numpy as np;from tensorflow.keras.models import Model;from sklearn.model_selection import train_test_split;from tensorflow.keras import layers, losses;import tensorflow as tf;import matplotlib.pyplot as plt;from sklearn.metrics import accuracy_score, precision_score, recall_score;from tensorflow.keras.datasets import fashion_mnist;import pandas as pd', function_to_run='fashion_mnist.load_data()', method_object=None, object_signature=None, function_args=[], function_kwargs={})
((x_train, _), (x_test, _)) = fashion_mnist.load_data()
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0
print(x_train.shape)
print(x_test.shape)
latent_dim = 64

class Autoencoder(Model):

    def __init__(self, latent_dim):
        super(Autoencoder, self).__init__()
        self.latent_dim = latent_dim
        custom_method(imports='import numpy as np;from tensorflow.keras.models import Model;from sklearn.model_selection import train_test_split;from tensorflow.keras import layers, losses;import tensorflow as tf;import matplotlib.pyplot as plt;from sklearn.metrics import accuracy_score, precision_score, recall_score;from tensorflow.keras.datasets import fashion_mnist;import pandas as pd', function_to_run='tf.keras.Sequential(*args)', method_object=None, object_signature=None, function_args=[eval("[\n      layers.Flatten(),\n      layers.Dense(latent_dim, activation='relu'),\n    ]")], function_kwargs={})
        self.encoder = tf.keras.Sequential([layers.Flatten(), layers.Dense(latent_dim, activation='relu')])
        custom_method(imports='import numpy as np;from tensorflow.keras.models import Model;from sklearn.model_selection import train_test_split;from tensorflow.keras import layers, losses;import tensorflow as tf;import matplotlib.pyplot as plt;from sklearn.metrics import accuracy_score, precision_score, recall_score;from tensorflow.keras.datasets import fashion_mnist;import pandas as pd', function_to_run='tf.keras.Sequential(*args)', method_object=None, object_signature=None, function_args=[eval("[\n      layers.Dense(784, activation='sigmoid'),\n      layers.Reshape((28, 28))\n    ]")], function_kwargs={})
        self.decoder = tf.keras.Sequential([layers.Dense(784, activation='sigmoid'), layers.Reshape((28, 28))])

    def call(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded
autoencoder = Autoencoder(latent_dim)
custom_method(imports='import numpy as np;from tensorflow.keras.models import Model;from sklearn.model_selection import train_test_split;from tensorflow.keras import layers, losses;import tensorflow as tf;import matplotlib.pyplot as plt;from sklearn.metrics import accuracy_score, precision_score, recall_score;from tensorflow.keras.datasets import fashion_mnist;import pandas as pd', function_to_run='obj.compile(**kwargs)', method_object=eval('autoencoder'), object_signature='Autoencoder(latent_dim)', function_args=[], function_kwargs={'optimizer': eval("'adam'"), 'loss': eval('losses.MeanSquaredError()')}, custom_class="class Autoencoder(Model):\n  def __init__(self, latent_dim):\n    super(Autoencoder, self).__init__()\n    self.latent_dim = latent_dim   \n    self.encoder = tf.keras.Sequential([\n      layers.Flatten(),\n      layers.Dense(latent_dim, activation='relu'),\n    ])\n    self.decoder = tf.keras.Sequential([\n      layers.Dense(784, activation='sigmoid'),\n      layers.Reshape((28, 28))\n    ])\n\n  def call(self, x):\n    encoded = self.encoder(x)\n    decoded = self.decoder(encoded)\n    return decoded")
autoencoder.compile(optimizer='adam', loss=losses.MeanSquaredError())
custom_method(imports='import numpy as np;from tensorflow.keras.models import Model;from sklearn.model_selection import train_test_split;from tensorflow.keras import layers, losses;import tensorflow as tf;import matplotlib.pyplot as plt;from sklearn.metrics import accuracy_score, precision_score, recall_score;from tensorflow.keras.datasets import fashion_mnist;import pandas as pd', function_to_run='obj.fit(*args, **kwargs)', method_object=eval('autoencoder'), object_signature='Autoencoder(latent_dim)', function_args=[eval('x_train'), eval('x_train')], function_kwargs={'epochs': eval('10'), 'shuffle': eval('True'), 'validation_data': eval('(x_test, x_test)')}, custom_class="class Autoencoder(Model):\n  def __init__(self, latent_dim):\n    super(Autoencoder, self).__init__()\n    self.latent_dim = latent_dim   \n    self.encoder = tf.keras.Sequential([\n      layers.Flatten(),\n      layers.Dense(latent_dim, activation='relu'),\n    ])\n    self.decoder = tf.keras.Sequential([\n      layers.Dense(784, activation='sigmoid'),\n      layers.Reshape((28, 28))\n    ])\n\n  def call(self, x):\n    encoded = self.encoder(x)\n    decoded = self.decoder(encoded)\n    return decoded")
autoencoder.fit(x_train, x_train, epochs=10, shuffle=True, validation_data=(x_test, x_test))
custom_method(imports='import numpy as np;from tensorflow.keras.models import Model;from sklearn.model_selection import train_test_split;from tensorflow.keras import layers, losses;import tensorflow as tf;import matplotlib.pyplot as plt;from sklearn.metrics import accuracy_score, precision_score, recall_score;from tensorflow.keras.datasets import fashion_mnist;import pandas as pd', function_to_run='obj.encoder(x_test).numpy()', method_object=eval('autoencoder'), object_signature=None, function_args=[], function_kwargs={}, custom_class="class Autoencoder(Model):\n  def __init__(self, latent_dim):\n    super(Autoencoder, self).__init__()\n    self.latent_dim = latent_dim   \n    self.encoder = tf.keras.Sequential([\n      layers.Flatten(),\n      layers.Dense(latent_dim, activation='relu'),\n    ])\n    self.decoder = tf.keras.Sequential([\n      layers.Dense(784, activation='sigmoid'),\n      layers.Reshape((28, 28))\n    ])\n\n  def call(self, x):\n    encoded = self.encoder(x)\n    decoded = self.decoder(encoded)\n    return decoded")
encoded_imgs = autoencoder.encoder(x_test).numpy()
custom_method(imports='import numpy as np;from tensorflow.keras.models import Model;from sklearn.model_selection import train_test_split;from tensorflow.keras import layers, losses;import tensorflow as tf;import matplotlib.pyplot as plt;from sklearn.metrics import accuracy_score, precision_score, recall_score;from tensorflow.keras.datasets import fashion_mnist;import pandas as pd', function_to_run='obj.decoder(encoded_imgs).numpy()', method_object=eval('autoencoder'), object_signature=None, function_args=[], function_kwargs={}, custom_class="class Autoencoder(Model):\n  def __init__(self, latent_dim):\n    super(Autoencoder, self).__init__()\n    self.latent_dim = latent_dim   \n    self.encoder = tf.keras.Sequential([\n      layers.Flatten(),\n      layers.Dense(latent_dim, activation='relu'),\n    ])\n    self.decoder = tf.keras.Sequential([\n      layers.Dense(784, activation='sigmoid'),\n      layers.Reshape((28, 28))\n    ])\n\n  def call(self, x):\n    encoded = self.encoder(x)\n    decoded = self.decoder(encoded)\n    return decoded")
decoded_imgs = autoencoder.decoder(encoded_imgs).numpy()
n = 10
plt.figure(figsize=(20, 4))
for i in range(n):
    ax = plt.subplot(2, n, i + 1)
    plt.imshow(x_test[i])
    plt.title('original')
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    ax = plt.subplot(2, n, i + 1 + n)
    plt.imshow(decoded_imgs[i])
    plt.title('reconstructed')
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()
custom_method(imports='import numpy as np;from tensorflow.keras.models import Model;from sklearn.model_selection import train_test_split;from tensorflow.keras import layers, losses;import tensorflow as tf;import matplotlib.pyplot as plt;from sklearn.metrics import accuracy_score, precision_score, recall_score;from tensorflow.keras.datasets import fashion_mnist;import pandas as pd', function_to_run='fashion_mnist.load_data()', method_object=None, object_signature=None, function_args=[], function_kwargs={})
((x_train, _), (x_test, _)) = fashion_mnist.load_data()
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0
x_train = x_train[..., tf.newaxis]
x_test = x_test[..., tf.newaxis]
print(x_train.shape)
noise_factor = 0.2
x_train_noisy = x_train + noise_factor * tf.random.normal(shape=x_train.shape)
x_test_noisy = x_test + noise_factor * tf.random.normal(shape=x_test.shape)
custom_method(imports='import numpy as np;from tensorflow.keras.models import Model;from sklearn.model_selection import train_test_split;from tensorflow.keras import layers, losses;import tensorflow as tf;import matplotlib.pyplot as plt;from sklearn.metrics import accuracy_score, precision_score, recall_score;from tensorflow.keras.datasets import fashion_mnist;import pandas as pd', function_to_run='tf.clip_by_value(*args, **kwargs)', method_object=None, object_signature=None, function_args=[eval('x_train_noisy')], function_kwargs={'clip_value_min': eval('0.'), 'clip_value_max': eval('1.')})
x_train_noisy = tf.clip_by_value(x_train_noisy, clip_value_min=0.0, clip_value_max=1.0)
custom_method(imports='import numpy as np;from tensorflow.keras.models import Model;from sklearn.model_selection import train_test_split;from tensorflow.keras import layers, losses;import tensorflow as tf;import matplotlib.pyplot as plt;from sklearn.metrics import accuracy_score, precision_score, recall_score;from tensorflow.keras.datasets import fashion_mnist;import pandas as pd', function_to_run='tf.clip_by_value(*args, **kwargs)', method_object=None, object_signature=None, function_args=[eval('x_test_noisy')], function_kwargs={'clip_value_min': eval('0.'), 'clip_value_max': eval('1.')})
x_test_noisy = tf.clip_by_value(x_test_noisy, clip_value_min=0.0, clip_value_max=1.0)
n = 10
plt.figure(figsize=(20, 2))
for i in range(n):
    ax = plt.subplot(1, n, i + 1)
    plt.title('original + noise')
    plt.imshow(tf.squeeze(x_test_noisy[i]))
    plt.gray()
plt.show()

class Denoise(Model):

    def __init__(self):
        super(Denoise, self).__init__()
        custom_method(imports='import numpy as np;from tensorflow.keras.models import Model;from sklearn.model_selection import train_test_split;from tensorflow.keras import layers, losses;import tensorflow as tf;import matplotlib.pyplot as plt;from sklearn.metrics import accuracy_score, precision_score, recall_score;from tensorflow.keras.datasets import fashion_mnist;import pandas as pd', function_to_run='tf.keras.Sequential(*args)', method_object=None, object_signature=None, function_args=[eval("[\n      layers.Input(shape=(28, 28, 1)),\n      layers.Conv2D(16, (3, 3), activation='relu', padding='same', strides=2),\n      layers.Conv2D(8, (3, 3), activation='relu', padding='same', strides=2)]")], function_kwargs={})
        self.encoder = tf.keras.Sequential([layers.Input(shape=(28, 28, 1)), layers.Conv2D(16, (3, 3), activation='relu', padding='same', strides=2), layers.Conv2D(8, (3, 3), activation='relu', padding='same', strides=2)])
        custom_method(imports='import numpy as np;from tensorflow.keras.models import Model;from sklearn.model_selection import train_test_split;from tensorflow.keras import layers, losses;import tensorflow as tf;import matplotlib.pyplot as plt;from sklearn.metrics import accuracy_score, precision_score, recall_score;from tensorflow.keras.datasets import fashion_mnist;import pandas as pd', function_to_run='tf.keras.Sequential(*args)', method_object=None, object_signature=None, function_args=[eval("[\n      layers.Conv2DTranspose(8, kernel_size=3, strides=2, activation='relu', padding='same'),\n      layers.Conv2DTranspose(16, kernel_size=3, strides=2, activation='relu', padding='same'),\n      layers.Conv2D(1, kernel_size=(3, 3), activation='sigmoid', padding='same')]")], function_kwargs={})
        self.decoder = tf.keras.Sequential([layers.Conv2DTranspose(8, kernel_size=3, strides=2, activation='relu', padding='same'), layers.Conv2DTranspose(16, kernel_size=3, strides=2, activation='relu', padding='same'), layers.Conv2D(1, kernel_size=(3, 3), activation='sigmoid', padding='same')])

    def call(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded
autoencoder = Denoise()
custom_method(imports='import numpy as np;from tensorflow.keras.models import Model;from sklearn.model_selection import train_test_split;from tensorflow.keras import layers, losses;import tensorflow as tf;import matplotlib.pyplot as plt;from sklearn.metrics import accuracy_score, precision_score, recall_score;from tensorflow.keras.datasets import fashion_mnist;import pandas as pd', function_to_run='obj.compile(**kwargs)', method_object=eval('autoencoder'), object_signature='Denoise()', function_args=[], function_kwargs={'optimizer': eval("'adam'"), 'loss': eval('losses.MeanSquaredError()')}, custom_class="class Denoise(Model):\n  def __init__(self):\n    super(Denoise, self).__init__()\n    self.encoder = tf.keras.Sequential([\n      layers.Input(shape=(28, 28, 1)),\n      layers.Conv2D(16, (3, 3), activation='relu', padding='same', strides=2),\n      layers.Conv2D(8, (3, 3), activation='relu', padding='same', strides=2)])\n\n    self.decoder = tf.keras.Sequential([\n      layers.Conv2DTranspose(8, kernel_size=3, strides=2, activation='relu', padding='same'),\n      layers.Conv2DTranspose(16, kernel_size=3, strides=2, activation='relu', padding='same'),\n      layers.Conv2D(1, kernel_size=(3, 3), activation='sigmoid', padding='same')])\n\n  def call(self, x):\n    encoded = self.encoder(x)\n    decoded = self.decoder(encoded)\n    return decoded")
autoencoder.compile(optimizer='adam', loss=losses.MeanSquaredError())
custom_method(imports='import numpy as np;from tensorflow.keras.models import Model;from sklearn.model_selection import train_test_split;from tensorflow.keras import layers, losses;import tensorflow as tf;import matplotlib.pyplot as plt;from sklearn.metrics import accuracy_score, precision_score, recall_score;from tensorflow.keras.datasets import fashion_mnist;import pandas as pd', function_to_run='obj.fit(*args, **kwargs)', method_object=eval('autoencoder'), object_signature='Denoise()', function_args=[eval('x_train_noisy'), eval('x_train')], function_kwargs={'epochs': eval('10'), 'shuffle': eval('True'), 'validation_data': eval('(x_test_noisy, x_test)')}, custom_class="class Denoise(Model):\n  def __init__(self):\n    super(Denoise, self).__init__()\n    self.encoder = tf.keras.Sequential([\n      layers.Input(shape=(28, 28, 1)),\n      layers.Conv2D(16, (3, 3), activation='relu', padding='same', strides=2),\n      layers.Conv2D(8, (3, 3), activation='relu', padding='same', strides=2)])\n\n    self.decoder = tf.keras.Sequential([\n      layers.Conv2DTranspose(8, kernel_size=3, strides=2, activation='relu', padding='same'),\n      layers.Conv2DTranspose(16, kernel_size=3, strides=2, activation='relu', padding='same'),\n      layers.Conv2D(1, kernel_size=(3, 3), activation='sigmoid', padding='same')])\n\n  def call(self, x):\n    encoded = self.encoder(x)\n    decoded = self.decoder(encoded)\n    return decoded")
autoencoder.fit(x_train_noisy, x_train, epochs=10, shuffle=True, validation_data=(x_test_noisy, x_test))
custom_method(imports='import numpy as np;from tensorflow.keras.models import Model;from sklearn.model_selection import train_test_split;from tensorflow.keras import layers, losses;import tensorflow as tf;import matplotlib.pyplot as plt;from sklearn.metrics import accuracy_score, precision_score, recall_score;from tensorflow.keras.datasets import fashion_mnist;import pandas as pd', function_to_run='obj.encoder.summary()', method_object=eval('autoencoder'), object_signature='Denoise()', function_args=[], function_kwargs={}, custom_class="class Denoise(Model):\n  def __init__(self):\n    super(Denoise, self).__init__()\n    self.encoder = tf.keras.Sequential([\n      layers.Input(shape=(28, 28, 1)),\n      layers.Conv2D(16, (3, 3), activation='relu', padding='same', strides=2),\n      layers.Conv2D(8, (3, 3), activation='relu', padding='same', strides=2)])\n\n    self.decoder = tf.keras.Sequential([\n      layers.Conv2DTranspose(8, kernel_size=3, strides=2, activation='relu', padding='same'),\n      layers.Conv2DTranspose(16, kernel_size=3, strides=2, activation='relu', padding='same'),\n      layers.Conv2D(1, kernel_size=(3, 3), activation='sigmoid', padding='same')])\n\n  def call(self, x):\n    encoded = self.encoder(x)\n    decoded = self.decoder(encoded)\n    return decoded")
autoencoder.encoder.summary()
custom_method(imports='import numpy as np;from tensorflow.keras.models import Model;from sklearn.model_selection import train_test_split;from tensorflow.keras import layers, losses;import tensorflow as tf;import matplotlib.pyplot as plt;from sklearn.metrics import accuracy_score, precision_score, recall_score;from tensorflow.keras.datasets import fashion_mnist;import pandas as pd', function_to_run='obj.decoder.summary()', method_object=eval('autoencoder'), object_signature='Denoise()', function_args=[], function_kwargs={}, custom_class="class Denoise(Model):\n  def __init__(self):\n    super(Denoise, self).__init__()\n    self.encoder = tf.keras.Sequential([\n      layers.Input(shape=(28, 28, 1)),\n      layers.Conv2D(16, (3, 3), activation='relu', padding='same', strides=2),\n      layers.Conv2D(8, (3, 3), activation='relu', padding='same', strides=2)])\n\n    self.decoder = tf.keras.Sequential([\n      layers.Conv2DTranspose(8, kernel_size=3, strides=2, activation='relu', padding='same'),\n      layers.Conv2DTranspose(16, kernel_size=3, strides=2, activation='relu', padding='same'),\n      layers.Conv2D(1, kernel_size=(3, 3), activation='sigmoid', padding='same')])\n\n  def call(self, x):\n    encoded = self.encoder(x)\n    decoded = self.decoder(encoded)\n    return decoded")
autoencoder.decoder.summary()
custom_method(imports='import numpy as np;from tensorflow.keras.models import Model;from sklearn.model_selection import train_test_split;from tensorflow.keras import layers, losses;import tensorflow as tf;import matplotlib.pyplot as plt;from sklearn.metrics import accuracy_score, precision_score, recall_score;from tensorflow.keras.datasets import fashion_mnist;import pandas as pd', function_to_run='obj.encoder(x_test_noisy).numpy()', method_object=eval('autoencoder'), object_signature=None, function_args=[], function_kwargs={}, custom_class="class Denoise(Model):\n  def __init__(self):\n    super(Denoise, self).__init__()\n    self.encoder = tf.keras.Sequential([\n      layers.Input(shape=(28, 28, 1)),\n      layers.Conv2D(16, (3, 3), activation='relu', padding='same', strides=2),\n      layers.Conv2D(8, (3, 3), activation='relu', padding='same', strides=2)])\n\n    self.decoder = tf.keras.Sequential([\n      layers.Conv2DTranspose(8, kernel_size=3, strides=2, activation='relu', padding='same'),\n      layers.Conv2DTranspose(16, kernel_size=3, strides=2, activation='relu', padding='same'),\n      layers.Conv2D(1, kernel_size=(3, 3), activation='sigmoid', padding='same')])\n\n  def call(self, x):\n    encoded = self.encoder(x)\n    decoded = self.decoder(encoded)\n    return decoded")
encoded_imgs = autoencoder.encoder(x_test_noisy).numpy()
custom_method(imports='import numpy as np;from tensorflow.keras.models import Model;from sklearn.model_selection import train_test_split;from tensorflow.keras import layers, losses;import tensorflow as tf;import matplotlib.pyplot as plt;from sklearn.metrics import accuracy_score, precision_score, recall_score;from tensorflow.keras.datasets import fashion_mnist;import pandas as pd', function_to_run='obj.decoder(encoded_imgs).numpy()', method_object=eval('autoencoder'), object_signature=None, function_args=[], function_kwargs={}, custom_class="class Denoise(Model):\n  def __init__(self):\n    super(Denoise, self).__init__()\n    self.encoder = tf.keras.Sequential([\n      layers.Input(shape=(28, 28, 1)),\n      layers.Conv2D(16, (3, 3), activation='relu', padding='same', strides=2),\n      layers.Conv2D(8, (3, 3), activation='relu', padding='same', strides=2)])\n\n    self.decoder = tf.keras.Sequential([\n      layers.Conv2DTranspose(8, kernel_size=3, strides=2, activation='relu', padding='same'),\n      layers.Conv2DTranspose(16, kernel_size=3, strides=2, activation='relu', padding='same'),\n      layers.Conv2D(1, kernel_size=(3, 3), activation='sigmoid', padding='same')])\n\n  def call(self, x):\n    encoded = self.encoder(x)\n    decoded = self.decoder(encoded)\n    return decoded")
decoded_imgs = autoencoder.decoder(encoded_imgs).numpy()
n = 10
plt.figure(figsize=(20, 4))
for i in range(n):
    ax = plt.subplot(2, n, i + 1)
    plt.title('original + noise')
    plt.imshow(tf.squeeze(x_test_noisy[i]))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    bx = plt.subplot(2, n, i + n + 1)
    plt.title('reconstructed')
    plt.imshow(tf.squeeze(decoded_imgs[i]))
    plt.gray()
    bx.get_xaxis().set_visible(False)
    bx.get_yaxis().set_visible(False)
plt.show()
dataframe = pd.read_csv('http://storage.googleapis.com/download.tensorflow.org/data/ecg.csv', header=None)
raw_data = dataframe.values
dataframe.head()
labels = raw_data[:, -1]
data = raw_data[:, 0:-1]
(train_data, test_data, train_labels, test_labels) = train_test_split(data, labels, test_size=0.2, random_state=21)
custom_method(imports='import numpy as np;from tensorflow.keras.models import Model;from sklearn.model_selection import train_test_split;from tensorflow.keras import layers, losses;import tensorflow as tf;import matplotlib.pyplot as plt;from sklearn.metrics import accuracy_score, precision_score, recall_score;from tensorflow.keras.datasets import fashion_mnist;import pandas as pd', function_to_run='tf.reduce_min(*args)', method_object=None, object_signature=None, function_args=[eval('train_data')], function_kwargs={})
min_val = tf.reduce_min(train_data)
custom_method(imports='import numpy as np;from tensorflow.keras.models import Model;from sklearn.model_selection import train_test_split;from tensorflow.keras import layers, losses;import tensorflow as tf;import matplotlib.pyplot as plt;from sklearn.metrics import accuracy_score, precision_score, recall_score;from tensorflow.keras.datasets import fashion_mnist;import pandas as pd', function_to_run='tf.reduce_max(*args)', method_object=None, object_signature=None, function_args=[eval('train_data')], function_kwargs={})
max_val = tf.reduce_max(train_data)
train_data = (train_data - min_val) / (max_val - min_val)
test_data = (test_data - min_val) / (max_val - min_val)
custom_method(imports='import numpy as np;from tensorflow.keras.models import Model;from sklearn.model_selection import train_test_split;from tensorflow.keras import layers, losses;import tensorflow as tf;import matplotlib.pyplot as plt;from sklearn.metrics import accuracy_score, precision_score, recall_score;from tensorflow.keras.datasets import fashion_mnist;import pandas as pd', function_to_run='tf.cast(*args)', method_object=None, object_signature=None, function_args=[eval('train_data'), eval('tf.float32')], function_kwargs={})
train_data = tf.cast(train_data, tf.float32)
custom_method(imports='import numpy as np;from tensorflow.keras.models import Model;from sklearn.model_selection import train_test_split;from tensorflow.keras import layers, losses;import tensorflow as tf;import matplotlib.pyplot as plt;from sklearn.metrics import accuracy_score, precision_score, recall_score;from tensorflow.keras.datasets import fashion_mnist;import pandas as pd', function_to_run='tf.cast(*args)', method_object=None, object_signature=None, function_args=[eval('test_data'), eval('tf.float32')], function_kwargs={})
test_data = tf.cast(test_data, tf.float32)
train_labels = train_labels.astype(bool)
test_labels = test_labels.astype(bool)
normal_train_data = train_data[train_labels]
normal_test_data = test_data[test_labels]
anomalous_train_data = train_data[~train_labels]
anomalous_test_data = test_data[~test_labels]
plt.grid()
plt.plot(np.arange(140), normal_train_data[0])
plt.title('A Normal ECG')
plt.show()
plt.grid()
plt.plot(np.arange(140), anomalous_train_data[0])
plt.title('An Anomalous ECG')
plt.show()

class AnomalyDetector(Model):

    def __init__(self):
        super(AnomalyDetector, self).__init__()
        custom_method(imports='import numpy as np;from tensorflow.keras.models import Model;from sklearn.model_selection import train_test_split;from tensorflow.keras import layers, losses;import tensorflow as tf;import matplotlib.pyplot as plt;from sklearn.metrics import accuracy_score, precision_score, recall_score;from tensorflow.keras.datasets import fashion_mnist;import pandas as pd', function_to_run='tf.keras.Sequential(*args)', method_object=None, object_signature=None, function_args=[eval('[\n      layers.Dense(32, activation="relu"),\n      layers.Dense(16, activation="relu"),\n      layers.Dense(8, activation="relu")]')], function_kwargs={})
        self.encoder = tf.keras.Sequential([layers.Dense(32, activation='relu'), layers.Dense(16, activation='relu'), layers.Dense(8, activation='relu')])
        custom_method(imports='import numpy as np;from tensorflow.keras.models import Model;from sklearn.model_selection import train_test_split;from tensorflow.keras import layers, losses;import tensorflow as tf;import matplotlib.pyplot as plt;from sklearn.metrics import accuracy_score, precision_score, recall_score;from tensorflow.keras.datasets import fashion_mnist;import pandas as pd', function_to_run='tf.keras.Sequential(*args)', method_object=None, object_signature=None, function_args=[eval('[\n      layers.Dense(16, activation="relu"),\n      layers.Dense(32, activation="relu"),\n      layers.Dense(140, activation="sigmoid")]')], function_kwargs={})
        self.decoder = tf.keras.Sequential([layers.Dense(16, activation='relu'), layers.Dense(32, activation='relu'), layers.Dense(140, activation='sigmoid')])

    def call(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded
autoencoder = AnomalyDetector()
custom_method(imports='import numpy as np;from tensorflow.keras.models import Model;from sklearn.model_selection import train_test_split;from tensorflow.keras import layers, losses;import tensorflow as tf;import matplotlib.pyplot as plt;from sklearn.metrics import accuracy_score, precision_score, recall_score;from tensorflow.keras.datasets import fashion_mnist;import pandas as pd', function_to_run='obj.compile(**kwargs)', method_object=eval('autoencoder'), object_signature='AnomalyDetector()', function_args=[], function_kwargs={'optimizer': eval("'adam'"), 'loss': eval("'mae'")}, custom_class='class AnomalyDetector(Model):\n  def __init__(self):\n    super(AnomalyDetector, self).__init__()\n    self.encoder = tf.keras.Sequential([\n      layers.Dense(32, activation="relu"),\n      layers.Dense(16, activation="relu"),\n      layers.Dense(8, activation="relu")])\n    \n    self.decoder = tf.keras.Sequential([\n      layers.Dense(16, activation="relu"),\n      layers.Dense(32, activation="relu"),\n      layers.Dense(140, activation="sigmoid")])\n    \n  def call(self, x):\n    encoded = self.encoder(x)\n    decoded = self.decoder(encoded)\n    return decoded')
autoencoder.compile(optimizer='adam', loss='mae')
custom_method(imports='import numpy as np;from tensorflow.keras.models import Model;from sklearn.model_selection import train_test_split;from tensorflow.keras import layers, losses;import tensorflow as tf;import matplotlib.pyplot as plt;from sklearn.metrics import accuracy_score, precision_score, recall_score;from tensorflow.keras.datasets import fashion_mnist;import pandas as pd', function_to_run='obj.fit(*args, **kwargs)', method_object=eval('autoencoder'), object_signature=None, function_args=[eval('normal_train_data'), eval('normal_train_data')], function_kwargs={'epochs': eval('20'), 'batch_size': eval('512'), 'validation_data': eval('(test_data, test_data)'), 'shuffle': eval('True')}, custom_class='class AnomalyDetector(Model):\n  def __init__(self):\n    super(AnomalyDetector, self).__init__()\n    self.encoder = tf.keras.Sequential([\n      layers.Dense(32, activation="relu"),\n      layers.Dense(16, activation="relu"),\n      layers.Dense(8, activation="relu")])\n    \n    self.decoder = tf.keras.Sequential([\n      layers.Dense(16, activation="relu"),\n      layers.Dense(32, activation="relu"),\n      layers.Dense(140, activation="sigmoid")])\n    \n  def call(self, x):\n    encoded = self.encoder(x)\n    decoded = self.decoder(encoded)\n    return decoded')
history = autoencoder.fit(normal_train_data, normal_train_data, epochs=20, batch_size=512, validation_data=(test_data, test_data), shuffle=True)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.legend()
custom_method(imports='import numpy as np;from tensorflow.keras.models import Model;from sklearn.model_selection import train_test_split;from tensorflow.keras import layers, losses;import tensorflow as tf;import matplotlib.pyplot as plt;from sklearn.metrics import accuracy_score, precision_score, recall_score;from tensorflow.keras.datasets import fashion_mnist;import pandas as pd', function_to_run='obj.encoder(normal_test_data).numpy()', method_object=eval('autoencoder'), object_signature=None, function_args=[], function_kwargs={}, custom_class='class AnomalyDetector(Model):\n  def __init__(self):\n    super(AnomalyDetector, self).__init__()\n    self.encoder = tf.keras.Sequential([\n      layers.Dense(32, activation="relu"),\n      layers.Dense(16, activation="relu"),\n      layers.Dense(8, activation="relu")])\n    \n    self.decoder = tf.keras.Sequential([\n      layers.Dense(16, activation="relu"),\n      layers.Dense(32, activation="relu"),\n      layers.Dense(140, activation="sigmoid")])\n    \n  def call(self, x):\n    encoded = self.encoder(x)\n    decoded = self.decoder(encoded)\n    return decoded')
encoded_data = autoencoder.encoder(normal_test_data).numpy()
custom_method(imports='import numpy as np;from tensorflow.keras.models import Model;from sklearn.model_selection import train_test_split;from tensorflow.keras import layers, losses;import tensorflow as tf;import matplotlib.pyplot as plt;from sklearn.metrics import accuracy_score, precision_score, recall_score;from tensorflow.keras.datasets import fashion_mnist;import pandas as pd', function_to_run='obj.decoder(encoded_data).numpy()', method_object=eval('autoencoder'), object_signature=None, function_args=[], function_kwargs={}, custom_class='class AnomalyDetector(Model):\n  def __init__(self):\n    super(AnomalyDetector, self).__init__()\n    self.encoder = tf.keras.Sequential([\n      layers.Dense(32, activation="relu"),\n      layers.Dense(16, activation="relu"),\n      layers.Dense(8, activation="relu")])\n    \n    self.decoder = tf.keras.Sequential([\n      layers.Dense(16, activation="relu"),\n      layers.Dense(32, activation="relu"),\n      layers.Dense(140, activation="sigmoid")])\n    \n  def call(self, x):\n    encoded = self.encoder(x)\n    decoded = self.decoder(encoded)\n    return decoded')
decoded_data = autoencoder.decoder(encoded_data).numpy()
plt.plot(normal_test_data[0], 'b')
plt.plot(decoded_data[0], 'r')
plt.fill_between(np.arange(140), decoded_data[0], normal_test_data[0], color='lightcoral')
plt.legend(labels=['Input', 'Reconstruction', 'Error'])
plt.show()
custom_method(imports='import numpy as np;from tensorflow.keras.models import Model;from sklearn.model_selection import train_test_split;from tensorflow.keras import layers, losses;import tensorflow as tf;import matplotlib.pyplot as plt;from sklearn.metrics import accuracy_score, precision_score, recall_score;from tensorflow.keras.datasets import fashion_mnist;import pandas as pd', function_to_run='obj.encoder(anomalous_test_data).numpy()', method_object=eval('autoencoder'), object_signature=None, function_args=[], function_kwargs={}, custom_class='class AnomalyDetector(Model):\n  def __init__(self):\n    super(AnomalyDetector, self).__init__()\n    self.encoder = tf.keras.Sequential([\n      layers.Dense(32, activation="relu"),\n      layers.Dense(16, activation="relu"),\n      layers.Dense(8, activation="relu")])\n    \n    self.decoder = tf.keras.Sequential([\n      layers.Dense(16, activation="relu"),\n      layers.Dense(32, activation="relu"),\n      layers.Dense(140, activation="sigmoid")])\n    \n  def call(self, x):\n    encoded = self.encoder(x)\n    decoded = self.decoder(encoded)\n    return decoded')
encoded_data = autoencoder.encoder(anomalous_test_data).numpy()
custom_method(imports='import numpy as np;from tensorflow.keras.models import Model;from sklearn.model_selection import train_test_split;from tensorflow.keras import layers, losses;import tensorflow as tf;import matplotlib.pyplot as plt;from sklearn.metrics import accuracy_score, precision_score, recall_score;from tensorflow.keras.datasets import fashion_mnist;import pandas as pd', function_to_run='obj.decoder(encoded_data).numpy()', method_object=eval('autoencoder'), object_signature=None, function_args=[], function_kwargs={}, custom_class='class AnomalyDetector(Model):\n  def __init__(self):\n    super(AnomalyDetector, self).__init__()\n    self.encoder = tf.keras.Sequential([\n      layers.Dense(32, activation="relu"),\n      layers.Dense(16, activation="relu"),\n      layers.Dense(8, activation="relu")])\n    \n    self.decoder = tf.keras.Sequential([\n      layers.Dense(16, activation="relu"),\n      layers.Dense(32, activation="relu"),\n      layers.Dense(140, activation="sigmoid")])\n    \n  def call(self, x):\n    encoded = self.encoder(x)\n    decoded = self.decoder(encoded)\n    return decoded')
decoded_data = autoencoder.decoder(encoded_data).numpy()
plt.plot(anomalous_test_data[0], 'b')
plt.plot(decoded_data[0], 'r')
plt.fill_between(np.arange(140), decoded_data[0], anomalous_test_data[0], color='lightcoral')
plt.legend(labels=['Input', 'Reconstruction', 'Error'])
plt.show()
custom_method(imports='import numpy as np;from tensorflow.keras.models import Model;from sklearn.model_selection import train_test_split;from tensorflow.keras import layers, losses;import tensorflow as tf;import matplotlib.pyplot as plt;from sklearn.metrics import accuracy_score, precision_score, recall_score;from tensorflow.keras.datasets import fashion_mnist;import pandas as pd', function_to_run='obj.predict(*args)', method_object=eval('autoencoder'), object_signature=None, function_args=[eval('normal_train_data')], function_kwargs={}, custom_class='class AnomalyDetector(Model):\n  def __init__(self):\n    super(AnomalyDetector, self).__init__()\n    self.encoder = tf.keras.Sequential([\n      layers.Dense(32, activation="relu"),\n      layers.Dense(16, activation="relu"),\n      layers.Dense(8, activation="relu")])\n    \n    self.decoder = tf.keras.Sequential([\n      layers.Dense(16, activation="relu"),\n      layers.Dense(32, activation="relu"),\n      layers.Dense(140, activation="sigmoid")])\n    \n  def call(self, x):\n    encoded = self.encoder(x)\n    decoded = self.decoder(encoded)\n    return decoded')
reconstructions = autoencoder.predict(normal_train_data)
custom_method(imports='import numpy as np;from tensorflow.keras.models import Model;from sklearn.model_selection import train_test_split;from tensorflow.keras import layers, losses;import tensorflow as tf;import matplotlib.pyplot as plt;from sklearn.metrics import accuracy_score, precision_score, recall_score;from tensorflow.keras.datasets import fashion_mnist;import pandas as pd', function_to_run='tf.keras.losses.mae(*args)', method_object=None, object_signature=None, function_args=[eval('reconstructions'), eval('normal_train_data')], function_kwargs={})
train_loss = tf.keras.losses.mae(reconstructions, normal_train_data)
plt.hist(train_loss[None, :], bins=50)
plt.xlabel('Train loss')
plt.ylabel('No of examples')
plt.show()
threshold = np.mean(train_loss) + np.std(train_loss)
print('Threshold: ', threshold)
custom_method(imports='import numpy as np;from tensorflow.keras.models import Model;from sklearn.model_selection import train_test_split;from tensorflow.keras import layers, losses;import tensorflow as tf;import matplotlib.pyplot as plt;from sklearn.metrics import accuracy_score, precision_score, recall_score;from tensorflow.keras.datasets import fashion_mnist;import pandas as pd', function_to_run='obj.predict(*args)', method_object=eval('autoencoder'), object_signature=None, function_args=[eval('anomalous_test_data')], function_kwargs={}, custom_class='class AnomalyDetector(Model):\n  def __init__(self):\n    super(AnomalyDetector, self).__init__()\n    self.encoder = tf.keras.Sequential([\n      layers.Dense(32, activation="relu"),\n      layers.Dense(16, activation="relu"),\n      layers.Dense(8, activation="relu")])\n    \n    self.decoder = tf.keras.Sequential([\n      layers.Dense(16, activation="relu"),\n      layers.Dense(32, activation="relu"),\n      layers.Dense(140, activation="sigmoid")])\n    \n  def call(self, x):\n    encoded = self.encoder(x)\n    decoded = self.decoder(encoded)\n    return decoded')
reconstructions = autoencoder.predict(anomalous_test_data)
custom_method(imports='import numpy as np;from tensorflow.keras.models import Model;from sklearn.model_selection import train_test_split;from tensorflow.keras import layers, losses;import tensorflow as tf;import matplotlib.pyplot as plt;from sklearn.metrics import accuracy_score, precision_score, recall_score;from tensorflow.keras.datasets import fashion_mnist;import pandas as pd', function_to_run='tf.keras.losses.mae(*args)', method_object=None, object_signature=None, function_args=[eval('reconstructions'), eval('anomalous_test_data')], function_kwargs={})
test_loss = tf.keras.losses.mae(reconstructions, anomalous_test_data)
plt.hist(test_loss[None, :], bins=50)
plt.xlabel('Test loss')
plt.ylabel('No of examples')
plt.show()

def predict(model, data, threshold):
    reconstructions = model(data)
    custom_method(imports='import numpy as np;from tensorflow.keras.models import Model;from sklearn.model_selection import train_test_split;from tensorflow.keras import layers, losses;import tensorflow as tf;import matplotlib.pyplot as plt;from sklearn.metrics import accuracy_score, precision_score, recall_score;from tensorflow.keras.datasets import fashion_mnist;import pandas as pd', function_to_run='tf.keras.losses.mae(*args)', method_object=None, object_signature=None, function_args=[eval('reconstructions'), eval('data')], function_kwargs={})
    loss = tf.keras.losses.mae(reconstructions, data)
    return tf.math.less(loss, threshold)

def print_stats(predictions, labels):
    print('Accuracy = {}'.format(accuracy_score(labels, predictions)))
    print('Precision = {}'.format(precision_score(labels, predictions)))
    print('Recall = {}'.format(recall_score(labels, predictions)))
preds = predict(autoencoder, test_data, threshold)
print_stats(preds, test_labels)
