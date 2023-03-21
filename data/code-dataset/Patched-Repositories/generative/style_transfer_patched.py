import os
import tensorflow as tf
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

def custom_method(func, imports: str, function_to_run: str, method_object=None, function_args: list=None, function_kwargs: dict=None, max_wait_secs=MAX_WAIT_S, custom_class=None, wait_after_run_secs=WAIT_AFTER_RUN_S):
    result = send_request(imports=imports, function_to_run=function_to_run, function_args=function_args, function_kwargs=function_kwargs, max_wait_secs=max_wait_secs, wait_after_run_secs=wait_after_run_secs, method_object=method_object, custom_class=custom_class, experiment_file_path=EXPERIMENT_FILE_PATH)
    return func
if __name__ == '__main__':
    print(EXPERIMENT_FILE_PATH)
os.environ['TFHUB_MODEL_LOAD_FORMAT'] = 'COMPRESSED'
import IPython.display as display
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['figure.figsize'] = (12, 12)
mpl.rcParams['axes.grid'] = False
import numpy as np
import PIL.Image
import time
import functools

def tensor_to_image(tensor):
    tensor = tensor * 255
    tensor = np.array(tensor, dtype=np.uint8)
    if np.ndim(tensor) > 3:
        assert tensor.shape[0] == 1
        tensor = tensor[0]
    return PIL.Image.fromarray(tensor)
content_path = custom_method(
tf.keras.utils.get_file('YellowLabradorLooking_new.jpg', 'https://storage.googleapis.com/download.tensorflow.org/example_images/YellowLabradorLooking_new.jpg'), imports='import numpy as np;import PIL.Image;import tensorflow_hub as hub;from google.colab import files;import matplotlib.pyplot as plt;import tensorflow as tf;import matplotlib as mpl;import IPython.display as display;import time;import os;import functools', function_to_run='tf.keras.utils.get_file(*args)', method_object=None, function_args=[eval("'YellowLabradorLooking_new.jpg'"), eval("'https://storage.googleapis.com/download.tensorflow.org/example_images/YellowLabradorLooking_new.jpg'")], function_kwargs={}, max_wait_secs=0)
style_path = custom_method(
tf.keras.utils.get_file('kandinsky5.jpg', 'https://storage.googleapis.com/download.tensorflow.org/example_images/Vassily_Kandinsky%2C_1913_-_Composition_7.jpg'), imports='import numpy as np;import PIL.Image;import tensorflow_hub as hub;from google.colab import files;import matplotlib.pyplot as plt;import tensorflow as tf;import matplotlib as mpl;import IPython.display as display;import time;import os;import functools', function_to_run='tf.keras.utils.get_file(*args)', method_object=None, function_args=[eval("'kandinsky5.jpg'"), eval("'https://storage.googleapis.com/download.tensorflow.org/example_images/Vassily_Kandinsky%2C_1913_-_Composition_7.jpg'")], function_kwargs={}, max_wait_secs=0)

def load_img(path_to_img):
    max_dim = 512
    img = custom_method(
    tf.io.read_file(path_to_img), imports='import numpy as np;import PIL.Image;import tensorflow_hub as hub;from google.colab import files;import matplotlib.pyplot as plt;import tensorflow as tf;import matplotlib as mpl;import IPython.display as display;import time;import os;import functools', function_to_run='tf.io.read_file(*args)', method_object=None, function_args=[eval('path_to_img')], function_kwargs={}, max_wait_secs=0)
    img = custom_method(
    tf.image.decode_image(img, channels=3), imports='import numpy as np;import PIL.Image;import tensorflow_hub as hub;from google.colab import files;import matplotlib.pyplot as plt;import tensorflow as tf;import matplotlib as mpl;import IPython.display as display;import time;import os;import functools', function_to_run='tf.image.decode_image(*args, **kwargs)', method_object=None, function_args=[eval('img')], function_kwargs={'channels': eval('3')}, max_wait_secs=0)
    img = custom_method(
    tf.image.convert_image_dtype(img, tf.float32), imports='import numpy as np;import PIL.Image;import tensorflow_hub as hub;from google.colab import files;import matplotlib.pyplot as plt;import tensorflow as tf;import matplotlib as mpl;import IPython.display as display;import time;import os;import functools', function_to_run='tf.image.convert_image_dtype(*args)', method_object=None, function_args=[eval('img'), eval('tf.float32')], function_kwargs={}, max_wait_secs=0)
    shape = custom_method(
    tf.cast(tf.shape(img)[:-1], tf.float32), imports='import numpy as np;import PIL.Image;import tensorflow_hub as hub;from google.colab import files;import matplotlib.pyplot as plt;import tensorflow as tf;import matplotlib as mpl;import IPython.display as display;import time;import os;import functools', function_to_run='tf.cast(*args)', method_object=None, function_args=[eval('tf.shape(img)[:-1]'), eval('tf.float32')], function_kwargs={}, max_wait_secs=0)
    long_dim = custom_method(
    max(shape), imports='import numpy as np;import PIL.Image;import tensorflow_hub as hub;from google.colab import files;import matplotlib.pyplot as plt;import tensorflow as tf;import matplotlib as mpl;import IPython.display as display;import time;import os;import functools', function_to_run='obj(*args)', method_object=eval('max'), function_args=[eval('shape')], function_kwargs={}, max_wait_secs=0, custom_class=None)
    scale = max_dim / long_dim
    new_shape = custom_method(
    tf.cast(shape * scale, tf.int32), imports='import numpy as np;import PIL.Image;import tensorflow_hub as hub;from google.colab import files;import matplotlib.pyplot as plt;import tensorflow as tf;import matplotlib as mpl;import IPython.display as display;import time;import os;import functools', function_to_run='tf.cast(*args)', method_object=None, function_args=[eval('shape * scale'), eval('tf.int32')], function_kwargs={}, max_wait_secs=0)
    img = custom_method(
    tf.image.resize(img, new_shape), imports='import numpy as np;import PIL.Image;import tensorflow_hub as hub;from google.colab import files;import matplotlib.pyplot as plt;import tensorflow as tf;import matplotlib as mpl;import IPython.display as display;import time;import os;import functools', function_to_run='tf.image.resize(*args)', method_object=None, function_args=[eval('img'), eval('new_shape')], function_kwargs={}, max_wait_secs=0)
    img = img[tf.newaxis, :]
    return img

def imshow(image, title=None):
    if len(image.shape) > 3:
        image = custom_method(
        tf.squeeze(image, axis=0), imports='import numpy as np;import PIL.Image;import tensorflow_hub as hub;from google.colab import files;import matplotlib.pyplot as plt;import tensorflow as tf;import matplotlib as mpl;import IPython.display as display;import time;import os;import functools', function_to_run='tf.squeeze(*args, **kwargs)', method_object=None, function_args=[eval('image')], function_kwargs={'axis': eval('0')}, max_wait_secs=0)
    plt.imshow(image)
    if title:
        plt.title(title)
content_image = custom_method(
load_img(content_path), imports='import numpy as np;import PIL.Image;import tensorflow_hub as hub;from google.colab import files;import matplotlib.pyplot as plt;import tensorflow as tf;import matplotlib as mpl;import IPython.display as display;import time;import os;import functools', function_to_run='obj(*args)', method_object=eval('load_img'), function_args=[eval('content_path')], function_kwargs={}, max_wait_secs=0, custom_class=None)
style_image = custom_method(
load_img(style_path), imports='import numpy as np;import PIL.Image;import tensorflow_hub as hub;from google.colab import files;import matplotlib.pyplot as plt;import tensorflow as tf;import matplotlib as mpl;import IPython.display as display;import time;import os;import functools', function_to_run='obj(*args)', method_object=eval('load_img'), function_args=[eval('style_path')], function_kwargs={}, max_wait_secs=0, custom_class=None)
plt.subplot(1, 2, 1)
imshow(content_image, 'Content Image')
plt.subplot(1, 2, 2)
imshow(style_image, 'Style Image')
import tensorflow_hub as hub
hub_model = hub.load('https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/2')
stylized_image = custom_method(
hub_model(tf.constant(content_image), tf.constant(style_image)), imports='import numpy as np;import PIL.Image;import tensorflow_hub as hub;from google.colab import files;import matplotlib.pyplot as plt;import tensorflow as tf;import matplotlib as mpl;import IPython.display as display;import time;import os;import functools', function_to_run='obj(*args)', method_object=eval('hub_model'), function_args=[eval('tf.constant(content_image)'), eval('tf.constant(style_image)')], function_kwargs={}, max_wait_secs=0, custom_class=None)[0]
custom_method(
tensor_to_image(stylized_image), imports='import numpy as np;import PIL.Image;import tensorflow_hub as hub;from google.colab import files;import matplotlib.pyplot as plt;import tensorflow as tf;import matplotlib as mpl;import IPython.display as display;import time;import os;import functools', function_to_run='obj(*args)', method_object=eval('tensor_to_image'), function_args=[eval('stylized_image')], function_kwargs={}, max_wait_secs=0, custom_class=None)
x = custom_method(
tf.keras.applications.vgg19.preprocess_input(content_image * 255), imports='import numpy as np;import PIL.Image;import tensorflow_hub as hub;from google.colab import files;import matplotlib.pyplot as plt;import tensorflow as tf;import matplotlib as mpl;import IPython.display as display;import time;import os;import functools', function_to_run='tf.keras.applications.vgg19.preprocess_input(*args)', method_object=None, function_args=[eval('content_image*255')], function_kwargs={}, max_wait_secs=0)
x = custom_method(
tf.image.resize(x, (224, 224)), imports='import numpy as np;import PIL.Image;import tensorflow_hub as hub;from google.colab import files;import matplotlib.pyplot as plt;import tensorflow as tf;import matplotlib as mpl;import IPython.display as display;import time;import os;import functools', function_to_run='tf.image.resize(*args)', method_object=None, function_args=[eval('x'), eval('(224, 224)')], function_kwargs={}, max_wait_secs=0)
vgg = custom_method(
tf.keras.applications.VGG19(include_top=True, weights='imagenet'), imports='import numpy as np;import PIL.Image;import tensorflow_hub as hub;from google.colab import files;import matplotlib.pyplot as plt;import tensorflow as tf;import matplotlib as mpl;import IPython.display as display;import time;import os;import functools', function_to_run='tf.keras.applications.VGG19(**kwargs)', method_object=None, function_args=[], function_kwargs={'include_top': eval('True'), 'weights': eval("'imagenet'")}, max_wait_secs=0)
prediction_probabilities = custom_method(
vgg(x), imports='import numpy as np;import PIL.Image;import tensorflow_hub as hub;from google.colab import files;import matplotlib.pyplot as plt;import tensorflow as tf;import matplotlib as mpl;import IPython.display as display;import time;import os;import functools', function_to_run='obj(*args)', method_object=eval('vgg'), function_args=[eval('x')], function_kwargs={}, max_wait_secs=0, custom_class=None)
prediction_probabilities.shape
predicted_top_5 = custom_method(
tf.keras.applications.vgg19.decode_predictions(prediction_probabilities.numpy()), imports='import numpy as np;import PIL.Image;import tensorflow_hub as hub;from google.colab import files;import matplotlib.pyplot as plt;import tensorflow as tf;import matplotlib as mpl;import IPython.display as display;import time;import os;import functools', function_to_run='tf.keras.applications.vgg19.decode_predictions(*args)', method_object=None, function_args=[eval('prediction_probabilities.numpy()')], function_kwargs={}, max_wait_secs=0)[0]
[(class_name, prob) for (number, class_name, prob) in predicted_top_5]
vgg = custom_method(
tf.keras.applications.VGG19(include_top=False, weights='imagenet'), imports='import numpy as np;import PIL.Image;import tensorflow_hub as hub;from google.colab import files;import matplotlib.pyplot as plt;import tensorflow as tf;import matplotlib as mpl;import IPython.display as display;import time;import os;import functools', function_to_run='tf.keras.applications.VGG19(**kwargs)', method_object=None, function_args=[], function_kwargs={'include_top': eval('False'), 'weights': eval("'imagenet'")}, max_wait_secs=0)
print()
for layer in vgg.layers:
    print(layer.name)
content_layers = ['block5_conv2']
style_layers = ['block1_conv1', 'block2_conv1', 'block3_conv1', 'block4_conv1', 'block5_conv1']
num_content_layers = len(content_layers)
num_style_layers = len(style_layers)

def vgg_layers(layer_names):
    """ Creates a VGG model that returns a list of intermediate output values."""
    vgg = custom_method(
    tf.keras.applications.VGG19(include_top=False, weights='imagenet'), imports='import numpy as np;import PIL.Image;import tensorflow_hub as hub;from google.colab import files;import matplotlib.pyplot as plt;import tensorflow as tf;import matplotlib as mpl;import IPython.display as display;import time;import os;import functools', function_to_run='tf.keras.applications.VGG19(**kwargs)', method_object=None, function_args=[], function_kwargs={'include_top': eval('False'), 'weights': eval("'imagenet'")}, max_wait_secs=0)
    vgg.trainable = False
    outputs = [custom_method(
    vgg.get_layer(name), imports='import numpy as np;import PIL.Image;import tensorflow_hub as hub;from google.colab import files;import matplotlib.pyplot as plt;import tensorflow as tf;import matplotlib as mpl;import IPython.display as display;import time;import os;import functools', function_to_run='obj.get_layer(*args)', method_object=eval('vgg'), function_args=[eval('name')], function_kwargs={}, max_wait_secs=0, custom_class=None).output for name in layer_names]
    model = custom_method(
    tf.keras.Model([vgg.input], outputs), imports='import numpy as np;import PIL.Image;import tensorflow_hub as hub;from google.colab import files;import matplotlib.pyplot as plt;import tensorflow as tf;import matplotlib as mpl;import IPython.display as display;import time;import os;import functools', function_to_run='tf.keras.Model(*args)', method_object=None, function_args=[eval('[vgg.input]'), eval('outputs')], function_kwargs={}, max_wait_secs=0)
    return model
style_extractor = custom_method(
vgg_layers(style_layers), imports='import numpy as np;import PIL.Image;import tensorflow_hub as hub;from google.colab import files;import matplotlib.pyplot as plt;import tensorflow as tf;import matplotlib as mpl;import IPython.display as display;import time;import os;import functools', function_to_run='obj(*args)', method_object=eval('vgg_layers'), function_args=[eval('style_layers')], function_kwargs={}, max_wait_secs=0, custom_class=None)
style_outputs = custom_method(
style_extractor(style_image * 255), imports='import numpy as np;import PIL.Image;import tensorflow_hub as hub;from google.colab import files;import matplotlib.pyplot as plt;import tensorflow as tf;import matplotlib as mpl;import IPython.display as display;import time;import os;import functools', function_to_run='obj(*args)', method_object=eval('style_extractor'), function_args=[eval('style_image*255')], function_kwargs={}, max_wait_secs=0, custom_class=None)
for (name, output) in zip(style_layers, style_outputs):
    print(name)
    print('  shape: ', output.numpy().shape)
    print('  min: ', output.numpy().min())
    print('  max: ', output.numpy().max())
    print('  mean: ', output.numpy().mean())
    print()

def gram_matrix(input_tensor):
    result = custom_method(
    tf.linalg.einsum('bijc,bijd->bcd', input_tensor, input_tensor), imports='import numpy as np;import PIL.Image;import tensorflow_hub as hub;from google.colab import files;import matplotlib.pyplot as plt;import tensorflow as tf;import matplotlib as mpl;import IPython.display as display;import time;import os;import functools', function_to_run='tf.linalg.einsum(*args)', method_object=None, function_args=[eval("'bijc,bijd->bcd'"), eval('input_tensor'), eval('input_tensor')], function_kwargs={}, max_wait_secs=0)
    input_shape = custom_method(
    tf.shape(input_tensor), imports='import numpy as np;import PIL.Image;import tensorflow_hub as hub;from google.colab import files;import matplotlib.pyplot as plt;import tensorflow as tf;import matplotlib as mpl;import IPython.display as display;import time;import os;import functools', function_to_run='tf.shape(*args)', method_object=None, function_args=[eval('input_tensor')], function_kwargs={}, max_wait_secs=0)
    num_locations = custom_method(
    tf.cast(input_shape[1] * input_shape[2], tf.float32), imports='import numpy as np;import PIL.Image;import tensorflow_hub as hub;from google.colab import files;import matplotlib.pyplot as plt;import tensorflow as tf;import matplotlib as mpl;import IPython.display as display;import time;import os;import functools', function_to_run='tf.cast(*args)', method_object=None, function_args=[eval('input_shape[1]*input_shape[2]'), eval('tf.float32')], function_kwargs={}, max_wait_secs=0)
    return result / num_locations

class StyleContentModel(tf.keras.models.Model):

    def __init__(self, style_layers, content_layers):
        super(StyleContentModel, self).__init__()
        self.vgg = custom_method(
        vgg_layers(style_layers + content_layers), imports='import numpy as np;import PIL.Image;import tensorflow_hub as hub;from google.colab import files;import matplotlib.pyplot as plt;import tensorflow as tf;import matplotlib as mpl;import IPython.display as display;import time;import os;import functools', function_to_run='obj(*args)', method_object=eval('vgg_layers'), function_args=[eval('style_layers + content_layers')], function_kwargs={}, max_wait_secs=0, custom_class=None)
        self.style_layers = style_layers
        self.content_layers = content_layers
        self.num_style_layers = len(style_layers)
        self.vgg.trainable = False

    def call(self, inputs):
        """Expects float input in [0,1]"""
        inputs = inputs * 255.0
        preprocessed_input = custom_method(
        tf.keras.applications.vgg19.preprocess_input(inputs), imports='import numpy as np;import PIL.Image;import tensorflow_hub as hub;from google.colab import files;import matplotlib.pyplot as plt;import tensorflow as tf;import matplotlib as mpl;import IPython.display as display;import time;import os;import functools', function_to_run='tf.keras.applications.vgg19.preprocess_input(*args)', method_object=None, function_args=[eval('inputs')], function_kwargs={}, max_wait_secs=0)
        outputs = self.vgg(preprocessed_input)
        (style_outputs, content_outputs) = (outputs[:self.num_style_layers], outputs[self.num_style_layers:])
        style_outputs = [custom_method(
        gram_matrix(style_output), imports='import numpy as np;import PIL.Image;import tensorflow_hub as hub;from google.colab import files;import matplotlib.pyplot as plt;import tensorflow as tf;import matplotlib as mpl;import IPython.display as display;import time;import os;import functools', function_to_run='obj(*args)', method_object=eval('gram_matrix'), function_args=[eval('style_output')], function_kwargs={}, max_wait_secs=0, custom_class=None) for style_output in style_outputs]
        content_dict = {content_name: value for (content_name, value) in zip(self.content_layers, content_outputs)}
        style_dict = {style_name: value for (style_name, value) in zip(self.style_layers, style_outputs)}
        return {'content': content_dict, 'style': style_dict}
extractor = StyleContentModel(style_layers, content_layers)
results = custom_method(
extractor(tf.constant(content_image)), imports='import numpy as np;import PIL.Image;import tensorflow_hub as hub;from google.colab import files;import matplotlib.pyplot as plt;import tensorflow as tf;import matplotlib as mpl;import IPython.display as display;import time;import os;import functools', function_to_run='obj(*args)', method_object=eval('extractor'), function_args=[eval('tf.constant(content_image)')], function_kwargs={}, max_wait_secs=0, custom_class='class StyleContentModel(tf.keras.models.Model):\n  def __init__(self, style_layers, content_layers):\n    super(StyleContentModel, self).__init__()\n    self.vgg = vgg_layers(style_layers + content_layers)\n    self.style_layers = style_layers\n    self.content_layers = content_layers\n    self.num_style_layers = len(style_layers)\n    self.vgg.trainable = False\n\n  def call(self, inputs):\n    "Expects float input in [0,1]"\n    inputs = inputs*255.0\n    preprocessed_input = tf.keras.applications.vgg19.preprocess_input(inputs)\n    outputs = self.vgg(preprocessed_input)\n    style_outputs, content_outputs = (outputs[:self.num_style_layers],\n                                      outputs[self.num_style_layers:])\n\n    style_outputs = [gram_matrix(style_output)\n                     for style_output in style_outputs]\n\n    content_dict = {content_name: value\n                    for content_name, value\n                    in zip(self.content_layers, content_outputs)}\n\n    style_dict = {style_name: value\n                  for style_name, value\n                  in zip(self.style_layers, style_outputs)}\n\n    return {\'content\': content_dict, \'style\': style_dict}')
print('Styles:')
for (name, output) in sorted(results['style'].items()):
    print('  ', name)
    print('    shape: ', output.numpy().shape)
    print('    min: ', output.numpy().min())
    print('    max: ', output.numpy().max())
    print('    mean: ', output.numpy().mean())
    print()
print('Contents:')
for (name, output) in sorted(results['content'].items()):
    print('  ', name)
    print('    shape: ', output.numpy().shape)
    print('    min: ', output.numpy().min())
    print('    max: ', output.numpy().max())
    print('    mean: ', output.numpy().mean())
style_targets = custom_method(
extractor(style_image), imports='import numpy as np;import PIL.Image;import tensorflow_hub as hub;from google.colab import files;import matplotlib.pyplot as plt;import tensorflow as tf;import matplotlib as mpl;import IPython.display as display;import time;import os;import functools', function_to_run='obj(*args)', method_object=eval('extractor'), function_args=[eval('style_image')], function_kwargs={}, max_wait_secs=0, custom_class='class StyleContentModel(tf.keras.models.Model):\n  def __init__(self, style_layers, content_layers):\n    super(StyleContentModel, self).__init__()\n    self.vgg = vgg_layers(style_layers + content_layers)\n    self.style_layers = style_layers\n    self.content_layers = content_layers\n    self.num_style_layers = len(style_layers)\n    self.vgg.trainable = False\n\n  def call(self, inputs):\n    "Expects float input in [0,1]"\n    inputs = inputs*255.0\n    preprocessed_input = tf.keras.applications.vgg19.preprocess_input(inputs)\n    outputs = self.vgg(preprocessed_input)\n    style_outputs, content_outputs = (outputs[:self.num_style_layers],\n                                      outputs[self.num_style_layers:])\n\n    style_outputs = [gram_matrix(style_output)\n                     for style_output in style_outputs]\n\n    content_dict = {content_name: value\n                    for content_name, value\n                    in zip(self.content_layers, content_outputs)}\n\n    style_dict = {style_name: value\n                  for style_name, value\n                  in zip(self.style_layers, style_outputs)}\n\n    return {\'content\': content_dict, \'style\': style_dict}')['style']
content_targets = custom_method(
extractor(content_image), imports='import numpy as np;import PIL.Image;import tensorflow_hub as hub;from google.colab import files;import matplotlib.pyplot as plt;import tensorflow as tf;import matplotlib as mpl;import IPython.display as display;import time;import os;import functools', function_to_run='obj(*args)', method_object=eval('extractor'), function_args=[eval('content_image')], function_kwargs={}, max_wait_secs=0, custom_class='class StyleContentModel(tf.keras.models.Model):\n  def __init__(self, style_layers, content_layers):\n    super(StyleContentModel, self).__init__()\n    self.vgg = vgg_layers(style_layers + content_layers)\n    self.style_layers = style_layers\n    self.content_layers = content_layers\n    self.num_style_layers = len(style_layers)\n    self.vgg.trainable = False\n\n  def call(self, inputs):\n    "Expects float input in [0,1]"\n    inputs = inputs*255.0\n    preprocessed_input = tf.keras.applications.vgg19.preprocess_input(inputs)\n    outputs = self.vgg(preprocessed_input)\n    style_outputs, content_outputs = (outputs[:self.num_style_layers],\n                                      outputs[self.num_style_layers:])\n\n    style_outputs = [gram_matrix(style_output)\n                     for style_output in style_outputs]\n\n    content_dict = {content_name: value\n                    for content_name, value\n                    in zip(self.content_layers, content_outputs)}\n\n    style_dict = {style_name: value\n                  for style_name, value\n                  in zip(self.style_layers, style_outputs)}\n\n    return {\'content\': content_dict, \'style\': style_dict}')['content']
image = custom_method(
tf.Variable(content_image), imports='import numpy as np;import PIL.Image;import tensorflow_hub as hub;from google.colab import files;import matplotlib.pyplot as plt;import tensorflow as tf;import matplotlib as mpl;import IPython.display as display;import time;import os;import functools', function_to_run='tf.Variable(*args)', method_object=None, function_args=[eval('content_image')], function_kwargs={}, max_wait_secs=0)

def clip_0_1(image):
    return custom_method(
    tf.clip_by_value(image, clip_value_min=0.0, clip_value_max=1.0), imports='import numpy as np;import PIL.Image;import tensorflow_hub as hub;from google.colab import files;import matplotlib.pyplot as plt;import tensorflow as tf;import matplotlib as mpl;import IPython.display as display;import time;import os;import functools', function_to_run='tf.clip_by_value(*args, **kwargs)', method_object=None, function_args=[eval('image')], function_kwargs={'clip_value_min': eval('0.0'), 'clip_value_max': eval('1.0')}, max_wait_secs=0)
opt = custom_method(
tf.keras.optimizers.Adam(learning_rate=0.02, beta_1=0.99, epsilon=0.1), imports='import numpy as np;import PIL.Image;import tensorflow_hub as hub;from google.colab import files;import matplotlib.pyplot as plt;import tensorflow as tf;import matplotlib as mpl;import IPython.display as display;import time;import os;import functools', function_to_run='tf.keras.optimizers.Adam(**kwargs)', method_object=None, function_args=[], function_kwargs={'learning_rate': eval('0.02'), 'beta_1': eval('0.99'), 'epsilon': eval('1e-1')}, max_wait_secs=0)
style_weight = 0.01
content_weight = 10000.0

def style_content_loss(outputs):
    style_outputs = outputs['style']
    content_outputs = outputs['content']
    style_loss = custom_method(
    tf.add_n([tf.reduce_mean((style_outputs[name] - style_targets[name]) ** 2) for name in style_outputs.keys()]), imports='import numpy as np;import PIL.Image;import tensorflow_hub as hub;from google.colab import files;import matplotlib.pyplot as plt;import tensorflow as tf;import matplotlib as mpl;import IPython.display as display;import time;import os;import functools', function_to_run='tf.add_n(*args)', method_object=None, function_args=[eval('[tf.reduce_mean((style_outputs[name]-style_targets[name])**2) \n                           for name in style_outputs.keys()]')], function_kwargs={}, max_wait_secs=0)
    style_loss *= style_weight / num_style_layers
    content_loss = custom_method(
    tf.add_n([tf.reduce_mean((content_outputs[name] - content_targets[name]) ** 2) for name in content_outputs.keys()]), imports='import numpy as np;import PIL.Image;import tensorflow_hub as hub;from google.colab import files;import matplotlib.pyplot as plt;import tensorflow as tf;import matplotlib as mpl;import IPython.display as display;import time;import os;import functools', function_to_run='tf.add_n(*args)', method_object=None, function_args=[eval('[tf.reduce_mean((content_outputs[name]-content_targets[name])**2) \n                             for name in content_outputs.keys()]')], function_kwargs={}, max_wait_secs=0)
    content_loss *= content_weight / num_content_layers
    loss = style_loss + content_loss
    return loss

@custom_method(
tf.function(), imports='import numpy as np;import PIL.Image;import tensorflow_hub as hub;from google.colab import files;import matplotlib.pyplot as plt;import tensorflow as tf;import matplotlib as mpl;import IPython.display as display;import time;import os;import functools', function_to_run='tf.function()', method_object=None, function_args=[], function_kwargs={}, max_wait_secs=0)
def train_step(image):
    with custom_method(
    tf.GradientTape(), imports='import numpy as np;import PIL.Image;import tensorflow_hub as hub;from google.colab import files;import matplotlib.pyplot as plt;import tensorflow as tf;import matplotlib as mpl;import IPython.display as display;import time;import os;import functools', function_to_run='tf.GradientTape()', method_object=None, function_args=[], function_kwargs={}, max_wait_secs=0) as tape:
        outputs = custom_method(
        extractor(image), imports='import numpy as np;import PIL.Image;import tensorflow_hub as hub;from google.colab import files;import matplotlib.pyplot as plt;import tensorflow as tf;import matplotlib as mpl;import IPython.display as display;import time;import os;import functools', function_to_run='obj(*args)', method_object=eval('extractor'), function_args=[eval('image')], function_kwargs={}, max_wait_secs=0, custom_class='class StyleContentModel(tf.keras.models.Model):\n  def __init__(self, style_layers, content_layers):\n    super(StyleContentModel, self).__init__()\n    self.vgg = vgg_layers(style_layers + content_layers)\n    self.style_layers = style_layers\n    self.content_layers = content_layers\n    self.num_style_layers = len(style_layers)\n    self.vgg.trainable = False\n\n  def call(self, inputs):\n    "Expects float input in [0,1]"\n    inputs = inputs*255.0\n    preprocessed_input = tf.keras.applications.vgg19.preprocess_input(inputs)\n    outputs = self.vgg(preprocessed_input)\n    style_outputs, content_outputs = (outputs[:self.num_style_layers],\n                                      outputs[self.num_style_layers:])\n\n    style_outputs = [gram_matrix(style_output)\n                     for style_output in style_outputs]\n\n    content_dict = {content_name: value\n                    for content_name, value\n                    in zip(self.content_layers, content_outputs)}\n\n    style_dict = {style_name: value\n                  for style_name, value\n                  in zip(self.style_layers, style_outputs)}\n\n    return {\'content\': content_dict, \'style\': style_dict}')
        loss = custom_method(
        style_content_loss(outputs), imports='import numpy as np;import PIL.Image;import tensorflow_hub as hub;from google.colab import files;import matplotlib.pyplot as plt;import tensorflow as tf;import matplotlib as mpl;import IPython.display as display;import time;import os;import functools', function_to_run='obj(*args)', method_object=eval('style_content_loss'), function_args=[eval('outputs')], function_kwargs={}, max_wait_secs=0, custom_class=None)
    grad = tape.gradient(loss, image)
    custom_method(
    opt.apply_gradients([(grad, image)]), imports='import numpy as np;import PIL.Image;import tensorflow_hub as hub;from google.colab import files;import matplotlib.pyplot as plt;import tensorflow as tf;import matplotlib as mpl;import IPython.display as display;import time;import os;import functools', function_to_run='obj.apply_gradients(*args)', method_object=eval('opt'), function_args=[eval('[(grad, image)]')], function_kwargs={}, max_wait_secs=0, custom_class=None)
    custom_method(
    image.assign(clip_0_1(image)), imports='import numpy as np;import PIL.Image;import tensorflow_hub as hub;from google.colab import files;import matplotlib.pyplot as plt;import tensorflow as tf;import matplotlib as mpl;import IPython.display as display;import time;import os;import functools', function_to_run='obj.assign(*args)', method_object=eval('image'), function_args=[eval('clip_0_1(image)')], function_kwargs={}, max_wait_secs=0, custom_class=None)
train_step(image)
train_step(image)
train_step(image)
custom_method(
tensor_to_image(image), imports='import numpy as np;import PIL.Image;import tensorflow_hub as hub;from google.colab import files;import matplotlib.pyplot as plt;import tensorflow as tf;import matplotlib as mpl;import IPython.display as display;import time;import os;import functools', function_to_run='obj(*args)', method_object=eval('tensor_to_image'), function_args=[eval('image')], function_kwargs={}, max_wait_secs=0, custom_class=None)
import time
start = time.time()
epochs = 10
steps_per_epoch = 100
step = 0
for n in range(epochs):
    for m in range(steps_per_epoch):
        step += 1
        train_step(image)
        print('.', end='', flush=True)
    display.clear_output(wait=True)
    display.display(tensor_to_image(image))
    print('Train step: {}'.format(step))
end = time.time()
print('Total time: {:.1f}'.format(end - start))

def high_pass_x_y(image):
    x_var = image[:, :, 1:, :] - image[:, :, :-1, :]
    y_var = image[:, 1:, :, :] - image[:, :-1, :, :]
    return (x_var, y_var)
(x_deltas, y_deltas) = custom_method(
high_pass_x_y(content_image), imports='import numpy as np;import PIL.Image;import tensorflow_hub as hub;from google.colab import files;import matplotlib.pyplot as plt;import tensorflow as tf;import matplotlib as mpl;import IPython.display as display;import time;import os;import functools', function_to_run='obj(*args)', method_object=eval('high_pass_x_y'), function_args=[eval('content_image')], function_kwargs={}, max_wait_secs=0, custom_class=None)
plt.figure(figsize=(14, 10))
plt.subplot(2, 2, 1)
imshow(clip_0_1(2 * y_deltas + 0.5), 'Horizontal Deltas: Original')
plt.subplot(2, 2, 2)
imshow(clip_0_1(2 * x_deltas + 0.5), 'Vertical Deltas: Original')
(x_deltas, y_deltas) = custom_method(
high_pass_x_y(image), imports='import numpy as np;import PIL.Image;import tensorflow_hub as hub;from google.colab import files;import matplotlib.pyplot as plt;import tensorflow as tf;import matplotlib as mpl;import IPython.display as display;import time;import os;import functools', function_to_run='obj(*args)', method_object=eval('high_pass_x_y'), function_args=[eval('image')], function_kwargs={}, max_wait_secs=0, custom_class=None)
plt.subplot(2, 2, 3)
imshow(clip_0_1(2 * y_deltas + 0.5), 'Horizontal Deltas: Styled')
plt.subplot(2, 2, 4)
imshow(clip_0_1(2 * x_deltas + 0.5), 'Vertical Deltas: Styled')
plt.figure(figsize=(14, 10))
sobel = custom_method(
tf.image.sobel_edges(content_image), imports='import numpy as np;import PIL.Image;import tensorflow_hub as hub;from google.colab import files;import matplotlib.pyplot as plt;import tensorflow as tf;import matplotlib as mpl;import IPython.display as display;import time;import os;import functools', function_to_run='tf.image.sobel_edges(*args)', method_object=None, function_args=[eval('content_image')], function_kwargs={}, max_wait_secs=0)
plt.subplot(1, 2, 1)
imshow(clip_0_1(sobel[..., 0] / 4 + 0.5), 'Horizontal Sobel-edges')
plt.subplot(1, 2, 2)
imshow(clip_0_1(sobel[..., 1] / 4 + 0.5), 'Vertical Sobel-edges')

def total_variation_loss(image):
    (x_deltas, y_deltas) = custom_method(
    high_pass_x_y(image), imports='import numpy as np;import PIL.Image;import tensorflow_hub as hub;from google.colab import files;import matplotlib.pyplot as plt;import tensorflow as tf;import matplotlib as mpl;import IPython.display as display;import time;import os;import functools', function_to_run='obj(*args)', method_object=eval('high_pass_x_y'), function_args=[eval('image')], function_kwargs={}, max_wait_secs=0, custom_class=None)
    return custom_method(
    tf.reduce_sum(tf.abs(x_deltas)), imports='import numpy as np;import PIL.Image;import tensorflow_hub as hub;from google.colab import files;import matplotlib.pyplot as plt;import tensorflow as tf;import matplotlib as mpl;import IPython.display as display;import time;import os;import functools', function_to_run='tf.reduce_sum(*args)', method_object=None, function_args=[eval('tf.abs(x_deltas)')], function_kwargs={}, max_wait_secs=0) + custom_method(
    tf.reduce_sum(tf.abs(y_deltas)), imports='import numpy as np;import PIL.Image;import tensorflow_hub as hub;from google.colab import files;import matplotlib.pyplot as plt;import tensorflow as tf;import matplotlib as mpl;import IPython.display as display;import time;import os;import functools', function_to_run='tf.reduce_sum(*args)', method_object=None, function_args=[eval('tf.abs(y_deltas)')], function_kwargs={}, max_wait_secs=0)
custom_method(
total_variation_loss(image).numpy(), imports='import numpy as np;import PIL.Image;import tensorflow_hub as hub;from google.colab import files;import matplotlib.pyplot as plt;import tensorflow as tf;import matplotlib as mpl;import IPython.display as display;import time;import os;import functools', function_to_run='total_variation_loss(obj).numpy()', method_object=eval('image'), function_args=[], function_kwargs={}, max_wait_secs=0, custom_class=None)
custom_method(
tf.image.total_variation(image).numpy(), imports='import numpy as np;import PIL.Image;import tensorflow_hub as hub;from google.colab import files;import matplotlib.pyplot as plt;import tensorflow as tf;import matplotlib as mpl;import IPython.display as display;import time;import os;import functools', function_to_run='tf.image.total_variation(image).numpy()', method_object=None, function_args=[], function_kwargs={}, max_wait_secs=0)
total_variation_weight = 30

@custom_method(
tf.function(), imports='import numpy as np;import PIL.Image;import tensorflow_hub as hub;from google.colab import files;import matplotlib.pyplot as plt;import tensorflow as tf;import matplotlib as mpl;import IPython.display as display;import time;import os;import functools', function_to_run='tf.function()', method_object=None, function_args=[], function_kwargs={}, max_wait_secs=0)
def train_step(image):
    with custom_method(
    tf.GradientTape(), imports='import numpy as np;import PIL.Image;import tensorflow_hub as hub;from google.colab import files;import matplotlib.pyplot as plt;import tensorflow as tf;import matplotlib as mpl;import IPython.display as display;import time;import os;import functools', function_to_run='tf.GradientTape()', method_object=None, function_args=[], function_kwargs={}, max_wait_secs=0) as tape:
        outputs = custom_method(
        extractor(image), imports='import numpy as np;import PIL.Image;import tensorflow_hub as hub;from google.colab import files;import matplotlib.pyplot as plt;import tensorflow as tf;import matplotlib as mpl;import IPython.display as display;import time;import os;import functools', function_to_run='obj(*args)', method_object=eval('extractor'), function_args=[eval('image')], function_kwargs={}, max_wait_secs=0, custom_class='class StyleContentModel(tf.keras.models.Model):\n  def __init__(self, style_layers, content_layers):\n    super(StyleContentModel, self).__init__()\n    self.vgg = vgg_layers(style_layers + content_layers)\n    self.style_layers = style_layers\n    self.content_layers = content_layers\n    self.num_style_layers = len(style_layers)\n    self.vgg.trainable = False\n\n  def call(self, inputs):\n    "Expects float input in [0,1]"\n    inputs = inputs*255.0\n    preprocessed_input = tf.keras.applications.vgg19.preprocess_input(inputs)\n    outputs = self.vgg(preprocessed_input)\n    style_outputs, content_outputs = (outputs[:self.num_style_layers],\n                                      outputs[self.num_style_layers:])\n\n    style_outputs = [gram_matrix(style_output)\n                     for style_output in style_outputs]\n\n    content_dict = {content_name: value\n                    for content_name, value\n                    in zip(self.content_layers, content_outputs)}\n\n    style_dict = {style_name: value\n                  for style_name, value\n                  in zip(self.style_layers, style_outputs)}\n\n    return {\'content\': content_dict, \'style\': style_dict}')
        loss = custom_method(
        style_content_loss(outputs), imports='import numpy as np;import PIL.Image;import tensorflow_hub as hub;from google.colab import files;import matplotlib.pyplot as plt;import tensorflow as tf;import matplotlib as mpl;import IPython.display as display;import time;import os;import functools', function_to_run='obj(*args)', method_object=eval('style_content_loss'), function_args=[eval('outputs')], function_kwargs={}, max_wait_secs=0, custom_class=None)
        loss += total_variation_weight * custom_method(
        tf.image.total_variation(image), imports='import numpy as np;import PIL.Image;import tensorflow_hub as hub;from google.colab import files;import matplotlib.pyplot as plt;import tensorflow as tf;import matplotlib as mpl;import IPython.display as display;import time;import os;import functools', function_to_run='tf.image.total_variation(*args)', method_object=None, function_args=[eval('image')], function_kwargs={}, max_wait_secs=0)
    grad = tape.gradient(loss, image)
    custom_method(
    opt.apply_gradients([(grad, image)]), imports='import numpy as np;import PIL.Image;import tensorflow_hub as hub;from google.colab import files;import matplotlib.pyplot as plt;import tensorflow as tf;import matplotlib as mpl;import IPython.display as display;import time;import os;import functools', function_to_run='obj.apply_gradients(*args)', method_object=eval('opt'), function_args=[eval('[(grad, image)]')], function_kwargs={}, max_wait_secs=0, custom_class=None)
    custom_method(
    image.assign(clip_0_1(image)), imports='import numpy as np;import PIL.Image;import tensorflow_hub as hub;from google.colab import files;import matplotlib.pyplot as plt;import tensorflow as tf;import matplotlib as mpl;import IPython.display as display;import time;import os;import functools', function_to_run='obj.assign(*args)', method_object=eval('image'), function_args=[eval('clip_0_1(image)')], function_kwargs={}, max_wait_secs=0, custom_class=None)
opt = custom_method(
tf.keras.optimizers.Adam(learning_rate=0.02, beta_1=0.99, epsilon=0.1), imports='import numpy as np;import PIL.Image;import tensorflow_hub as hub;from google.colab import files;import matplotlib.pyplot as plt;import tensorflow as tf;import matplotlib as mpl;import IPython.display as display;import time;import os;import functools', function_to_run='tf.keras.optimizers.Adam(**kwargs)', method_object=None, function_args=[], function_kwargs={'learning_rate': eval('0.02'), 'beta_1': eval('0.99'), 'epsilon': eval('1e-1')}, max_wait_secs=0)
image = custom_method(
tf.Variable(content_image), imports='import numpy as np;import PIL.Image;import tensorflow_hub as hub;from google.colab import files;import matplotlib.pyplot as plt;import tensorflow as tf;import matplotlib as mpl;import IPython.display as display;import time;import os;import functools', function_to_run='tf.Variable(*args)', method_object=None, function_args=[eval('content_image')], function_kwargs={}, max_wait_secs=0)
import time
start = time.time()
epochs = 10
steps_per_epoch = 100
step = 0
for n in range(epochs):
    for m in range(steps_per_epoch):
        step += 1
        train_step(image)
        print('.', end='', flush=True)
    display.clear_output(wait=True)
    display.display(tensor_to_image(image))
    print('Train step: {}'.format(step))
end = time.time()
print('Total time: {:.1f}'.format(end - start))
file_name = 'stylized-image.png'
custom_method(
tensor_to_image(image).save(file_name), imports='import numpy as np;import PIL.Image;import tensorflow_hub as hub;from google.colab import files;import matplotlib.pyplot as plt;import tensorflow as tf;import matplotlib as mpl;import IPython.display as display;import time;import os;import functools', function_to_run='tensor_to_obj(image).save(*args)', method_object=eval('image'), function_args=[eval('file_name')], function_kwargs={}, max_wait_secs=0, custom_class=None)
try:
    from google.colab import files
except ImportError:
    pass
else:
    files.download(file_name)
