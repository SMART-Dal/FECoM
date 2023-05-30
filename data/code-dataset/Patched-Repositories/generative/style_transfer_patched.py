import os
import tensorflow as tf
import os
from pathlib import Path
import dill as pickle
import sys
import numpy as np
from tool.client.client_config import EXPERIMENT_DIR, MAX_WAIT_S, WAIT_AFTER_RUN_S
from tool.server.send_request import send_request
from tool.server.function_details import FunctionDetails
import json
current_path = os.path.abspath(__file__)
experiment_number = sys.argv[1]
experiment_project = sys.argv[2]
EXPERIMENT_FILE_PATH = EXPERIMENT_DIR / 'method-level' / experiment_project / f'experiment-{experiment_number}.json'
skip_calls_file_path = EXPERIMENT_FILE_PATH.parent / 'skip_calls.json'
if skip_calls_file_path.exists():
    with open(skip_calls_file_path, 'r') as f:
        skip_calls = json.load(f)
else:
    skip_calls = []
    with open(skip_calls_file_path, 'w') as f:
        json.dump(skip_calls, f)

def custom_method(imports: str, function_to_run: str, method_object=None, object_signature=None, function_args: list=None, function_kwargs: dict=None, custom_class=None):
    if skip_calls is not None and any((call['function_to_run'] == function_to_run and np.array_equal(call['function_args'], function_args) and (call['function_kwargs'] == function_kwargs) for call in skip_calls)):
        print('skipping call: ', function_to_run)
        return
    result = send_request(imports=imports, function_to_run=function_to_run, function_args=function_args, function_kwargs=function_kwargs, max_wait_secs=MAX_WAIT_S, wait_after_run_secs=WAIT_AFTER_RUN_S, method_object=method_object, object_signature=object_signature, custom_class=custom_class, experiment_file_path=EXPERIMENT_FILE_PATH)
    if result is not None and isinstance(result, dict) and (len(result) == 1):
        energy_data = next(iter(result.values()))
        if skip_calls is not None and 'start_time_perf' in energy_data['times'] and ('end_time_perf' in energy_data['times']) and ('start_time_nvidia' in energy_data['times']) and ('end_time_nvidia' in energy_data['times']) and (energy_data['times']['start_time_perf'] == energy_data['times']['end_time_perf']) and (energy_data['times']['start_time_nvidia'] == energy_data['times']['end_time_nvidia']):
            call_to_skip = {'function_to_run': function_to_run, 'function_args': function_args, 'function_kwargs': function_kwargs}
            try:
                json.dumps(call_to_skip)
                if call_to_skip not in skip_calls:
                    skip_calls.append(call_to_skip)
                    with open(skip_calls_file_path, 'w') as f:
                        json.dump(skip_calls, f)
                    print('skipping call added, current list is: ', skip_calls)
                else:
                    print('Skipping call already exists.')
            except TypeError:
                print('Ignore: Skipping call is not JSON serializable, skipping append and dump.')
    else:
        print('Invalid dictionary object or does not have one key-value pair.')
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
custom_method(imports='from google.colab import files;import numpy as np;import matplotlib.pyplot as plt;import matplotlib as mpl;import tensorflow_hub as hub;import IPython.display as display;import tensorflow as tf;import functools;import os;import time;import PIL.Image', function_to_run='tf.keras.utils.get_file(*args)', method_object=None, object_signature=None, function_args=[eval("'YellowLabradorLooking_new.jpg'"), eval("'https://storage.googleapis.com/download.tensorflow.org/example_images/YellowLabradorLooking_new.jpg'")], function_kwargs={})
content_path = tf.keras.utils.get_file('YellowLabradorLooking_new.jpg', 'https://storage.googleapis.com/download.tensorflow.org/example_images/YellowLabradorLooking_new.jpg')
custom_method(imports='from google.colab import files;import numpy as np;import matplotlib.pyplot as plt;import matplotlib as mpl;import tensorflow_hub as hub;import IPython.display as display;import tensorflow as tf;import functools;import os;import time;import PIL.Image', function_to_run='tf.keras.utils.get_file(*args)', method_object=None, object_signature=None, function_args=[eval("'kandinsky5.jpg'"), eval("'https://storage.googleapis.com/download.tensorflow.org/example_images/Vassily_Kandinsky%2C_1913_-_Composition_7.jpg'")], function_kwargs={})
style_path = tf.keras.utils.get_file('kandinsky5.jpg', 'https://storage.googleapis.com/download.tensorflow.org/example_images/Vassily_Kandinsky%2C_1913_-_Composition_7.jpg')

def load_img(path_to_img):
    max_dim = 512
    custom_method(imports='from google.colab import files;import numpy as np;import matplotlib.pyplot as plt;import matplotlib as mpl;import tensorflow_hub as hub;import IPython.display as display;import tensorflow as tf;import functools;import os;import time;import PIL.Image', function_to_run='tf.io.read_file(*args)', method_object=None, object_signature=None, function_args=[eval('path_to_img')], function_kwargs={})
    img = tf.io.read_file(path_to_img)
    custom_method(imports='from google.colab import files;import numpy as np;import matplotlib.pyplot as plt;import matplotlib as mpl;import tensorflow_hub as hub;import IPython.display as display;import tensorflow as tf;import functools;import os;import time;import PIL.Image', function_to_run='tf.image.decode_image(*args, **kwargs)', method_object=None, object_signature=None, function_args=[eval('img')], function_kwargs={'channels': eval('3')})
    img = tf.image.decode_image(img, channels=3)
    custom_method(imports='from google.colab import files;import numpy as np;import matplotlib.pyplot as plt;import matplotlib as mpl;import tensorflow_hub as hub;import IPython.display as display;import tensorflow as tf;import functools;import os;import time;import PIL.Image', function_to_run='tf.image.convert_image_dtype(*args)', method_object=None, object_signature=None, function_args=[eval('img'), eval('tf.float32')], function_kwargs={})
    img = tf.image.convert_image_dtype(img, tf.float32)
    custom_method(imports='from google.colab import files;import numpy as np;import matplotlib.pyplot as plt;import matplotlib as mpl;import tensorflow_hub as hub;import IPython.display as display;import tensorflow as tf;import functools;import os;import time;import PIL.Image', function_to_run='tf.cast(*args)', method_object=None, object_signature=None, function_args=[eval('tf.shape(img)[:-1]'), eval('tf.float32')], function_kwargs={})
    shape = tf.cast(tf.shape(img)[:-1], tf.float32)
    long_dim = max(shape)
    scale = max_dim / long_dim
    custom_method(imports='from google.colab import files;import numpy as np;import matplotlib.pyplot as plt;import matplotlib as mpl;import tensorflow_hub as hub;import IPython.display as display;import tensorflow as tf;import functools;import os;import time;import PIL.Image', function_to_run='tf.cast(*args)', method_object=None, object_signature=None, function_args=[eval('shape * scale'), eval('tf.int32')], function_kwargs={})
    new_shape = tf.cast(shape * scale, tf.int32)
    custom_method(imports='from google.colab import files;import numpy as np;import matplotlib.pyplot as plt;import matplotlib as mpl;import tensorflow_hub as hub;import IPython.display as display;import tensorflow as tf;import functools;import os;import time;import PIL.Image', function_to_run='tf.image.resize(*args)', method_object=None, object_signature=None, function_args=[eval('img'), eval('new_shape')], function_kwargs={})
    img = tf.image.resize(img, new_shape)
    img = img[tf.newaxis, :]
    return img

def imshow(image, title=None):
    if len(image.shape) > 3:
        custom_method(imports='from google.colab import files;import numpy as np;import matplotlib.pyplot as plt;import matplotlib as mpl;import tensorflow_hub as hub;import IPython.display as display;import tensorflow as tf;import functools;import os;import time;import PIL.Image', function_to_run='tf.squeeze(*args, **kwargs)', method_object=None, object_signature=None, function_args=[eval('image')], function_kwargs={'axis': eval('0')})
        image = tf.squeeze(image, axis=0)
    plt.imshow(image)
    if title:
        plt.title(title)
content_image = load_img(content_path)
style_image = load_img(style_path)
plt.subplot(1, 2, 1)
imshow(content_image, 'Content Image')
plt.subplot(1, 2, 2)
imshow(style_image, 'Style Image')
import tensorflow_hub as hub
hub_model = hub.load('https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/2')
stylized_image = hub_model(tf.constant(content_image), tf.constant(style_image))[0]
tensor_to_image(stylized_image)
custom_method(imports='from google.colab import files;import numpy as np;import matplotlib.pyplot as plt;import matplotlib as mpl;import tensorflow_hub as hub;import IPython.display as display;import tensorflow as tf;import functools;import os;import time;import PIL.Image', function_to_run='tf.keras.applications.vgg19.preprocess_input(*args)', method_object=None, object_signature=None, function_args=[eval('content_image*255')], function_kwargs={})
x = tf.keras.applications.vgg19.preprocess_input(content_image * 255)
custom_method(imports='from google.colab import files;import numpy as np;import matplotlib.pyplot as plt;import matplotlib as mpl;import tensorflow_hub as hub;import IPython.display as display;import tensorflow as tf;import functools;import os;import time;import PIL.Image', function_to_run='tf.image.resize(*args)', method_object=None, object_signature=None, function_args=[eval('x'), eval('(224, 224)')], function_kwargs={})
x = tf.image.resize(x, (224, 224))
custom_method(imports='from google.colab import files;import numpy as np;import matplotlib.pyplot as plt;import matplotlib as mpl;import tensorflow_hub as hub;import IPython.display as display;import tensorflow as tf;import functools;import os;import time;import PIL.Image', function_to_run='tf.keras.applications.VGG19(**kwargs)', method_object=None, object_signature=None, function_args=[], function_kwargs={'include_top': eval('True'), 'weights': eval("'imagenet'")})
vgg = tf.keras.applications.VGG19(include_top=True, weights='imagenet')
custom_method(imports='from google.colab import files;import numpy as np;import matplotlib.pyplot as plt;import matplotlib as mpl;import tensorflow_hub as hub;import IPython.display as display;import tensorflow as tf;import functools;import os;import time;import PIL.Image', function_to_run='obj(*args)', method_object=eval('vgg'), object_signature=None, function_args=[eval('x')], function_kwargs={}, custom_class=None)
prediction_probabilities = vgg(x)
prediction_probabilities.shape
predicted_top_5 = tf.keras.applications.vgg19.decode_predictions(prediction_probabilities.numpy())[0]
[(class_name, prob) for (number, class_name, prob) in predicted_top_5]
custom_method(imports='from google.colab import files;import numpy as np;import matplotlib.pyplot as plt;import matplotlib as mpl;import tensorflow_hub as hub;import IPython.display as display;import tensorflow as tf;import functools;import os;import time;import PIL.Image', function_to_run='tf.keras.applications.VGG19(**kwargs)', method_object=None, object_signature=None, function_args=[], function_kwargs={'include_top': eval('False'), 'weights': eval("'imagenet'")})
vgg = tf.keras.applications.VGG19(include_top=False, weights='imagenet')
print()
for layer in vgg.layers:
    print(layer.name)
content_layers = ['block5_conv2']
style_layers = ['block1_conv1', 'block2_conv1', 'block3_conv1', 'block4_conv1', 'block5_conv1']
num_content_layers = len(content_layers)
num_style_layers = len(style_layers)

def vgg_layers(layer_names):
    """ Creates a VGG model that returns a list of intermediate output values."""
    custom_method(imports='from google.colab import files;import numpy as np;import matplotlib.pyplot as plt;import matplotlib as mpl;import tensorflow_hub as hub;import IPython.display as display;import tensorflow as tf;import functools;import os;import time;import PIL.Image', function_to_run='tf.keras.applications.VGG19(**kwargs)', method_object=None, object_signature=None, function_args=[], function_kwargs={'include_top': eval('False'), 'weights': eval("'imagenet'")})
    vgg = tf.keras.applications.VGG19(include_top=False, weights='imagenet')
    vgg.trainable = False
    outputs = [vgg.get_layer(name).output for name in layer_names]
    custom_method(imports='from google.colab import files;import numpy as np;import matplotlib.pyplot as plt;import matplotlib as mpl;import tensorflow_hub as hub;import IPython.display as display;import tensorflow as tf;import functools;import os;import time;import PIL.Image', function_to_run='tf.keras.Model(*args)', method_object=None, object_signature=None, function_args=[eval('[vgg.input]'), eval('outputs')], function_kwargs={})
    model = tf.keras.Model([vgg.input], outputs)
    return model
style_extractor = vgg_layers(style_layers)
style_outputs = style_extractor(style_image * 255)
for (name, output) in zip(style_layers, style_outputs):
    print(name)
    print('  shape: ', output.numpy().shape)
    print('  min: ', output.numpy().min())
    print('  max: ', output.numpy().max())
    print('  mean: ', output.numpy().mean())
    print()

def gram_matrix(input_tensor):
    custom_method(imports='from google.colab import files;import numpy as np;import matplotlib.pyplot as plt;import matplotlib as mpl;import tensorflow_hub as hub;import IPython.display as display;import tensorflow as tf;import functools;import os;import time;import PIL.Image', function_to_run='tf.linalg.einsum(*args)', method_object=None, object_signature=None, function_args=[eval("'bijc,bijd->bcd'"), eval('input_tensor'), eval('input_tensor')], function_kwargs={})
    result = tf.linalg.einsum('bijc,bijd->bcd', input_tensor, input_tensor)
    custom_method(imports='from google.colab import files;import numpy as np;import matplotlib.pyplot as plt;import matplotlib as mpl;import tensorflow_hub as hub;import IPython.display as display;import tensorflow as tf;import functools;import os;import time;import PIL.Image', function_to_run='tf.shape(*args)', method_object=None, object_signature=None, function_args=[eval('input_tensor')], function_kwargs={})
    input_shape = tf.shape(input_tensor)
    custom_method(imports='from google.colab import files;import numpy as np;import matplotlib.pyplot as plt;import matplotlib as mpl;import tensorflow_hub as hub;import IPython.display as display;import tensorflow as tf;import functools;import os;import time;import PIL.Image', function_to_run='tf.cast(*args)', method_object=None, object_signature=None, function_args=[eval('input_shape[1]*input_shape[2]'), eval('tf.float32')], function_kwargs={})
    num_locations = tf.cast(input_shape[1] * input_shape[2], tf.float32)
    return result / num_locations

class StyleContentModel(tf.keras.models.Model):

    def __init__(self, style_layers, content_layers):
        super(StyleContentModel, self).__init__()
        self.vgg = vgg_layers(style_layers + content_layers)
        self.style_layers = style_layers
        self.content_layers = content_layers
        self.num_style_layers = len(style_layers)
        self.vgg.trainable = False

    def call(self, inputs):
        """Expects float input in [0,1]"""
        inputs = inputs * 255.0
        custom_method(imports='from google.colab import files;import numpy as np;import matplotlib.pyplot as plt;import matplotlib as mpl;import tensorflow_hub as hub;import IPython.display as display;import tensorflow as tf;import functools;import os;import time;import PIL.Image', function_to_run='tf.keras.applications.vgg19.preprocess_input(*args)', method_object=None, object_signature=None, function_args=[eval('inputs')], function_kwargs={})
        preprocessed_input = tf.keras.applications.vgg19.preprocess_input(inputs)
        outputs = self.vgg(preprocessed_input)
        (style_outputs, content_outputs) = (outputs[:self.num_style_layers], outputs[self.num_style_layers:])
        style_outputs = [gram_matrix(style_output) for style_output in style_outputs]
        content_dict = {content_name: value for (content_name, value) in zip(self.content_layers, content_outputs)}
        style_dict = {style_name: value for (style_name, value) in zip(self.style_layers, style_outputs)}
        return {'content': content_dict, 'style': style_dict}
extractor = StyleContentModel(style_layers, content_layers)
custom_method(imports='from google.colab import files;import numpy as np;import matplotlib.pyplot as plt;import matplotlib as mpl;import tensorflow_hub as hub;import IPython.display as display;import tensorflow as tf;import functools;import os;import time;import PIL.Image', function_to_run='obj(*args)', method_object=eval('extractor'), object_signature=None, function_args=[eval('tf.constant(content_image)')], function_kwargs={}, custom_class='class StyleContentModel(tf.keras.models.Model):\n  def __init__(self, style_layers, content_layers):\n    super(StyleContentModel, self).__init__()\n    self.vgg = vgg_layers(style_layers + content_layers)\n    self.style_layers = style_layers\n    self.content_layers = content_layers\n    self.num_style_layers = len(style_layers)\n    self.vgg.trainable = False\n\n  def call(self, inputs):\n    "Expects float input in [0,1]"\n    inputs = inputs*255.0\n    preprocessed_input = tf.keras.applications.vgg19.preprocess_input(inputs)\n    outputs = self.vgg(preprocessed_input)\n    style_outputs, content_outputs = (outputs[:self.num_style_layers],\n                                      outputs[self.num_style_layers:])\n\n    style_outputs = [gram_matrix(style_output)\n                     for style_output in style_outputs]\n\n    content_dict = {content_name: value\n                    for content_name, value\n                    in zip(self.content_layers, content_outputs)}\n\n    style_dict = {style_name: value\n                  for style_name, value\n                  in zip(self.style_layers, style_outputs)}\n\n    return {\'content\': content_dict, \'style\': style_dict}')
results = extractor(tf.constant(content_image))
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
style_targets = extractor(style_image)['style']
content_targets = extractor(content_image)['content']
custom_method(imports='from google.colab import files;import numpy as np;import matplotlib.pyplot as plt;import matplotlib as mpl;import tensorflow_hub as hub;import IPython.display as display;import tensorflow as tf;import functools;import os;import time;import PIL.Image', function_to_run='tf.Variable(*args)', method_object=None, object_signature=None, function_args=[eval('content_image')], function_kwargs={})
image = tf.Variable(content_image)

def clip_0_1(image):
    return tf.clip_by_value(image, clip_value_min=0.0, clip_value_max=1.0)
custom_method(imports='from google.colab import files;import numpy as np;import matplotlib.pyplot as plt;import matplotlib as mpl;import tensorflow_hub as hub;import IPython.display as display;import tensorflow as tf;import functools;import os;import time;import PIL.Image', function_to_run='tf.keras.optimizers.Adam(**kwargs)', method_object=None, object_signature=None, function_args=[], function_kwargs={'learning_rate': eval('0.02'), 'beta_1': eval('0.99'), 'epsilon': eval('1e-1')})
opt = tf.keras.optimizers.Adam(learning_rate=0.02, beta_1=0.99, epsilon=0.1)
style_weight = 0.01
content_weight = 10000.0

def style_content_loss(outputs):
    style_outputs = outputs['style']
    content_outputs = outputs['content']
    custom_method(imports='from google.colab import files;import numpy as np;import matplotlib.pyplot as plt;import matplotlib as mpl;import tensorflow_hub as hub;import IPython.display as display;import tensorflow as tf;import functools;import os;import time;import PIL.Image', function_to_run='tf.add_n(*args)', method_object=None, object_signature=None, function_args=[eval('[tf.reduce_mean((style_outputs[name]-style_targets[name])**2) \n                           for name in style_outputs.keys()]')], function_kwargs={})
    style_loss = tf.add_n([tf.reduce_mean((style_outputs[name] - style_targets[name]) ** 2) for name in style_outputs.keys()])
    style_loss *= style_weight / num_style_layers
    custom_method(imports='from google.colab import files;import numpy as np;import matplotlib.pyplot as plt;import matplotlib as mpl;import tensorflow_hub as hub;import IPython.display as display;import tensorflow as tf;import functools;import os;import time;import PIL.Image', function_to_run='tf.add_n(*args)', method_object=None, object_signature=None, function_args=[eval('[tf.reduce_mean((content_outputs[name]-content_targets[name])**2) \n                             for name in content_outputs.keys()]')], function_kwargs={})
    content_loss = tf.add_n([tf.reduce_mean((content_outputs[name] - content_targets[name]) ** 2) for name in content_outputs.keys()])
    content_loss *= content_weight / num_content_layers
    loss = style_loss + content_loss
    return loss

@tf.function()
def train_step(image):
    with tf.GradientTape() as tape:
        custom_method(imports='from google.colab import files;import numpy as np;import matplotlib.pyplot as plt;import matplotlib as mpl;import tensorflow_hub as hub;import IPython.display as display;import tensorflow as tf;import functools;import os;import time;import PIL.Image', function_to_run='obj(*args)', method_object=eval('extractor'), object_signature=None, function_args=[eval('image')], function_kwargs={}, custom_class='class StyleContentModel(tf.keras.models.Model):\n  def __init__(self, style_layers, content_layers):\n    super(StyleContentModel, self).__init__()\n    self.vgg = vgg_layers(style_layers + content_layers)\n    self.style_layers = style_layers\n    self.content_layers = content_layers\n    self.num_style_layers = len(style_layers)\n    self.vgg.trainable = False\n\n  def call(self, inputs):\n    "Expects float input in [0,1]"\n    inputs = inputs*255.0\n    preprocessed_input = tf.keras.applications.vgg19.preprocess_input(inputs)\n    outputs = self.vgg(preprocessed_input)\n    style_outputs, content_outputs = (outputs[:self.num_style_layers],\n                                      outputs[self.num_style_layers:])\n\n    style_outputs = [gram_matrix(style_output)\n                     for style_output in style_outputs]\n\n    content_dict = {content_name: value\n                    for content_name, value\n                    in zip(self.content_layers, content_outputs)}\n\n    style_dict = {style_name: value\n                  for style_name, value\n                  in zip(self.style_layers, style_outputs)}\n\n    return {\'content\': content_dict, \'style\': style_dict}')
        outputs = extractor(image)
        loss = style_content_loss(outputs)
    grad = tape.gradient(loss, image)
    custom_method(imports='from google.colab import files;import numpy as np;import matplotlib.pyplot as plt;import matplotlib as mpl;import tensorflow_hub as hub;import IPython.display as display;import tensorflow as tf;import functools;import os;import time;import PIL.Image', function_to_run='obj.apply_gradients(*args)', method_object=eval('opt'), object_signature=None, function_args=[eval('[(grad, image)]')], function_kwargs={}, custom_class=None)
    opt.apply_gradients([(grad, image)])
    custom_method(imports='from google.colab import files;import numpy as np;import matplotlib.pyplot as plt;import matplotlib as mpl;import tensorflow_hub as hub;import IPython.display as display;import tensorflow as tf;import functools;import os;import time;import PIL.Image', function_to_run='obj.assign(*args)', method_object=eval('image'), object_signature=None, function_args=[eval('clip_0_1(image)')], function_kwargs={}, custom_class=None)
    image.assign(clip_0_1(image))
train_step(image)
train_step(image)
train_step(image)
tensor_to_image(image)
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
(x_deltas, y_deltas) = high_pass_x_y(content_image)
plt.figure(figsize=(14, 10))
plt.subplot(2, 2, 1)
imshow(clip_0_1(2 * y_deltas + 0.5), 'Horizontal Deltas: Original')
plt.subplot(2, 2, 2)
imshow(clip_0_1(2 * x_deltas + 0.5), 'Vertical Deltas: Original')
(x_deltas, y_deltas) = high_pass_x_y(image)
plt.subplot(2, 2, 3)
imshow(clip_0_1(2 * y_deltas + 0.5), 'Horizontal Deltas: Styled')
plt.subplot(2, 2, 4)
imshow(clip_0_1(2 * x_deltas + 0.5), 'Vertical Deltas: Styled')
plt.figure(figsize=(14, 10))
custom_method(imports='from google.colab import files;import numpy as np;import matplotlib.pyplot as plt;import matplotlib as mpl;import tensorflow_hub as hub;import IPython.display as display;import tensorflow as tf;import functools;import os;import time;import PIL.Image', function_to_run='tf.image.sobel_edges(*args)', method_object=None, object_signature=None, function_args=[eval('content_image')], function_kwargs={})
sobel = tf.image.sobel_edges(content_image)
plt.subplot(1, 2, 1)
imshow(clip_0_1(sobel[..., 0] / 4 + 0.5), 'Horizontal Sobel-edges')
plt.subplot(1, 2, 2)
imshow(clip_0_1(sobel[..., 1] / 4 + 0.5), 'Vertical Sobel-edges')

def total_variation_loss(image):
    (x_deltas, y_deltas) = high_pass_x_y(image)
    return tf.reduce_sum(tf.abs(x_deltas)) + tf.reduce_sum(tf.abs(y_deltas))
total_variation_loss(image).numpy()
custom_method(imports='from google.colab import files;import numpy as np;import matplotlib.pyplot as plt;import matplotlib as mpl;import tensorflow_hub as hub;import IPython.display as display;import tensorflow as tf;import functools;import os;import time;import PIL.Image', function_to_run='tf.image.total_variation(image).numpy()', method_object=None, object_signature=None, function_args=[], function_kwargs={})
tf.image.total_variation(image).numpy()
total_variation_weight = 30

@tf.function()
def train_step(image):
    with tf.GradientTape() as tape:
        custom_method(imports='from google.colab import files;import numpy as np;import matplotlib.pyplot as plt;import matplotlib as mpl;import tensorflow_hub as hub;import IPython.display as display;import tensorflow as tf;import functools;import os;import time;import PIL.Image', function_to_run='obj(*args)', method_object=eval('extractor'), object_signature=None, function_args=[eval('image')], function_kwargs={}, custom_class='class StyleContentModel(tf.keras.models.Model):\n  def __init__(self, style_layers, content_layers):\n    super(StyleContentModel, self).__init__()\n    self.vgg = vgg_layers(style_layers + content_layers)\n    self.style_layers = style_layers\n    self.content_layers = content_layers\n    self.num_style_layers = len(style_layers)\n    self.vgg.trainable = False\n\n  def call(self, inputs):\n    "Expects float input in [0,1]"\n    inputs = inputs*255.0\n    preprocessed_input = tf.keras.applications.vgg19.preprocess_input(inputs)\n    outputs = self.vgg(preprocessed_input)\n    style_outputs, content_outputs = (outputs[:self.num_style_layers],\n                                      outputs[self.num_style_layers:])\n\n    style_outputs = [gram_matrix(style_output)\n                     for style_output in style_outputs]\n\n    content_dict = {content_name: value\n                    for content_name, value\n                    in zip(self.content_layers, content_outputs)}\n\n    style_dict = {style_name: value\n                  for style_name, value\n                  in zip(self.style_layers, style_outputs)}\n\n    return {\'content\': content_dict, \'style\': style_dict}')
        outputs = extractor(image)
        loss = style_content_loss(outputs)
        loss += total_variation_weight * tf.image.total_variation(image)
    grad = tape.gradient(loss, image)
    custom_method(imports='from google.colab import files;import numpy as np;import matplotlib.pyplot as plt;import matplotlib as mpl;import tensorflow_hub as hub;import IPython.display as display;import tensorflow as tf;import functools;import os;import time;import PIL.Image', function_to_run='obj.apply_gradients(*args)', method_object=eval('opt'), object_signature=None, function_args=[eval('[(grad, image)]')], function_kwargs={}, custom_class=None)
    opt.apply_gradients([(grad, image)])
    custom_method(imports='from google.colab import files;import numpy as np;import matplotlib.pyplot as plt;import matplotlib as mpl;import tensorflow_hub as hub;import IPython.display as display;import tensorflow as tf;import functools;import os;import time;import PIL.Image', function_to_run='obj.assign(*args)', method_object=eval('image'), object_signature=None, function_args=[eval('clip_0_1(image)')], function_kwargs={}, custom_class=None)
    image.assign(clip_0_1(image))
custom_method(imports='from google.colab import files;import numpy as np;import matplotlib.pyplot as plt;import matplotlib as mpl;import tensorflow_hub as hub;import IPython.display as display;import tensorflow as tf;import functools;import os;import time;import PIL.Image', function_to_run='tf.keras.optimizers.Adam(**kwargs)', method_object=None, object_signature=None, function_args=[], function_kwargs={'learning_rate': eval('0.02'), 'beta_1': eval('0.99'), 'epsilon': eval('1e-1')})
opt = tf.keras.optimizers.Adam(learning_rate=0.02, beta_1=0.99, epsilon=0.1)
custom_method(imports='from google.colab import files;import numpy as np;import matplotlib.pyplot as plt;import matplotlib as mpl;import tensorflow_hub as hub;import IPython.display as display;import tensorflow as tf;import functools;import os;import time;import PIL.Image', function_to_run='tf.Variable(*args)', method_object=None, object_signature=None, function_args=[eval('content_image')], function_kwargs={})
image = tf.Variable(content_image)
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
tensor_to_image(image).save(file_name)
try:
    from google.colab import files
except ImportError:
    pass
else:
    files.download(file_name)
