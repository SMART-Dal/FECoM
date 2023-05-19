import tensorflow as tf
import numpy as np
import matplotlib as mpl
import IPython.display as display
import PIL.Image
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

def custom_method(imports: str, function_to_run: str, method_object=None, object_signature=None, function_args: list=None, function_kwargs: dict=None, custom_class=None):
    result = send_request(imports=imports, function_to_run=function_to_run, function_args=function_args, function_kwargs=function_kwargs, max_wait_secs=MAX_WAIT_S, wait_after_run_secs=WAIT_AFTER_RUN_S, method_object=method_object, object_signature=object_signature, custom_class=custom_class, experiment_file_path=EXPERIMENT_FILE_PATH)
url = 'https://storage.googleapis.com/download.tensorflow.org/example_images/YellowLabradorLooking_new.jpg'

def download(url, max_dim=None):
    name = url.split('/')[-1]
    custom_method(imports='import PIL.Image;import IPython.display as display;import numpy as np;import time;import matplotlib as mpl;import tensorflow as tf', function_to_run='tf.keras.utils.get_file(*args, **kwargs)', method_object=None, object_signature=None, function_args=[eval('name')], function_kwargs={'origin': eval('url')})
    image_path = tf.keras.utils.get_file(name, origin=url)
    img = PIL.Image.open(image_path)
    if max_dim:
        custom_method(imports='import PIL.Image;import IPython.display as display;import numpy as np;import time;import matplotlib as mpl;import tensorflow as tf', function_to_run='obj.thumbnail(*args)', method_object=eval('img'), object_signature=None, function_args=[eval('(max_dim, max_dim)')], function_kwargs={}, custom_class=None)
        img.thumbnail((max_dim, max_dim))
    return np.array(img)

def deprocess(img):
    img = 255 * (img + 1.0) / 2.0
    return tf.cast(img, tf.uint8)

def show(img):
    display.display(PIL.Image.fromarray(np.array(img)))
original_img = download(url, max_dim=500)
show(original_img)
display.display(display.HTML('Image cc-by: <a "href=https://commons.wikimedia.org/wiki/File:Felis_catus-cat_on_snow.jpg">Von.grzanka</a>'))
custom_method(imports='import PIL.Image;import IPython.display as display;import numpy as np;import time;import matplotlib as mpl;import tensorflow as tf', function_to_run='tf.keras.applications.InceptionV3(**kwargs)', method_object=None, object_signature=None, function_args=[], function_kwargs={'include_top': eval('False'), 'weights': eval("'imagenet'")})
base_model = tf.keras.applications.InceptionV3(include_top=False, weights='imagenet')
names = ['mixed3', 'mixed5']
layers = [base_model.get_layer(name).output for name in names]
custom_method(imports='import PIL.Image;import IPython.display as display;import numpy as np;import time;import matplotlib as mpl;import tensorflow as tf', function_to_run='tf.keras.Model(**kwargs)', method_object=None, object_signature=None, function_args=[], function_kwargs={'inputs': eval('base_model.input'), 'outputs': eval('layers')})
dream_model = tf.keras.Model(inputs=base_model.input, outputs=layers)

def calc_loss(img, model):
    custom_method(imports='import PIL.Image;import IPython.display as display;import numpy as np;import time;import matplotlib as mpl;import tensorflow as tf', function_to_run='tf.expand_dims(*args, **kwargs)', method_object=None, object_signature=None, function_args=[eval('img')], function_kwargs={'axis': eval('0')})
    img_batch = tf.expand_dims(img, axis=0)
    layer_activations = model(img_batch)
    if len(layer_activations) == 1:
        layer_activations = [layer_activations]
    losses = []
    for act in layer_activations:
        custom_method(imports='import PIL.Image;import IPython.display as display;import numpy as np;import time;import matplotlib as mpl;import tensorflow as tf', function_to_run='tf.math.reduce_mean(*args)', method_object=None, object_signature=None, function_args=[eval('act')], function_kwargs={})
        loss = tf.math.reduce_mean(act)
        losses.append(loss)
    return tf.reduce_sum(losses)

class DeepDream(tf.Module):

    def __init__(self, model):
        self.model = model

    @tf.function(input_signature=(tf.TensorSpec(shape=[None, None, 3], dtype=tf.float32), tf.TensorSpec(shape=[], dtype=tf.int32), tf.TensorSpec(shape=[], dtype=tf.float32)))
    def __call__(self, img, steps, step_size):
        print('Tracing')
        custom_method(imports='import PIL.Image;import IPython.display as display;import numpy as np;import time;import matplotlib as mpl;import tensorflow as tf', function_to_run='tf.constant(*args)', method_object=None, object_signature=None, function_args=[eval('0.0')], function_kwargs={})
        loss = tf.constant(0.0)
        for n in tf.range(steps):
            with tf.GradientTape() as tape:
                tape.watch(img)
                loss = calc_loss(img, self.model)
            gradients = tape.gradient(loss, img)
            gradients /= tf.math.reduce_std(gradients) + 1e-08
            img = img + gradients * step_size
            custom_method(imports='import PIL.Image;import IPython.display as display;import numpy as np;import time;import matplotlib as mpl;import tensorflow as tf', function_to_run='tf.clip_by_value(*args)', method_object=None, object_signature=None, function_args=[eval('img'), eval('-1'), eval('1')], function_kwargs={})
            img = tf.clip_by_value(img, -1, 1)
        return (loss, img)
deepdream = DeepDream(dream_model)

def run_deep_dream_simple(img, steps=100, step_size=0.01):
    custom_method(imports='import PIL.Image;import IPython.display as display;import numpy as np;import time;import matplotlib as mpl;import tensorflow as tf', function_to_run='tf.keras.applications.inception_v3.preprocess_input(*args)', method_object=None, object_signature=None, function_args=[eval('img')], function_kwargs={})
    img = tf.keras.applications.inception_v3.preprocess_input(img)
    custom_method(imports='import PIL.Image;import IPython.display as display;import numpy as np;import time;import matplotlib as mpl;import tensorflow as tf', function_to_run='tf.convert_to_tensor(*args)', method_object=None, object_signature=None, function_args=[eval('img')], function_kwargs={})
    img = tf.convert_to_tensor(img)
    custom_method(imports='import PIL.Image;import IPython.display as display;import numpy as np;import time;import matplotlib as mpl;import tensorflow as tf', function_to_run='tf.convert_to_tensor(*args)', method_object=None, object_signature=None, function_args=[eval('step_size')], function_kwargs={})
    step_size = tf.convert_to_tensor(step_size)
    steps_remaining = steps
    step = 0
    while steps_remaining:
        if steps_remaining > 100:
            custom_method(imports='import PIL.Image;import IPython.display as display;import numpy as np;import time;import matplotlib as mpl;import tensorflow as tf', function_to_run='tf.constant(*args)', method_object=None, object_signature=None, function_args=[eval('100')], function_kwargs={})
            run_steps = tf.constant(100)
        else:
            custom_method(imports='import PIL.Image;import IPython.display as display;import numpy as np;import time;import matplotlib as mpl;import tensorflow as tf', function_to_run='tf.constant(*args)', method_object=None, object_signature=None, function_args=[eval('steps_remaining')], function_kwargs={})
            run_steps = tf.constant(steps_remaining)
        steps_remaining -= run_steps
        step += run_steps
        custom_method(imports='import PIL.Image;import IPython.display as display;import numpy as np;import time;import matplotlib as mpl;import tensorflow as tf', function_to_run='obj(*args)', method_object=eval('deepdream'), object_signature=None, function_args=[eval('img'), eval('run_steps'), eval('tf.constant(step_size)')], function_kwargs={}, custom_class='class DeepDream(tf.Module):\n  def __init__(self, model):\n    self.model = model\n\n  @tf.function(\n      input_signature=(\n        tf.TensorSpec(shape=[None,None,3], dtype=tf.float32),\n        tf.TensorSpec(shape=[], dtype=tf.int32),\n        tf.TensorSpec(shape=[], dtype=tf.float32),)\n  )\n  def __call__(self, img, steps, step_size):\n      print("Tracing")\n      loss = tf.constant(0.0)\n      for n in tf.range(steps):\n        with tf.GradientTape() as tape:\n          tape.watch(img)\n          loss = calc_loss(img, self.model)\n\n        gradients = tape.gradient(loss, img)\n\n        gradients /= tf.math.reduce_std(gradients) + 1e-8 \n        \n        img = img + gradients*step_size\n        img = tf.clip_by_value(img, -1, 1)\n\n      return loss, img')
        (loss, img) = deepdream(img, run_steps, tf.constant(step_size))
        display.clear_output(wait=True)
        show(deprocess(img))
        print('Step {}, loss {}'.format(step, loss))
    result = deprocess(img)
    display.clear_output(wait=True)
    show(result)
    return result
dream_img = run_deep_dream_simple(img=original_img, steps=100, step_size=0.01)
import time
start = time.time()
OCTAVE_SCALE = 1.3
custom_method(imports='import PIL.Image;import IPython.display as display;import numpy as np;import time;import matplotlib as mpl;import tensorflow as tf', function_to_run='tf.constant(*args)', method_object=None, object_signature=None, function_args=[eval('np.array(original_img)')], function_kwargs={})
img = tf.constant(np.array(original_img))
base_shape = tf.shape(img)[:-1]
custom_method(imports='import PIL.Image;import IPython.display as display;import numpy as np;import time;import matplotlib as mpl;import tensorflow as tf', function_to_run='tf.cast(*args)', method_object=None, object_signature=None, function_args=[eval('base_shape'), eval('tf.float32')], function_kwargs={})
float_base_shape = tf.cast(base_shape, tf.float32)
for n in range(-2, 3):
    custom_method(imports='import PIL.Image;import IPython.display as display;import numpy as np;import time;import matplotlib as mpl;import tensorflow as tf', function_to_run='tf.cast(*args)', method_object=None, object_signature=None, function_args=[eval('float_base_shape*(OCTAVE_SCALE**n)'), eval('tf.int32')], function_kwargs={})
    new_shape = tf.cast(float_base_shape * OCTAVE_SCALE ** n, tf.int32)
    custom_method(imports='import PIL.Image;import IPython.display as display;import numpy as np;import time;import matplotlib as mpl;import tensorflow as tf', function_to_run='tf.image.resize(img, new_shape).numpy()', method_object=None, object_signature=None, function_args=[], function_kwargs={})
    img = tf.image.resize(img, new_shape).numpy()
    img = run_deep_dream_simple(img=img, steps=50, step_size=0.01)
display.clear_output(wait=True)
custom_method(imports='import PIL.Image;import IPython.display as display;import numpy as np;import time;import matplotlib as mpl;import tensorflow as tf', function_to_run='tf.image.resize(*args)', method_object=None, object_signature=None, function_args=[eval('img'), eval('base_shape')], function_kwargs={})
img = tf.image.resize(img, base_shape)
custom_method(imports='import PIL.Image;import IPython.display as display;import numpy as np;import time;import matplotlib as mpl;import tensorflow as tf', function_to_run='tf.image.convert_image_dtype(*args, **kwargs)', method_object=None, object_signature=None, function_args=[eval('img/255.0')], function_kwargs={'dtype': eval('tf.uint8')})
img = tf.image.convert_image_dtype(img / 255.0, dtype=tf.uint8)
show(img)
end = time.time()
end - start

def random_roll(img, maxroll):
    custom_method(imports='import PIL.Image;import IPython.display as display;import numpy as np;import time;import matplotlib as mpl;import tensorflow as tf', function_to_run='tf.random.uniform(**kwargs)', method_object=None, object_signature=None, function_args=[], function_kwargs={'shape': eval('[2]'), 'minval': eval('-maxroll'), 'maxval': eval('maxroll'), 'dtype': eval('tf.int32')})
    shift = tf.random.uniform(shape=[2], minval=-maxroll, maxval=maxroll, dtype=tf.int32)
    custom_method(imports='import PIL.Image;import IPython.display as display;import numpy as np;import time;import matplotlib as mpl;import tensorflow as tf', function_to_run='tf.roll(*args, **kwargs)', method_object=None, object_signature=None, function_args=[eval('img')], function_kwargs={'shift': eval('shift'), 'axis': eval('[0,1]')})
    img_rolled = tf.roll(img, shift=shift, axis=[0, 1])
    return (shift, img_rolled)
(shift, img_rolled) = random_roll(np.array(original_img), 512)
show(img_rolled)

class TiledGradients(tf.Module):

    def __init__(self, model):
        self.model = model

    @tf.function(input_signature=(tf.TensorSpec(shape=[None, None, 3], dtype=tf.float32), tf.TensorSpec(shape=[2], dtype=tf.int32), tf.TensorSpec(shape=[], dtype=tf.int32)))
    def __call__(self, img, img_size, tile_size=512):
        (shift, img_rolled) = random_roll(img, tile_size)
        custom_method(imports='import PIL.Image;import IPython.display as display;import numpy as np;import time;import matplotlib as mpl;import tensorflow as tf', function_to_run='tf.zeros_like(*args)', method_object=None, object_signature=None, function_args=[eval('img_rolled')], function_kwargs={})
        gradients = tf.zeros_like(img_rolled)
        xs = tf.range(0, img_size[1], tile_size)[:-1]
        if not tf.cast(len(xs), bool):
            custom_method(imports='import PIL.Image;import IPython.display as display;import numpy as np;import time;import matplotlib as mpl;import tensorflow as tf', function_to_run='tf.constant(*args)', method_object=None, object_signature=None, function_args=[eval('[0]')], function_kwargs={})
            xs = tf.constant([0])
        ys = tf.range(0, img_size[0], tile_size)[:-1]
        if not tf.cast(len(ys), bool):
            custom_method(imports='import PIL.Image;import IPython.display as display;import numpy as np;import time;import matplotlib as mpl;import tensorflow as tf', function_to_run='tf.constant(*args)', method_object=None, object_signature=None, function_args=[eval('[0]')], function_kwargs={})
            ys = tf.constant([0])
        for x in xs:
            for y in ys:
                with tf.GradientTape() as tape:
                    tape.watch(img_rolled)
                    img_tile = img_rolled[y:y + tile_size, x:x + tile_size]
                    loss = calc_loss(img_tile, self.model)
                gradients = gradients + tape.gradient(loss, img_rolled)
        custom_method(imports='import PIL.Image;import IPython.display as display;import numpy as np;import time;import matplotlib as mpl;import tensorflow as tf', function_to_run='tf.roll(*args, **kwargs)', method_object=None, object_signature=None, function_args=[eval('gradients')], function_kwargs={'shift': eval('-shift'), 'axis': eval('[0,1]')})
        gradients = tf.roll(gradients, shift=-shift, axis=[0, 1])
        gradients /= tf.math.reduce_std(gradients) + 1e-08
        return gradients
get_tiled_gradients = TiledGradients(dream_model)

def run_deep_dream_with_octaves(img, steps_per_octave=100, step_size=0.01, octaves=range(-2, 3), octave_scale=1.3):
    custom_method(imports='import PIL.Image;import IPython.display as display;import numpy as np;import time;import matplotlib as mpl;import tensorflow as tf', function_to_run='tf.shape(*args)', method_object=None, object_signature=None, function_args=[eval('img')], function_kwargs={})
    base_shape = tf.shape(img)
    custom_method(imports='import PIL.Image;import IPython.display as display;import numpy as np;import time;import matplotlib as mpl;import tensorflow as tf', function_to_run='tf.keras.utils.img_to_array(*args)', method_object=None, object_signature=None, function_args=[eval('img')], function_kwargs={})
    img = tf.keras.utils.img_to_array(img)
    custom_method(imports='import PIL.Image;import IPython.display as display;import numpy as np;import time;import matplotlib as mpl;import tensorflow as tf', function_to_run='tf.keras.applications.inception_v3.preprocess_input(*args)', method_object=None, object_signature=None, function_args=[eval('img')], function_kwargs={})
    img = tf.keras.applications.inception_v3.preprocess_input(img)
    initial_shape = img.shape[:-1]
    custom_method(imports='import PIL.Image;import IPython.display as display;import numpy as np;import time;import matplotlib as mpl;import tensorflow as tf', function_to_run='tf.image.resize(*args)', method_object=None, object_signature=None, function_args=[eval('img'), eval('initial_shape')], function_kwargs={})
    img = tf.image.resize(img, initial_shape)
    for octave in octaves:
        new_size = tf.cast(tf.convert_to_tensor(base_shape[:-1]), tf.float32) * octave_scale ** octave
        custom_method(imports='import PIL.Image;import IPython.display as display;import numpy as np;import time;import matplotlib as mpl;import tensorflow as tf', function_to_run='tf.cast(*args)', method_object=None, object_signature=None, function_args=[eval('new_size'), eval('tf.int32')], function_kwargs={})
        new_size = tf.cast(new_size, tf.int32)
        custom_method(imports='import PIL.Image;import IPython.display as display;import numpy as np;import time;import matplotlib as mpl;import tensorflow as tf', function_to_run='tf.image.resize(*args)', method_object=None, object_signature=None, function_args=[eval('img'), eval('new_size')], function_kwargs={})
        img = tf.image.resize(img, new_size)
        for step in range(steps_per_octave):
            custom_method(imports='import PIL.Image;import IPython.display as display;import numpy as np;import time;import matplotlib as mpl;import tensorflow as tf', function_to_run='obj(*args)', method_object=eval('get_tiled_gradients'), object_signature=None, function_args=[eval('img'), eval('new_size')], function_kwargs={}, custom_class='class TiledGradients(tf.Module):\n  def __init__(self, model):\n    self.model = model\n\n  @tf.function(\n      input_signature=(\n        tf.TensorSpec(shape=[None,None,3], dtype=tf.float32),\n        tf.TensorSpec(shape=[2], dtype=tf.int32),\n        tf.TensorSpec(shape=[], dtype=tf.int32),)\n  )\n  def __call__(self, img, img_size, tile_size=512):\n    shift, img_rolled = random_roll(img, tile_size)\n\n    gradients = tf.zeros_like(img_rolled)\n    \n    xs = tf.range(0, img_size[1], tile_size)[:-1]\n    if not tf.cast(len(xs), bool):\n      xs = tf.constant([0])\n    ys = tf.range(0, img_size[0], tile_size)[:-1]\n    if not tf.cast(len(ys), bool):\n      ys = tf.constant([0])\n\n    for x in xs:\n      for y in ys:\n        with tf.GradientTape() as tape:\n          tape.watch(img_rolled)\n\n          img_tile = img_rolled[y:y+tile_size, x:x+tile_size]\n          loss = calc_loss(img_tile, self.model)\n\n        gradients = gradients + tape.gradient(loss, img_rolled)\n\n    gradients = tf.roll(gradients, shift=-shift, axis=[0,1])\n\n    gradients /= tf.math.reduce_std(gradients) + 1e-8 \n\n    return gradients')
            gradients = get_tiled_gradients(img, new_size)
            img = img + gradients * step_size
            custom_method(imports='import PIL.Image;import IPython.display as display;import numpy as np;import time;import matplotlib as mpl;import tensorflow as tf', function_to_run='tf.clip_by_value(*args)', method_object=None, object_signature=None, function_args=[eval('img'), eval('-1'), eval('1')], function_kwargs={})
            img = tf.clip_by_value(img, -1, 1)
            if step % 10 == 0:
                display.clear_output(wait=True)
                show(deprocess(img))
                print('Octave {}, Step {}'.format(octave, step))
    result = deprocess(img)
    return result
img = run_deep_dream_with_octaves(img=original_img, step_size=0.01)
display.clear_output(wait=True)
custom_method(imports='import PIL.Image;import IPython.display as display;import numpy as np;import time;import matplotlib as mpl;import tensorflow as tf', function_to_run='tf.image.resize(*args)', method_object=None, object_signature=None, function_args=[eval('img'), eval('base_shape')], function_kwargs={})
img = tf.image.resize(img, base_shape)
custom_method(imports='import PIL.Image;import IPython.display as display;import numpy as np;import time;import matplotlib as mpl;import tensorflow as tf', function_to_run='tf.image.convert_image_dtype(*args, **kwargs)', method_object=None, object_signature=None, function_args=[eval('img/255.0')], function_kwargs={'dtype': eval('tf.uint8')})
img = tf.image.convert_image_dtype(img / 255.0, dtype=tf.uint8)
show(img)
