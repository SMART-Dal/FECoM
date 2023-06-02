import tensorflow as tf
import numpy as np
import matplotlib as mpl
import IPython.display as display
import PIL.Image
import sys
from tool.client.client_config import EXPERIMENT_DIR
from tool.server.local_execution import before_execution as before_execution_INSERTED_INTO_SCRIPT
from tool.server.local_execution import after_execution as after_execution_INSERTED_INTO_SCRIPT
experiment_number = sys.argv[1]
experiment_project = sys.argv[2]
EXPERIMENT_FILE_PATH = EXPERIMENT_DIR / 'local-execution' / experiment_project / f'experiment-{experiment_number}.json'
url = 'https://storage.googleapis.com/download.tensorflow.org/example_images/YellowLabradorLooking_new.jpg'

def download(url, max_dim=None):
    name = url.split('/')[-1]
    start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT()
    image_path = tf.keras.utils.get_file(name, origin=url)
    after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tf.keras.utils.get_file()', method_object=None, function_args=[name], function_kwargs={'origin': url})
    img = PIL.Image.open(image_path)
    if max_dim:
        start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT()
        img.thumbnail((max_dim, max_dim))
        after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='PIL.Image.open.thumbnail()', method_object=img, function_args=[(max_dim, max_dim)], function_kwargs=None)
    return np.array(img)

def deprocess(img):
    img = 255 * (img + 1.0) / 2.0
    return tf.cast(img, tf.uint8)

def show(img):
    display.display(PIL.Image.fromarray(np.array(img)))
original_img = download(url, max_dim=500)
show(original_img)
display.display(display.HTML('Image cc-by: <a "href=https://commons.wikimedia.org/wiki/File:Felis_catus-cat_on_snow.jpg">Von.grzanka</a>'))
start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT()
base_model = tf.keras.applications.InceptionV3(include_top=False, weights='imagenet')
after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tf.keras.applications.InceptionV3()', method_object=None, function_args=None, function_kwargs={'include_top': False, 'weights': 'imagenet'})
names = ['mixed3', 'mixed5']
layers = [base_model.get_layer(name).output for name in names]
start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT()
dream_model = tf.keras.Model(inputs=base_model.input, outputs=layers)
after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tf.keras.Model()', method_object=None, function_args=None, function_kwargs={'inputs': base_model.input, 'outputs': layers})

def calc_loss(img, model):
    start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT()
    img_batch = tf.expand_dims(img, axis=0)
    after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tf.expand_dims()', method_object=None, function_args=[img], function_kwargs={'axis': 0})
    layer_activations = model(img_batch)
    if len(layer_activations) == 1:
        layer_activations = [layer_activations]
    losses = []
    for act in layer_activations:
        start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT()
        loss = tf.math.reduce_mean(act)
        after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tf.math.reduce_mean()', method_object=None, function_args=[act], function_kwargs=None)
        losses.append(loss)
    return tf.reduce_sum(losses)

class DeepDream(tf.Module):

    def __init__(self, model):
        self.model = model

    @tf.function(input_signature=(tf.TensorSpec(shape=[None, None, 3], dtype=tf.float32), tf.TensorSpec(shape=[], dtype=tf.int32), tf.TensorSpec(shape=[], dtype=tf.float32)))
    def __call__(self, img, steps, step_size):
        print('Tracing')
        start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT()
        loss = tf.constant(0.0)
        after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tf.constant()', method_object=None, function_args=[0.0], function_kwargs=None)
        for n in tf.range(steps):
            with tf.GradientTape() as tape:
                tape.watch(img)
                loss = calc_loss(img, self.model)
            gradients = tape.gradient(loss, img)
            gradients /= tf.math.reduce_std(gradients) + 1e-08
            img = img + gradients * step_size
            start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT()
            img = tf.clip_by_value(img, -1, 1)
            after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tf.clip_by_value()', method_object=None, function_args=[img, -1, 1], function_kwargs=None)
        return (loss, img)
deepdream = DeepDream(dream_model)

def run_deep_dream_simple(img, steps=100, step_size=0.01):
    start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT()
    img = tf.keras.applications.inception_v3.preprocess_input(img)
    after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tf.keras.applications.inception_v3.preprocess_input()', method_object=None, function_args=[img], function_kwargs=None)
    start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT()
    img = tf.convert_to_tensor(img)
    after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tf.convert_to_tensor()', method_object=None, function_args=[img], function_kwargs=None)
    start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT()
    step_size = tf.convert_to_tensor(step_size)
    after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tf.convert_to_tensor()', method_object=None, function_args=[step_size], function_kwargs=None)
    steps_remaining = steps
    step = 0
    while steps_remaining:
        if steps_remaining > 100:
            start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT()
            run_steps = tf.constant(100)
            after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tf.constant()', method_object=None, function_args=[100], function_kwargs=None)
        else:
            start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT()
            run_steps = tf.constant(steps_remaining)
            after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tf.constant()', method_object=None, function_args=[steps_remaining], function_kwargs=None)
        steps_remaining -= run_steps
        step += run_steps
        start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT()
        (loss, img) = deepdream(img, run_steps, tf.constant(step_size))
        after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='DeepDream()', method_object=deepdream, function_args=[img, run_steps, tf.constant(step_size)], function_kwargs=None)
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
start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT()
img = tf.constant(np.array(original_img))
after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tf.constant()', method_object=None, function_args=[np.array(original_img)], function_kwargs=None)
base_shape = tf.shape(img)[:-1]
start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT()
float_base_shape = tf.cast(base_shape, tf.float32)
after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tf.cast()', method_object=None, function_args=[base_shape, tf.float32], function_kwargs=None)
for n in range(-2, 3):
    start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT()
    new_shape = tf.cast(float_base_shape * OCTAVE_SCALE ** n, tf.int32)
    after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tf.cast()', method_object=None, function_args=[float_base_shape * OCTAVE_SCALE ** n, tf.int32], function_kwargs=None)
    start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT()
    img = tf.image.resize(img, new_shape).numpy()
    after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tf.image.resize(img, new_shape).numpy()', method_object=None, function_args=None, function_kwargs=None)
    img = run_deep_dream_simple(img=img, steps=50, step_size=0.01)
display.clear_output(wait=True)
start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT()
img = tf.image.resize(img, base_shape)
after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tf.image.resize()', method_object=None, function_args=[img, base_shape], function_kwargs=None)
start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT()
img = tf.image.convert_image_dtype(img / 255.0, dtype=tf.uint8)
after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tf.image.convert_image_dtype()', method_object=None, function_args=[img / 255.0], function_kwargs={'dtype': tf.uint8})
show(img)
end = time.time()
end - start

def random_roll(img, maxroll):
    start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT()
    shift = tf.random.uniform(shape=[2], minval=-maxroll, maxval=maxroll, dtype=tf.int32)
    after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tf.random.uniform()', method_object=None, function_args=None, function_kwargs={'shape': [2], 'minval': -maxroll, 'maxval': maxroll, 'dtype': tf.int32})
    start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT()
    img_rolled = tf.roll(img, shift=shift, axis=[0, 1])
    after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tf.roll()', method_object=None, function_args=[img], function_kwargs={'shift': shift, 'axis': [0, 1]})
    return (shift, img_rolled)
(shift, img_rolled) = random_roll(np.array(original_img), 512)
show(img_rolled)

class TiledGradients(tf.Module):

    def __init__(self, model):
        self.model = model

    @tf.function(input_signature=(tf.TensorSpec(shape=[None, None, 3], dtype=tf.float32), tf.TensorSpec(shape=[2], dtype=tf.int32), tf.TensorSpec(shape=[], dtype=tf.int32)))
    def __call__(self, img, img_size, tile_size=512):
        (shift, img_rolled) = random_roll(img, tile_size)
        start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT()
        gradients = tf.zeros_like(img_rolled)
        after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tf.zeros_like()', method_object=None, function_args=[img_rolled], function_kwargs=None)
        xs = tf.range(0, img_size[1], tile_size)[:-1]
        if not tf.cast(len(xs), bool):
            start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT()
            xs = tf.constant([0])
            after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tf.constant()', method_object=None, function_args=[[0]], function_kwargs=None)
        ys = tf.range(0, img_size[0], tile_size)[:-1]
        if not tf.cast(len(ys), bool):
            start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT()
            ys = tf.constant([0])
            after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tf.constant()', method_object=None, function_args=[[0]], function_kwargs=None)
        for x in xs:
            for y in ys:
                with tf.GradientTape() as tape:
                    tape.watch(img_rolled)
                    img_tile = img_rolled[y:y + tile_size, x:x + tile_size]
                    loss = calc_loss(img_tile, self.model)
                gradients = gradients + tape.gradient(loss, img_rolled)
        start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT()
        gradients = tf.roll(gradients, shift=-shift, axis=[0, 1])
        after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tf.roll()', method_object=None, function_args=[gradients], function_kwargs={'shift': -shift, 'axis': [0, 1]})
        gradients /= tf.math.reduce_std(gradients) + 1e-08
        return gradients
get_tiled_gradients = TiledGradients(dream_model)

def run_deep_dream_with_octaves(img, steps_per_octave=100, step_size=0.01, octaves=range(-2, 3), octave_scale=1.3):
    start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT()
    base_shape = tf.shape(img)
    after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tf.shape()', method_object=None, function_args=[img], function_kwargs=None)
    start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT()
    img = tf.keras.utils.img_to_array(img)
    after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tf.keras.utils.img_to_array()', method_object=None, function_args=[img], function_kwargs=None)
    start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT()
    img = tf.keras.applications.inception_v3.preprocess_input(img)
    after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tf.keras.applications.inception_v3.preprocess_input()', method_object=None, function_args=[img], function_kwargs=None)
    initial_shape = img.shape[:-1]
    start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT()
    img = tf.image.resize(img, initial_shape)
    after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tf.image.resize()', method_object=None, function_args=[img, initial_shape], function_kwargs=None)
    for octave in octaves:
        new_size = tf.cast(tf.convert_to_tensor(base_shape[:-1]), tf.float32) * octave_scale ** octave
        start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT()
        new_size = tf.cast(new_size, tf.int32)
        after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tf.cast()', method_object=None, function_args=[new_size, tf.int32], function_kwargs=None)
        start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT()
        img = tf.image.resize(img, new_size)
        after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tf.image.resize()', method_object=None, function_args=[img, new_size], function_kwargs=None)
        for step in range(steps_per_octave):
            start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT()
            gradients = get_tiled_gradients(img, new_size)
            after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='TiledGradients()', method_object=get_tiled_gradients, function_args=[img, new_size], function_kwargs=None)
            img = img + gradients * step_size
            start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT()
            img = tf.clip_by_value(img, -1, 1)
            after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tf.clip_by_value()', method_object=None, function_args=[img, -1, 1], function_kwargs=None)
            if step % 10 == 0:
                display.clear_output(wait=True)
                show(deprocess(img))
                print('Octave {}, Step {}'.format(octave, step))
    result = deprocess(img)
    return result
img = run_deep_dream_with_octaves(img=original_img, step_size=0.01)
display.clear_output(wait=True)
start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT()
img = tf.image.resize(img, base_shape)
after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tf.image.resize()', method_object=None, function_args=[img, base_shape], function_kwargs=None)
start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT()
img = tf.image.convert_image_dtype(img / 255.0, dtype=tf.uint8)
after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tf.image.convert_image_dtype()', method_object=None, function_args=[img / 255.0], function_kwargs={'dtype': tf.uint8})
show(img)
