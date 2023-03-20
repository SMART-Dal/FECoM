import matplotlib.pylab as plt
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import os
from pathlib import Path
import dill as pickle
import sys
from tool.client.client_config import EXPERIMENT_DIR, MAX_WAIT_S, WAIT_AFTER_RUN_S
from tool.server.send_request import send_request
from tool.server.function_details import FunctionDetails
current_path = os.path.abspath(__file__)
(immediate_folder, file_name) = os.path.split(current_path)
immediate_folder = os.path.basename(immediate_folder)
experiment_number = int(sys.argv[0])
experiment_file_name = os.path.splitext(file_name)[0]
EXPERIMENT_FILE_PATH = EXPERIMENT_DIR / 'method-level' / immediate_folder / experiment_file_name / f'experiment-{experiment_number}.json'

def custom_method(func, imports: str, function_to_run: str, method_object=None, function_args: list=None, function_kwargs: dict=None, max_wait_secs=MAX_WAIT_S, custom_class=None, wait_after_run_secs=WAIT_AFTER_RUN_S):
    result = send_request(imports=imports, function_to_run=function_to_run, function_args=function_args, function_kwargs=function_kwargs, max_wait_secs=max_wait_secs, method_object=method_object, custom_class=custom_class, experiment_file_path=EXPERIMENT_FILE_PATH)
    return func
if __name__ == '__main__':
    print(EXPERIMENT_FILE_PATH)
model = custom_method(
tf.keras.Sequential([hub.KerasLayer(name='inception_v1', handle='https://tfhub.dev/google/imagenet/inception_v1/classification/4', trainable=False)]), imports='import matplotlib.pylab as plt;import tensorflow as tf;import numpy as np;import tensorflow_hub as hub', function_to_run='tf.keras.Sequential(*args)', method_object=None, function_args=[eval("[\n    hub.KerasLayer(\n        name='inception_v1',\n        handle='https://tfhub.dev/google/imagenet/inception_v1/classification/4',\n        trainable=False),\n]")], function_kwargs={}, max_wait_secs=0)
custom_method(
model.build([None, 224, 224, 3]), imports='import matplotlib.pylab as plt;import tensorflow as tf;import numpy as np;import tensorflow_hub as hub', function_to_run='obj.build(*args)', method_object=eval('model'), function_args=[eval('[None, 224, 224, 3]')], function_kwargs={}, max_wait_secs=0, custom_class=None)
custom_method(
model.summary(), imports='import matplotlib.pylab as plt;import tensorflow as tf;import numpy as np;import tensorflow_hub as hub', function_to_run='obj.summary()', method_object=eval('model'), function_args=[], function_kwargs={}, max_wait_secs=0, custom_class=None)

def load_imagenet_labels(file_path):
    labels_file = custom_method(
    tf.keras.utils.get_file('ImageNetLabels.txt', file_path), imports='import matplotlib.pylab as plt;import tensorflow as tf;import numpy as np;import tensorflow_hub as hub', function_to_run='tf.keras.utils.get_file(*args)', method_object=None, function_args=[eval("'ImageNetLabels.txt'"), eval('file_path')], function_kwargs={}, max_wait_secs=0)
    with open(labels_file) as reader:
        f = reader.read()
        labels = f.splitlines()
    return np.array(labels)
imagenet_labels = custom_method(
load_imagenet_labels('https://storage.googleapis.com/download.tensorflow.org/data/ImageNetLabels.txt'), imports='import matplotlib.pylab as plt;import tensorflow as tf;import numpy as np;import tensorflow_hub as hub', function_to_run='obj(*args)', method_object=eval('load_imagenet_labels'), function_args=[eval("'https://storage.googleapis.com/download.tensorflow.org/data/ImageNetLabels.txt'")], function_kwargs={}, max_wait_secs=0, custom_class=None)

def read_image(file_name):
    image = custom_method(
    tf.io.read_file(file_name), imports='import matplotlib.pylab as plt;import tensorflow as tf;import numpy as np;import tensorflow_hub as hub', function_to_run='tf.io.read_file(*args)', method_object=None, function_args=[eval('file_name')], function_kwargs={}, max_wait_secs=0)
    image = custom_method(
    tf.io.decode_jpeg(image, channels=3), imports='import matplotlib.pylab as plt;import tensorflow as tf;import numpy as np;import tensorflow_hub as hub', function_to_run='tf.io.decode_jpeg(*args, **kwargs)', method_object=None, function_args=[eval('image')], function_kwargs={'channels': eval('3')}, max_wait_secs=0)
    image = custom_method(
    tf.image.convert_image_dtype(image, tf.float32), imports='import matplotlib.pylab as plt;import tensorflow as tf;import numpy as np;import tensorflow_hub as hub', function_to_run='tf.image.convert_image_dtype(*args)', method_object=None, function_args=[eval('image'), eval('tf.float32')], function_kwargs={}, max_wait_secs=0)
    image = custom_method(
    tf.image.resize_with_pad(image, target_height=224, target_width=224), imports='import matplotlib.pylab as plt;import tensorflow as tf;import numpy as np;import tensorflow_hub as hub', function_to_run='tf.image.resize_with_pad(*args, **kwargs)', method_object=None, function_args=[eval('image')], function_kwargs={'target_height': eval('224'), 'target_width': eval('224')}, max_wait_secs=0)
    return image
img_url = {'Fireboat': 'http://storage.googleapis.com/download.tensorflow.org/example_images/San_Francisco_fireboat_showing_off.jpg', 'Giant Panda': 'http://storage.googleapis.com/download.tensorflow.org/example_images/Giant_Panda_2.jpeg'}
img_paths = {name: custom_method(
tf.keras.utils.get_file(name, url), imports='import matplotlib.pylab as plt;import tensorflow as tf;import numpy as np;import tensorflow_hub as hub', function_to_run='tf.keras.utils.get_file(*args)', method_object=None, function_args=[eval('name'), eval('url')], function_kwargs={}, max_wait_secs=0) for (name, url) in img_url.items()}
img_name_tensors = {name: custom_method(
read_image(img_path), imports='import matplotlib.pylab as plt;import tensorflow as tf;import numpy as np;import tensorflow_hub as hub', function_to_run='obj(*args)', method_object=eval('read_image'), function_args=[eval('img_path')], function_kwargs={}, max_wait_secs=0, custom_class=None) for (name, img_path) in img_paths.items()}
plt.figure(figsize=(8, 8))
for (n, (name, img_tensors)) in enumerate(img_name_tensors.items()):
    ax = plt.subplot(1, 2, n + 1)
    custom_method(
    ax.imshow(img_tensors), imports='import matplotlib.pylab as plt;import tensorflow as tf;import numpy as np;import tensorflow_hub as hub', function_to_run='obj.imshow(*args)', method_object=eval('ax'), function_args=[eval('img_tensors')], function_kwargs={}, max_wait_secs=0, custom_class=None)
    custom_method(
    ax.set_title(name), imports='import matplotlib.pylab as plt;import tensorflow as tf;import numpy as np;import tensorflow_hub as hub', function_to_run='obj.set_title(*args)', method_object=eval('ax'), function_args=[eval('name')], function_kwargs={}, max_wait_secs=0, custom_class=None)
    custom_method(
    ax.axis('off'), imports='import matplotlib.pylab as plt;import tensorflow as tf;import numpy as np;import tensorflow_hub as hub', function_to_run='obj.axis(*args)', method_object=eval('ax'), function_args=[eval("'off'")], function_kwargs={}, max_wait_secs=0, custom_class=None)
plt.tight_layout()

def top_k_predictions(img, k=3):
    image_batch = custom_method(
    tf.expand_dims(img, 0), imports='import matplotlib.pylab as plt;import tensorflow as tf;import numpy as np;import tensorflow_hub as hub', function_to_run='tf.expand_dims(*args)', method_object=None, function_args=[eval('img'), eval('0')], function_kwargs={}, max_wait_secs=0)
    predictions = custom_method(
    model(image_batch), imports='import matplotlib.pylab as plt;import tensorflow as tf;import numpy as np;import tensorflow_hub as hub', function_to_run='obj(*args)', method_object=eval('model'), function_args=[eval('image_batch')], function_kwargs={}, max_wait_secs=0, custom_class=None)
    probs = custom_method(
    tf.nn.softmax(predictions, axis=-1), imports='import matplotlib.pylab as plt;import tensorflow as tf;import numpy as np;import tensorflow_hub as hub', function_to_run='tf.nn.softmax(*args, **kwargs)', method_object=None, function_args=[eval('predictions')], function_kwargs={'axis': eval('-1')}, max_wait_secs=0)
    (top_probs, top_idxs) = custom_method(
    tf.math.top_k(input=probs, k=k), imports='import matplotlib.pylab as plt;import tensorflow as tf;import numpy as np;import tensorflow_hub as hub', function_to_run='tf.math.top_k(**kwargs)', method_object=None, function_args=[], function_kwargs={'input': eval('probs'), 'k': eval('k')}, max_wait_secs=0)
    top_labels = imagenet_labels[tuple(top_idxs)]
    return (top_labels, top_probs[0])
for (name, img_tensor) in img_name_tensors.items():
    plt.imshow(img_tensor)
    plt.title(name, fontweight='bold')
    plt.axis('off')
    plt.show()
    (pred_label, pred_prob) = custom_method(
    top_k_predictions(img_tensor), imports='import matplotlib.pylab as plt;import tensorflow as tf;import numpy as np;import tensorflow_hub as hub', function_to_run='obj(*args)', method_object=eval('top_k_predictions'), function_args=[eval('img_tensor')], function_kwargs={}, max_wait_secs=0, custom_class=None)
    for (label, prob) in zip(pred_label, pred_prob):
        print(f'{label}: {prob:0.1%}')

def f(x):
    """A simplified model function."""
    return custom_method(
    tf.where(x < 0.8, x, 0.8), imports='import matplotlib.pylab as plt;import tensorflow as tf;import numpy as np;import tensorflow_hub as hub', function_to_run='tf.where(*args)', method_object=None, function_args=[eval('x < 0.8'), eval('x'), eval('0.8')], function_kwargs={}, max_wait_secs=0)

def interpolated_path(x):
    """A straight line path."""
    return custom_method(
    tf.zeros_like(x), imports='import matplotlib.pylab as plt;import tensorflow as tf;import numpy as np;import tensorflow_hub as hub', function_to_run='tf.zeros_like(*args)', method_object=None, function_args=[eval('x')], function_kwargs={}, max_wait_secs=0)
x = custom_method(
tf.linspace(start=0.0, stop=1.0, num=6), imports='import matplotlib.pylab as plt;import tensorflow as tf;import numpy as np;import tensorflow_hub as hub', function_to_run='tf.linspace(**kwargs)', method_object=None, function_args=[], function_kwargs={'start': eval('0.0'), 'stop': eval('1.0'), 'num': eval('6')}, max_wait_secs=0)
y = f(x)
fig = plt.figure(figsize=(12, 5))
ax0 = fig.add_subplot(121)
custom_method(
ax0.plot(x, f(x), marker='o'), imports='import matplotlib.pylab as plt;import tensorflow as tf;import numpy as np;import tensorflow_hub as hub', function_to_run='obj.plot(*args, **kwargs)', method_object=eval('ax0'), function_args=[eval('x'), eval('f(x)')], function_kwargs={'marker': eval("'o'")}, max_wait_secs=0, custom_class=None)
custom_method(
ax0.set_title('Gradients saturate over F(x)', fontweight='bold'), imports='import matplotlib.pylab as plt;import tensorflow as tf;import numpy as np;import tensorflow_hub as hub', function_to_run='obj.set_title(*args, **kwargs)', method_object=eval('ax0'), function_args=[eval("'Gradients saturate over F(x)'")], function_kwargs={'fontweight': eval("'bold'")}, max_wait_secs=0, custom_class=None)
custom_method(
ax0.text(0.2, 0.5, 'Gradients > 0 = \n x is important'), imports='import matplotlib.pylab as plt;import tensorflow as tf;import numpy as np;import tensorflow_hub as hub', function_to_run='obj.text(*args)', method_object=eval('ax0'), function_args=[eval('0.2'), eval('0.5'), eval("'Gradients > 0 = \\n x is important'")], function_kwargs={}, max_wait_secs=0, custom_class=None)
custom_method(
ax0.text(0.7, 0.85, 'Gradients = 0 \n x not important'), imports='import matplotlib.pylab as plt;import tensorflow as tf;import numpy as np;import tensorflow_hub as hub', function_to_run='obj.text(*args)', method_object=eval('ax0'), function_args=[eval('0.7'), eval('0.85'), eval("'Gradients = 0 \\n x not important'")], function_kwargs={}, max_wait_secs=0, custom_class=None)
custom_method(
ax0.set_yticks(tf.range(0, 1.5, 0.5)), imports='import matplotlib.pylab as plt;import tensorflow as tf;import numpy as np;import tensorflow_hub as hub', function_to_run='obj.set_yticks(*args)', method_object=eval('ax0'), function_args=[eval('tf.range(0, 1.5, 0.5)')], function_kwargs={}, max_wait_secs=0, custom_class=None)
custom_method(
ax0.set_xticks(tf.range(0, 1.5, 0.5)), imports='import matplotlib.pylab as plt;import tensorflow as tf;import numpy as np;import tensorflow_hub as hub', function_to_run='obj.set_xticks(*args)', method_object=eval('ax0'), function_args=[eval('tf.range(0, 1.5, 0.5)')], function_kwargs={}, max_wait_secs=0, custom_class=None)
custom_method(
ax0.set_ylabel('F(x) - model true class predicted probability'), imports='import matplotlib.pylab as plt;import tensorflow as tf;import numpy as np;import tensorflow_hub as hub', function_to_run='obj.set_ylabel(*args)', method_object=eval('ax0'), function_args=[eval("'F(x) - model true class predicted probability'")], function_kwargs={}, max_wait_secs=0, custom_class=None)
custom_method(
ax0.set_xlabel('x - (pixel value)'), imports='import matplotlib.pylab as plt;import tensorflow as tf;import numpy as np;import tensorflow_hub as hub', function_to_run='obj.set_xlabel(*args)', method_object=eval('ax0'), function_args=[eval("'x - (pixel value)'")], function_kwargs={}, max_wait_secs=0, custom_class=None)
ax1 = fig.add_subplot(122)
custom_method(
ax1.plot(x, f(x), marker='o'), imports='import matplotlib.pylab as plt;import tensorflow as tf;import numpy as np;import tensorflow_hub as hub', function_to_run='obj.plot(*args, **kwargs)', method_object=eval('ax1'), function_args=[eval('x'), eval('f(x)')], function_kwargs={'marker': eval("'o'")}, max_wait_secs=0, custom_class=None)
custom_method(
ax1.plot(x, interpolated_path(x), marker='>'), imports='import matplotlib.pylab as plt;import tensorflow as tf;import numpy as np;import tensorflow_hub as hub', function_to_run='obj.plot(*args, **kwargs)', method_object=eval('ax1'), function_args=[eval('x'), eval('interpolated_path(x)')], function_kwargs={'marker': eval("'>'")}, max_wait_secs=0, custom_class=None)
custom_method(
ax1.set_title('IG intuition', fontweight='bold'), imports='import matplotlib.pylab as plt;import tensorflow as tf;import numpy as np;import tensorflow_hub as hub', function_to_run='obj.set_title(*args, **kwargs)', method_object=eval('ax1'), function_args=[eval("'IG intuition'")], function_kwargs={'fontweight': eval("'bold'")}, max_wait_secs=0, custom_class=None)
custom_method(
ax1.text(0.25, 0.1, 'Accumulate gradients along path'), imports='import matplotlib.pylab as plt;import tensorflow as tf;import numpy as np;import tensorflow_hub as hub', function_to_run='obj.text(*args)', method_object=eval('ax1'), function_args=[eval('0.25'), eval('0.1'), eval("'Accumulate gradients along path'")], function_kwargs={}, max_wait_secs=0, custom_class=None)
custom_method(
ax1.set_ylabel('F(x) - model true class predicted probability'), imports='import matplotlib.pylab as plt;import tensorflow as tf;import numpy as np;import tensorflow_hub as hub', function_to_run='obj.set_ylabel(*args)', method_object=eval('ax1'), function_args=[eval("'F(x) - model true class predicted probability'")], function_kwargs={}, max_wait_secs=0, custom_class=None)
custom_method(
ax1.set_xlabel('x - (pixel value)'), imports='import matplotlib.pylab as plt;import tensorflow as tf;import numpy as np;import tensorflow_hub as hub', function_to_run='obj.set_xlabel(*args)', method_object=eval('ax1'), function_args=[eval("'x - (pixel value)'")], function_kwargs={}, max_wait_secs=0, custom_class=None)
custom_method(
ax1.set_yticks(tf.range(0, 1.5, 0.5)), imports='import matplotlib.pylab as plt;import tensorflow as tf;import numpy as np;import tensorflow_hub as hub', function_to_run='obj.set_yticks(*args)', method_object=eval('ax1'), function_args=[eval('tf.range(0, 1.5, 0.5)')], function_kwargs={}, max_wait_secs=0, custom_class=None)
custom_method(
ax1.set_xticks(tf.range(0, 1.5, 0.5)), imports='import matplotlib.pylab as plt;import tensorflow as tf;import numpy as np;import tensorflow_hub as hub', function_to_run='obj.set_xticks(*args)', method_object=eval('ax1'), function_args=[eval('tf.range(0, 1.5, 0.5)')], function_kwargs={}, max_wait_secs=0, custom_class=None)
custom_method(
ax1.annotate('Baseline', xy=(0.0, 0.0), xytext=(0.0, 0.2), arrowprops=dict(facecolor='black', shrink=0.1)), imports='import matplotlib.pylab as plt;import tensorflow as tf;import numpy as np;import tensorflow_hub as hub', function_to_run='obj.annotate(*args, **kwargs)', method_object=eval('ax1'), function_args=[eval("'Baseline'")], function_kwargs={'xy': eval('(0.0, 0.0)'), 'xytext': eval('(0.0, 0.2)'), 'arrowprops': eval("dict(facecolor='black', shrink=0.1)")}, max_wait_secs=0, custom_class=None)
custom_method(
ax1.annotate('Input', xy=(1.0, 0.0), xytext=(0.95, 0.2), arrowprops=dict(facecolor='black', shrink=0.1)), imports='import matplotlib.pylab as plt;import tensorflow as tf;import numpy as np;import tensorflow_hub as hub', function_to_run='obj.annotate(*args, **kwargs)', method_object=eval('ax1'), function_args=[eval("'Input'")], function_kwargs={'xy': eval('(1.0, 0.0)'), 'xytext': eval('(0.95, 0.2)'), 'arrowprops': eval("dict(facecolor='black', shrink=0.1)")}, max_wait_secs=0, custom_class=None)
plt.show()
baseline = custom_method(
tf.zeros(shape=(224, 224, 3)), imports='import matplotlib.pylab as plt;import tensorflow as tf;import numpy as np;import tensorflow_hub as hub', function_to_run='tf.zeros(**kwargs)', method_object=None, function_args=[], function_kwargs={'shape': eval('(224,224,3)')}, max_wait_secs=0)
plt.imshow(baseline)
plt.title('Baseline')
plt.axis('off')
plt.show()
m_steps = 50
alphas = custom_method(
tf.linspace(start=0.0, stop=1.0, num=m_steps + 1), imports='import matplotlib.pylab as plt;import tensorflow as tf;import numpy as np;import tensorflow_hub as hub', function_to_run='tf.linspace(**kwargs)', method_object=None, function_args=[], function_kwargs={'start': eval('0.0'), 'stop': eval('1.0'), 'num': eval('m_steps+1')}, max_wait_secs=0)

def interpolate_images(baseline, image, alphas):
    alphas_x = alphas[:, tf.newaxis, tf.newaxis, tf.newaxis]
    baseline_x = custom_method(
    tf.expand_dims(baseline, axis=0), imports='import matplotlib.pylab as plt;import tensorflow as tf;import numpy as np;import tensorflow_hub as hub', function_to_run='tf.expand_dims(*args, **kwargs)', method_object=None, function_args=[eval('baseline')], function_kwargs={'axis': eval('0')}, max_wait_secs=0)
    input_x = custom_method(
    tf.expand_dims(image, axis=0), imports='import matplotlib.pylab as plt;import tensorflow as tf;import numpy as np;import tensorflow_hub as hub', function_to_run='tf.expand_dims(*args, **kwargs)', method_object=None, function_args=[eval('image')], function_kwargs={'axis': eval('0')}, max_wait_secs=0)
    delta = input_x - baseline_x
    images = baseline_x + alphas_x * delta
    return images
interpolated_images = custom_method(
interpolate_images(baseline=baseline, image=img_name_tensors['Fireboat'], alphas=alphas), imports='import matplotlib.pylab as plt;import tensorflow as tf;import numpy as np;import tensorflow_hub as hub', function_to_run='obj(**kwargs)', method_object=eval('interpolate_images'), function_args=[], function_kwargs={'baseline': eval('baseline'), 'image': eval("img_name_tensors['Fireboat']"), 'alphas': eval('alphas')}, max_wait_secs=0, custom_class=None)
fig = plt.figure(figsize=(20, 20))
i = 0
for (alpha, image) in zip(alphas[0::10], interpolated_images[0::10]):
    i += 1
    plt.subplot(1, len(alphas[0::10]), i)
    plt.title(f'alpha: {alpha:.1f}')
    plt.imshow(image)
    plt.axis('off')
plt.tight_layout()

def compute_gradients(images, target_class_idx):
    with custom_method(
    tf.GradientTape(), imports='import matplotlib.pylab as plt;import tensorflow as tf;import numpy as np;import tensorflow_hub as hub', function_to_run='tf.GradientTape()', method_object=None, function_args=[], function_kwargs={}, max_wait_secs=0) as tape:
        tape.watch(images)
        logits = custom_method(
        model(images), imports='import matplotlib.pylab as plt;import tensorflow as tf;import numpy as np;import tensorflow_hub as hub', function_to_run='obj(*args)', method_object=eval('model'), function_args=[eval('images')], function_kwargs={}, max_wait_secs=0, custom_class=None)
        probs = custom_method(
        tf.nn.softmax(logits, axis=-1), imports='import matplotlib.pylab as plt;import tensorflow as tf;import numpy as np;import tensorflow_hub as hub', function_to_run='tf.nn.softmax(*args, **kwargs)', method_object=None, function_args=[eval('logits')], function_kwargs={'axis': eval('-1')}, max_wait_secs=0)[:, target_class_idx]
    return tape.gradient(probs, images)
path_gradients = compute_gradients(images=interpolated_images, target_class_idx=555)
print(path_gradients.shape)
pred = custom_method(
model(interpolated_images), imports='import matplotlib.pylab as plt;import tensorflow as tf;import numpy as np;import tensorflow_hub as hub', function_to_run='obj(*args)', method_object=eval('model'), function_args=[eval('interpolated_images')], function_kwargs={}, max_wait_secs=0, custom_class=None)
pred_proba = custom_method(
tf.nn.softmax(pred, axis=-1), imports='import matplotlib.pylab as plt;import tensorflow as tf;import numpy as np;import tensorflow_hub as hub', function_to_run='tf.nn.softmax(*args, **kwargs)', method_object=None, function_args=[eval('pred')], function_kwargs={'axis': eval('-1')}, max_wait_secs=0)[:, 555]
plt.figure(figsize=(10, 4))
ax1 = plt.subplot(1, 2, 1)
custom_method(
ax1.plot(alphas, pred_proba), imports='import matplotlib.pylab as plt;import tensorflow as tf;import numpy as np;import tensorflow_hub as hub', function_to_run='obj.plot(*args)', method_object=eval('ax1'), function_args=[eval('alphas'), eval('pred_proba')], function_kwargs={}, max_wait_secs=0, custom_class=None)
custom_method(
ax1.set_title('Target class predicted probability over alpha'), imports='import matplotlib.pylab as plt;import tensorflow as tf;import numpy as np;import tensorflow_hub as hub', function_to_run='obj.set_title(*args)', method_object=eval('ax1'), function_args=[eval("'Target class predicted probability over alpha'")], function_kwargs={}, max_wait_secs=0, custom_class=None)
custom_method(
ax1.set_ylabel('model p(target class)'), imports='import matplotlib.pylab as plt;import tensorflow as tf;import numpy as np;import tensorflow_hub as hub', function_to_run='obj.set_ylabel(*args)', method_object=eval('ax1'), function_args=[eval("'model p(target class)'")], function_kwargs={}, max_wait_secs=0, custom_class=None)
custom_method(
ax1.set_xlabel('alpha'), imports='import matplotlib.pylab as plt;import tensorflow as tf;import numpy as np;import tensorflow_hub as hub', function_to_run='obj.set_xlabel(*args)', method_object=eval('ax1'), function_args=[eval("'alpha'")], function_kwargs={}, max_wait_secs=0, custom_class=None)
custom_method(
ax1.set_ylim([0, 1]), imports='import matplotlib.pylab as plt;import tensorflow as tf;import numpy as np;import tensorflow_hub as hub', function_to_run='obj.set_ylim(*args)', method_object=eval('ax1'), function_args=[eval('[0, 1]')], function_kwargs={}, max_wait_secs=0, custom_class=None)
ax2 = plt.subplot(1, 2, 2)
average_grads = custom_method(
tf.reduce_mean(path_gradients, axis=[1, 2, 3]), imports='import matplotlib.pylab as plt;import tensorflow as tf;import numpy as np;import tensorflow_hub as hub', function_to_run='tf.reduce_mean(*args, **kwargs)', method_object=None, function_args=[eval('path_gradients')], function_kwargs={'axis': eval('[1, 2, 3]')}, max_wait_secs=0)
average_grads_norm = (average_grads - custom_method(
tf.math.reduce_min(average_grads), imports='import matplotlib.pylab as plt;import tensorflow as tf;import numpy as np;import tensorflow_hub as hub', function_to_run='tf.math.reduce_min(*args)', method_object=None, function_args=[eval('average_grads')], function_kwargs={}, max_wait_secs=0)) / (custom_method(
tf.math.reduce_max(average_grads), imports='import matplotlib.pylab as plt;import tensorflow as tf;import numpy as np;import tensorflow_hub as hub', function_to_run='tf.math.reduce_max(*args)', method_object=None, function_args=[eval('average_grads')], function_kwargs={}, max_wait_secs=0) - custom_method(
tf.reduce_min(average_grads), imports='import matplotlib.pylab as plt;import tensorflow as tf;import numpy as np;import tensorflow_hub as hub', function_to_run='tf.reduce_min(*args)', method_object=None, function_args=[eval('average_grads')], function_kwargs={}, max_wait_secs=0))
custom_method(
ax2.plot(alphas, average_grads_norm), imports='import matplotlib.pylab as plt;import tensorflow as tf;import numpy as np;import tensorflow_hub as hub', function_to_run='obj.plot(*args)', method_object=eval('ax2'), function_args=[eval('alphas'), eval('average_grads_norm')], function_kwargs={}, max_wait_secs=0, custom_class=None)
custom_method(
ax2.set_title('Average pixel gradients (normalized) over alpha'), imports='import matplotlib.pylab as plt;import tensorflow as tf;import numpy as np;import tensorflow_hub as hub', function_to_run='obj.set_title(*args)', method_object=eval('ax2'), function_args=[eval("'Average pixel gradients (normalized) over alpha'")], function_kwargs={}, max_wait_secs=0, custom_class=None)
custom_method(
ax2.set_ylabel('Average pixel gradients'), imports='import matplotlib.pylab as plt;import tensorflow as tf;import numpy as np;import tensorflow_hub as hub', function_to_run='obj.set_ylabel(*args)', method_object=eval('ax2'), function_args=[eval("'Average pixel gradients'")], function_kwargs={}, max_wait_secs=0, custom_class=None)
custom_method(
ax2.set_xlabel('alpha'), imports='import matplotlib.pylab as plt;import tensorflow as tf;import numpy as np;import tensorflow_hub as hub', function_to_run='obj.set_xlabel(*args)', method_object=eval('ax2'), function_args=[eval("'alpha'")], function_kwargs={}, max_wait_secs=0, custom_class=None)
custom_method(
ax2.set_ylim([0, 1]), imports='import matplotlib.pylab as plt;import tensorflow as tf;import numpy as np;import tensorflow_hub as hub', function_to_run='obj.set_ylim(*args)', method_object=eval('ax2'), function_args=[eval('[0, 1]')], function_kwargs={}, max_wait_secs=0, custom_class=None)

def integral_approximation(gradients):
    grads = (gradients[:-1] + gradients[1:]) / custom_method(
    tf.constant(2.0), imports='import matplotlib.pylab as plt;import tensorflow as tf;import numpy as np;import tensorflow_hub as hub', function_to_run='tf.constant(*args)', method_object=None, function_args=[eval('2.0')], function_kwargs={}, max_wait_secs=0)
    integrated_gradients = custom_method(
    tf.math.reduce_mean(grads, axis=0), imports='import matplotlib.pylab as plt;import tensorflow as tf;import numpy as np;import tensorflow_hub as hub', function_to_run='tf.math.reduce_mean(*args, **kwargs)', method_object=None, function_args=[eval('grads')], function_kwargs={'axis': eval('0')}, max_wait_secs=0)
    return integrated_gradients
ig = custom_method(
integral_approximation(gradients=path_gradients), imports='import matplotlib.pylab as plt;import tensorflow as tf;import numpy as np;import tensorflow_hub as hub', function_to_run='obj(**kwargs)', method_object=eval('integral_approximation'), function_args=[], function_kwargs={'gradients': eval('path_gradients')}, max_wait_secs=0, custom_class=None)
print(ig.shape)

def integrated_gradients(baseline, image, target_class_idx, m_steps=50, batch_size=32):
    alphas = custom_method(
    tf.linspace(start=0.0, stop=1.0, num=m_steps + 1), imports='import matplotlib.pylab as plt;import tensorflow as tf;import numpy as np;import tensorflow_hub as hub', function_to_run='tf.linspace(**kwargs)', method_object=None, function_args=[], function_kwargs={'start': eval('0.0'), 'stop': eval('1.0'), 'num': eval('m_steps+1')}, max_wait_secs=0)
    gradient_batches = []
    for alpha in custom_method(
    tf.range(0, len(alphas), batch_size), imports='import matplotlib.pylab as plt;import tensorflow as tf;import numpy as np;import tensorflow_hub as hub', function_to_run='tf.range(*args)', method_object=None, function_args=[eval('0'), eval('len(alphas)'), eval('batch_size')], function_kwargs={}, max_wait_secs=0):
        from_ = alpha
        to = custom_method(
        tf.minimum(from_ + batch_size, len(alphas)), imports='import matplotlib.pylab as plt;import tensorflow as tf;import numpy as np;import tensorflow_hub as hub', function_to_run='tf.minimum(*args)', method_object=None, function_args=[eval('from_ + batch_size'), eval('len(alphas)')], function_kwargs={}, max_wait_secs=0)
        alpha_batch = alphas[from_:to]
        gradient_batch = one_batch(baseline, image, alpha_batch, target_class_idx)
        gradient_batches.append(gradient_batch)
    total_gradients = custom_method(
    tf.concat(gradient_batches, axis=0), imports='import matplotlib.pylab as plt;import tensorflow as tf;import numpy as np;import tensorflow_hub as hub', function_to_run='tf.concat(*args, **kwargs)', method_object=None, function_args=[eval('gradient_batches')], function_kwargs={'axis': eval('0')}, max_wait_secs=0)
    avg_gradients = custom_method(
    integral_approximation(gradients=total_gradients), imports='import matplotlib.pylab as plt;import tensorflow as tf;import numpy as np;import tensorflow_hub as hub', function_to_run='obj(**kwargs)', method_object=eval('integral_approximation'), function_args=[], function_kwargs={'gradients': eval('total_gradients')}, max_wait_secs=0, custom_class=None)
    integrated_gradients = (image - baseline) * avg_gradients
    return integrated_gradients

@tf.function
def one_batch(baseline, image, alpha_batch, target_class_idx):
    interpolated_path_input_batch = custom_method(
    interpolate_images(baseline=baseline, image=image, alphas=alpha_batch), imports='import matplotlib.pylab as plt;import tensorflow as tf;import numpy as np;import tensorflow_hub as hub', function_to_run='obj(**kwargs)', method_object=eval('interpolate_images'), function_args=[], function_kwargs={'baseline': eval('baseline'), 'image': eval('image'), 'alphas': eval('alpha_batch')}, max_wait_secs=0, custom_class=None)
    gradient_batch = compute_gradients(images=interpolated_path_input_batch, target_class_idx=target_class_idx)
    return gradient_batch
ig_attributions = custom_method(
integrated_gradients(baseline=baseline, image=img_name_tensors['Fireboat'], target_class_idx=555, m_steps=240), imports='import matplotlib.pylab as plt;import tensorflow as tf;import numpy as np;import tensorflow_hub as hub', function_to_run='obj(**kwargs)', method_object=eval('integrated_gradients'), function_args=[], function_kwargs={'baseline': eval('baseline'), 'image': eval("img_name_tensors['Fireboat']"), 'target_class_idx': eval('555'), 'm_steps': eval('240')}, max_wait_secs=0, custom_class=None)
print(ig_attributions.shape)

def plot_img_attributions(baseline, image, target_class_idx, m_steps=50, cmap=None, overlay_alpha=0.4):
    attributions = custom_method(
    integrated_gradients(baseline=baseline, image=image, target_class_idx=target_class_idx, m_steps=m_steps), imports='import matplotlib.pylab as plt;import tensorflow as tf;import numpy as np;import tensorflow_hub as hub', function_to_run='obj(**kwargs)', method_object=eval('integrated_gradients'), function_args=[], function_kwargs={'baseline': eval('baseline'), 'image': eval('image'), 'target_class_idx': eval('target_class_idx'), 'm_steps': eval('m_steps')}, max_wait_secs=0, custom_class=None)
    attribution_mask = custom_method(
    tf.reduce_sum(tf.math.abs(attributions), axis=-1), imports='import matplotlib.pylab as plt;import tensorflow as tf;import numpy as np;import tensorflow_hub as hub', function_to_run='tf.reduce_sum(*args, **kwargs)', method_object=None, function_args=[eval('tf.math.abs(attributions)')], function_kwargs={'axis': eval('-1')}, max_wait_secs=0)
    (fig, axs) = plt.subplots(nrows=2, ncols=2, squeeze=False, figsize=(8, 8))
    custom_method(
    axs[0, 0].set_title('Baseline image'), imports='import matplotlib.pylab as plt;import tensorflow as tf;import numpy as np;import tensorflow_hub as hub', function_to_run='obj[0, 0].set_title(*args)', method_object=eval('axs'), function_args=[eval("'Baseline image'")], function_kwargs={}, max_wait_secs=0, custom_class=None)
    custom_method(
    axs[0, 0].imshow(baseline), imports='import matplotlib.pylab as plt;import tensorflow as tf;import numpy as np;import tensorflow_hub as hub', function_to_run='obj[0, 0].imshow(*args)', method_object=eval('axs'), function_args=[eval('baseline')], function_kwargs={}, max_wait_secs=0, custom_class=None)
    custom_method(
    axs[0, 0].axis('off'), imports='import matplotlib.pylab as plt;import tensorflow as tf;import numpy as np;import tensorflow_hub as hub', function_to_run='obj[0, 0].axis(*args)', method_object=eval('axs'), function_args=[eval("'off'")], function_kwargs={}, max_wait_secs=0, custom_class=None)
    custom_method(
    axs[0, 1].set_title('Original image'), imports='import matplotlib.pylab as plt;import tensorflow as tf;import numpy as np;import tensorflow_hub as hub', function_to_run='obj[0, 1].set_title(*args)', method_object=eval('axs'), function_args=[eval("'Original image'")], function_kwargs={}, max_wait_secs=0, custom_class=None)
    custom_method(
    axs[0, 1].imshow(image), imports='import matplotlib.pylab as plt;import tensorflow as tf;import numpy as np;import tensorflow_hub as hub', function_to_run='obj[0, 1].imshow(*args)', method_object=eval('axs'), function_args=[eval('image')], function_kwargs={}, max_wait_secs=0, custom_class=None)
    custom_method(
    axs[0, 1].axis('off'), imports='import matplotlib.pylab as plt;import tensorflow as tf;import numpy as np;import tensorflow_hub as hub', function_to_run='obj[0, 1].axis(*args)', method_object=eval('axs'), function_args=[eval("'off'")], function_kwargs={}, max_wait_secs=0, custom_class=None)
    custom_method(
    axs[1, 0].set_title('Attribution mask'), imports='import matplotlib.pylab as plt;import tensorflow as tf;import numpy as np;import tensorflow_hub as hub', function_to_run='obj[1, 0].set_title(*args)', method_object=eval('axs'), function_args=[eval("'Attribution mask'")], function_kwargs={}, max_wait_secs=0, custom_class=None)
    custom_method(
    axs[1, 0].imshow(attribution_mask, cmap=cmap), imports='import matplotlib.pylab as plt;import tensorflow as tf;import numpy as np;import tensorflow_hub as hub', function_to_run='obj[1, 0].imshow(*args, **kwargs)', method_object=eval('axs'), function_args=[eval('attribution_mask')], function_kwargs={'cmap': eval('cmap')}, max_wait_secs=0, custom_class=None)
    custom_method(
    axs[1, 0].axis('off'), imports='import matplotlib.pylab as plt;import tensorflow as tf;import numpy as np;import tensorflow_hub as hub', function_to_run='obj[1, 0].axis(*args)', method_object=eval('axs'), function_args=[eval("'off'")], function_kwargs={}, max_wait_secs=0, custom_class=None)
    custom_method(
    axs[1, 1].set_title('Overlay'), imports='import matplotlib.pylab as plt;import tensorflow as tf;import numpy as np;import tensorflow_hub as hub', function_to_run='obj[1, 1].set_title(*args)', method_object=eval('axs'), function_args=[eval("'Overlay'")], function_kwargs={}, max_wait_secs=0, custom_class=None)
    custom_method(
    axs[1, 1].imshow(attribution_mask, cmap=cmap), imports='import matplotlib.pylab as plt;import tensorflow as tf;import numpy as np;import tensorflow_hub as hub', function_to_run='obj[1, 1].imshow(*args, **kwargs)', method_object=eval('axs'), function_args=[eval('attribution_mask')], function_kwargs={'cmap': eval('cmap')}, max_wait_secs=0, custom_class=None)
    custom_method(
    axs[1, 1].imshow(image, alpha=overlay_alpha), imports='import matplotlib.pylab as plt;import tensorflow as tf;import numpy as np;import tensorflow_hub as hub', function_to_run='obj[1, 1].imshow(*args, **kwargs)', method_object=eval('axs'), function_args=[eval('image')], function_kwargs={'alpha': eval('overlay_alpha')}, max_wait_secs=0, custom_class=None)
    custom_method(
    axs[1, 1].axis('off'), imports='import matplotlib.pylab as plt;import tensorflow as tf;import numpy as np;import tensorflow_hub as hub', function_to_run='obj[1, 1].axis(*args)', method_object=eval('axs'), function_args=[eval("'off'")], function_kwargs={}, max_wait_secs=0, custom_class=None)
    plt.tight_layout()
    return fig
_ = plot_img_attributions(image=img_name_tensors['Fireboat'], baseline=baseline, target_class_idx=555, m_steps=240, cmap=plt.cm.inferno, overlay_alpha=0.4)
_ = plot_img_attributions(image=img_name_tensors['Giant Panda'], baseline=baseline, target_class_idx=389, m_steps=55, cmap=plt.cm.viridis, overlay_alpha=0.5)
