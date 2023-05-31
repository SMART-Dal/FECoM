import matplotlib.pylab as plt
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
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
custom_method(imports='import tensorflow as tf;import tensorflow_hub as hub;import numpy as np;import matplotlib.pylab as plt', function_to_run='tf.keras.Sequential(*args)', method_object=None, object_signature=None, function_args=[eval("[\n    hub.KerasLayer(\n        name='inception_v1',\n        handle='https://tfhub.dev/google/imagenet/inception_v1/classification/4',\n        trainable=False),\n]")], function_kwargs={})
model = tf.keras.Sequential([hub.KerasLayer(name='inception_v1', handle='https://tfhub.dev/google/imagenet/inception_v1/classification/4', trainable=False)])
custom_method(imports='import tensorflow as tf;import tensorflow_hub as hub;import numpy as np;import matplotlib.pylab as plt', function_to_run='obj.build(*args)', method_object=eval('model'), object_signature=None, function_args=[eval('[None, 224, 224, 3]')], function_kwargs={}, custom_class=None)
model.build([None, 224, 224, 3])
custom_method(imports='import tensorflow as tf;import tensorflow_hub as hub;import numpy as np;import matplotlib.pylab as plt', function_to_run='obj.summary()', method_object=eval('model'), object_signature=None, function_args=[], function_kwargs={}, custom_class=None)
model.summary()

def load_imagenet_labels(file_path):
    custom_method(imports='import tensorflow as tf;import tensorflow_hub as hub;import numpy as np;import matplotlib.pylab as plt', function_to_run='tf.keras.utils.get_file(*args)', method_object=None, object_signature=None, function_args=[eval("'ImageNetLabels.txt'"), eval('file_path')], function_kwargs={})
    labels_file = tf.keras.utils.get_file('ImageNetLabels.txt', file_path)
    with open(labels_file) as reader:
        f = reader.read()
        labels = f.splitlines()
    return np.array(labels)
imagenet_labels = load_imagenet_labels('https://storage.googleapis.com/download.tensorflow.org/data/ImageNetLabels.txt')

def read_image(file_name):
    custom_method(imports='import tensorflow as tf;import tensorflow_hub as hub;import numpy as np;import matplotlib.pylab as plt', function_to_run='tf.io.read_file(*args)', method_object=None, object_signature=None, function_args=[eval('file_name')], function_kwargs={})
    image = tf.io.read_file(file_name)
    custom_method(imports='import tensorflow as tf;import tensorflow_hub as hub;import numpy as np;import matplotlib.pylab as plt', function_to_run='tf.io.decode_jpeg(*args, **kwargs)', method_object=None, object_signature=None, function_args=[eval('image')], function_kwargs={'channels': eval('3')})
    image = tf.io.decode_jpeg(image, channels=3)
    custom_method(imports='import tensorflow as tf;import tensorflow_hub as hub;import numpy as np;import matplotlib.pylab as plt', function_to_run='tf.image.convert_image_dtype(*args)', method_object=None, object_signature=None, function_args=[eval('image'), eval('tf.float32')], function_kwargs={})
    image = tf.image.convert_image_dtype(image, tf.float32)
    custom_method(imports='import tensorflow as tf;import tensorflow_hub as hub;import numpy as np;import matplotlib.pylab as plt', function_to_run='tf.image.resize_with_pad(*args, **kwargs)', method_object=None, object_signature=None, function_args=[eval('image')], function_kwargs={'target_height': eval('224'), 'target_width': eval('224')})
    image = tf.image.resize_with_pad(image, target_height=224, target_width=224)
    return image
img_url = {'Fireboat': 'http://storage.googleapis.com/download.tensorflow.org/example_images/San_Francisco_fireboat_showing_off.jpg', 'Giant Panda': 'http://storage.googleapis.com/download.tensorflow.org/example_images/Giant_Panda_2.jpeg'}
img_paths = {name: tf.keras.utils.get_file(name, url) for (name, url) in img_url.items()}
img_name_tensors = {name: read_image(img_path) for (name, img_path) in img_paths.items()}
plt.figure(figsize=(8, 8))
for (n, (name, img_tensors)) in enumerate(img_name_tensors.items()):
    ax = plt.subplot(1, 2, n + 1)
    ax.imshow(img_tensors)
    ax.set_title(name)
    ax.axis('off')
plt.tight_layout()

def top_k_predictions(img, k=3):
    custom_method(imports='import tensorflow as tf;import tensorflow_hub as hub;import numpy as np;import matplotlib.pylab as plt', function_to_run='tf.expand_dims(*args)', method_object=None, object_signature=None, function_args=[eval('img'), eval('0')], function_kwargs={})
    image_batch = tf.expand_dims(img, 0)
    custom_method(imports='import tensorflow as tf;import tensorflow_hub as hub;import numpy as np;import matplotlib.pylab as plt', function_to_run='obj(*args)', method_object=eval('model'), object_signature=None, function_args=[eval('image_batch')], function_kwargs={}, custom_class=None)
    predictions = model(image_batch)
    custom_method(imports='import tensorflow as tf;import tensorflow_hub as hub;import numpy as np;import matplotlib.pylab as plt', function_to_run='tf.nn.softmax(*args, **kwargs)', method_object=None, object_signature=None, function_args=[eval('predictions')], function_kwargs={'axis': eval('-1')})
    probs = tf.nn.softmax(predictions, axis=-1)
    custom_method(imports='import tensorflow as tf;import tensorflow_hub as hub;import numpy as np;import matplotlib.pylab as plt', function_to_run='tf.math.top_k(**kwargs)', method_object=None, object_signature=None, function_args=[], function_kwargs={'input': eval('probs'), 'k': eval('k')})
    (top_probs, top_idxs) = tf.math.top_k(input=probs, k=k)
    top_labels = imagenet_labels[tuple(top_idxs)]
    return (top_labels, top_probs[0])
for (name, img_tensor) in img_name_tensors.items():
    plt.imshow(img_tensor)
    plt.title(name, fontweight='bold')
    plt.axis('off')
    plt.show()
    (pred_label, pred_prob) = top_k_predictions(img_tensor)
    for (label, prob) in zip(pred_label, pred_prob):
        print(f'{label}: {prob:0.1%}')

def f(x):
    """A simplified model function."""
    return tf.where(x < 0.8, x, 0.8)

def interpolated_path(x):
    """A straight line path."""
    return tf.zeros_like(x)
custom_method(imports='import tensorflow as tf;import tensorflow_hub as hub;import numpy as np;import matplotlib.pylab as plt', function_to_run='tf.linspace(**kwargs)', method_object=None, object_signature=None, function_args=[], function_kwargs={'start': eval('0.0'), 'stop': eval('1.0'), 'num': eval('6')})
x = tf.linspace(start=0.0, stop=1.0, num=6)
y = f(x)
fig = plt.figure(figsize=(12, 5))
ax0 = fig.add_subplot(121)
ax0.plot(x, f(x), marker='o')
ax0.set_title('Gradients saturate over F(x)', fontweight='bold')
ax0.text(0.2, 0.5, 'Gradients > 0 = \n x is important')
ax0.text(0.7, 0.85, 'Gradients = 0 \n x not important')
ax0.set_yticks(tf.range(0, 1.5, 0.5))
ax0.set_xticks(tf.range(0, 1.5, 0.5))
ax0.set_ylabel('F(x) - model true class predicted probability')
ax0.set_xlabel('x - (pixel value)')
ax1 = fig.add_subplot(122)
ax1.plot(x, f(x), marker='o')
ax1.plot(x, interpolated_path(x), marker='>')
ax1.set_title('IG intuition', fontweight='bold')
ax1.text(0.25, 0.1, 'Accumulate gradients along path')
ax1.set_ylabel('F(x) - model true class predicted probability')
ax1.set_xlabel('x - (pixel value)')
ax1.set_yticks(tf.range(0, 1.5, 0.5))
ax1.set_xticks(tf.range(0, 1.5, 0.5))
ax1.annotate('Baseline', xy=(0.0, 0.0), xytext=(0.0, 0.2), arrowprops=dict(facecolor='black', shrink=0.1))
ax1.annotate('Input', xy=(1.0, 0.0), xytext=(0.95, 0.2), arrowprops=dict(facecolor='black', shrink=0.1))
plt.show()
custom_method(imports='import tensorflow as tf;import tensorflow_hub as hub;import numpy as np;import matplotlib.pylab as plt', function_to_run='tf.zeros(**kwargs)', method_object=None, object_signature=None, function_args=[], function_kwargs={'shape': eval('(224,224,3)')})
baseline = tf.zeros(shape=(224, 224, 3))
plt.imshow(baseline)
plt.title('Baseline')
plt.axis('off')
plt.show()
m_steps = 50
custom_method(imports='import tensorflow as tf;import tensorflow_hub as hub;import numpy as np;import matplotlib.pylab as plt', function_to_run='tf.linspace(**kwargs)', method_object=None, object_signature=None, function_args=[], function_kwargs={'start': eval('0.0'), 'stop': eval('1.0'), 'num': eval('m_steps+1')})
alphas = tf.linspace(start=0.0, stop=1.0, num=m_steps + 1)

def interpolate_images(baseline, image, alphas):
    alphas_x = alphas[:, tf.newaxis, tf.newaxis, tf.newaxis]
    custom_method(imports='import tensorflow as tf;import tensorflow_hub as hub;import numpy as np;import matplotlib.pylab as plt', function_to_run='tf.expand_dims(*args, **kwargs)', method_object=None, object_signature=None, function_args=[eval('baseline')], function_kwargs={'axis': eval('0')})
    baseline_x = tf.expand_dims(baseline, axis=0)
    custom_method(imports='import tensorflow as tf;import tensorflow_hub as hub;import numpy as np;import matplotlib.pylab as plt', function_to_run='tf.expand_dims(*args, **kwargs)', method_object=None, object_signature=None, function_args=[eval('image')], function_kwargs={'axis': eval('0')})
    input_x = tf.expand_dims(image, axis=0)
    delta = input_x - baseline_x
    images = baseline_x + alphas_x * delta
    return images
interpolated_images = interpolate_images(baseline=baseline, image=img_name_tensors['Fireboat'], alphas=alphas)
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
    with tf.GradientTape() as tape:
        tape.watch(images)
        custom_method(imports='import tensorflow as tf;import tensorflow_hub as hub;import numpy as np;import matplotlib.pylab as plt', function_to_run='obj(*args)', method_object=eval('model'), object_signature=None, function_args=[eval('images')], function_kwargs={}, custom_class=None)
        logits = model(images)
        probs = tf.nn.softmax(logits, axis=-1)[:, target_class_idx]
    return tape.gradient(probs, images)
path_gradients = compute_gradients(images=interpolated_images, target_class_idx=555)
print(path_gradients.shape)
custom_method(imports='import tensorflow as tf;import tensorflow_hub as hub;import numpy as np;import matplotlib.pylab as plt', function_to_run='obj(*args)', method_object=eval('model'), object_signature=None, function_args=[eval('interpolated_images')], function_kwargs={}, custom_class=None)
pred = model(interpolated_images)
pred_proba = tf.nn.softmax(pred, axis=-1)[:, 555]
plt.figure(figsize=(10, 4))
ax1 = plt.subplot(1, 2, 1)
ax1.plot(alphas, pred_proba)
ax1.set_title('Target class predicted probability over alpha')
ax1.set_ylabel('model p(target class)')
ax1.set_xlabel('alpha')
ax1.set_ylim([0, 1])
ax2 = plt.subplot(1, 2, 2)
custom_method(imports='import tensorflow as tf;import tensorflow_hub as hub;import numpy as np;import matplotlib.pylab as plt', function_to_run='tf.reduce_mean(*args, **kwargs)', method_object=None, object_signature=None, function_args=[eval('path_gradients')], function_kwargs={'axis': eval('[1, 2, 3]')})
average_grads = tf.reduce_mean(path_gradients, axis=[1, 2, 3])
average_grads_norm = (average_grads - tf.math.reduce_min(average_grads)) / (tf.math.reduce_max(average_grads) - tf.reduce_min(average_grads))
ax2.plot(alphas, average_grads_norm)
ax2.set_title('Average pixel gradients (normalized) over alpha')
ax2.set_ylabel('Average pixel gradients')
ax2.set_xlabel('alpha')
ax2.set_ylim([0, 1])

def integral_approximation(gradients):
    grads = (gradients[:-1] + gradients[1:]) / tf.constant(2.0)
    custom_method(imports='import tensorflow as tf;import tensorflow_hub as hub;import numpy as np;import matplotlib.pylab as plt', function_to_run='tf.math.reduce_mean(*args, **kwargs)', method_object=None, object_signature=None, function_args=[eval('grads')], function_kwargs={'axis': eval('0')})
    integrated_gradients = tf.math.reduce_mean(grads, axis=0)
    return integrated_gradients
ig = integral_approximation(gradients=path_gradients)
print(ig.shape)

def integrated_gradients(baseline, image, target_class_idx, m_steps=50, batch_size=32):
    custom_method(imports='import tensorflow as tf;import tensorflow_hub as hub;import numpy as np;import matplotlib.pylab as plt', function_to_run='tf.linspace(**kwargs)', method_object=None, object_signature=None, function_args=[], function_kwargs={'start': eval('0.0'), 'stop': eval('1.0'), 'num': eval('m_steps+1')})
    alphas = tf.linspace(start=0.0, stop=1.0, num=m_steps + 1)
    gradient_batches = []
    for alpha in tf.range(0, len(alphas), batch_size):
        from_ = alpha
        custom_method(imports='import tensorflow as tf;import tensorflow_hub as hub;import numpy as np;import matplotlib.pylab as plt', function_to_run='tf.minimum(*args)', method_object=None, object_signature=None, function_args=[eval('from_ + batch_size'), eval('len(alphas)')], function_kwargs={})
        to = tf.minimum(from_ + batch_size, len(alphas))
        alpha_batch = alphas[from_:to]
        gradient_batch = one_batch(baseline, image, alpha_batch, target_class_idx)
        gradient_batches.append(gradient_batch)
    custom_method(imports='import tensorflow as tf;import tensorflow_hub as hub;import numpy as np;import matplotlib.pylab as plt', function_to_run='tf.concat(*args, **kwargs)', method_object=None, object_signature=None, function_args=[eval('gradient_batches')], function_kwargs={'axis': eval('0')})
    total_gradients = tf.concat(gradient_batches, axis=0)
    avg_gradients = integral_approximation(gradients=total_gradients)
    integrated_gradients = (image - baseline) * avg_gradients
    return integrated_gradients

@tf.function
def one_batch(baseline, image, alpha_batch, target_class_idx):
    interpolated_path_input_batch = interpolate_images(baseline=baseline, image=image, alphas=alpha_batch)
    gradient_batch = compute_gradients(images=interpolated_path_input_batch, target_class_idx=target_class_idx)
    return gradient_batch
custom_method(imports='import tensorflow as tf;import tensorflow_hub as hub;import numpy as np;import matplotlib.pylab as plt', function_to_run='obj(**kwargs)', method_object=eval('integrated_gradients'), object_signature=None, function_args=[], function_kwargs={'baseline': eval('baseline'), 'image': eval("img_name_tensors['Fireboat']"), 'target_class_idx': eval('555'), 'm_steps': eval('240')}, custom_class=None)
ig_attributions = integrated_gradients(baseline=baseline, image=img_name_tensors['Fireboat'], target_class_idx=555, m_steps=240)
print(ig_attributions.shape)

def plot_img_attributions(baseline, image, target_class_idx, m_steps=50, cmap=None, overlay_alpha=0.4):
    custom_method(imports='import tensorflow as tf;import tensorflow_hub as hub;import numpy as np;import matplotlib.pylab as plt', function_to_run='obj(**kwargs)', method_object=eval('integrated_gradients'), object_signature=None, function_args=[], function_kwargs={'baseline': eval('baseline'), 'image': eval('image'), 'target_class_idx': eval('target_class_idx'), 'm_steps': eval('m_steps')}, custom_class=None)
    attributions = integrated_gradients(baseline=baseline, image=image, target_class_idx=target_class_idx, m_steps=m_steps)
    custom_method(imports='import tensorflow as tf;import tensorflow_hub as hub;import numpy as np;import matplotlib.pylab as plt', function_to_run='tf.reduce_sum(*args, **kwargs)', method_object=None, object_signature=None, function_args=[eval('tf.math.abs(attributions)')], function_kwargs={'axis': eval('-1')})
    attribution_mask = tf.reduce_sum(tf.math.abs(attributions), axis=-1)
    (fig, axs) = plt.subplots(nrows=2, ncols=2, squeeze=False, figsize=(8, 8))
    axs[0, 0].set_title('Baseline image')
    axs[0, 0].imshow(baseline)
    axs[0, 0].axis('off')
    axs[0, 1].set_title('Original image')
    axs[0, 1].imshow(image)
    axs[0, 1].axis('off')
    axs[1, 0].set_title('Attribution mask')
    axs[1, 0].imshow(attribution_mask, cmap=cmap)
    axs[1, 0].axis('off')
    axs[1, 1].set_title('Overlay')
    axs[1, 1].imshow(attribution_mask, cmap=cmap)
    axs[1, 1].imshow(image, alpha=overlay_alpha)
    axs[1, 1].axis('off')
    plt.tight_layout()
    return fig
_ = plot_img_attributions(image=img_name_tensors['Fireboat'], baseline=baseline, target_class_idx=555, m_steps=240, cmap=plt.cm.inferno, overlay_alpha=0.4)
_ = plot_img_attributions(image=img_name_tensors['Giant Panda'], baseline=baseline, target_class_idx=389, m_steps=55, cmap=plt.cm.viridis, overlay_alpha=0.5)
