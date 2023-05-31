import tensorflow as tf
import matplotlib as mpl
import matplotlib.pyplot as plt
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
mpl.rcParams['figure.figsize'] = (8, 8)
mpl.rcParams['axes.grid'] = False
custom_method(imports='import matplotlib.pyplot as plt;import matplotlib as mpl;import tensorflow as tf', function_to_run='tf.keras.applications.MobileNetV2(**kwargs)', method_object=None, object_signature=None, function_args=[], function_kwargs={'include_top': eval('True'), 'weights': eval("'imagenet'")})
pretrained_model = tf.keras.applications.MobileNetV2(include_top=True, weights='imagenet')
pretrained_model.trainable = False
decode_predictions = tf.keras.applications.mobilenet_v2.decode_predictions

def preprocess(image):
    custom_method(imports='import matplotlib.pyplot as plt;import matplotlib as mpl;import tensorflow as tf', function_to_run='tf.cast(*args)', method_object=None, object_signature=None, function_args=[eval('image'), eval('tf.float32')], function_kwargs={})
    image = tf.cast(image, tf.float32)
    custom_method(imports='import matplotlib.pyplot as plt;import matplotlib as mpl;import tensorflow as tf', function_to_run='tf.image.resize(*args)', method_object=None, object_signature=None, function_args=[eval('image'), eval('(224, 224)')], function_kwargs={})
    image = tf.image.resize(image, (224, 224))
    custom_method(imports='import matplotlib.pyplot as plt;import matplotlib as mpl;import tensorflow as tf', function_to_run='tf.keras.applications.mobilenet_v2.preprocess_input(*args)', method_object=None, object_signature=None, function_args=[eval('image')], function_kwargs={})
    image = tf.keras.applications.mobilenet_v2.preprocess_input(image)
    image = image[None, ...]
    return image

def get_imagenet_label(probs):
    return decode_predictions(probs, top=1)[0][0]
custom_method(imports='import matplotlib.pyplot as plt;import matplotlib as mpl;import tensorflow as tf', function_to_run='tf.keras.utils.get_file(*args)', method_object=None, object_signature=None, function_args=[eval("'YellowLabradorLooking_new.jpg'"), eval("'https://storage.googleapis.com/download.tensorflow.org/example_images/YellowLabradorLooking_new.jpg'")], function_kwargs={})
image_path = tf.keras.utils.get_file('YellowLabradorLooking_new.jpg', 'https://storage.googleapis.com/download.tensorflow.org/example_images/YellowLabradorLooking_new.jpg')
custom_method(imports='import matplotlib.pyplot as plt;import matplotlib as mpl;import tensorflow as tf', function_to_run='tf.io.read_file(*args)', method_object=None, object_signature=None, function_args=[eval('image_path')], function_kwargs={})
image_raw = tf.io.read_file(image_path)
custom_method(imports='import matplotlib.pyplot as plt;import matplotlib as mpl;import tensorflow as tf', function_to_run='tf.image.decode_image(*args)', method_object=None, object_signature=None, function_args=[eval('image_raw')], function_kwargs={})
image = tf.image.decode_image(image_raw)
image = preprocess(image)
custom_method(imports='import matplotlib.pyplot as plt;import matplotlib as mpl;import tensorflow as tf', function_to_run='obj.predict(*args)', method_object=eval('pretrained_model'), object_signature=None, function_args=[eval('image')], function_kwargs={}, custom_class=None)
image_probs = pretrained_model.predict(image)
plt.figure()
plt.imshow(image[0] * 0.5 + 0.5)
(_, image_class, class_confidence) = get_imagenet_label(image_probs)
plt.title('{} : {:.2f}% Confidence'.format(image_class, class_confidence * 100))
plt.show()
custom_method(imports='import matplotlib.pyplot as plt;import matplotlib as mpl;import tensorflow as tf', function_to_run='tf.keras.losses.CategoricalCrossentropy()', method_object=None, object_signature=None, function_args=[], function_kwargs={})
loss_object = tf.keras.losses.CategoricalCrossentropy()

def create_adversarial_pattern(input_image, input_label):
    with tf.GradientTape() as tape:
        tape.watch(input_image)
        custom_method(imports='import matplotlib.pyplot as plt;import matplotlib as mpl;import tensorflow as tf', function_to_run='obj(*args)', method_object=eval('pretrained_model'), object_signature=None, function_args=[eval('input_image')], function_kwargs={}, custom_class=None)
        prediction = pretrained_model(input_image)
        custom_method(imports='import matplotlib.pyplot as plt;import matplotlib as mpl;import tensorflow as tf', function_to_run='obj(*args)', method_object=eval('loss_object'), object_signature=None, function_args=[eval('input_label'), eval('prediction')], function_kwargs={}, custom_class=None)
        loss = loss_object(input_label, prediction)
    gradient = tape.gradient(loss, input_image)
    custom_method(imports='import matplotlib.pyplot as plt;import matplotlib as mpl;import tensorflow as tf', function_to_run='tf.sign(*args)', method_object=None, object_signature=None, function_args=[eval('gradient')], function_kwargs={})
    signed_grad = tf.sign(gradient)
    return signed_grad
labrador_retriever_index = 208
custom_method(imports='import matplotlib.pyplot as plt;import matplotlib as mpl;import tensorflow as tf', function_to_run='tf.one_hot(*args)', method_object=None, object_signature=None, function_args=[eval('labrador_retriever_index'), eval('image_probs.shape[-1]')], function_kwargs={})
label = tf.one_hot(labrador_retriever_index, image_probs.shape[-1])
custom_method(imports='import matplotlib.pyplot as plt;import matplotlib as mpl;import tensorflow as tf', function_to_run='tf.reshape(*args)', method_object=None, object_signature=None, function_args=[eval('label'), eval('(1, image_probs.shape[-1])')], function_kwargs={})
label = tf.reshape(label, (1, image_probs.shape[-1]))
perturbations = create_adversarial_pattern(image, label)
plt.imshow(perturbations[0] * 0.5 + 0.5)

def display_images(image, description):
    (_, label, confidence) = get_imagenet_label(pretrained_model.predict(image))
    plt.figure()
    plt.imshow(image[0] * 0.5 + 0.5)
    plt.title('{} \n {} : {:.2f}% Confidence'.format(description, label, confidence * 100))
    plt.show()
epsilons = [0, 0.01, 0.1, 0.15]
descriptions = ['Epsilon = {:0.3f}'.format(eps) if eps else 'Input' for eps in epsilons]
for (i, eps) in enumerate(epsilons):
    adv_x = image + eps * perturbations
    custom_method(imports='import matplotlib.pyplot as plt;import matplotlib as mpl;import tensorflow as tf', function_to_run='tf.clip_by_value(*args)', method_object=None, object_signature=None, function_args=[eval('adv_x'), eval('-1'), eval('1')], function_kwargs={})
    adv_x = tf.clip_by_value(adv_x, -1, 1)
    display_images(adv_x, descriptions[i])
