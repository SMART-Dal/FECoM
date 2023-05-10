import tensorflow as tf
import matplotlib as mpl
import matplotlib.pyplot as plt
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
    return func
mpl.rcParams['figure.figsize'] = (8, 8)
mpl.rcParams['axes.grid'] = False
pretrained_model = custom_method(
tf.keras.applications.MobileNetV2(include_top=True, weights='imagenet'), imports='import matplotlib as mpl;import matplotlib.pyplot as plt;import tensorflow as tf', function_to_run='tf.keras.applications.MobileNetV2(**kwargs)', method_object=None, object_signature=None, function_args=[], function_kwargs={'include_top': eval('True'), 'weights': eval("'imagenet'")})
pretrained_model.trainable = False
decode_predictions = tf.keras.applications.mobilenet_v2.decode_predictions

def preprocess(image):
    image = custom_method(
    tf.cast(image, tf.float32), imports='import matplotlib as mpl;import matplotlib.pyplot as plt;import tensorflow as tf', function_to_run='tf.cast(*args)', method_object=None, object_signature=None, function_args=[eval('image'), eval('tf.float32')], function_kwargs={})
    image = custom_method(
    tf.image.resize(image, (224, 224)), imports='import matplotlib as mpl;import matplotlib.pyplot as plt;import tensorflow as tf', function_to_run='tf.image.resize(*args)', method_object=None, object_signature=None, function_args=[eval('image'), eval('(224, 224)')], function_kwargs={})
    image = custom_method(
    tf.keras.applications.mobilenet_v2.preprocess_input(image), imports='import matplotlib as mpl;import matplotlib.pyplot as plt;import tensorflow as tf', function_to_run='tf.keras.applications.mobilenet_v2.preprocess_input(*args)', method_object=None, object_signature=None, function_args=[eval('image')], function_kwargs={})
    image = image[None, ...]
    return image

def get_imagenet_label(probs):
    return decode_predictions(probs, top=1)[0][0]
image_path = custom_method(
tf.keras.utils.get_file('YellowLabradorLooking_new.jpg', 'https://storage.googleapis.com/download.tensorflow.org/example_images/YellowLabradorLooking_new.jpg'), imports='import matplotlib as mpl;import matplotlib.pyplot as plt;import tensorflow as tf', function_to_run='tf.keras.utils.get_file(*args)', method_object=None, object_signature=None, function_args=[eval("'YellowLabradorLooking_new.jpg'"), eval("'https://storage.googleapis.com/download.tensorflow.org/example_images/YellowLabradorLooking_new.jpg'")], function_kwargs={})
image_raw = custom_method(
tf.io.read_file(image_path), imports='import matplotlib as mpl;import matplotlib.pyplot as plt;import tensorflow as tf', function_to_run='tf.io.read_file(*args)', method_object=None, object_signature=None, function_args=[eval('image_path')], function_kwargs={})
image = custom_method(
tf.image.decode_image(image_raw), imports='import matplotlib as mpl;import matplotlib.pyplot as plt;import tensorflow as tf', function_to_run='tf.image.decode_image(*args)', method_object=None, object_signature=None, function_args=[eval('image_raw')], function_kwargs={})
image = preprocess(image)
image_probs = custom_method(
pretrained_model.predict(image), imports='import matplotlib as mpl;import matplotlib.pyplot as plt;import tensorflow as tf', function_to_run='obj.predict(*args)', method_object=eval('pretrained_model'), object_signature=None, function_args=[eval('image')], function_kwargs={}, custom_class=None)
plt.figure()
plt.imshow(image[0] * 0.5 + 0.5)
(_, image_class, class_confidence) = get_imagenet_label(image_probs)
plt.title('{} : {:.2f}% Confidence'.format(image_class, class_confidence * 100))
plt.show()
loss_object = custom_method(
tf.keras.losses.CategoricalCrossentropy(), imports='import matplotlib as mpl;import matplotlib.pyplot as plt;import tensorflow as tf', function_to_run='tf.keras.losses.CategoricalCrossentropy()', method_object=None, object_signature=None, function_args=[], function_kwargs={})

def create_adversarial_pattern(input_image, input_label):
    with custom_method(
    tf.GradientTape(), imports='import matplotlib as mpl;import matplotlib.pyplot as plt;import tensorflow as tf', function_to_run='tf.GradientTape()', method_object=None, object_signature=None, function_args=[], function_kwargs={}) as tape:
        tape.watch(input_image)
        prediction = custom_method(
        pretrained_model(input_image), imports='import matplotlib as mpl;import matplotlib.pyplot as plt;import tensorflow as tf', function_to_run='obj(*args)', method_object=eval('pretrained_model'), object_signature=None, function_args=[eval('input_image')], function_kwargs={}, custom_class=None)
        loss = custom_method(
        loss_object(input_label, prediction), imports='import matplotlib as mpl;import matplotlib.pyplot as plt;import tensorflow as tf', function_to_run='obj(*args)', method_object=eval('loss_object'), object_signature=None, function_args=[eval('input_label'), eval('prediction')], function_kwargs={}, custom_class=None)
    gradient = tape.gradient(loss, input_image)
    signed_grad = custom_method(
    tf.sign(gradient), imports='import matplotlib as mpl;import matplotlib.pyplot as plt;import tensorflow as tf', function_to_run='tf.sign(*args)', method_object=None, object_signature=None, function_args=[eval('gradient')], function_kwargs={})
    return signed_grad
labrador_retriever_index = 208
label = custom_method(
tf.one_hot(labrador_retriever_index, image_probs.shape[-1]), imports='import matplotlib as mpl;import matplotlib.pyplot as plt;import tensorflow as tf', function_to_run='tf.one_hot(*args)', method_object=None, object_signature=None, function_args=[eval('labrador_retriever_index'), eval('image_probs.shape[-1]')], function_kwargs={})
label = custom_method(
tf.reshape(label, (1, image_probs.shape[-1])), imports='import matplotlib as mpl;import matplotlib.pyplot as plt;import tensorflow as tf', function_to_run='tf.reshape(*args)', method_object=None, object_signature=None, function_args=[eval('label'), eval('(1, image_probs.shape[-1])')], function_kwargs={})
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
    adv_x = custom_method(
    tf.clip_by_value(adv_x, -1, 1), imports='import matplotlib as mpl;import matplotlib.pyplot as plt;import tensorflow as tf', function_to_run='tf.clip_by_value(*args)', method_object=None, object_signature=None, function_args=[eval('adv_x'), eval('-1'), eval('1')], function_kwargs={})
    display_images(adv_x, descriptions[i])
