import tensorflow_datasets as tfds
import tensorflow as tf
import os, json
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
custom_method(
tf.compat.v1.disable_eager_execution(), imports='import tensorflow_datasets as tfds;import os, json;import tensorflow as tf', function_to_run='tf.compat.v1.disable_eager_execution()', method_object=None, function_args=[], function_kwargs={}, max_wait_secs=0)
BUFFER_SIZE = 10000
BATCH_SIZE = 64

def input_fn(mode, input_context=None):
    (datasets, info) = custom_method(
    tfds.load(name='mnist', with_info=True, as_supervised=True), imports='import tensorflow_datasets as tfds;import os, json;import tensorflow as tf', function_to_run='tfds.load(**kwargs)', method_object=None, function_args=[], function_kwargs={'name': eval("'mnist'"), 'with_info': eval('True'), 'as_supervised': eval('True')}, max_wait_secs=0)
    mnist_dataset = datasets['train'] if mode == tf.estimator.ModeKeys.TRAIN else datasets['test']

    def scale(image, label):
        image = custom_method(
        tf.cast(image, tf.float32), imports='import tensorflow_datasets as tfds;import os, json;import tensorflow as tf', function_to_run='tf.cast(*args)', method_object=None, function_args=[eval('image'), eval('tf.float32')], function_kwargs={}, max_wait_secs=0)
        image /= 255
        return (image, label)
    if input_context:
        mnist_dataset = mnist_dataset.shard(input_context.num_input_pipelines, input_context.input_pipeline_id)
    return mnist_dataset.map(scale).cache().shuffle(BUFFER_SIZE).batch(BATCH_SIZE)
LEARNING_RATE = 0.0001

def model_fn(features, labels, mode):
    model = custom_method(
    tf.keras.Sequential([tf.keras.layers.Conv2D(32, 3, activation='relu', input_shape=(28, 28, 1)), tf.keras.layers.MaxPooling2D(), tf.keras.layers.Flatten(), tf.keras.layers.Dense(64, activation='relu'), tf.keras.layers.Dense(10)]), imports='import tensorflow_datasets as tfds;import os, json;import tensorflow as tf', function_to_run='tf.keras.Sequential(*args)', method_object=None, function_args=[eval("[\n      tf.keras.layers.Conv2D(32, 3, activation='relu', input_shape=(28, 28, 1)),\n      tf.keras.layers.MaxPooling2D(),\n      tf.keras.layers.Flatten(),\n      tf.keras.layers.Dense(64, activation='relu'),\n      tf.keras.layers.Dense(10)\n  ]")], function_kwargs={}, max_wait_secs=0)
    logits = custom_method(
    model(features, training=False), imports='import tensorflow_datasets as tfds;import os, json;import tensorflow as tf', function_to_run='obj(*args, **kwargs)', method_object=eval('model'), function_args=[eval('features')], function_kwargs={'training': eval('False')}, max_wait_secs=0, custom_class=None)
    if mode == tf.estimator.ModeKeys.PREDICT:
        predictions = {'logits': logits}
        return custom_method(
        tf.estimator.EstimatorSpec(labels=labels, predictions=predictions), imports='import tensorflow_datasets as tfds;import os, json;import tensorflow as tf', function_to_run='tf.estimator.EstimatorSpec(**kwargs)', method_object=None, function_args=[], function_kwargs={'labels': eval('labels'), 'predictions': eval('predictions')}, max_wait_secs=0)
    optimizer = custom_method(
    tf.compat.v1.train.GradientDescentOptimizer(learning_rate=LEARNING_RATE), imports='import tensorflow_datasets as tfds;import os, json;import tensorflow as tf', function_to_run='tf.compat.v1.train.GradientDescentOptimizer(**kwargs)', method_object=None, function_args=[], function_kwargs={'learning_rate': eval('LEARNING_RATE')}, max_wait_secs=0)
    loss = custom_method(
    tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction=tf.keras.losses.Reduction.NONE)(labels, logits), imports='import tensorflow_datasets as tfds;import os, json;import tensorflow as tf', function_to_run='tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction=tf.keras.losses.Reduction.NONE)(*args)', method_object=None, function_args=[eval('labels'), eval('logits')], function_kwargs={}, max_wait_secs=0)
    loss = custom_method(
    tf.reduce_sum(loss), imports='import tensorflow_datasets as tfds;import os, json;import tensorflow as tf', function_to_run='tf.reduce_sum(*args)', method_object=None, function_args=[eval('loss')], function_kwargs={}, max_wait_secs=0) * (1.0 / BATCH_SIZE)
    if mode == tf.estimator.ModeKeys.EVAL:
        return custom_method(
        tf.estimator.EstimatorSpec(mode, loss=loss), imports='import tensorflow_datasets as tfds;import os, json;import tensorflow as tf', function_to_run='tf.estimator.EstimatorSpec(*args, **kwargs)', method_object=None, function_args=[eval('mode')], function_kwargs={'loss': eval('loss')}, max_wait_secs=0)
    return custom_method(
    tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=optimizer.minimize(loss, tf.compat.v1.train.get_or_create_global_step())), imports='import tensorflow_datasets as tfds;import os, json;import tensorflow as tf', function_to_run='tf.estimator.EstimatorSpec(**kwargs)', method_object=None, function_args=[], function_kwargs={'mode': eval('mode'), 'loss': eval('loss'), 'train_op': eval('optimizer.minimize(\n          loss, tf.compat.v1.train.get_or_create_global_step())')}, max_wait_secs=0)
strategy = custom_method(
tf.distribute.experimental.MultiWorkerMirroredStrategy(), imports='import tensorflow_datasets as tfds;import os, json;import tensorflow as tf', function_to_run='tf.distribute.experimental.MultiWorkerMirroredStrategy()', method_object=None, function_args=[], function_kwargs={}, max_wait_secs=0)
config = custom_method(
tf.estimator.RunConfig(train_distribute=strategy), imports='import tensorflow_datasets as tfds;import os, json;import tensorflow as tf', function_to_run='tf.estimator.RunConfig(**kwargs)', method_object=None, function_args=[], function_kwargs={'train_distribute': eval('strategy')}, max_wait_secs=0)
classifier = custom_method(
tf.estimator.Estimator(model_fn=model_fn, model_dir='/tmp/multiworker', config=config), imports='import tensorflow_datasets as tfds;import os, json;import tensorflow as tf', function_to_run='tf.estimator.Estimator(**kwargs)', method_object=None, function_args=[], function_kwargs={'model_fn': eval('model_fn'), 'model_dir': eval("'/tmp/multiworker'"), 'config': eval('config')}, max_wait_secs=0)
custom_method(
tf.estimator.train_and_evaluate(classifier, train_spec=tf.estimator.TrainSpec(input_fn=input_fn), eval_spec=tf.estimator.EvalSpec(input_fn=input_fn)), imports='import tensorflow_datasets as tfds;import os, json;import tensorflow as tf', function_to_run='tf.estimator.train_and_evaluate(*args, **kwargs)', method_object=None, function_args=[eval('classifier')], function_kwargs={'train_spec': eval('tf.estimator.TrainSpec(input_fn=input_fn)'), 'eval_spec': eval('tf.estimator.EvalSpec(input_fn=input_fn)')}, max_wait_secs=0)
