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
experiment_number = sys.argv[1]
experiment_project = sys.argv[2]
EXPERIMENT_FILE_PATH = EXPERIMENT_DIR / 'method-level' / experiment_project / f'experiment-{experiment_number}.json'

def custom_method(func, imports: str, function_to_run: str, method_object=None, object_signature=None, function_args: list=None, function_kwargs: dict=None, custom_class=None):
    result = send_request(imports=imports, function_to_run=function_to_run, function_args=function_args, function_kwargs=function_kwargs, max_wait_secs=MAX_WAIT_S, wait_after_run_secs=WAIT_AFTER_RUN_S, method_object=method_object, object_signature=object_signature, custom_class=custom_class, experiment_file_path=EXPERIMENT_FILE_PATH)
custom_method(imports='import tensorflow as tf;import os, json;import tensorflow_datasets as tfds', function_to_run='tf.compat.v1.disable_eager_execution()', method_object=None, object_signature=None, function_args=[], function_kwargs={})
tf.compat.v1.disable_eager_execution()
BUFFER_SIZE = 10000
BATCH_SIZE = 64

def input_fn(mode, input_context=None):
    (datasets, info) = tfds.load(name='mnist', with_info=True, as_supervised=True)
    mnist_dataset = datasets['train'] if mode == tf.estimator.ModeKeys.TRAIN else datasets['test']

    def scale(image, label):
        custom_method(imports='import tensorflow as tf;import os, json;import tensorflow_datasets as tfds', function_to_run='tf.cast(*args)', method_object=None, object_signature=None, function_args=[eval('image'), eval('tf.float32')], function_kwargs={})
        image = tf.cast(image, tf.float32)
        image /= 255
        return (image, label)
    if input_context:
        mnist_dataset = mnist_dataset.shard(input_context.num_input_pipelines, input_context.input_pipeline_id)
    return mnist_dataset.map(scale).cache().shuffle(BUFFER_SIZE).batch(BATCH_SIZE)
LEARNING_RATE = 0.0001

def model_fn(features, labels, mode):
    custom_method(imports='import tensorflow as tf;import os, json;import tensorflow_datasets as tfds', function_to_run='tf.keras.Sequential(*args)', method_object=None, object_signature=None, function_args=[eval("[\n      tf.keras.layers.Conv2D(32, 3, activation='relu', input_shape=(28, 28, 1)),\n      tf.keras.layers.MaxPooling2D(),\n      tf.keras.layers.Flatten(),\n      tf.keras.layers.Dense(64, activation='relu'),\n      tf.keras.layers.Dense(10)\n  ]")], function_kwargs={})
    model = tf.keras.Sequential([tf.keras.layers.Conv2D(32, 3, activation='relu', input_shape=(28, 28, 1)), tf.keras.layers.MaxPooling2D(), tf.keras.layers.Flatten(), tf.keras.layers.Dense(64, activation='relu'), tf.keras.layers.Dense(10)])
    custom_method(imports='import tensorflow as tf;import os, json;import tensorflow_datasets as tfds', function_to_run='obj(*args, **kwargs)', method_object=eval('model'), object_signature=None, function_args=[eval('features')], function_kwargs={'training': eval('False')}, custom_class=None)
    logits = model(features, training=False)
    if mode == tf.estimator.ModeKeys.PREDICT:
        predictions = {'logits': logits}
        return tf.estimator.EstimatorSpec(labels=labels, predictions=predictions)
    custom_method(imports='import tensorflow as tf;import os, json;import tensorflow_datasets as tfds', function_to_run='tf.compat.v1.train.GradientDescentOptimizer(**kwargs)', method_object=None, object_signature=None, function_args=[], function_kwargs={'learning_rate': eval('LEARNING_RATE')})
    optimizer = tf.compat.v1.train.GradientDescentOptimizer(learning_rate=LEARNING_RATE)
    custom_method(imports='import tensorflow as tf;import os, json;import tensorflow_datasets as tfds', function_to_run='tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction=tf.keras.losses.Reduction.NONE)(*args)', method_object=None, object_signature=None, function_args=[eval('labels'), eval('logits')], function_kwargs={})
    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction=tf.keras.losses.Reduction.NONE)(labels, logits)
    loss = tf.reduce_sum(loss) * (1.0 / BATCH_SIZE)
    if mode == tf.estimator.ModeKeys.EVAL:
        return tf.estimator.EstimatorSpec(mode, loss=loss)
    return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=optimizer.minimize(loss, tf.compat.v1.train.get_or_create_global_step()))
custom_method(imports='import tensorflow as tf;import os, json;import tensorflow_datasets as tfds', function_to_run='tf.distribute.experimental.MultiWorkerMirroredStrategy()', method_object=None, object_signature=None, function_args=[], function_kwargs={})
strategy = tf.distribute.experimental.MultiWorkerMirroredStrategy()
custom_method(imports='import tensorflow as tf;import os, json;import tensorflow_datasets as tfds', function_to_run='tf.estimator.RunConfig(**kwargs)', method_object=None, object_signature=None, function_args=[], function_kwargs={'train_distribute': eval('strategy')})
config = tf.estimator.RunConfig(train_distribute=strategy)
custom_method(imports='import tensorflow as tf;import os, json;import tensorflow_datasets as tfds', function_to_run='tf.estimator.Estimator(**kwargs)', method_object=None, object_signature=None, function_args=[], function_kwargs={'model_fn': eval('model_fn'), 'model_dir': eval("'/tmp/multiworker'"), 'config': eval('config')})
classifier = tf.estimator.Estimator(model_fn=model_fn, model_dir='/tmp/multiworker', config=config)
custom_method(imports='import tensorflow as tf;import os, json;import tensorflow_datasets as tfds', function_to_run='tf.estimator.train_and_evaluate(*args, **kwargs)', method_object=None, object_signature=None, function_args=[eval('classifier')], function_kwargs={'train_spec': eval('tf.estimator.TrainSpec(input_fn=input_fn)'), 'eval_spec': eval('tf.estimator.EvalSpec(input_fn=input_fn)')})
tf.estimator.train_and_evaluate(classifier, train_spec=tf.estimator.TrainSpec(input_fn=input_fn), eval_spec=tf.estimator.EvalSpec(input_fn=input_fn))
