import tensorflow as tf
import numpy as np
import tensorflow_datasets as tfds
import sys
from tool.client.client_config import EXPERIMENT_DIR
from tool.server.local_execution import before_execution as before_execution_INSERTED_INTO_SCRIPT
from tool.server.local_execution import after_execution as after_execution_INSERTED_INTO_SCRIPT
experiment_number = sys.argv[1]
experiment_project = sys.argv[2]
EXPERIMENT_FILE_PATH = EXPERIMENT_DIR / 'method-level' / experiment_project / f'experiment-{experiment_number}.json'
start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT()
model = tf.keras.models.Sequential([tf.keras.layers.Dense(16, activation='relu', input_shape=(4,)), tf.keras.layers.Dropout(0.2), tf.keras.layers.Dense(3)])
after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tf.keras.models.Sequential(*args)', method_object=None, object_signature=None, function_args=[[tf.keras.layers.Dense(16, activation='relu', input_shape=(4,)), tf.keras.layers.Dropout(0.2), tf.keras.layers.Dense(3)]], function_kwargs={})
start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT()
model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), optimizer='adam')
after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='obj.compile(**kwargs)', method_object=model, object_signature=None, function_args=[], function_kwargs={'loss': tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), 'optimizer': 'adam'})
start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT()
model.summary()
after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='obj.summary()', method_object=model, object_signature=None, function_args=[], function_kwargs={})

def input_fn():
    import tensorflow_datasets as tfds
    split = tfds.Split.TRAIN
    dataset = tfds.load('iris', split=split, as_supervised=True)
    dataset = dataset.map(lambda features, labels: ({'dense_2_input': features}, labels))
    dataset = dataset.batch(32).repeat()
    return dataset
for (features_batch, labels_batch) in input_fn().take(1):
    print(features_batch)
    print(labels_batch)
import tempfile
model_dir = tempfile.mkdtemp()
start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT()
keras_estimator = tf.keras.estimator.model_to_estimator(keras_model=model, model_dir=model_dir)
after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tf.keras.estimator.model_to_estimator(**kwargs)', method_object=None, object_signature=None, function_args=[], function_kwargs={'keras_model': model, 'model_dir': model_dir})
start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT()
keras_estimator.train(input_fn=input_fn, steps=500)
after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='obj.train(**kwargs)', method_object=keras_estimator, object_signature=None, function_args=[], function_kwargs={'input_fn': input_fn, 'steps': 500})
start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT()
eval_result = keras_estimator.evaluate(input_fn=input_fn, steps=10)
after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='obj.evaluate(**kwargs)', method_object=keras_estimator, object_signature=None, function_args=[], function_kwargs={'input_fn': input_fn, 'steps': 10})
print('Eval result: {}'.format(eval_result))
