import tensorflow as tf
import pandas as pd
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
CSV_COLUMN_NAMES = ['SepalLength', 'SepalWidth', 'PetalLength', 'PetalWidth', 'Species']
SPECIES = ['Setosa', 'Versicolor', 'Virginica']
custom_method(imports='import pandas as pd;import tensorflow as tf', function_to_run='tf.keras.utils.get_file(*args)', method_object=None, object_signature=None, function_args=[eval('"iris_training.csv"'), eval('"https://storage.googleapis.com/download.tensorflow.org/data/iris_training.csv"')], function_kwargs={})
train_path = tf.keras.utils.get_file('iris_training.csv', 'https://storage.googleapis.com/download.tensorflow.org/data/iris_training.csv')
custom_method(imports='import pandas as pd;import tensorflow as tf', function_to_run='tf.keras.utils.get_file(*args)', method_object=None, object_signature=None, function_args=[eval('"iris_test.csv"'), eval('"https://storage.googleapis.com/download.tensorflow.org/data/iris_test.csv"')], function_kwargs={})
test_path = tf.keras.utils.get_file('iris_test.csv', 'https://storage.googleapis.com/download.tensorflow.org/data/iris_test.csv')
train = pd.read_csv(train_path, names=CSV_COLUMN_NAMES, header=0)
test = pd.read_csv(test_path, names=CSV_COLUMN_NAMES, header=0)
train.head()
train_y = train.pop('Species')
test_y = test.pop('Species')
train.head()

def input_evaluation_set():
    features = {'SepalLength': np.array([6.4, 5.0]), 'SepalWidth': np.array([2.8, 2.3]), 'PetalLength': np.array([5.6, 3.3]), 'PetalWidth': np.array([2.2, 1.0])}
    labels = np.array([2, 1])
    return (features, labels)

def input_fn(features, labels, training=True, batch_size=256):
    """An input function for training or evaluating"""
    custom_method(imports='import pandas as pd;import tensorflow as tf', function_to_run='tf.data.Dataset.from_tensor_slices(*args)', method_object=None, object_signature=None, function_args=[eval('(dict(features), labels)')], function_kwargs={})
    dataset = tf.data.Dataset.from_tensor_slices((dict(features), labels))
    if training:
        custom_method(imports='import pandas as pd;import tensorflow as tf', function_to_run='obj.shuffle(1000).repeat()', method_object=eval('dataset'), object_signature=None, function_args=[], function_kwargs={}, custom_class=None)
        dataset = dataset.shuffle(1000).repeat()
    return dataset.batch(batch_size)
my_feature_columns = []
for key in train.keys():
    my_feature_columns.append(tf.feature_column.numeric_column(key=key))
custom_method(imports='import pandas as pd;import tensorflow as tf', function_to_run='tf.estimator.DNNClassifier(**kwargs)', method_object=None, object_signature=None, function_args=[], function_kwargs={'feature_columns': eval('my_feature_columns'), 'hidden_units': eval('[30, 10]'), 'n_classes': eval('3')})
classifier = tf.estimator.DNNClassifier(feature_columns=my_feature_columns, hidden_units=[30, 10], n_classes=3)
custom_method(imports='import pandas as pd;import tensorflow as tf', function_to_run='obj.train(**kwargs)', method_object=eval('classifier'), object_signature=None, function_args=[], function_kwargs={'input_fn': eval('lambda: input_fn(train, train_y, training=True)'), 'steps': eval('5000')}, custom_class=None)
classifier.train(input_fn=lambda : input_fn(train, train_y, training=True), steps=5000)
custom_method(imports='import pandas as pd;import tensorflow as tf', function_to_run='obj.evaluate(**kwargs)', method_object=eval('classifier'), object_signature=None, function_args=[], function_kwargs={'input_fn': eval('lambda: input_fn(test, test_y, training=False)')}, custom_class=None)
eval_result = classifier.evaluate(input_fn=lambda : input_fn(test, test_y, training=False))
print('\nTest set accuracy: {accuracy:0.3f}\n'.format(**eval_result))
expected = ['Setosa', 'Versicolor', 'Virginica']
predict_x = {'SepalLength': [5.1, 5.9, 6.9], 'SepalWidth': [3.3, 3.0, 3.1], 'PetalLength': [1.7, 4.2, 5.4], 'PetalWidth': [0.5, 1.5, 2.1]}

def input_fn(features, batch_size=256):
    """An input function for prediction."""
    return tf.data.Dataset.from_tensor_slices(dict(features)).batch(batch_size)
custom_method(imports='import pandas as pd;import tensorflow as tf', function_to_run='obj.predict(**kwargs)', method_object=eval('classifier'), object_signature=None, function_args=[], function_kwargs={'input_fn': eval('lambda: input_fn(predict_x)')}, custom_class=None)
predictions = classifier.predict(input_fn=lambda : input_fn(predict_x))
for (pred_dict, expec) in zip(predictions, expected):
    class_id = pred_dict['class_ids'][0]
    probability = pred_dict['probabilities'][class_id]
    print('Prediction is "{}" ({:.1f}%), expected "{}"'.format(SPECIES[class_id], 100 * probability, expec))
