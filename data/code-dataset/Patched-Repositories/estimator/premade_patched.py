import tensorflow as tf
import pandas as pd
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

def custom_method(func, imports: str, function_to_run: str, method_object=None, object_signature=None, function_args: list=None, function_kwargs: dict=None, max_wait_secs=MAX_WAIT_S, custom_class=None, wait_after_run_secs=WAIT_AFTER_RUN_S):
    result = send_request(imports=imports, function_to_run=function_to_run, function_args=function_args, function_kwargs=function_kwargs, max_wait_secs=max_wait_secs, wait_after_run_secs=wait_after_run_secs, method_object=method_object, object_signature=object_signature, custom_class=custom_class, experiment_file_path=EXPERIMENT_FILE_PATH)
    return func
CSV_COLUMN_NAMES = ['SepalLength', 'SepalWidth', 'PetalLength', 'PetalWidth', 'Species']
SPECIES = ['Setosa', 'Versicolor', 'Virginica']
train_path = custom_method(
tf.keras.utils.get_file('iris_training.csv', 'https://storage.googleapis.com/download.tensorflow.org/data/iris_training.csv'), imports='import pandas as pd;import tensorflow as tf', function_to_run='tf.keras.utils.get_file(*args)', method_object=None, object_signature=None, function_args=[eval('"iris_training.csv"'), eval('"https://storage.googleapis.com/download.tensorflow.org/data/iris_training.csv"')], function_kwargs={}, max_wait_secs=0)
test_path = custom_method(
tf.keras.utils.get_file('iris_test.csv', 'https://storage.googleapis.com/download.tensorflow.org/data/iris_test.csv'), imports='import pandas as pd;import tensorflow as tf', function_to_run='tf.keras.utils.get_file(*args)', method_object=None, object_signature=None, function_args=[eval('"iris_test.csv"'), eval('"https://storage.googleapis.com/download.tensorflow.org/data/iris_test.csv"')], function_kwargs={}, max_wait_secs=0)
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
    dataset = custom_method(
    tf.data.Dataset.from_tensor_slices((dict(features), labels)), imports='import pandas as pd;import tensorflow as tf', function_to_run='tf.data.Dataset.from_tensor_slices(*args)', method_object=None, object_signature=None, function_args=[eval('(dict(features), labels)')], function_kwargs={}, max_wait_secs=0)
    if training:
        dataset = custom_method(
        dataset.shuffle(1000).repeat(), imports='import pandas as pd;import tensorflow as tf', function_to_run='obj.shuffle(1000).repeat()', method_object=eval('dataset'), object_signature='tf.data.Dataset.from_tensor_slices', function_args=[], function_kwargs={}, max_wait_secs=0, custom_class=None)
    return custom_method(
    dataset.batch(batch_size), imports='import pandas as pd;import tensorflow as tf', function_to_run='obj.batch(*args)', method_object=eval('dataset'), object_signature='tf.data.Dataset.from_tensor_slices', function_args=[eval('batch_size')], function_kwargs={}, max_wait_secs=0, custom_class=None)
my_feature_columns = []
for key in train.keys():
    my_feature_columns.append(tf.feature_column.numeric_column(key=key))
classifier = custom_method(
tf.estimator.DNNClassifier(feature_columns=my_feature_columns, hidden_units=[30, 10], n_classes=3), imports='import pandas as pd;import tensorflow as tf', function_to_run='tf.estimator.DNNClassifier(**kwargs)', method_object=None, object_signature=None, function_args=[], function_kwargs={'feature_columns': eval('my_feature_columns'), 'hidden_units': eval('[30, 10]'), 'n_classes': eval('3')}, max_wait_secs=0)
custom_method(
classifier.train(input_fn=lambda : input_fn(train, train_y, training=True), steps=5000), imports='import pandas as pd;import tensorflow as tf', function_to_run='obj.train(**kwargs)', method_object=eval('classifier'), object_signature='tf.estimator.DNNClassifier', function_args=[], function_kwargs={'input_fn': eval('lambda: input_fn(train, train_y, training=True)'), 'steps': eval('5000')}, max_wait_secs=0, custom_class=None)
eval_result = custom_method(
classifier.evaluate(input_fn=lambda : input_fn(test, test_y, training=False)), imports='import pandas as pd;import tensorflow as tf', function_to_run='obj.evaluate(**kwargs)', method_object=eval('classifier'), object_signature='tf.estimator.DNNClassifier', function_args=[], function_kwargs={'input_fn': eval('lambda: input_fn(test, test_y, training=False)')}, max_wait_secs=0, custom_class=None)
print('\nTest set accuracy: {accuracy:0.3f}\n'.format(**eval_result))
expected = ['Setosa', 'Versicolor', 'Virginica']
predict_x = {'SepalLength': [5.1, 5.9, 6.9], 'SepalWidth': [3.3, 3.0, 3.1], 'PetalLength': [1.7, 4.2, 5.4], 'PetalWidth': [0.5, 1.5, 2.1]}

def input_fn(features, batch_size=256):
    """An input function for prediction."""
    return custom_method(
    tf.data.Dataset.from_tensor_slices(dict(features)).batch(batch_size), imports='import pandas as pd;import tensorflow as tf', function_to_run='tf.data.Dataset.from_tensor_slices(dict(features)).batch(*args)', method_object=None, object_signature=None, function_args=[eval('batch_size')], function_kwargs={}, max_wait_secs=0)
predictions = custom_method(
classifier.predict(input_fn=lambda : input_fn(predict_x)), imports='import pandas as pd;import tensorflow as tf', function_to_run='obj.predict(**kwargs)', method_object=eval('classifier'), object_signature='tf.estimator.DNNClassifier', function_args=[], function_kwargs={'input_fn': eval('lambda: input_fn(predict_x)')}, max_wait_secs=0, custom_class=None)
for (pred_dict, expec) in zip(predictions, expected):
    class_id = pred_dict['class_ids'][0]
    probability = pred_dict['probabilities'][class_id]
    print('Prediction is "{}" ({:.1f}%), expected "{}"'.format(SPECIES[class_id], 100 * probability, expec))
