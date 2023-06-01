import tensorflow as tf
import pandas as pd
import sys
from tool.client.client_config import EXPERIMENT_DIR
from tool.server.local_execution import before_execution as before_execution_INSERTED_INTO_SCRIPT
from tool.server.local_execution import after_execution as after_execution_INSERTED_INTO_SCRIPT
experiment_number = sys.argv[1]
experiment_project = sys.argv[2]
EXPERIMENT_FILE_PATH = EXPERIMENT_DIR / 'method-level' / experiment_project / f'experiment-{experiment_number}.json'
CSV_COLUMN_NAMES = ['SepalLength', 'SepalWidth', 'PetalLength', 'PetalWidth', 'Species']
SPECIES = ['Setosa', 'Versicolor', 'Virginica']
start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT()
train_path = tf.keras.utils.get_file('iris_training.csv', 'https://storage.googleapis.com/download.tensorflow.org/data/iris_training.csv')
after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tf.keras.utils.get_file(*args)', method_object=None, object_signature=None, function_args=['iris_training.csv', 'https://storage.googleapis.com/download.tensorflow.org/data/iris_training.csv'], function_kwargs={})
start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT()
test_path = tf.keras.utils.get_file('iris_test.csv', 'https://storage.googleapis.com/download.tensorflow.org/data/iris_test.csv')
after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tf.keras.utils.get_file(*args)', method_object=None, object_signature=None, function_args=['iris_test.csv', 'https://storage.googleapis.com/download.tensorflow.org/data/iris_test.csv'], function_kwargs={})
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
    start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT()
    dataset = tf.data.Dataset.from_tensor_slices((dict(features), labels))
    after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tf.data.Dataset.from_tensor_slices(*args)', method_object=None, object_signature=None, function_args=[(dict(features), labels)], function_kwargs={})
    if training:
        start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT()
        dataset = dataset.shuffle(1000).repeat()
        after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='obj.shuffle(1000).repeat()', method_object=dataset, object_signature=None, function_args=[], function_kwargs={})
    return dataset.batch(batch_size)
my_feature_columns = []
for key in train.keys():
    my_feature_columns.append(tf.feature_column.numeric_column(key=key))
start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT()
classifier = tf.estimator.DNNClassifier(feature_columns=my_feature_columns, hidden_units=[30, 10], n_classes=3)
after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tf.estimator.DNNClassifier(**kwargs)', method_object=None, object_signature=None, function_args=[], function_kwargs={'feature_columns': my_feature_columns, 'hidden_units': [30, 10], 'n_classes': 3})
start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT()
classifier.train(input_fn=lambda : input_fn(train, train_y, training=True), steps=5000)
after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='obj.train(**kwargs)', method_object=classifier, object_signature=None, function_args=[], function_kwargs={'input_fn': lambda : input_fn(train, train_y, training=True), 'steps': 5000})
start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT()
eval_result = classifier.evaluate(input_fn=lambda : input_fn(test, test_y, training=False))
after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='obj.evaluate(**kwargs)', method_object=classifier, object_signature=None, function_args=[], function_kwargs={'input_fn': lambda : input_fn(test, test_y, training=False)})
print('\nTest set accuracy: {accuracy:0.3f}\n'.format(**eval_result))
expected = ['Setosa', 'Versicolor', 'Virginica']
predict_x = {'SepalLength': [5.1, 5.9, 6.9], 'SepalWidth': [3.3, 3.0, 3.1], 'PetalLength': [1.7, 4.2, 5.4], 'PetalWidth': [0.5, 1.5, 2.1]}

def input_fn(features, batch_size=256):
    """An input function for prediction."""
    return tf.data.Dataset.from_tensor_slices(dict(features)).batch(batch_size)
start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT()
predictions = classifier.predict(input_fn=lambda : input_fn(predict_x))
after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='obj.predict(**kwargs)', method_object=classifier, object_signature=None, function_args=[], function_kwargs={'input_fn': lambda : input_fn(predict_x)})
for (pred_dict, expec) in zip(predictions, expected):
    class_id = pred_dict['class_ids'][0]
    probability = pred_dict['probabilities'][class_id]
    print('Prediction is "{}" ({:.1f}%), expected "{}"'.format(SPECIES[class_id], 100 * probability, expec))
