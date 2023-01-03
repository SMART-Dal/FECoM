import pickle
import requests

def custom_method(func, imports, function_to_run, method_object, function_args, function_kwargs, max_wait_secs):
    method_details = {'imports': imports, 'function': function_to_run, 'method_object': method_object, 'args': function_args, 'kwargs': function_kwargs, 'max_wait_secs': max_wait_secs}
    data = pickle.dumps(method_details)
    # resp = requests.post(url, data=data, headers={'Content-Type': 'application/octet-stream'})
    return func
import tensorflow as tf
mnist = tf.keras.datasets.mnist
((x_train, y_train), (x_test, y_test)) = mnist.load_data()
(x_train, x_test) = (x_train / 255.0, x_test / 255.0)
model = custom_method(
tf.keras.models.Sequential([tf.keras.layers.Flatten(input_shape=(28, 28)), tf.keras.layers.Dense(128, activation='relu'), tf.keras.layers.Dropout(0.2), tf.keras.layers.Dense(10)]), imports='import tensorflow as tf', function_to_run='tf.keras.models.Sequential(*args)', method_object=None, function_args=[eval("[\n  tf.keras.layers.Flatten(input_shape=(28, 28)),\n  tf.keras.layers.Dense(128, activation='relu'),\n  tf.keras.layers.Dropout(0.2),\n  tf.keras.layers.Dense(10)\n]")], function_kwargs={}, max_wait_secs=30)
predictions = model(x_train[:1]).numpy()
custom_method(
tf.nn.softmax(predictions).numpy(), imports='import tensorflow as tf', function_to_run='tf.nn.softmax(predictions).numpy()', method_object=None, function_args=[], function_kwargs={}, max_wait_secs=30)
loss_fn = custom_method(
tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), imports='import tensorflow as tf', function_to_run='tf.keras.losses.SparseCategoricalCrossentropy(**kwargs)', method_object=None, function_args=[], function_kwargs={'from_logits': eval('True')}, max_wait_secs=30)
loss_fn(y_train[:1], predictions).numpy()
custom_method(
model.compile(optimizer='adam', loss=loss_fn, metrics=['accuracy']), imports='import tensorflow as tf', function_to_run='obj.compile(**kwargs)', method_object='model', function_args=[], function_kwargs={'optimizer': eval("'adam'"), 'loss': eval('loss_fn'), 'metrics': eval("['accuracy']")}, max_wait_secs=30)
custom_method(
model.fit(x_train, y_train, epochs=5), imports='import tensorflow as tf', function_to_run='obj.fit(*args, **kwargs)', method_object='model', function_args=[eval('x_train'), eval('y_train')], function_kwargs={'epochs': eval('5')}, max_wait_secs=30)
custom_method(
model.evaluate(x_test, y_test, verbose=2), imports='import tensorflow as tf', function_to_run='obj.evaluate(*args, **kwargs)', method_object='model', function_args=[eval('x_test'), eval('y_test')], function_kwargs={'verbose': eval('2')}, max_wait_secs=30)
probability_model = custom_method(
tf.keras.Sequential([model, tf.keras.layers.Softmax()]), imports='import tensorflow as tf', function_to_run='tf.keras.Sequential(*args)', method_object=None, function_args=[eval('[\n  model,\n  tf.keras.layers.Softmax()\n]')], function_kwargs={}, max_wait_secs=30)
custom_method(
probability_model(x_test[:5]), imports='import tensorflow as tf', function_to_run='obj(*args)', method_object='probability_model', function_args=[eval('x_test[:5]')], function_kwargs={}, max_wait_secs=30)
