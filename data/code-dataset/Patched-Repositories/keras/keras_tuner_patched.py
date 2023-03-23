import tensorflow as tf
from tensorflow import keras
import keras_tuner as kt
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
    result = send_request(imports=imports, function_to_run=function_to_run, function_args=function_args, function_kwargs=function_kwargs, max_wait_secs=max_wait_secs, wait_after_run_secs=wait_after_run_secs, method_object=method_object, object_signature=object_signature, custom_class=custom_class, experiment_file_path=EXPERIMENT_FILE_PATH)
    return func
((img_train, label_train), (img_test, label_test)) = custom_method(
keras.datasets.fashion_mnist.load_data(), imports='import tensorflow as tf;import keras_tuner as kt;from tensorflow import keras', function_to_run='keras.datasets.fashion_mnist.load_data()', method_object=None, object_signature=None, function_args=[], function_kwargs={})
img_train = img_train.astype('float32') / 255.0
img_test = img_test.astype('float32') / 255.0

def model_builder(hp):
    model = custom_method(
    keras.Sequential(), imports='import tensorflow as tf;import keras_tuner as kt;from tensorflow import keras', function_to_run='keras.Sequential()', method_object=None, object_signature=None, function_args=[], function_kwargs={})
    custom_method(
    model.add(keras.layers.Flatten(input_shape=(28, 28))), imports='import tensorflow as tf;import keras_tuner as kt;from tensorflow import keras', function_to_run='obj.add(*args)', method_object=eval('model'), object_signature='keras.Sequential', function_args=[eval('keras.layers.Flatten(input_shape=(28, 28))')], function_kwargs={}, custom_class=None)
    hp_units = hp.Int('units', min_value=32, max_value=512, step=32)
    custom_method(
    model.add(keras.layers.Dense(units=hp_units, activation='relu')), imports='import tensorflow as tf;import keras_tuner as kt;from tensorflow import keras', function_to_run='obj.add(*args)', method_object=eval('model'), object_signature='keras.Sequential', function_args=[eval("keras.layers.Dense(units=hp_units, activation='relu')")], function_kwargs={}, custom_class=None)
    custom_method(
    model.add(keras.layers.Dense(10)), imports='import tensorflow as tf;import keras_tuner as kt;from tensorflow import keras', function_to_run='obj.add(*args)', method_object=eval('model'), object_signature='keras.Sequential', function_args=[eval('keras.layers.Dense(10)')], function_kwargs={}, custom_class=None)
    hp_learning_rate = hp.Choice('learning_rate', values=[0.01, 0.001, 0.0001])
    custom_method(
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=hp_learning_rate), loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=['accuracy']), imports='import tensorflow as tf;import keras_tuner as kt;from tensorflow import keras', function_to_run='obj.compile(**kwargs)', method_object=eval('model'), object_signature='keras.Sequential', function_args=[], function_kwargs={'optimizer': eval('keras.optimizers.Adam(learning_rate=hp_learning_rate)'), 'loss': eval('keras.losses.SparseCategoricalCrossentropy(from_logits=True)'), 'metrics': eval("['accuracy']")}, custom_class=None)
    return model
tuner = kt.Hyperband(model_builder, objective='val_accuracy', max_epochs=10, factor=3, directory='my_dir', project_name='intro_to_kt')
stop_early = custom_method(
tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5), imports='import tensorflow as tf;import keras_tuner as kt;from tensorflow import keras', function_to_run='tf.keras.callbacks.EarlyStopping(**kwargs)', method_object=None, object_signature=None, function_args=[], function_kwargs={'monitor': eval("'val_loss'"), 'patience': eval('5')})
tuner.search(img_train, label_train, epochs=50, validation_split=0.2, callbacks=[stop_early])
best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
print(f"\nThe hyperparameter search is complete. The optimal number of units in the first densely-connected\nlayer is {best_hps.get('units')} and the optimal learning rate for the optimizer\nis {best_hps.get('learning_rate')}.\n")
model = tuner.hypermodel.build(best_hps)
history = custom_method(
model.fit(img_train, label_train, epochs=50, validation_split=0.2), imports='import tensorflow as tf;import keras_tuner as kt;from tensorflow import keras', function_to_run='obj.fit(*args, **kwargs)', method_object=eval('model'), object_signature='keras.Sequential', function_args=[eval('img_train'), eval('label_train')], function_kwargs={'epochs': eval('50'), 'validation_split': eval('0.2')}, custom_class=None)
val_acc_per_epoch = history.history['val_accuracy']
best_epoch = val_acc_per_epoch.index(max(val_acc_per_epoch)) + 1
print('Best epoch: %d' % (best_epoch,))
hypermodel = tuner.hypermodel.build(best_hps)
hypermodel.fit(img_train, label_train, epochs=best_epoch, validation_split=0.2)
eval_result = hypermodel.evaluate(img_test, label_test)
print('[test loss, test accuracy]:', eval_result)
