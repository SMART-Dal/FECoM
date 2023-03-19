import tensorflow as tf
from tensorflow import keras
import keras_tuner as kt
import os
from pathlib import Path
import dill as pickle
from tool.client.client_config import EXPERIMENT_DIR, EXPERIMENT_TAG, MAX_WAIT_S, WAIT_AFTER_RUN_S
from tool.server.send_request import send_request
from tool.server.function_details import FunctionDetails
current_path = os.path.abspath(__file__)
(immediate_folder, file_name) = os.path.split(current_path)
immediate_folder = os.path.basename(immediate_folder)
experiment_file_name = os.path.splitext(file_name)[0]
EXPERIMENT_FILE_PATH = EXPERIMENT_DIR / immediate_folder / EXPERIMENT_TAG / (experiment_file_name + '-energy.json')

def custom_method(func, imports: str, function_to_run: str, method_object=None, function_args: list=None, function_kwargs: dict=None, max_wait_secs=MAX_WAIT_S, custom_class=None, wait_after_run_secs=WAIT_AFTER_RUN_S):
    result = send_request(imports=imports, function_to_run=function_to_run, function_args=function_args, function_kwargs=function_kwargs, max_wait_secs=max_wait_secs, method_object=method_object, custom_class=custom_class, experiment_file_path=EXPERIMENT_FILE_PATH)
    return func
if __name__ == '__main__':
    print(EXPERIMENT_FILE_PATH)
((img_train, label_train), (img_test, label_test)) = custom_method(
keras.datasets.fashion_mnist.load_data(), imports='import keras_tuner as kt;from tensorflow import keras;import tensorflow as tf', function_to_run='keras.datasets.fashion_mnist.load_data()', method_object=None, function_args=[], function_kwargs={}, max_wait_secs=0)
img_train = img_train.astype('float32') / 255.0
img_test = img_test.astype('float32') / 255.0

def model_builder(hp):
    model = custom_method(
    keras.Sequential(), imports='import keras_tuner as kt;from tensorflow import keras;import tensorflow as tf', function_to_run='keras.Sequential()', method_object=None, function_args=[], function_kwargs={}, max_wait_secs=0)
    custom_method(
    model.add(keras.layers.Flatten(input_shape=(28, 28))), imports='import keras_tuner as kt;from tensorflow import keras;import tensorflow as tf', function_to_run='obj.add(*args)', method_object=eval('model'), function_args=[eval('keras.layers.Flatten(input_shape=(28, 28))')], function_kwargs={}, max_wait_secs=0, custom_class=None)
    hp_units = hp.Int('units', min_value=32, max_value=512, step=32)
    custom_method(
    model.add(keras.layers.Dense(units=hp_units, activation='relu')), imports='import keras_tuner as kt;from tensorflow import keras;import tensorflow as tf', function_to_run='obj.add(*args)', method_object=eval('model'), function_args=[eval("keras.layers.Dense(units=hp_units, activation='relu')")], function_kwargs={}, max_wait_secs=0, custom_class=None)
    custom_method(
    model.add(keras.layers.Dense(10)), imports='import keras_tuner as kt;from tensorflow import keras;import tensorflow as tf', function_to_run='obj.add(*args)', method_object=eval('model'), function_args=[eval('keras.layers.Dense(10)')], function_kwargs={}, max_wait_secs=0, custom_class=None)
    hp_learning_rate = hp.Choice('learning_rate', values=[0.01, 0.001, 0.0001])
    custom_method(
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=hp_learning_rate), loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=['accuracy']), imports='import keras_tuner as kt;from tensorflow import keras;import tensorflow as tf', function_to_run='obj.compile(**kwargs)', method_object=eval('model'), function_args=[], function_kwargs={'optimizer': eval('keras.optimizers.Adam(learning_rate=hp_learning_rate)'), 'loss': eval('keras.losses.SparseCategoricalCrossentropy(from_logits=True)'), 'metrics': eval("['accuracy']")}, max_wait_secs=0, custom_class=None)
    return model
tuner = kt.Hyperband(model_builder, objective='val_accuracy', max_epochs=10, factor=3, directory='my_dir', project_name='intro_to_kt')
stop_early = custom_method(
tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5), imports='import keras_tuner as kt;from tensorflow import keras;import tensorflow as tf', function_to_run='tf.keras.callbacks.EarlyStopping(**kwargs)', method_object=None, function_args=[], function_kwargs={'monitor': eval("'val_loss'"), 'patience': eval('5')}, max_wait_secs=0)
tuner.search(img_train, label_train, epochs=50, validation_split=0.2, callbacks=[stop_early])
best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
print(f"\nThe hyperparameter search is complete. The optimal number of units in the first densely-connected\nlayer is {best_hps.get('units')} and the optimal learning rate for the optimizer\nis {best_hps.get('learning_rate')}.\n")
model = tuner.hypermodel.build(best_hps)
history = custom_method(
model.fit(img_train, label_train, epochs=50, validation_split=0.2), imports='import keras_tuner as kt;from tensorflow import keras;import tensorflow as tf', function_to_run='obj.fit(*args, **kwargs)', method_object=eval('model'), function_args=[eval('img_train'), eval('label_train')], function_kwargs={'epochs': eval('50'), 'validation_split': eval('0.2')}, max_wait_secs=0, custom_class=None)
val_acc_per_epoch = history.history['val_accuracy']
best_epoch = val_acc_per_epoch.index(max(val_acc_per_epoch)) + 1
print('Best epoch: %d' % (best_epoch,))
hypermodel = tuner.hypermodel.build(best_hps)
custom_method(
hypermodel.fit(img_train, label_train, epochs=best_epoch, validation_split=0.2), imports='import keras_tuner as kt;from tensorflow import keras;import tensorflow as tf', function_to_run='obj.fit(*args, **kwargs)', method_object=eval('hypermodel'), function_args=[eval('img_train'), eval('label_train')], function_kwargs={'epochs': eval('best_epoch'), 'validation_split': eval('0.2')}, max_wait_secs=0, custom_class=None)
eval_result = custom_method(
hypermodel.evaluate(img_test, label_test), imports='import keras_tuner as kt;from tensorflow import keras;import tensorflow as tf', function_to_run='obj.evaluate(*args)', method_object=eval('hypermodel'), function_args=[eval('img_test'), eval('label_test')], function_kwargs={}, max_wait_secs=0, custom_class=None)
print('[test loss, test accuracy]:', eval_result)
