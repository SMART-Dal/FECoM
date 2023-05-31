import tensorflow as tf
from tensorflow import keras
import keras_tuner as kt
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
custom_method(imports='import tensorflow as tf;from tensorflow import keras;import keras_tuner as kt', function_to_run='keras.datasets.fashion_mnist.load_data()', method_object=None, object_signature=None, function_args=[], function_kwargs={})
((img_train, label_train), (img_test, label_test)) = keras.datasets.fashion_mnist.load_data()
img_train = img_train.astype('float32') / 255.0
img_test = img_test.astype('float32') / 255.0

def model_builder(hp):
    custom_method(imports='import tensorflow as tf;from tensorflow import keras;import keras_tuner as kt', function_to_run='keras.Sequential()', method_object=None, object_signature=None, function_args=[], function_kwargs={})
    model = keras.Sequential()
    custom_method(imports='import tensorflow as tf;from tensorflow import keras;import keras_tuner as kt', function_to_run='obj.add(*args)', method_object=eval('model'), object_signature=None, function_args=[eval('keras.layers.Flatten(input_shape=(28, 28))')], function_kwargs={}, custom_class=None)
    model.add(keras.layers.Flatten(input_shape=(28, 28)))
    hp_units = hp.Int('units', min_value=32, max_value=512, step=32)
    custom_method(imports='import tensorflow as tf;from tensorflow import keras;import keras_tuner as kt', function_to_run='obj.add(*args)', method_object=eval('model'), object_signature=None, function_args=[eval("keras.layers.Dense(units=hp_units, activation='relu')")], function_kwargs={}, custom_class=None)
    model.add(keras.layers.Dense(units=hp_units, activation='relu'))
    custom_method(imports='import tensorflow as tf;from tensorflow import keras;import keras_tuner as kt', function_to_run='obj.add(*args)', method_object=eval('model'), object_signature=None, function_args=[eval('keras.layers.Dense(10)')], function_kwargs={}, custom_class=None)
    model.add(keras.layers.Dense(10))
    hp_learning_rate = hp.Choice('learning_rate', values=[0.01, 0.001, 0.0001])
    custom_method(imports='import tensorflow as tf;from tensorflow import keras;import keras_tuner as kt', function_to_run='obj.compile(**kwargs)', method_object=eval('model'), object_signature=None, function_args=[], function_kwargs={'optimizer': eval('keras.optimizers.Adam(learning_rate=hp_learning_rate)'), 'loss': eval('keras.losses.SparseCategoricalCrossentropy(from_logits=True)'), 'metrics': eval("['accuracy']")}, custom_class=None)
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=hp_learning_rate), loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=['accuracy'])
    return model
tuner = kt.Hyperband(model_builder, objective='val_accuracy', max_epochs=10, factor=3, directory='my_dir', project_name='intro_to_kt')
custom_method(imports='import tensorflow as tf;from tensorflow import keras;import keras_tuner as kt', function_to_run='tf.keras.callbacks.EarlyStopping(**kwargs)', method_object=None, object_signature=None, function_args=[], function_kwargs={'monitor': eval("'val_loss'"), 'patience': eval('5')})
stop_early = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)
tuner.search(img_train, label_train, epochs=50, validation_split=0.2, callbacks=[stop_early])
best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
print(f"\nThe hyperparameter search is complete. The optimal number of units in the first densely-connected\nlayer is {best_hps.get('units')} and the optimal learning rate for the optimizer\nis {best_hps.get('learning_rate')}.\n")
model = tuner.hypermodel.build(best_hps)
custom_method(imports='import tensorflow as tf;from tensorflow import keras;import keras_tuner as kt', function_to_run='obj.fit(*args, **kwargs)', method_object=eval('model'), object_signature=None, function_args=[eval('img_train'), eval('label_train')], function_kwargs={'epochs': eval('50'), 'validation_split': eval('0.2')}, custom_class=None)
history = model.fit(img_train, label_train, epochs=50, validation_split=0.2)
val_acc_per_epoch = history.history['val_accuracy']
best_epoch = val_acc_per_epoch.index(max(val_acc_per_epoch)) + 1
print('Best epoch: %d' % (best_epoch,))
hypermodel = tuner.hypermodel.build(best_hps)
hypermodel.fit(img_train, label_train, epochs=best_epoch, validation_split=0.2)
eval_result = hypermodel.evaluate(img_test, label_test)
print('[test loss, test accuracy]:', eval_result)
