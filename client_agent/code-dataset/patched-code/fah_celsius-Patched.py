import pickle
import requests
import sys
sys.path.append('../../../server')
from send_request import send_request

def custom_method(func, imports: str, function_to_run: str, method_object=None, function_args: list=None, function_kwargs: dict=None, max_wait_secs=0, custom_class=None):
    result = send_request(imports, function_to_run, function_args, function_kwargs, max_wait_secs, method_object, custom_class)
    return func
import tensorflow as tf
import numpy as np
import logging
logger = custom_method(
tf.get_logger(), imports='import logging;import tensorflow as tf;import numpy as np;import matplotlib.pyplot as plt', function_to_run='tf.get_logger()', method_object=None, function_args=[], function_kwargs={}, max_wait_secs=0)
custom_method(
logger.setLevel(logging.ERROR), imports='import logging;import tensorflow as tf;import numpy as np;import matplotlib.pyplot as plt', function_to_run='obj.setLevel(*args)', method_object='logger', function_args=[eval('logging.ERROR')], function_kwargs={}, max_wait_secs=0, custom_class=None)
celsius_q = np.array([-40, -10, 0, 8, 15, 22, 38], dtype=float)
fahrenheit_a = np.array([-40, 14, 32, 46, 59, 72, 100], dtype=float)
for (i, c) in enumerate(celsius_q):
    print('{} degrees Celsius = {} degrees Fahrenheit'.format(c, fahrenheit_a[i]))
l0 = custom_method(
tf.keras.layers.Dense(units=1, input_shape=[1]), imports='import logging;import tensorflow as tf;import numpy as np;import matplotlib.pyplot as plt', function_to_run='tf.keras.layers.Dense(**kwargs)', method_object=None, function_args=[], function_kwargs={'units': eval('1'), 'input_shape': eval('[1]')}, max_wait_secs=0)
model = custom_method(
tf.keras.Sequential([l0]), imports='import logging;import tensorflow as tf;import numpy as np;import matplotlib.pyplot as plt', function_to_run='tf.keras.Sequential(*args)', method_object=None, function_args=[eval('[l0]')], function_kwargs={}, max_wait_secs=0)
custom_method(
model.compile(loss='mean_squared_error', optimizer=tf.keras.optimizers.Adam(0.1)), imports='import logging;import tensorflow as tf;import numpy as np;import matplotlib.pyplot as plt', function_to_run='obj.compile(**kwargs)', method_object='model', function_args=[], function_kwargs={'loss': eval("'mean_squared_error'"), 'optimizer': eval('tf.keras.optimizers.Adam(0.1)')}, max_wait_secs=0, custom_class=None)
history = custom_method(
model.fit(celsius_q, fahrenheit_a, epochs=500, verbose=False), imports='import logging;import tensorflow as tf;import numpy as np;import matplotlib.pyplot as plt', function_to_run='obj.fit(*args, **kwargs)', method_object='model', function_args=[eval('celsius_q'), eval('fahrenheit_a')], function_kwargs={'epochs': eval('500'), 'verbose': eval('False')}, max_wait_secs=0, custom_class=None)
print('Finished training the model')
import matplotlib.pyplot as plt
plt.xlabel('Epoch Number')
plt.ylabel('Loss Magnitude')
plt.plot(history.history['loss'])
print(model.predict([100.0]))
print('These are the layer variables: {}'.format(l0.get_weights()))
l0 = custom_method(
tf.keras.layers.Dense(units=4, input_shape=[1]), imports='import logging;import tensorflow as tf;import numpy as np;import matplotlib.pyplot as plt', function_to_run='tf.keras.layers.Dense(**kwargs)', method_object=None, function_args=[], function_kwargs={'units': eval('4'), 'input_shape': eval('[1]')}, max_wait_secs=0)
l1 = custom_method(
tf.keras.layers.Dense(units=4), imports='import logging;import tensorflow as tf;import numpy as np;import matplotlib.pyplot as plt', function_to_run='tf.keras.layers.Dense(**kwargs)', method_object=None, function_args=[], function_kwargs={'units': eval('4')}, max_wait_secs=0)
l2 = custom_method(
tf.keras.layers.Dense(units=1), imports='import logging;import tensorflow as tf;import numpy as np;import matplotlib.pyplot as plt', function_to_run='tf.keras.layers.Dense(**kwargs)', method_object=None, function_args=[], function_kwargs={'units': eval('1')}, max_wait_secs=0)
model = custom_method(
tf.keras.Sequential([l0, l1, l2]), imports='import logging;import tensorflow as tf;import numpy as np;import matplotlib.pyplot as plt', function_to_run='tf.keras.Sequential(*args)', method_object=None, function_args=[eval('[l0, l1, l2]')], function_kwargs={}, max_wait_secs=0)
custom_method(
model.compile(loss='mean_squared_error', optimizer=tf.keras.optimizers.Adam(0.1)), imports='import logging;import tensorflow as tf;import numpy as np;import matplotlib.pyplot as plt', function_to_run='obj.compile(**kwargs)', method_object='model', function_args=[], function_kwargs={'loss': eval("'mean_squared_error'"), 'optimizer': eval('tf.keras.optimizers.Adam(0.1)')}, max_wait_secs=0, custom_class=None)
custom_method(
model.fit(celsius_q, fahrenheit_a, epochs=500, verbose=False), imports='import logging;import tensorflow as tf;import numpy as np;import matplotlib.pyplot as plt', function_to_run='obj.fit(*args, **kwargs)', method_object='model', function_args=[eval('celsius_q'), eval('fahrenheit_a')], function_kwargs={'epochs': eval('500'), 'verbose': eval('False')}, max_wait_secs=0, custom_class=None)
print('Finished training the model')
print(model.predict([100.0]))
print('Model predicts that 100 degrees Celsius is: {} degrees Fahrenheit'.format(model.predict([100.0])))
print('These are the l0 variables: {}'.format(l0.get_weights()))
print('These are the l1 variables: {}'.format(l1.get_weights()))
print('These are the l2 variables: {}'.format(l2.get_weights()))
