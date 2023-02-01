import pickle
import requests
import sys
sys.path.append('../../../server')
from send_request import send_request

def custom_method(func, imports: str, function_to_run: str, method_object=None, function_args: list=None, function_kwargs: dict=None, max_wait_secs=0, custom_class=None):
    result = send_request(imports=imports, function_to_run=function_to_run, function_args=function_args, function_kwargs=function_kwargs, max_wait_secs=max_wait_secs, method_object=method_object, custom_class=custom_class)
    return func
import tensorflow as tf
custom_method(
tf.executing_eagerly(), imports='import tensorflow as tf', function_to_run='tf.executing_eagerly()', method_object=None, function_args=[], function_kwargs={}, max_wait_secs=0)
a = custom_method(
tf.constant([1, 2, 3, 4]), imports='import tensorflow as tf', function_to_run='tf.constant(*args)', method_object=None, function_args=[eval('[1, 2, 3, 4]')], function_kwargs={}, max_wait_secs=0)
b = custom_method(
tf.constant([10, 20, 30, 40]), imports='import tensorflow as tf', function_to_run='tf.constant(*args)', method_object=None, function_args=[eval('[10, 20, 30, 40]')], function_kwargs={}, max_wait_secs=0)
c = a + b
print(c)
