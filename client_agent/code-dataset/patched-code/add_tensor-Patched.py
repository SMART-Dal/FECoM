import pickle
import requests
import sys
sys.path.append('../../../server')
from send_request import send_request

def custom_method(func, imports: str, function_to_run: str, method_object=None, function_args: list=None, function_kwargs: dict=None, max_wait_secs=0, custom_class=None):
    result = send_request(imports, function_to_run, function_args, function_kwargs, max_wait_secs, method_object, custom_class)
    return func
import tensorflow as tf
custom_method(
tf.compat.v1.disable_eager_execution(), imports='import tensorflow as tf', function_to_run='tf.compat.v1.disable_eager_execution()', method_object=None, function_args=[], function_kwargs={}, max_wait_secs=0)
const1 = custom_method(
tf.constant([[1, 2, 3], [1, 2, 3]]), imports='import tensorflow as tf', function_to_run='tf.constant(*args)', method_object=None, function_args=[eval('[[1,2,3], [1,2,3]]')], function_kwargs={}, max_wait_secs=0)
const2 = custom_method(
tf.constant([[3, 4, 5], [3, 4, 5]]), imports='import tensorflow as tf', function_to_run='tf.constant(*args)', method_object=None, function_args=[eval('[[3,4,5], [3,4,5]]')], function_kwargs={}, max_wait_secs=0)
result = custom_method(
tf.add(const1, const2), imports='import tensorflow as tf', function_to_run='tf.add(*args)', method_object=None, function_args=[eval('const1'), eval('const2')], function_kwargs={}, max_wait_secs=0)
with custom_method(
tf.compat.v1.Session(), imports='import tensorflow as tf', function_to_run='tf.compat.v1.Session()', method_object=None, function_args=[], function_kwargs={}, max_wait_secs=0) as sess:
    sess.run(tf.compat.v1.global_variables_initializer())
    output = sess.run(result)
    print(output)
