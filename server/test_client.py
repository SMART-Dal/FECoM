"""
Test client to send requests to the server
"""


from __future__ import print_function

import numpy as np
import tensorflow as tf
import pickle
import requests
from config import API_PATH, DEBUG, SERVER_HOST, SERVER_PORT

url = "http://"+SERVER_HOST+":"+str(SERVER_PORT)+API_PATH

def send_request(imports: str, function_to_run: str, function_args: list, function_kwargs: dict):
    """
    Send a request to execute any function and show the result
    TODO continue here: test compatibility with other functions and modules
    """

    method_details = {
        "imports": imports,
        "function": function_to_run,
        "args": function_args,
        "kwargs": function_kwargs
    }

    if DEBUG:
        print(f"sending {function_to_run} request to {url}")

    data = pickle.dumps(method_details)
    
    resp = requests.post(url, data=data, headers={'Content-Type': 'application/octet-stream'})

    result = pickle.loads(resp.content)

    # if HTTP status code is 500, the server could not reach a stable state.
    # now, simply raise an error. TODO: send a new request instead.
    if resp.status_code == 500:
        raise TimeoutError(result)
    
    if DEBUG:
        print(f"Result: {result}")
    
    return result


def send_matmul_request():
    """
    Send a request to perform numpy.matmul on two arrays and show the result
    """
    
    arr1 = np.random.rand(100,100)
    arr2 = np.random.rand(100,100)

    imports = "import numpy as np"
    function_to_run = "np.matmul(*args)"
    function_args = [arr1, arr2]
    function_kwargs = None

    result = send_request(imports, function_to_run, function_args, function_kwargs)

    if DEBUG:
        print(f"Result shape: {result.shape}")
        print(f"Means (arr1,arr2,result): ({arr1.mean()},{arr2.mean()},{result.mean()})")

def send_rfft_request():
    """
    Send a request to perform numpy.fft.rfft on an array and show the result
    """
    
    arr1 = np.random.rand(100,100)

    imports = "import numpy as np"
    function_to_run = "np.fft.rfft(*args)"
    function_args = [arr1.tolist(), 200, -1]
    function_kwargs = None

    result = send_request(imports, function_to_run, function_args, function_kwargs)
    
    if DEBUG:
        print(f"Result shape: {result.shape}")
        print(f"Means (arr1,result): ({arr1.mean()},{result.mean()})")

def send_tf_random_uniform_request():
    imports = "import tensorflow as tf"
    function_to_run = "tf.random.uniform(*args, **kwargs)"
    function_args = [[10, 1]]
    function_kwargs = {
        "minval": -1,
        "maxval": 1,
        "dtype": tf.float32
    }
    
    send_request(imports, function_to_run, function_args, function_kwargs)

def send_tf_nested_Variable_request():
    imports = "import tensorflow as tf"
    function_to_run = "tf.Variable(*args)"
    function_args = [tf.random.uniform([10, 1], minval = -1, maxval = 1, dtype = tf.float32)]
    function_kwargs = None
    
    send_request(imports, function_to_run, function_args, function_kwargs)
    

while True:
    a =  input("\n\npress 1 for np.matmul, 2 for np.rfft, 3 for tf.random.uniform or 4 for tf.Variable...")
    if a == "1":
        send_matmul_request()
    elif a == "2":
        send_rfft_request()
    elif a == "3":
        send_tf_random_uniform_request()
    elif a == "4":
        send_tf_nested_Variable_request()
    else:
        print("press a valid key")