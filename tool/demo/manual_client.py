"""
Test client to send requests to the server
"""

import numpy as np
import tensorflow as tf

from tool.server.server_config import DEBUG
from tool.server.send_request import send_request


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
    max_wait_secs = 0

    result = send_request(imports, function_to_run, function_args, function_kwargs, max_wait_secs)

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
    max_wait_secs = 0

    result = send_request(imports, function_to_run, function_args, function_kwargs, max_wait_secs)
    
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
    max_wait_secs = 0
    
    send_request(imports, function_to_run, function_args, function_kwargs, max_wait_secs)

def send_tf_nested_Variable_request():
    imports = "import tensorflow as tf"
    function_to_run = "tf.Variable(*args)"
    function_args = [tf.random.uniform([10, 1], minval = -1, maxval = 1, dtype = tf.float32)]
    function_kwargs = None
    max_wait_secs = 0
    
    send_request(imports, function_to_run, function_args, function_kwargs, max_wait_secs)
    

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