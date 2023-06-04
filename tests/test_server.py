import numpy as np
import tensorflow as tf
import requests

from tool.measurement.send_request import send_request
from tool.measurement.measurement_config import SERVER_HOST, SERVER_PORT

def test_server_is_deployed():
    url = "https://"+SERVER_HOST+":"+str(SERVER_PORT)+"/"
    response = requests.get(url)
    assert(response.content=='<h1>Application Deployed!</h1>')

def test_matmul_request():
    arr1 = np.random.rand(100,100)
    arr2 = np.random.rand(100,100)

    imports = "import numpy as np"
    function_to_run = "np.matmul(*args)"
    function_args = [arr1, arr2]
    function_kwargs = None

    result = send_request(imports, function_to_run, function_args, function_kwargs, return_result=True)[function_to_run]

    assert(type(result["return"])==np.ndarray)
    assert(result["return"].shape==(100,100))

def test_rfft_request():
    
    arr1 = np.random.rand(100,100)

    imports = "import numpy as np"
    function_to_run = "np.fft.rfft(*args)"
    function_args = [arr1.tolist(), 200, -1]
    function_kwargs = None

    result = send_request(imports, function_to_run, function_args, function_kwargs, return_result=True)[function_to_run]

    assert(type(result["return"])==np.ndarray)
    assert(result["return"].shape==(100,101))

def test_tf_random_uniform_request():
    imports = "import tensorflow as tf"
    function_to_run = "tf.random.uniform(*args, **kwargs)"
    function_args = [[10, 1]]
    function_kwargs = {
        "minval": -1,
        "maxval": 1,
        "dtype": tf.float32
    }
    
    result = send_request(imports, function_to_run, function_args, function_kwargs, return_result=True)[function_to_run]

    assert(result["return"].shape==(10,1))

def test_tf_nested_Variable_request():
    imports = "import tensorflow as tf"
    function_to_run = "tf.Variable(*args)"
    function_args = [tf.random.uniform([10, 1], minval = -1, maxval = 1, dtype = tf.float32)]
    function_kwargs = None
    
    result = send_request(imports, function_to_run, function_args, function_kwargs, return_result=True)[function_to_run]

    assert(result["return"].shape==(10,1))