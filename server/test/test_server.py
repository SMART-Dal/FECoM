import numpy as np
import tensorflow as tf
# add parent directory to path
import sys
sys.path.insert(0,'..')
from send_request import send_request

def test_matmul_request():
    arr1 = np.random.rand(100,100)
    arr2 = np.random.rand(100,100)

    imports = "import numpy as np"
    function_to_run = "np.matmul(*args)"
    function_args = [arr1, arr2]
    function_kwargs = None
    max_wait_secs = 0

    result = send_request(imports, function_to_run, function_args, function_kwargs, max_wait_secs)
    
    assert(type(result)==np.ndarray)
    assert(result.shape==(100,100))

def test_rfft_request():
    
    arr1 = np.random.rand(100,100)

    imports = "import numpy as np"
    function_to_run = "np.fft.rfft(*args)"
    function_args = [arr1.tolist(), 200, -1]
    function_kwargs = None
    max_wait_secs = 0

    result = send_request(imports, function_to_run, function_args, function_kwargs, max_wait_secs)

    assert(type(result)==np.ndarray)
    assert(result.shape==(100,101))

def test_tf_random_uniform_request():
    imports = "import tensorflow as tf"
    function_to_run = "tf.random.uniform(*args, **kwargs)"
    function_args = [[10, 1]]
    function_kwargs = {
        "minval": -1,
        "maxval": 1,
        "dtype": tf.float32
    }
    max_wait_secs = 0
    
    result = send_request(imports, function_to_run, function_args, function_kwargs, max_wait_secs)

    assert(result.shape==(10,1))

def test_tf_nested_Variable_request():
    imports = "import tensorflow as tf"
    function_to_run = "tf.Variable(*args)"
    function_args = [tf.random.uniform([10, 1], minval = -1, maxval = 1, dtype = tf.float32)]
    function_kwargs = None
    max_wait_secs = 0
    
    result = send_request(imports, function_to_run, function_args, function_kwargs, max_wait_secs)

    assert(result.shape==(10,1))