# Server Application
The server receives client requests to run ML methods (and in the future also measure their energy consumption).
The test client sends sample requests to test the server's methods.

## Setup
Create a python virtual environment in the `/server` directory:  
``` python3 -m venv venv```   
Activate the venv (Linux):  
```source venv/bin/activate```  
Activate the venv (Windows):  
```venv\Scripts\activate```    
Install dependencies (pandas, matplotlib & scikit-learn are optional for energy measurement processing & graphing):  
```pip install flask numpy requests tensorflow torch torchvision torchaudio pandas matplotlib scikit-learn```

## Run
Start the client:  
```python3 test_client.py```  
Start the server in a separate terminal (you have to activate the venv here as well):  
```python3 server.py```  
Press return in the client terminal to send a request to the server and receive the result as a response.

## API Spec
The API (path specified in `config.py`) expects a POST request with request.data being a pickled python dictionary in the following format:
```
method_details = {
    "imports": imports,
    "function": function_to_run,
    "method_object": method_object,
    "args": function_args,
    "kwargs": function_kwargs,
    "max_wait_secs": max_wait_secs
}

# serialising with pickle
data = pickle.dumps(method_details)

# sending the POST request
resp = requests.post(url, data=data, headers={'Content-Type': 'application/octet-stream'})
```

The response `resp` contains the function return as a pickled python object that can be retrieved by calling `pickle.loads(resp.content)`. If a method (and not a function) was run, i.e. method_object is not `None`, this object is a dictionary:  
```
content = {
    "return": function_return,
    "method_object": method_object
}
```

The dictionary has the following variables:

### Imports
`imports` is a string of runnable python code that specifies the imports necessary to run the given function. E.g.  
```
imports = "import numpy as np"
```

### Function to run
`function_to_run` is a string of runnable python code that includes the full method signature with `*args` and/or `**kwargs` as the function arguments. If this is a method call, i.e. a function called on an object that has been instantiated previously in the code, the variable name has to be substituted with `obj`. E.g.  
```
"""
CASE 1: function call
"""
# only positional function arguments
function_to_run = "np.matmul(*args)"

# only keyword function arguments
function_to_run = "np.matmul(**kwargs)"

# both positional and keyword function arguments
function_to_run = "np.matmul(*args, **kwargs)"

"""
CASE 2: method call
"""
# substitute the object variable name with obj
function_to_run = "obj.compile(**kwargs)"
```

### Method object
`method_object` specifies the object to run the method on, if it is a method called on a previously initialised object and not a function. If it is a function, this parameter is set to `None`. This is any python object stored under name equal to the name specified in the `function_to_run` parameter. E.g.  
```
"""
CASE 1: function call
"""
method_object = None

"""
CASE 2: method call
"""
# function to run is a method invoked on the model object
function_to_run = "obj.compile(**kwargs)"

# model object initialised previously in code
model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(10)
    ])

# simply assign model to method_object
method_object = model
```

### Function arguments
`function_args` is an ordered `list` of the positional arguments that should be passed to the `function_to_run`. Each argument can be any kind of python object that can be serialised with pickle ([almost every object](https://machinelearningmastery.com/a-gentle-introduction-to-serialization-for-python/)). E.g.  
```
arr1 = np.random.rand(100,100)
arr2 = np.random.rand(100,100)
function_args = [arr1,arr2]

# this also works
function_args = [np.random.rand(100,100),np.random.rand(100,100)]
```

### Function keyword arguments
`function_kwargs` is a `dict` of the keyword arguments that should be passed to the `function_to_run`. The key-value pairs are of the form `"keyword": argument`. Each argument can be any kind of python object that can be serialised with pickle ([almost every object](https://machinelearningmastery.com/a-gentle-introduction-to-serialization-for-python/)). E.g.  
```
function_kwargs = {
    "minval": -1,
    "maxval": 1,
    "dtype": tf.float32
}
```

### Max wait seconds
`max_wait_secs` is an `int` specifying the number of seconds the server should wait for the system to reach a stable state. If a stable state is not reached within this time, the server will abort and return an error. The special value `0` tells the server to not check for stable state and simply execute the method, which can be useful for testing purposes. E.g.  
```
max_wait_secs = 30
```