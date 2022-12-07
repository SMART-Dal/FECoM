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
Install dependencies (matplotlib & pandas are optional for energy measurement processing & graphing):  
```pip install flask numpy requests tensorflow pandas matplotlib```

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
    "args": function_args,
    "kwargs": function_kwargs
}

# serialising with pickle
data = pickle.dumps(method_details)

# sending the POST request
resp = requests.post(url, data=data, headers={'Content-Type': 'application/octet-stream'})
```
The dictionary has the following variables:

`imports` is a string of runnable python code that specifies the imports necessary to run the given function. E.g.  
```
imports = "import numpy as np"
```

`function_to_run` is a string of runnable python code that includes the full method signature with `*args` and/or `**kwargs` as the function arguments. E.g.  
```
# only positional function arguments
function_to_run = "np.matmul(*args)"

# only keyword function arguments
function_to_run = "np.matmul(**kwargs)"

# both positional and keyword function arguments
function_to_run = "np.matmul(*args, **kwargs"
```

`function_args` is an ordered `list` of the positional arguments that should be passed to the `function_to_run`. Each argument can be any kind of python object that can be serialised with pickle ([almost every object](https://machinelearningmastery.com/a-gentle-introduction-to-serialization-for-python/)). E.g.  
```
arr1 = np.random.rand(100,100)
arr2 = np.random.rand(100,100)
function_args = [arr1,arr2]

# this also works
function_args = [np.random.rand(100,100),np.random.rand(100,100)]
```

`function_kwargs` is a `dict` of the keyword arguments that should be passed to the `function_to_run`. The key-value pairs are of the form `"keyword": argument`. Each argument can be any kind of python object that can be serialised with pickle ([almost every object](https://machinelearningmastery.com/a-gentle-introduction-to-serialization-for-python/)). E.g.  
```
function_kwargs = {
    "minval": -1,
    "maxval": 1,
    "dtype": tf.float32
}
```
