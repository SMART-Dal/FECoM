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
Install dependencies:  
```pip install flask numpy requests```

## Run
Start the client:  
```python3 test_client.py```  
Start the server in a separate terminal (you have to activate the venv here as well):  
```python3 server.py```  
Press return in the client terminal to send a request to the server and receive the result as a response.

## API Spec
The API (path specified in `config.py`) expects a POST request with request.json being a python dictionary in the following format:
```
{
    "imports": imports,
    "function": function_to_run,
    "args": function_args,
    "arg_type_conversions": arg_type_conversions,
    "return_type_conversion": return_type_conversion
}
```
With the following variables:

`imports` is a string of runnable python code that specifies the imports necessary to run the given function. E.g.  
```
imports = "import numpy as np"
```

`function_to_run` is a string of runnable python code that includes the full method signature with `*func_args` as the function argument. E.g.  
```
function_to_run = "np.matmul(*func_args)"
```

`function_args` is a list of the arguments that should be passed to the `function_to_run`. These arguments are json-serialisable. E.g.  
```
arr1 = np.random.rand(3,3)
arr2 = np.random.rand(3,3)
function_args = [arr1.tolist(),arr2.tolist()]
```

`arg_type_conversions` is a list of strings of runnable python code. Each string is either a function that specifies how to convert the json data into the data type required by the `function_to_run`, or `None` if the data is already in the required type. The conversion function has a single parameter `raw_data`, that is initialised on the server side to equal the corresponding argument from `function_args`. E.g.  
```
arg_type_conversions = ["np.array(raw_data)", "np.array(raw_data)"]
```

`return_type_conversion` is a string of runnable python code. It is either a function that specifies how to convert `func_return` (the return of `function_to_run`) into a json-serialisable data type, or `None` if `func_return` is already of a serialisable type. E.g.  
```
return_type_conversion = "func_return.tolist()"
```
