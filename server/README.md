# Server Application
The server receives client requests to run ML methods and measure their energy consumption.
Start the server by following the instructions below, wait a minute and then start sending requests from the client.

## Setup
Install [miniconda3](https://docs.conda.io/en/latest/miniconda.html)
In this directory, run the following command to create the required TensorFlow environment from the specified `environment.yml` file:  
```conda env create -f environment.yml```   
Activate the environemnt:  
```conda activate tf2```

## Run  
In the activated venv run this command to start the server:  
```python3 server.py```
To test if you have setup everything correctly, go to the `test` directory and follow the instructions in that directory's README file to run all tests. Make sure to do this in a new terminal with the activated environment. 

## API Spec
### Sending Requests
The API (path specified in `config.py`) expects a POST request with request.data being a pickled FunctionDetails object (defined in `function_details.py`) which has the following attributes:
```
# read the following as "attribute: type = default_value". Attributes with default values are not required to be specified at initialisation.

function_details = FunctionDetails(
    imports: str,
    function_to_run: str,
    args: list = None,
    kwargs: dict = None,
    max_wait_secs: int = 0,
    wait_after_run_secs: int = 0,
    return_result: bool = False,
    method_object: object = None,
    custom_class: str = None,
    module_name: str = None
)

# serialising with pickle
data = pickle.dumps(function_details)

# sending the POST request
resp = requests.post(url, data=data, headers={'Content-Type': 'application/octet-stream'})  
```  

### Processing the Response
The response `resp` contains a json object of the following format:  
```
{
    "energy_data": {
        "cpu": df_cpu_json,
        "ram": df_ram_json,
        "gpu": df_gpu_json
    },
    "start_time": start_time,
    "end_time": end_time
}
```  
Where each entry in `energy_json` is a Pandas DataFrame encoded as json through `pandas.DataFrame.to_json(orient='split')` which can be decoded into a DataFrame by calling `pd.read_json(energy_json, orient=‘split’)`. The other two entries `start_time` & `end_time` are integers indicating time since epoch in nanoseconds. If `return_result` is set to `True`, the response `resp` is a pickled python object that can be retrieved by calling `pickle.loads(resp.content)`. This object is a dictionary of the following form  
```
{
    "energy_data": energy_json,
    "start_time": start_time,
    "end_time": end_time
    "return": function_return,
    "method_object": method_object
}
```  
***The file `send_request.py` contains a function that can be used to send and receive such a request!***  

More on each of the FunctionDetails object's attributes:

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

### Function arguments
`args` is an ordered `list` of the positional arguments that should be passed to the `function_to_run`. Each argument can be any kind of python object that can be serialised with pickle ([almost every object](https://machinelearningmastery.com/a-gentle-introduction-to-serialization-for-python/)). E.g.  
```
arr1 = np.random.rand(100,100)
arr2 = np.random.rand(100,100)
args = [arr1,arr2]

# this also works
args = [np.random.rand(100,100),np.random.rand(100,100)]
```
*COMMENT: do we need the ability to also pass custom classes (see below) as args or kwargs?* 
### Function keyword arguments
`kwargs` is a `dict` of the keyword arguments that should be passed to the `function_to_run`. The key-value pairs are of the form `"keyword": argument`. Each argument can be any kind of python object that can be serialised with pickle ([almost every object](https://machinelearningmastery.com/a-gentle-introduction-to-serialization-for-python/)). E.g.  
```
kwargs = {
    "minval": -1,
    "maxval": 1,
    "dtype": tf.float32
}
```

### Max wait seconds
`max_wait_secs` is an `int` specifying the number of seconds the server should wait for the system to reach a stable state. If a stable state is not reached within this time, the server will abort and return an error. The special value `0` tells the server to not check for stable state and simply execute the method, which can be useful for testing purposes.  

### Wait after run seconds
`wait_after_run_secs` is an `int` specifying the number of seconds the server should wait after running the function. Setting this to a value other than 0 can be useful for determining whether there is additional energy consumption caused by a function call after execution.  

### Return result
`return result` is a `bool` set to `False` by default. It can be set to `True` for testing purposes to check whether the server runs the functions as expected. However, note the changes in the response format mentioned when set to `True`  

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

### Custom class
`custom_class` is a multi-line string of executable python code that defines a custom class, i.e. a class that cannot be imported from an installed module. This parameter is needed if the `method_object` is of such a custom class, since pickle cannot unpickle complex custom classes without their definition imported in the current module. An example of this is a PyTorch neural network inheriting from `nn.Module`. If `custom_class` is specified, `module_name` must also be specified.  
E.g.
```   
custom_class =  """class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x"""
```   

### Module name
`module_name` is a string that is the name of the module in which the `custom_class` was defined. This can be most easily determined by calling  
```  
module_name = method_object.__module__
```