# Server Application
The server receives client requests to run ML methods and measure their energy consumption.

Please find instructions for how to run the server in the repository's top-level README.md file.

# Sending Requests
The API (path specified in `tool/config.py`) expects a POST request with request.data being a pickled FunctionDetails object (defined in `function_details.py`) which has the following attributes:
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
    module_name: str = None,
    exec_not_eval: bool = False,
)

# serialising with pickle
data = pickle.dumps(function_details)

# sending the POST request
resp = requests.post(url, data=data, headers={'Content-Type': 'application/octet-stream'})  
```  

More on each of the FunctionDetails object's attributes:

## Imports
`imports` is a string of runnable python code that specifies the imports necessary to run the given function. E.g.  
```
imports = "import numpy as np"
```

## Function to run
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

## Function arguments
`args` is an ordered `list` of the positional arguments that should be passed to the `function_to_run`. Each argument can be any kind of python object that can be serialised with pickle ([almost every object](https://machinelearningmastery.com/a-gentle-introduction-to-serialization-for-python/)). E.g.  
```
arr1 = np.random.rand(100,100)
arr2 = np.random.rand(100,100)
args = [arr1,arr2]

# this also works
args = [np.random.rand(100,100),np.random.rand(100,100)]
```
*COMMENT: do we need the ability to also pass custom classes (see below) as args or kwargs?* 
## Function keyword arguments
`kwargs` is a `dict` of the keyword arguments that should be passed to the `function_to_run`. The key-value pairs are of the form `"keyword": argument`. Each argument can be any kind of python object that can be serialised with pickle ([almost every object](https://machinelearningmastery.com/a-gentle-introduction-to-serialization-for-python/)). E.g.  
```
kwargs = {
    "minval": -1,
    "maxval": 1,
    "dtype": tf.float32
}
```

## Max wait seconds
`max_wait_secs` is an `int` specifying the number of seconds the server should wait for the system to reach a stable state. If a stable state is not reached within this time, the server will abort and return an error. The special value `0` tells the server to not check for stable state and simply execute the method, which can be useful for testing purposes.  

## Wait after run seconds
`wait_after_run_secs` is an `int` specifying the number of seconds the server should wait after running the function. Setting this to a value other than 0 can be useful for determining whether there is additional energy consumption caused by a function call after execution.  

## Return result
`return result` is a `bool` set to `False` by default. It can be set to `True` for testing purposes to check whether the server runs the functions as expected. However, note the changes in the response format mentioned when set to `True`  

## Method object
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

## Custom class
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

## Module name
`module_name` is a string that is the name of the module in which the `custom_class` was defined. This can be most easily determined by calling  
```  
module_name = method_object.__module__
```

# Processing the Response
The response `resp` contains a json object of the following format:  
```
function_to_run: {
        "energy_data": {
            "cpu": df_cpu_json,
            "ram": df_ram_json,
            "gpu": df_gpu_json
        },
        "times": {
            "initial_start_time_server": initial_start_time_server,
            "start_time_server": start_time_server,
            "end_time_server": end_time_server,
            "start_time_perf": start_time_perf, 
            "end_time_perf": end_time_perf,
            "sys_start_time_perf": sys_start_time_perf,
            "start_time_nvidia": start_time_nvidia_normalised,
            "end_time_nvidia": end_time_nvidia_normalised,
            "sys_start_time_nvidia": sys_start_time_nvidia,
            "import_time": import_time,
            "begin_stable_check_time": begin_stable_check_time,
            "pickle_load_time": pickle_load_time
        },
        "cpu_temperatures": cpu_temperatures,
        "settings" = {
            "max_wait_s": max_wait_secs,
            "wait_after_run_s": wait_after_run_secs,
            "wait_per_stable_check_loop_s": WAIT_PER_STABLE_CHECK_LOOP_S,
            "tolerance": STABLE_CHECK_TOLERANCE,
            "check_last_n_points": CHECK_LAST_N_POINTS,
            "cpu_max_temp": CPU_MAXIMUM_TEMPERATURE,
            "gpu_max_temp": GPU_MAXIMUM_TEMPERATURE 
        }
        "input_sizes" {
            "args_size": args_size_bit,
            "kwargs_size": kwargs_size_bit,
            "object_size": object_size_bit
        }
    }
```  
Where `function_to_run` is the string specified in the request (see "Sending Requests" above and "Function to run" below).  

Each entry in `energy_data` is a Pandas DataFrame encoded as json through `pandas.DataFrame.to_json(orient='split')` which can be decoded into a DataFrame by calling `pd.read_json(df_json, orient=‘split’)`.  

The entries in `times` are start and end times measured in different ways. The `server` times are integers indicating time elapsed since epoch in nanoseconds and are the most accurate measure for start and end of execution. The `initial_start_time_server` is the time the server is first started from `start_measurement.py`. The `perf` and `nvidia` times are floats indicating time elapsed since start of energy measurement and are obtained by reading the most recent timestamp on the energy measurement data files generated by the `perf` and `nvidia-smi` tools, respectively. These are useful for aligning energy data with start/end times, in particular in combination with the `sys` start times which give the tools' start times logged by `start_measurement.py` in time elapsed since epoch in nanoseconds. The other times indicate key times measured by the server such as the begin of the stable check period.  

The single value called `cpu_temperatures` is a DataFrame encoded as json (see `energy_data` above) that condains the cpu temperature readings over time.  

The `settings` dictionary contains all relevant experiment settings. The capitalised values are constants defined in `server_config.py` and the others are defined in `client_config.py`. In theory, these settings can also be read by inspecting the config files, however, these might change between experiments to it can be useful to keep track of the settings in a more permanent way.  

Each entry in `input_sizes` is an integer indicating the approximate size of each input variable in bit. This approximation is obtained by calling `len(pickle.dumps(input_variable))`, i.e. by determining the length of a bitestring generated by serialising the `input_variable` using the built-in `pickle` library.

If `return_result` is set to `True`, the response `resp` is a pickled python object that can be retrieved by calling `pickle.loads(resp.content)`. This object is a dictionary formatted exactly like the json object above, but with two additional fields:  
```
"return": function_return,
"method_object": method_object
```  
This functionality serves debugging purposes and should not be used for energy measurement experiments since it can increase the response size drastically.  
***The file `send_request.py` contains a function that can be used to send and receive such a request!***  

# Server Status Codes
There are three status codes that the server might typically return:
- `HTTP 200 OK`: this indicates that everything went well
- `HTTP 500 Internal Server Error`: this indicates that the server did not reach a stable state within the specified amount of time
- `HTTP 401 Unauthorized`: this occurs if the authentication failed and will be an issue with the SSL configuration.

With status 500, the server will return a pickled python dictionary in the following format:  
```
results = {
    "energy_data": {
        "cpu": df_cpu_json,
        "ram": df_ram_json,
        "gpu": df_gpu_json
    },
    "error": e
}
```  
Where `energy_data` has the exact same format as explained above in "Processing the Response", and `error` is a Python Object containing the exact error message.