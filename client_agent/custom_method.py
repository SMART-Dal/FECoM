# import pickle
# import requests
# import sys

# import os

# # Define the path to the subdirectory, relative to the home directory
# subdir_path = os.path.join(os.path.expanduser("~"), "/home/saurabh/code-energy-consumption/server")

# # Get the current working directory
# # cwd = os.path.dirname(os.path.realpath(__file__))
# cwd = os.getcwd()

# # Get the relative path from the current working directory to the subdirectory
# rel_path = os.path.relpath(subdir_path, cwd)

# sys.path.append(rel_path)
# from send_request import send_single_thread_request

import pickle
import requests
import json

# server settings for localhost environment
DEV_HOST = "localhost"
DEV_PORT = 54321

# server settings for production environment
PROD_HOST = "129.173.67.60" # can also set to "0.0.0.0" on the server
PROD_PORT = 8080

### SET TO FALSE FOR DEV SETTINGS ON YOUR LOCAL MACHINE ###
PROD = True

# configure server settings here
API_PATH = "/api/run_experiment"
SERVER_HOST = PROD_HOST if PROD else DEV_HOST
SERVER_PORT = PROD_PORT if PROD else DEV_PORT
URL = "https://"+SERVER_HOST+":"+str(SERVER_PORT)+API_PATH

# set this to True to get print outs as the server receives and processes requests
DEBUG = True

class FunctionDetails():
    """
    Store the information needed for running a function on the server.
    The method_object is pickled upon setting for cases where this object
    is a custom class. FunctionDetails objects are serialised and sent to
    the server, where they are deserialised. The double-pickled
    method_object will remain serialised until the getter is called such
    that the custom class definition can be loaded first onto the server.
    """
    def __init__(self, imports: str, function_to_run: str, args: list = None, kwargs: dict = None, max_wait_secs: int = 0, wait_after_run_secs: int = 0, return_result: bool = False, method_object: object = None, custom_class: str = None, module_name: str = None):
        # basic function details
        self.imports = imports
        self.function_to_run = function_to_run
        self.args = args
        self.kwargs = kwargs
        
        # configuration for running the function
        self.max_wait_secs = max_wait_secs
        self.wait_after_run_secs = wait_after_run_secs
        self.return_result = return_result # parameter for testing purposes
        
        # required for running a method on an object
        self.__method_object = pickle.dumps(method_object)

        # required for running a method on an object of a custom class (TODO this could also be used for custom functions)
        self.custom_class = custom_class
        self.module_name = module_name
    
    @property
    def method_object(self):
        return pickle.loads(self.__method_object)

def send_single_thread_request(imports: str, function_to_run: str, function_args: list = None, function_kwargs: dict = None, max_wait_secs: int = 0, wait_after_run_secs: int = 0, return_result: bool = False, method_object = None, custom_class: str = None, username: str = "tim9220", password: str = "qQ32XALjF9JqFh!vF3xY"):
    """
    Send a request to execute any function and show the result
    """

    function_details = FunctionDetails(
        imports,
        function_to_run,
        function_args,
        function_kwargs,
        max_wait_secs,
        wait_after_run_secs,
        return_result,
        method_object,
        custom_class,
        method_object.__module__ if custom_class is not None else None
    )

    if DEBUG:
        print(f"sending {function_to_run} request to {URL}")

    run_data = pickle.dumps(function_details)

    # verify = False because the server uses a self-signed certificate
    # TODO this setting throws a warning, we need to set verify to the trusted certificate path instead.
    # But this didn't work for a self-signed certificate, since a certificate authority (CA) bundle is required
    
    run_resp = requests.post(URL, data=run_data, auth=(username, password), verify=False, headers={'Content-Type': 'application/octet-stream'})

    # if HTTP status code is 500, the server could not reach a stable state.
    # now, simply raise an error. TODO: send a new request instead.

    if run_resp.status_code == 500:
         raise TimeoutError(run_resp.content)
    elif run_resp.status_code == 401:
        raise RuntimeError(run_resp.content)

    if return_result:
        return pickle.loads(run_resp.content)
    else:
        # result = pickle.loads(result)
        result = run_resp.content

    try:
        with open('methodcall-energy-dataset.json', 'r') as f:
            existing_data = json.load(f)
    except:
        existing_data = []

    if result:
        try:
            data = json.loads(result)
            with open('methodcall-energy-dataset.json', 'w') as f:
                existing_data.append(data)
                json.dump(existing_data, f)
        except json.decoder.JSONDecodeError as e:
            print(f"Error decoding JSON: {e}")
        except:
            print("Error writing JSON to file")
    else:
        print("Response content is empty")
    
    # return run_resp.json()

# TODO (by Tim) we also need to pass the wait_after_run_secs argument to send_request to specify
# the number of seconds we want to wait after the function stops running on the server

def custom_method(func,imports: str, function_to_run: str, method_object=None, function_args: list = None, function_kwargs: dict = None,max_wait_secs=0, custom_class=None):
   result = send_single_thread_request(imports=imports, function_to_run=function_to_run, function_args=function_args, function_kwargs=function_kwargs, max_wait_secs=max_wait_secs, method_object=method_object, custom_class=custom_class)
   return func