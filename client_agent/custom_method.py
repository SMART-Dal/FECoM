import pickle
import requests
import sys

import os

# Define the path to the subdirectory, relative to the home directory
subdir_path = os.path.join(os.path.expanduser("~"), "/home/saurabh/code-energy-consumption/server")

# Get the current working directory
# cwd = os.path.dirname(os.path.realpath(__file__))
cwd = os.getcwd()

# Get the relative path from the current working directory to the subdirectory
rel_path = os.path.relpath(subdir_path, cwd)

sys.path.append(rel_path)
from send_request import send_single_thread_request


# TODO (by Tim) we also need to pass the wait_after_run_secs argument to send_request to specify
# the number of seconds we want to wait after the function stops running on the server

def custom_method(func,imports: str, function_to_run: str, method_object=None, function_args: list = None, function_kwargs: dict = None,max_wait_secs=0, custom_class=None):
   result = send_single_thread_request(imports=imports, function_to_run=function_to_run, function_args=function_args, function_kwargs=function_kwargs, max_wait_secs=max_wait_secs, method_object=method_object, custom_class=custom_class)
   return func