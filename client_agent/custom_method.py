import pickle
import requests
import sys
sys.path.append("../../../server")
from send_request import send_request, send_single_thread_request
# from dummy_send_request import dummy_send_request

# TODO (by Tim) we also need to pass the wait_after_run_secs argument to send_request to specify
# the number of seconds we want to wait after the function stops running on the server

def custom_method(func,imports: str, function_to_run: str, method_object=None, function_args: list = None, function_kwargs: dict = None,max_wait_secs=0, custom_class=None):
   result = send_single_thread_request(imports=imports, function_to_run=function_to_run, function_args=function_args, function_kwargs=function_kwargs, max_wait_secs=max_wait_secs, method_object=method_object, custom_class=custom_class)
   return func