import pickle
import requests

def custom_method(func,imports,function_to_run,method_object,function_args,function_kwargs,max_wait_secs):
   method_details = {
    "imports": imports,
    "function": function_to_run,
    "method_object": method_object,
    "args": function_args,
    "kwargs": function_kwargs,
    "max_wait_secs": max_wait_secs,
   }

   # serialising with pickle
   data = pickle.dumps(method_details)

   # sending the POST request
   resp = requests.post(url, data=data, headers={'Content-Type': 'application/octet-stream'})

   return func