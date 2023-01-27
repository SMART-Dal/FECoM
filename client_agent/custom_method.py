import pickle
import requests
import sys
sys.path.append("../../../server")
from send_request import send_request

def custom_method(func,imports: str, function_to_run: str, method_object=None, function_args: list = None, function_kwargs: dict = None,max_wait_secs=0, custom_class=None):
   # method_details = {
   #  "imports": imports,
   #  "function": function_to_run,
   #  "method_object": method_object,
   #  "args": function_args,
   #  "kwargs": function_kwargs,
   #  "max_wait_secs": max_wait_secs,
   # }

   # # serialising with pickle
   # data = pickle.dumps(method_details)

   # # sending the POST request
   # resp = requests.post(url, data=data, headers={'Content-Type': 'application/octet-stream'})

   result = send_request(imports, function_to_run, function_args, function_kwargs, max_wait_secs, method_object)

#    if DEBUG:
#         print(f"Result shape: {result.shape}")
#         print(f"Means (arr1,arr2,result): ({arr1.mean()},{arr2.mean()},{result.mean()})")

   return func