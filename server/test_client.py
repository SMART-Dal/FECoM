"""
Test client to send requests to the server
"""


from __future__ import print_function

import numpy as np
import pickle
import requests
from config import API_PATH, DEBUG, SERVER_HOST, SERVER_PORT

url = "http://"+SERVER_HOST+":"+str(SERVER_PORT)+API_PATH

def send_request(imports: str, function_to_run: str, function_args: list):
    """
    Send a request to execute any function and show the result
    TODO continue here: test compatibility with other functions and modules
    """

    method_details = {
        "imports": imports,
        "function": function_to_run,
        "args": function_args,
    }

    if DEBUG:
        print(f"sending {function_to_run} request to {url}")

    data = pickle.dumps(method_details)
    
    resp = requests.post(url, data=data, headers={'Content-Type': 'application/octet-stream'})
    
    result = pickle.loads(resp.content)
    
    if DEBUG:
        print(f"Result: {result}")


def send_matmul_request():
    """
    Send a request to perform numpy.matmul on two arrays and show the result
    """
    
    arr1 = np.random.rand(100,100)
    arr2 = np.random.rand(100,100)

    imports = "import numpy as np"
    function_to_run = "np.matmul(*args)"
    function_args = [arr1, arr2]

    method_details = {
        "imports": imports,
        "function": function_to_run,
        "args": function_args,
    }

    if DEBUG:
        print(f"sending matmul request to {url}")

    data = pickle.dumps(method_details)
    
    resp = requests.post(url, data=data, headers={'Content-Type': 'application/octet-stream'})
    
    result = pickle.loads(resp.content)
    
    if DEBUG:
        print(f"Result shape: {result.shape}")
        print(f"Result: {result}")
        print(f"Means (arr1,arr2,result): ({arr1.mean()},{arr2.mean()},{result.mean()})")

def send_rfft_request():
    """
    Send a request to perform numpy.fft.rfft on an array and show the result
    """
    
    arr1 = np.random.rand(100,100)

    imports = "import numpy as np"
    function_to_run = "np.fft.rfft(*args)"
    function_args = [arr1.tolist(), 200, -1]
    
    method_details = {
        "imports": imports,
        "function": function_to_run,
        "args": function_args,
    }

    if DEBUG:
        print(f"sending rfft request to {url}")

    data = pickle.dumps(method_details)
    
    resp = requests.post(url, data=data, headers={'Content-Type': 'application/octet-stream'})
    
    result = pickle.loads(resp.content)
    
    if DEBUG:
        print(f"Result shape: {result.shape}")
        print(f"Result: {result}")
        print(f"Means (arr1,result): ({arr1.mean()},{result.mean()})")
    

while True:
    a =  input("\n\npress 1 for matmul or 2 for rfft...")
    if a == "1":
        send_matmul_request()
    elif a == "2":
        send_rfft_request()
    else:
        print("press a valid key")