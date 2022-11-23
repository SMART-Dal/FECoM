"""
Test client to send requests to the server
"""


from __future__ import print_function

import numpy as np
import requests
from config import API_PATH, DEBUG, SERVER_HOST, SERVER_PORT

url = "http://"+SERVER_HOST+":"+str(SERVER_PORT)+API_PATH

def send_matmul_request():
    """
    Send a request to perform numpy.matmul on two arrays and show the result
    """
    
    arr1 = np.random.rand(3,3)
    arr2 = np.random.rand(3,3)

    method_details = {
        "library": "numpy",
        "method_sig": "matmul",
        "args": [arr1.tolist(),arr2.tolist()],
        "arg_types": ["numpy.array", "numpy.array"],
        "return_type": "numpy.array"
    }

    if DEBUG:
        print(f"sending matmul request to {url}")
    
    resp = requests.post(url, json=method_details)
    
    result = np.array(resp.json()["output"])
    
    if DEBUG:
        print(f"Result shape: {result.shape}")
        print(f"Result: {result}")
        print(f"Means (arr1,arr2,result): ({arr1.mean()},{arr2.mean()},{result.mean()})")

def send_rfft_request():
    """
    Send a request to perform numpy.fft.rfft on an array and show the result
    """
    
    arr1 = np.random.rand(100,100)

    method_details = {
        "library": "numpy.fft",
        "method_name": "rfft",
        "args": [arr1.tolist(), 200, -1],
        "arg_types": ["numpy.array", "int", "int"],
        "return_type": "complex numpy.array"
    }

    if DEBUG:
        print(f"sending rfft request to {url}")
    
    resp = requests.post(url, json=method_details)
    
    result = np.array(resp.json()["output"])
    
    if DEBUG:
        print(f"Result shape: {result.shape}")
        print(f"Result: {result}")
        print(f"Means (arr1,result): ({arr1.mean()},{result.mean()})")
    

while True:
    a =  input("\n\npress 1 for matmul or 2 for rfft...")
    if a == "1":
        send_matmul_request()
    if a == "2":
        send_rfft_request()
    else:
        print("press a valid key")