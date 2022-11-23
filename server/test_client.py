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
    Send a dummy request to perform numpy.matmul on two arrays and show the result
    """
    
    arr1 = np.random.rand(3,3)
    arr2 = np.random.rand(3,3)

    method_details = {
        "library": "numpy",
        "method_sig": "matmul",
        "method_params": None,
        "n_serial_data": 2
    }

    data = {
        'method_details': method_details,
        'arr1': arr1.tolist(),
        'arr2': arr2.tolist()
    }

    if DEBUG:
        print(f"sending request to {url}")
    
    resp = requests.post(url, json=data)
    
    result = np.array(resp.json()["output"])
    
    if DEBUG:
        print(f"Result shape: {result.shape}")
        print(f"Result: {result}")
        print(f"Means (arr1,arr2,result): ({arr1.mean()},{arr2.mean()},{result.mean()})")
    

while True:
    input("\n\npress return...")
    send_matmul_request()