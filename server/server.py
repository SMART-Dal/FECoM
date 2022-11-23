"""
Server to receive client requests to run ML methods and measure energy.
"""

import numpy as np
from config import API_PATH, DEBUG, SERVER_HOST, SERVER_PORT
from flask import Flask, Response, json, request


app = Flask(__name__)


@app.route(API_PATH, methods=["POST"])
def receive_matmul_request():
    data = request.json
    method_details = data["method_details"]
    arr1 = np.array(data["arr1"])
    arr2 = np.array(data["arr2"])
    
    if DEBUG:
        print(f"Received method details: {method_details}")
        print(f"Received arrays \n 1: {arr1.shape} \n 2: {arr2.shape}")
    
    if method_details["library"] == "numpy":
        method_to_run = getattr(np, method_details["method_sig"])
    
    output_arr = method_to_run(arr1,arr2)

    if DEBUG:
        print(f"Performed {str(method_to_run)} on input")
        print(f"Result: {output_arr.shape}")

    output = {"output": output_arr.tolist()}

    response = Response(
        response=json.dumps(output),
        status=200,
        mimetype='application/json'
    )

    return response


# start flask app
if __name__ == "__main__":
    app.run(host=SERVER_HOST, port=SERVER_PORT)