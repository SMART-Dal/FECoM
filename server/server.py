"""
Server to receive client requests to run ML methods and measure energy.
"""

import numpy as np
from config import API_PATH, DEBUG, SERVER_HOST, SERVER_PORT
from flask import Flask, Response, json, request


app = Flask(__name__)

def run_method(library: str, method_name: str, args: list, arg_types: list, return_type: str):
    method_to_run = None

    # (1) find the method to run
    # TODO: how can we generalise this? I.e. how can we automatically import the relevant module?
    if library == "numpy":
        method_to_run = getattr(np, method_name)
    elif library == "numpy.fft":
        method_to_run = getattr(np.fft, method_name)
    
    # (2) convert arguments to correct types
    for i, arg_type in enumerate(arg_types):
        if arg_type == "numpy.array":
            args[i] = np.array(args[i])
    
    # (3) run the method with the converted arguments
    output = method_to_run(*args)

    if DEBUG:
        print(f"Performed {str(method_to_run)} on input")
        print(f"Output: {output}")
    
    # (4) convert the return type into a json-serialisable format
    if return_type == "numpy.array":
        output = output.tolist()
    elif return_type == "complex numpy.array":
        output = np.real(output).tolist()

    return output


@app.route(API_PATH, methods=["POST"])
def run_method_and_return_result():
    method_details = request.json

    # check that the data format is correct
    assert("library" in method_details)
    assert("method_name" in method_details)
    assert("args" in method_details)
    assert("arg_types" in method_details)
    assert("return_type" in method_details)
    
    if DEBUG:
        print(f"Received method details: {method_details}")
    
    output = run_method(
        method_details["library"],
        method_details["method_name"],
        method_details["args"],
        method_details["arg_types"],
        method_details["return_type"]
    )

    output = {"output": output}

    response = Response(
        response=json.dumps(output),
        status=200,
        mimetype='application/json'
    )

    return response


# start flask app
if __name__ == "__main__":
    app.run(host=SERVER_HOST, port=SERVER_PORT)