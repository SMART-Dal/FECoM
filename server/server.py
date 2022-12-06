"""
Server to receive client requests to run ML methods and measure energy.
"""

import numpy as np
import pickle
from config import API_PATH, DEBUG, SERVER_HOST, SERVER_PORT
from flask import Flask, Response, json, request


app = Flask(__name__)

def run_method(imports: str, function_to_run: str, args: list, kwargs: dict):
    """
    Run the method given by function_to_run with the given arguments (args) and keyword arguments (kwargs).
    These two variables appear to not be used, however, they are used when evaluating the function_to_run
    since this is a string in the format function_signature(*args) or function_signature(*args, **kwargs).
    """
    # WARNING: potential security risk from exec and eval statements

    # (1) import relevant modules
    exec(imports)

    # (2) evaluate the function return. This is where we should measure energy.
    func_return = eval(function_to_run)

    if DEBUG:
        print(f"Performed {function_to_run} on input")
        print(f"Output: {func_return}")
    
    return func_return


@app.route(API_PATH, methods=["POST"])
def run_method_and_return_result():
    method_details = pickle.loads(request.data)
    
    if DEBUG:
        print(f"Received method details: {method_details}")
    
    output = run_method(
        method_details["imports"],
        method_details["function"],
        method_details["args"],
        method_details["kwargs"]
    )

    data = pickle.dumps(output)

    response = Response(
        response=data,
        status=200,
        mimetype='application/octet_stream'
    )

    return response


# start flask app
if __name__ == "__main__":
    app.run(host=SERVER_HOST, port=SERVER_PORT, debug=True)