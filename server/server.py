"""
Server to receive client requests to run ML methods and measure energy.
"""

import numpy as np
from config import API_PATH, DEBUG, SERVER_HOST, SERVER_PORT
from flask import Flask, Response, json, request


app = Flask(__name__)

def run_method(imports: str, function_to_run: str, args: list, arg_type_conversions: list, return_type_conversion: str):
    # WARNING: potential security risk from exec and eval statements

    # (1) import relevant modules
    exec(imports)

    # (2) convert the args to their correct types specified by arg_type_conversions
    func_args = [] # func_args is the argument list that will be passed to the function_to_run
    for i, raw_arg in enumerate(args):
        if arg_type_conversions[i] is None:
            func_args.append(raw_arg)
        else:
            # raw_data is the argument passed to the conversion function
            raw_data = raw_arg
            func_args.append(eval(arg_type_conversions[i]))
    
    assert(len(func_args) == len(args))

    # (3) evaluate the function return. This is where we should measure energy.
    func_return = eval(function_to_run)

    if DEBUG:
        print(f"Performed {function_to_run} on input")
        print(f"Output: {func_return}")
    
    # (4) convert the return type into a json-serialisable format
    if return_type_conversion is None:
        return func_return
    else:
        return eval(return_type_conversion)


@app.route(API_PATH, methods=["POST"])
def run_method_and_return_result():
    method_details = request.json
    
    if DEBUG:
        print(f"Received method details: {method_details}")
    
    output = run_method(
        method_details["imports"],
        method_details["function"],
        method_details["args"],
        method_details["arg_type_conversions"],
        method_details["return_type_conversion"]
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