"""
Server to receive client requests to run ML methods and measure energy.
"""

import numpy as np
import time
import statistics as stats
import pickle
from config import API_PATH, DEBUG, SERVER_HOST, SERVER_PORT
from flask import Flask, Response, request


app = Flask(__name__)

def is_stable_state(max_wait_secs: int):
    """
    Return True only when the system's energy consumption is stable.
    """
    # only consider the last n points
    n = 50

    for i in range(max_wait_secs*2):
        data_lines = []
        with open('./energy_measurement/perf.txt', 'r') as f:
            data_lines = f.read().splitlines(True)

        last_n_cpu_energies = [line.strip(' ').split(';')[1] for line in data_lines[2::2][-n:]]
        last_n_ram_energies = [line.strip(' ').split(';')[1] for line in data_lines[3::2][-n:]]
        print(f"CPU: {last_n_cpu_energies}")
        print(f"RAM: {last_n_ram_energies}")
        return True

        energy_vars = [stats.variance(last_n_energies[n/5]) for i in range(n/5)]

        time.sleep(0.5)
        

    return False

def run_method(imports: str, function_to_run: str, args: list, kwargs: dict, max_wait_secs: int):
    """
    Run the method given by function_to_run with the given arguments (args) and keyword arguments (kwargs).
    These two variables appear to not be used, however, they are used when evaluating the function_to_run
    since this is a string in the format function_signature(*args) or function_signature(*args, **kwargs).
    """
    # WARNING: potential security risk from exec and eval statements

    # (1) import relevant modules
    exec(imports)

    # # (2) continue only when the system has reached a stable state of energy consumption
    if not is_stable_state(max_wait_secs):
        raise TimeoutError(f"System could not reach a stable state within {max_wait_secs} seconds")

    # (3) evaluate the function return. This is where we should measure energy.
    func_return = eval(function_to_run)

    if DEBUG:
        print(f"Performed {function_to_run} on input")
        print(f"Output: {func_return}")
    
    return func_return


@app.route(API_PATH, methods=["POST"])
def run_method_and_return_result():
    MAX_WAIT_SECS = 100
    method_details = pickle.loads(request.data)
    
    if DEBUG:
        print(f"Received method details: {method_details}")
    
    output = run_method(
        method_details["imports"],
        method_details["function"],
        method_details["args"],
        method_details["kwargs"],
        MAX_WAIT_SECS
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