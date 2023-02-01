"""
Server to receive client requests to run ML methods and measure energy.
"""

import time
import os
import statistics as stats
import pickle
from pathlib import Path

from flask import Flask, Response, request
from flask_httpauth import HTTPBasicAuth
from werkzeug.security import check_password_hash

from config import API_PATH, DEBUG, SERVER_HOST, SERVER_PORT, CPU_STD_TO_MEAN, RAM_STD_TO_MEAN, GPU_STD_TO_MEAN, USERS, CA_CERT_PATH, CA_KEY_PATH
from function_details import FunctionDetails # shown unused but still required
import logging

app = Flask(__name__)
auth = HTTPBasicAuth()

logging.basicConfig(filename='flask.log', level=logging.DEBUG,
                    format='%(asctime)s %(levelname)s %(name)s %(message)s')

@auth.verify_password
def verify_password(username, password):
    if username in USERS.keys() and check_password_hash(USERS[username], password):
        # typically would return the user object here.
        app.logger.info('Auth successful')
        return True
    else:
        return False

@auth.error_handler
def auth_error(status_code):
    data = pickle.dumps(f"{status_code}: Access denied.")

    response = Response(
        response=data,
        status=status_code,
        mimetype='application/octet_stream'
    )
    return response

def load_last_n_cpu_ram_gpu(n: int, filepath: Path) -> tuple:
    """
    Helper method for is_stable_state to load the last n energy data points
    for CPU, RAM and GPU (in this order)
    """
    # load CPU & RAM data
    cpu_ram = []
    with open(filepath/"perf.txt", 'r') as f:
        cpu_ram = f.read().splitlines(True)
    
    # load GPU data
    gpu = []
    with open(filepath/"nvidia_smi.txt", 'r') as f:
        gpu = f.read().splitlines(True)

    # generate lists of data
    last_n_cpu_energies = [float(line.strip(' ').split(';')[1]) for line in cpu_ram[2::2][-n:]]
    last_n_ram_energies = [float(line.strip(' ').split(';')[1]) for line in cpu_ram[3::2][-n:]]
    last_n_gpu_energies = [float(line.split(' ')[2]) for line in gpu[-n:]]

    return last_n_cpu_energies, last_n_ram_energies, last_n_gpu_energies

def data_is_stable(data: list[float], tolerance: float, stable_std_mean_ratio: float) -> bool:
    return (stats.stdev(data) / stats.mean(data)) <= ((1 + tolerance)*stable_std_mean_ratio)

def server_is_stable(max_wait_secs: int) -> bool:
    """
    Return True only when the system's energy consumption is stable.
    """
    # For testing purposes
    if max_wait_secs == 0:
        return True

    # only consider the last n points
    n = 50

    # relative tolerance for difference between stable stdev/mean ratio and current ratio
    # e.g. 0.1 would mean allowing a ratio that's 10% higher than the stable stdev/mean ratio
    tolerance = 0.1

    filepath = Path("./energy_measurement/out/2022-11-23")

    # in each loop iteration, load new data, calculate statistics and check if the energy is stable.
    # try this for the specified number of seconds
    for _ in range(max_wait_secs*2):
        cpu_energies, ram_energies, gpu_energies = load_last_n_cpu_ram_gpu(n, filepath)

        # stable means that stdv/mean ratios are smaller or equal to the stable ratios determined experimentally
        # the tolerance value set above specifies how much relative deviation we want to allow
        cpu_stable = data_is_stable(cpu_energies, tolerance, CPU_STD_TO_MEAN)
        ram_stable = data_is_stable(ram_energies, tolerance, RAM_STD_TO_MEAN)
        gpu_stable = data_is_stable(gpu_energies, tolerance, GPU_STD_TO_MEAN)

        if DEBUG:
            print(f"CPU: {cpu_energies} \n stable: {cpu_stable} \n stdv: {stats.stdev(cpu_energies)} \n mean: {stats.mean(cpu_energies)}")
            print()
            print(f"RAM: {ram_energies} \n stable: {ram_stable} \nstdv: {stats.stdev(ram_energies)} \n mean: {stats.mean(ram_energies)}")
            print()
            print(f"GPU: {gpu_energies} \n stable: {gpu_stable} \nstdv: {stats.stdev(gpu_energies)} \n mean: {stats.mean(gpu_energies)}")

        if cpu_stable and ram_stable and gpu_stable:
            return True
        else:
            time.sleep(0.5)
        

    return False

def run_function(imports: str, function_to_run: str, method_object: object, args: list, kwargs: dict, max_wait_secs: int, wait_after_run_secs: int, return_result: bool):
    """
    Run the method given by function_to_run with the given arguments (args) and keyword arguments (kwargs).
    These two variables appear to not be used, however, they are used when evaluating the function_to_run
    since this is a string in the format function_signature(*args), function_signature(**kwargs) or
    function_signature(*args, **kwargs).
    """
    # WARNING: potential security risk from exec and eval statements

    # (1) import relevant modules
    app.logger.info("Imports value: %s", imports)
    exec(imports)

    # (2) if this is a method, initialise the local variable obj to hold the object the method will be called on
    if method_object is not None:
        obj = method_object

    # (3) continue only when the system has reached a stable state of energy consumption
    if not server_is_stable(max_wait_secs):
        raise TimeoutError(f"System could not reach a stable state within {max_wait_secs} seconds")

    # (4) evaluate the function return. This is where we should measure energy.
    start_time = time.time_ns()
    func_return = eval(function_to_run)
    end_time = time.time_ns()

    # (5) Wait some specified amount of time to measure potentially elevated energy consumption after the function has terminated
    print(f"waiting idle for {wait_after_run_secs} seconds")
    time.sleep(wait_after_run_secs)

    # (5) get the energy data since run_function has been invoked
    # TODO implement this

    # (6) return the energy data, times and status
    return_dict = {
        "energy_data": [], # TODO insert a dataframe here
        "start_time": start_time,
        "end_time": end_time,
        "status": "success"
    }

    if return_result:
        return_dict["return"] = func_return
        return_dict["method_object"] = obj
        # pickle to transmit TODO rethink this
        return_dict = pickle.dumps(return_dict)

    if DEBUG:
        print(f"Performed {function_to_run} on input")
        print(f"Output: {func_return}")
    
    return return_dict


@app.route(API_PATH, methods=["POST"])
@auth.login_required
def run_function_and_return_result():
    # (1) deserialise request data
    function_details = pickle.loads(request.data)
    
    if DEBUG:
        print(f"Received function details: {function_details}")

    # (2) if needed, create module with custom class definition and import it
    custom_class_file = None
    if function_details.custom_class is not None:
        custom_class_file = f"{function_details.module_name}.py"
        with open(custom_class_file, "w") as f:
            f.writelines(function_details.imports  + "\n")
            f.writelines(function_details.custom_class)
        
        exec(f"import {function_details.module_name}")
    
    # (3) Try reaching a stable state and running the function
    # TODO add energy measurement
    try:
        output = run_function(
            function_details.imports,
            function_details.function_to_run,
            function_details.method_object,
            function_details.args,
            function_details.kwargs,
            function_details.max_wait_secs,
            function_details.wait_after_run_secs,
            function_details.return_result
        )
        status = 200
    # TODO format the output dict the same way as in run_function
    except TimeoutError as e:
        output = e
        status = 500

    # (4) send response to client
    if function_details.return_result:
        response = Response(
            response=output,
            status=status,
            mimetype='application/octet_stream'
        )
    else:
        response = Response(
            json=output,
            status=status
        )

    # (5) if needed, delete the module created for the custom class definition
    if custom_class_file is not None:
        if os.path.isfile(custom_class_file):
            os.remove(custom_class_file)
        else:
           raise OSError("Could not remove custom class file")
    app.logger.info("response-value: %s", response)
    return response

@app.route("/")
def index():
    return '<h1>Application Deployed!</h1>'


# start flask app
if __name__ == "__main__":
    app.run(host=SERVER_HOST, port=SERVER_PORT, debug=False, ssl_context=(CA_CERT_PATH, CA_KEY_PATH))