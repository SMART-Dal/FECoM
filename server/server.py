"""
Server to receive client requests to run ML methods and measure energy.
"""

import time
from datetime import datetime
import os
import logging
import pickle
import statistics as stats
from pathlib import Path
import json
from typing import List

from flask import Flask, Response, request, jsonify
from flask_httpauth import HTTPBasicAuth
from werkzeug.security import check_password_hash

from config import API_PATH, DEBUG, SERVER_HOST, SERVER_PORT, CPU_STD_TO_MEAN, RAM_STD_TO_MEAN, GPU_STD_TO_MEAN, USERS, CA_CERT_PATH, CA_KEY_PATH, PERF_FILE, NVIDIA_SMI_FILE, START_EXECUTION, END_EXECUTION
from function_details import FunctionDetails # shown unused but still required since this is the class used for sending function details to the server
from measurement_parse import parse_nvidia_smi, parse_perf



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

def load_last_n_cpu_ram_gpu(n: int, perf_file: Path, nvidia_smi_file: Path) -> tuple:
    """
    Helper method for is_stable_state to load the last n energy data points
    for CPU, RAM and GPU (in this order)
    """
    # load CPU & RAM data
    cpu_ram = []
    with open(perf_file, 'r') as f:
        cpu_ram = f.read().splitlines(True)
    
    # load GPU data
    gpu = []
    with open(nvidia_smi_file, 'r') as f:
        gpu = f.read().splitlines(True)

    # generate lists of data
    last_n_cpu_energies = [float(line.strip(' ').split(';')[1]) for line in cpu_ram[2::2][-n:]]
    last_n_ram_energies = [float(line.strip(' ').split(';')[1]) for line in cpu_ram[3::2][-n:]]
    last_n_gpu_energies = [float(line.split(' ')[2]) for line in gpu[-n:]]

    return last_n_cpu_energies, last_n_ram_energies, last_n_gpu_energies

def data_is_stable(data: List[float], tolerance: float, stable_std_mean_ratio: float) -> bool:
    return (stats.stdev(data) / stats.mean(data)) <= ((1 + tolerance)*stable_std_mean_ratio)

def server_is_stable(max_wait_secs: int) -> bool:
    """
    Return True only when the system's energy consumption is stable.
    """
    # For testing purposes
    if max_wait_secs == 0:
        return True
    
    # re-calculate statistics every wait_per_loop_secs seconds
    # TODO play around with different values to see impact on energy data
    wait_per_loop_secs = 5

    # only consider the last n points, with perf stat/nvidia-smi interval of 0.5secs this corresponds to the last 10 seconds
    n = 20

    # relative tolerance for difference between stable stdev/mean ratio and current ratio
    # e.g. 0.1 would mean allowing a ratio that's 10% higher than the stable stdev/mean ratio
    tolerance = 1

    # in each loop iteration, load new data, calculate statistics and check if the energy is stable.
    # try this for the specified number of seconds
    for _ in range(int(max_wait_secs/wait_per_loop_secs)):
        cpu_energies, ram_energies, gpu_energies = load_last_n_cpu_ram_gpu(n, PERF_FILE, NVIDIA_SMI_FILE)

        # stable means that stdv/mean ratios are smaller or equal to the stable ratios determined experimentally
        # the tolerance value set above specifies how much relative deviation we want to allow
        cpu_stable = data_is_stable(cpu_energies, tolerance, CPU_STD_TO_MEAN)
        ram_stable = data_is_stable(ram_energies, tolerance, RAM_STD_TO_MEAN)
        gpu_stable = data_is_stable(gpu_energies, tolerance, GPU_STD_TO_MEAN)

        ## commented since it introduces further noise into the server which hinders it from reaching stable state
        # if DEBUG:
        #     print(f"CPU \n stable: {cpu_stable} \n stdv: {stats.stdev(cpu_energies)} \n mean: {stats.mean(cpu_energies)}")
        #     print()
        #     print(f"RAM \n stable: {ram_stable} \nstdv: {stats.stdev(ram_energies)} \n mean: {stats.mean(ram_energies)}")
        #     print()
        #     print(f"GPU \n stable: {gpu_stable} \nstdv: {stats.stdev(gpu_energies)} \n mean: {stats.mean(gpu_energies)}")

        if cpu_stable and ram_stable and gpu_stable:
            return True
        else:
            time.sleep(wait_per_loop_secs)
        

    return False

def get_current_times(perf_file: Path, nvidia_smi_file: Path):
    with open(perf_file, 'r') as f:
        last_line_perf = f.readlines()[-1]
    with open(nvidia_smi_file, 'r') as f:
        last_line_nvidia = f.readlines()[-1]
    
    time_perf = float(last_line_perf.strip(' \n').split(';')[0])
    time_nvidia = datetime.strptime(last_line_nvidia.strip('\n').split(',')[0], '%Y/%m/%d %H:%M:%S.%f').timestamp()

    return time_perf, time_nvidia

def write_start_or_end_symbol(perf_file: Path, nvidia_smi_file: Path, start: bool):
    raise DeprecationWarning("This method doesn't work as expected and should not be used")
    if start:
        symbol = START_EXECUTION + "\n"
    else:
        symbol = END_EXECUTION + "\n"
    print(symbol)
    with open(perf_file, 'a') as f:
        f.write(symbol)
    with open(nvidia_smi_file, 'a') as f:
        f.write(symbol)

def get_energy_data():
    df_cpu, df_ram = parse_perf(PERF_FILE)
    df_gpu = parse_nvidia_smi(NVIDIA_SMI_FILE)

    energy_data = {
        "cpu": df_cpu.to_json(orient="split"),
        "ram": df_ram.to_json(orient="split"),
        "gpu": df_gpu.to_json(orient="split") 
    }

    return energy_data, df_gpu

def run_function(imports: str, function_to_run: str, obj: object, args: list, kwargs: dict, max_wait_secs: int, wait_after_run_secs: int, return_result: bool):
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

    # (2) continue only when the system has reached a stable state of energy consumption
    if not server_is_stable(max_wait_secs):
        raise TimeoutError(f"System could not reach a stable state within {max_wait_secs} seconds")

    # (3) evaluate the function return. Mark the start & end times in the files and save their exact values.
    # TODO potentially correct here for the small time offset created by fetching the times for the files. We can use the server times for this.
    # (old) write_start_or_end_symbol(PERF_FILE, NVIDIA_SMI_FILE, start=True)
    start_time_perf, start_time_nvidia = get_current_times(PERF_FILE, NVIDIA_SMI_FILE)
    start_time_server = time.time_ns()
    func_return = eval(function_to_run)
    end_time_server = time.time_ns()
    end_time_perf, end_time_nvidia = get_current_times(PERF_FILE, NVIDIA_SMI_FILE)
    # (old) write_start_or_end_symbol(PERF_FILE, NVIDIA_SMI_FILE, start=False)

    # (4) Wait some specified amount of time to measure potentially elevated energy consumption after the function has terminated
    if DEBUG:
        print(f"waiting idle for {wait_after_run_secs} seconds")
    if wait_after_run_secs > 0:
        time.sleep(wait_after_run_secs)

    # (5) get the energy data & gather all start and end times
    # TODO only send energy data from the time when run_function is invoked
    # TODO clear the files after
    energy_data, df_gpu = get_energy_data()

    start_time_nvidia_normalised = start_time_nvidia - df_gpu["timestamp"].iloc[0]
    end_time_nvidia_normalised = end_time_nvidia - df_gpu["timestamp"].iloc[0]

    times = {
        "start_time_server": start_time_server,
        "end_time_server": end_time_server,
        "start_time_perf": start_time_perf, 
        "end_time_perf": end_time_perf,
        "start_time_nvidia": start_time_nvidia_normalised,
        "end_time_nvidia": end_time_nvidia_normalised
    }

    # (6) return the energy data, times and status
    return_dict = {
        "energy_data": energy_data,
        "times": times
    }
    # TODO From the meeting: add Total Consumption (still needed?)

    if return_result:
        return_dict["return"] = func_return
        return_dict["method_object"] = obj

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
    
    # (3) Try reaching a stable state and running the function.
    # The experimental results are stored in the results variable
    try:
        results = run_function(
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
        results = {
            "energy_data": get_energy_data(),
            "error": e
        }
        status = 500
    
    # (4) The function has executed successfully. Now add size data and format the return dictionary
    if status == 200:
        results["input_sizes"] = {
            "args_size": len(pickle.dumps(function_details.args)) if function_details.args is not None else None,
            "kwargs_size": len(pickle.dumps(function_details.kwargs)) if function_details.kwargs is not None else None,
            "object_size": len(pickle.dumps(function_details.method_object)) if function_details.method_object is not None else None
        }

        results = {function_details.function_to_run: results}

    # (5) form the response to send to the client, stored in the response variable
    if function_details.return_result or status != 200:
        if DEBUG:
            print("Pickling response to return result")
        response = Response(
            response=pickle.dumps(results),
            status=status,
            mimetype='application/octet_stream'
        )
    else:
        response = Response(
            response=json.dumps(results),
            status=status
        )

    # (6) if needed, delete the module created for the custom class definition
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
    app.run(host=SERVER_HOST, port=SERVER_PORT, debug=True, ssl_context=(CA_CERT_PATH, CA_KEY_PATH))