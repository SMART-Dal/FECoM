"""
Server to receive client requests to run ML methods and measure energy.
"""

import time
from datetime import datetime
import os
import logging
import dill as pickle
from statistics import mean, stdev
from pathlib import Path
import json
from typing import List
from flask import Flask, Response, request
from flask_httpauth import HTTPBasicAuth
from werkzeug.security import check_password_hash

# server settings
from tool.server.server_config import API_PATH, DEBUG, SERVER_HOST, SERVER_PORT, USERS, CA_CERT_PATH, CA_KEY_PATH
# file paths and separators
from tool.server.server_config import PERF_FILE, NVIDIA_SMI_FILE, EXECUTION_LOG_FILE, START_TIMES_FILE, CPU_TEMPERATURE_FILE, CPU_FILE_SEPARATOR
# stable state constants
from tool.server.server_config import CPU_STD_TO_MEAN, RAM_STD_TO_MEAN, GPU_STD_TO_MEAN, CPU_MAXIMUM_TEMPERATURE, GPU_MAXIMUM_TEMPERATURE
# stable state settings
from tool.server.server_config import WAIT_PER_STABLE_CHECK_LOOP_S, CHECK_LAST_N_POINTS, STABLE_CHECK_TOLERANCE, CPU_TEMPERATURE_INTERVAL_S
from tool.server.function_details import FunctionDetails # shown unused but still required since this is the class used for sending function details to the server
from tool.server.measurement_parse import parse_nvidia_smi, parse_perf, parse_cpu_temperature



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


def load_last_n_cpu_ram_gpu(n: int, perf_file: Path, nvidia_smi_file: Path, cpu_temperature_file: Path) -> tuple:
    """
    Helper method for is_stable_state to load the last n energy data points
    for CPU, RAM and GPU (in this order)
    """

    # load CPU & RAM data
    cpu_ram = []
    with open(perf_file, 'r') as f:
        # get all lines initially, since otherwise we cannot be sure which values are RAM and which are CPU
        cpu_ram = f.read().splitlines(True)

    # load GPU data
    gpu = []
    with open(nvidia_smi_file, 'r') as f:
        gpu = f.read().splitlines(True)[-n:]
    
    # load CPU temperature data
    cpu_temperature = []
    with open(cpu_temperature_file, 'r') as f:
        cpu_temperature = f.read().splitlines(True)[int(-n/CPU_TEMPERATURE_INTERVAL_S):]

    # generate lists of data
    last_n_cpu_energies = [float(line.strip(' ').split(CPU_FILE_SEPARATOR)[1]) for line in cpu_ram[2::2][-n:]]
    last_n_ram_energies = [float(line.strip(' ').split(CPU_FILE_SEPARATOR)[1]) for line in cpu_ram[3::2][-n:]]
    # create two lists from the nvidia-smi file, one with energy readings and one with temperatures
    last_n_gpu_energies = []
    last_n_gpu_temperatures = []
    for line in gpu:
        split_line = line.split(' ')
        last_n_gpu_energies.append(float(split_line[2]))
        last_n_gpu_temperatures.append(int(split_line[4]))

    # Note: CPU temperature reading frequency is lower/different to that of perf & nvidia-smi. Check config.py for the value.
    last_n_cpu_temperatures = [float(line.strip(' ').split(CPU_FILE_SEPARATOR)[1]) for line in cpu_temperature]

    return last_n_cpu_energies, last_n_ram_energies, last_n_gpu_energies, last_n_cpu_temperatures, last_n_gpu_temperatures


# compare the energy data's standard deviation/mean ratio to that found in a stable state and allow for a tolerance
def energy_is_stable(data: List[float], tolerance: float, stable_std_mean_ratio: float) -> bool:
    std_mean = stdev(data) / mean(data)
    tolerated = (1 + tolerance)*stable_std_mean_ratio
    is_stable = std_mean <= tolerated
    if DEBUG and not is_stable:
        print(f"Not stable: stdev/mean is {std_mean}, which is greater than {tolerated}")
    return is_stable


# compare the mean temperature to the maximum temperature we allow
def temperature_is_low(data: List[int], maximum_temperature: int):
    mean_temperature = mean(data)
    is_low = mean_temperature <= maximum_temperature
    if DEBUG and not is_low:
        print(f"Temperature too high: mean is {mean_temperature}, which is greater than {maximum_temperature}")
    return is_low


def server_is_stable(max_wait_secs: int, wait_per_loop_s: int, tolerance: float, check_last_n_points: int, cpu_max_temp: int, gpu_max_temp: int) -> bool:
    """
    Return True only when the system's energy consumption is stable.
    Settings that determine what "stable" means can be found in config.py.
    """
    # For testing purposes
    if max_wait_secs == 0:
        return True

    # in each loop iteration, load new data, calculate statistics and check if the energy is stable.
    # try this for the specified number of seconds
    for _ in range(int(max_wait_secs/wait_per_loop_s)):
        print(f"Waiting {wait_per_loop_s} seconds to reach stable state.\n")
        time.sleep(wait_per_loop_s)

        cpu_energies, ram_energies, gpu_energies, cpu_temperatures, gpu_temperatures = load_last_n_cpu_ram_gpu(check_last_n_points, PERF_FILE, NVIDIA_SMI_FILE, CPU_TEMPERATURE_FILE)
        if (
            temperature_is_low(gpu_temperatures, gpu_max_temp) and
            energy_is_stable(gpu_energies, tolerance, GPU_STD_TO_MEAN) and
            energy_is_stable(cpu_energies, tolerance, CPU_STD_TO_MEAN) and
            energy_is_stable(ram_energies, tolerance, RAM_STD_TO_MEAN) and
            temperature_is_low(cpu_temperatures, cpu_max_temp)
        ):
            print("Server is stable.")
            return True
        else:
            print("Server is not stable yet.")
            continue
    return False


def get_current_times(perf_file: Path, nvidia_smi_file: Path):
    with open(perf_file, 'r') as f:
        last_line_perf = f.readlines()[-1]
    with open(nvidia_smi_file, 'r') as f:
        last_line_nvidia = f.readlines()[-1]
    
    time_perf = float(last_line_perf.strip(' \n').split(';')[0])
    time_nvidia = datetime.strptime(last_line_nvidia.strip('\n').split(',')[0], '%Y/%m/%d %H:%M:%S.%f').timestamp()

    return time_perf, time_nvidia


def get_energy_data():
    df_cpu, df_ram = parse_perf(PERF_FILE)
    df_gpu = parse_nvidia_smi(NVIDIA_SMI_FILE)

    energy_data = {
        "cpu": df_cpu.to_json(orient="split"),
        "ram": df_ram.to_json(orient="split"),
        "gpu": df_gpu.to_json(orient="split") 
    }

    return energy_data, df_gpu


# get a dataframe with all the cpu temperature data and convert it to json
def get_cpu_temperature_data():
    df_cpu_temperature = parse_cpu_temperature(CPU_TEMPERATURE_FILE)
    return df_cpu_temperature.to_json(orient="split")


def run_function(imports: str, function_to_run: str, obj: object, args: list, kwargs: dict, max_wait_secs: int, wait_after_run_secs: int, return_result: bool):
    """
    Run the method given by function_to_run with the given arguments (args) and keyword arguments (kwargs).
    These two variables appear to not be used, however, they are used when evaluating the function_to_run
    since this is a string in the format function_signature(*args), function_signature(**kwargs) or
    function_signature(*args, **kwargs).
    """
    # WARNING: potential security risk from exec and eval statements

    # (1) import relevant modules
    import_time = time.time_ns()
    app.logger.info("Imports value: %s", imports)
    exec(imports)

    # (2) continue only when the system has reached a stable state of energy consumption
    begin_stable_check_time = time.time_ns()
    if not server_is_stable(max_wait_secs, WAIT_PER_STABLE_CHECK_LOOP_S, STABLE_CHECK_TOLERANCE, CHECK_LAST_N_POINTS, CPU_MAXIMUM_TEMPERATURE, GPU_MAXIMUM_TEMPERATURE):
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
    energy_data, df_gpu = get_energy_data()
    cpu_temperatures = get_cpu_temperature_data()

    # "normalise" nvidia-smi start/end times such that the first value in the gpu energy data has timestamp 0
    start_time_nvidia_normalised = start_time_nvidia - df_gpu["timestamp"].iloc[0]
    end_time_nvidia_normalised = end_time_nvidia - df_gpu["timestamp"].iloc[0]

    # get the start times generated by start_measurement.py
    with open(START_TIMES_FILE, 'r') as f:
        raw_times = f.readlines()
    
    # START_TIMES_FILE has format PERF_START <time_perf>\nNVIDIA_SMI_START <time_nvidia>\nSERVER_START <time_server>
    sys_start_time_perf, sys_start_time_nvidia, initial_start_time_server = [int(line.strip(' \n').split(" ")[1]) for line in raw_times]

    times = {
        "initial_start_time_server": initial_start_time_server,
        "start_time_server": start_time_server,
        "end_time_server": end_time_server,
        "start_time_perf": start_time_perf, 
        "end_time_perf": end_time_perf,
        "sys_start_time_perf": sys_start_time_perf,
        "start_time_nvidia": start_time_nvidia_normalised,
        "end_time_nvidia": end_time_nvidia_normalised,
        "sys_start_time_nvidia": sys_start_time_nvidia,
        "import_time": import_time,
        "begin_stable_check_time": begin_stable_check_time
    }

    # (6) collect all relevant settings
    settings = {
        "max_wait_s": max_wait_secs,
        "wait_after_run_s": wait_after_run_secs,
        "wait_per_stable_check_loop_s": WAIT_PER_STABLE_CHECK_LOOP_S,
        "tolerance": STABLE_CHECK_TOLERANCE,
        "check_last_n_points": CHECK_LAST_N_POINTS,
        "cpu_max_temp": CPU_MAXIMUM_TEMPERATURE,
        "gpu_max_temp": GPU_MAXIMUM_TEMPERATURE 
    }

    # (7) return the energy data, times, temperatures and settings
    return_dict = {
        "energy_data": energy_data,
        "times": times,
        "cpu_temperatures": cpu_temperatures,
        "settings": settings
    }

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
    pickle_load_time = time.time_ns()
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
            "energy_data": get_energy_data()[0],
            "error": e
        }
        status = 500
    
    # (4) The function has executed successfully. Now add size data, pickle load time and format the return dictionary
    # to be in the format {function_signature: results}
    if status == 200:
        results["input_sizes"] = {
            "args_size": len(pickle.dumps(function_details.args)) if function_details.args is not None else None,
            "kwargs_size": len(pickle.dumps(function_details.kwargs)) if function_details.kwargs is not None else None,
            "object_size": len(pickle.dumps(function_details.method_object)) if function_details.method_object is not None else None
        }

        results["times"]["pickle_load_time"] = pickle_load_time

        results = {function_details.function_to_run: results}

    # (5) form the response to send to the client, stored in the response variable
    # if return_result is True, we need to serialise the response since we will return an object that is potentially not json-serialisable.
    # if status is not 200, there was an error which also needs to be serialised.
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
        
    # (6) Write the method details to the execution log file with a time stamp (to keep entries unique) and status
    # This triggers the reload of perf & nvidia-smi, clearing the energy data from the execution of this function
    # (see start_measurement.py for implementation of this process) 
    with open(EXECUTION_LOG_FILE, 'a') as f:
        f.write(f"{function_details.function_to_run};{time.time_ns()};{status}\n")

    # (7) if needed, delete the module created for the custom class definition
    if custom_class_file is not None:
        if os.path.isfile(custom_class_file):
            os.remove(custom_class_file)
        else:
           raise OSError("Could not remove custom class file")
    
    app.logger.info("response-value: %s", response)

    return response


# for debugging purposes: allows quickly checking if the server was deployed correctly
@app.route("/")
def index():
    return '<h1>Application Deployed!</h1>'


# start flask app
if __name__ == "__main__":
    app.run(host=SERVER_HOST, port=SERVER_PORT, debug=False, ssl_context=(CA_CERT_PATH, CA_KEY_PATH))