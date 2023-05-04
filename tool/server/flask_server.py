"""
Server to receive client requests to run ML methods and measure energy.
"""

import time
import os
import logging
import atexit
import json
import subprocess
import shlex
from statistics import mean, stdev
from pathlib import Path
from datetime import datetime
from typing import List
from werkzeug.security import check_password_hash
import dill as pickle
from flask import Flask, Response, request
from flask_httpauth import HTTPBasicAuth
from tool.server.start_measurement import start_sensors, quit_process, unregister_and_quit_process

# server settings
from tool.server.server_config import API_PATH, DEBUG, SERVER_HOST, SERVER_PORT, USERS, CA_CERT_PATH, CA_KEY_PATH, TEMP_EXEC_CODE_FILE
# file paths and separators
from tool.server.server_config import PERF_FILE, NVIDIA_SMI_FILE, EXECUTION_LOG_FILE, START_TIMES_FILE, CPU_TEMPERATURE_FILE, CPU_FILE_SEPARATOR
# stable state constants
from tool.server.server_config import CPU_STD_TO_MEAN, RAM_STD_TO_MEAN, GPU_STD_TO_MEAN, CPU_MAXIMUM_TEMPERATURE, GPU_MAXIMUM_TEMPERATURE
# stable state settings
from tool.server.server_config import WAIT_PER_STABLE_CHECK_LOOP_S, CHECK_LAST_N_POINTS, STABLE_CHECK_TOLERANCE, CPU_TEMPERATURE_INTERVAL_S, MEASUREMENT_INTERVAL_S
from tool.server.function_details import FunctionDetails # shown unused but still required since this is the class used for sending function details to the server
from tool.server.measurement_parse import parse_nvidia_smi, parse_perf, parse_cpu_temperature

from tool.experiment.experiments import ExperimentKinds


app = Flask(__name__)
auth = HTTPBasicAuth()

logging.basicConfig(filename='flask.log', level=logging.DEBUG,
                    format='%(asctime)s %(levelname)s %(name)s %(message)s')

def print_server(message: str):
    print("[SERVER] " + message)

"""
Authentication
"""

@auth.verify_password
def verify_password(username, password):
    if username in USERS.keys() and check_password_hash(USERS[username], password):
        # typically would return the user object here.
        app.logger.info('Auth successful')
        return True
    else:
        app.logger.info(f'Auth failed for username {username}')
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

"""
Energy & temperature data loaders for stable check
"""

def load_last_n_cpu_temperatures(n: int, cpu_temperature_file: Path) -> list:
    """
    Helper method for cpu_temperature_is_low to load the last n CPU temperature data points.
    """
    cpu_temperature = []
    with open(cpu_temperature_file, 'r') as f:
        cpu_temperature = f.read().splitlines(True)[-n:]
    
    # Note: CPU temperature reading frequency is lower/different to that of perf & nvidia-smi. Check config.py for the value.
    last_n_cpu_temperatures = [float(line.strip(' ').split(CPU_FILE_SEPARATOR)[1]) for line in cpu_temperature]
    return last_n_cpu_temperatures

# load last n GPU data points from nvidia-smi file, used for GPU temperature & energy
def load_last_n_gpu_lines(n: int, nvidia_smi_file: Path):
    with open(nvidia_smi_file, 'r') as f:
        return f.read().splitlines(True)[-n:]


def load_last_n_gpu_temperatures(n: int, nvidia_smi_file: Path) -> list:
    gpu = load_last_n_gpu_lines(n, nvidia_smi_file)
    last_n_gpu_temperatures = [int(line.split(' ')[4]) for line in gpu]
    return last_n_gpu_temperatures


def load_last_n_cpu_ram_gpu_energies(n: int, perf_file: Path, nvidia_smi_file: Path) -> tuple:
    """
    Helper method for server_is_stable_check to load the last n energy data points
    for CPU, RAM and GPU (in this order)
    """

    # load CPU & RAM data
    cpu_ram = []
    with open(perf_file, 'r') as f:
        # get all lines initially, since otherwise we cannot be sure which values are RAM and which are CPU
        cpu_ram = f.read().splitlines(True)
    
    gpu = load_last_n_gpu_lines(n, nvidia_smi_file)

    # generate lists of data
    last_n_cpu_energies = [float(line.strip(' ').split(CPU_FILE_SEPARATOR)[1]) for line in cpu_ram[2::2][-n:]]
    last_n_ram_energies = [float(line.strip(' ').split(CPU_FILE_SEPARATOR)[1]) for line in cpu_ram[3::2][-n:]]
    last_n_gpu_energies = [float(line.split(' ')[2]) for line in gpu]

    return last_n_cpu_energies, last_n_ram_energies, last_n_gpu_energies

"""
Stable check: temperature and energy
"""

# compare the energy data's standard deviation/mean ratio to that found in a stable state and allow for a tolerance
def energy_is_stable(data: List[float], tolerance: float, stable_std_mean_ratio: float) -> bool:
    std_mean = stdev(data) / mean(data)
    tolerated = (1 + tolerance)*stable_std_mean_ratio
    is_stable = std_mean <= tolerated
    if DEBUG and not is_stable:
        print_server(f"Not stable: stdev/mean is {std_mean}, which is greater than {tolerated}")
    return is_stable


# compare the mean temperature to the maximum temperature we allow
def temperature_is_low(data: List[int], maximum_temperature: int):
    mean_temperature = mean(data)
    is_low = mean_temperature <= maximum_temperature
    if DEBUG and not is_low:
        print_server(f"Temperature too high: mean is {mean_temperature}, which is greater than {maximum_temperature}")
    return is_low


def server_is_stable_check(check_last_n_points: int, tolerance: float) -> bool:
    """
    Return True if all the energy data series are stable
    Settings that determine what "stable" means can be found in server_config.py.
    """
    cpu_energies, ram_energies, gpu_energies = load_last_n_cpu_ram_gpu_energies(check_last_n_points, PERF_FILE, NVIDIA_SMI_FILE)
    if (
        energy_is_stable(gpu_energies, tolerance, GPU_STD_TO_MEAN) and
        energy_is_stable(cpu_energies, tolerance, CPU_STD_TO_MEAN) and
        energy_is_stable(ram_energies, tolerance, RAM_STD_TO_MEAN)
    ):
        print_server("Success: Server is stable.")
        return True
    else:
        print_server("Server is not stable yet.")
        return False


def temperature_is_low_check(check_last_n_points: int, cpu_max_temp: int, gpu_max_temp: int) -> bool:
    """
    Tet the latest CPU & GPU temperatures and check that they are below threshold.
    Return True if both are below threshold, else return False. Settings can be found in server_config.py.
    """
    cpu_temperatures = load_last_n_cpu_temperatures(check_last_n_points, CPU_TEMPERATURE_FILE)
    gpu_temperatures = load_last_n_gpu_temperatures(check_last_n_points, NVIDIA_SMI_FILE)
    if (
        temperature_is_low(gpu_temperatures, gpu_max_temp) and
        temperature_is_low(cpu_temperatures, cpu_max_temp)
    ):
        print_server("Success: temperature is below threshold.")
        return True
    else:
        print_server("Temperature is too high.")
        return False 

def run_check_loop(no_initial_wait: bool, max_wait_secs: int, wait_per_loop_s: int, check_name: str, check_function: callable, *args):
    """
    Return True if the given check_function returns True at some point, else return False.
    """
    # For testing purposes
    if max_wait_secs == 0:
        return True
    
    # is the check already satisfied? Then we don't have to wait and enter the loop.
    if no_initial_wait and check_function(*args):
            return True

    # in each loop iteration, load new data, calculate statistics and perform the check.
    # try this for the specified number of seconds
    for _ in range(int(max_wait_secs/wait_per_loop_s)):
        print_server(f"Waiting {wait_per_loop_s} seconds to reach {check_name}.\n")
        time.sleep(wait_per_loop_s)
        if check_function(*args):
            return True
        
    return False

"""
Energy & temperature data loaders for the server response
"""

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


"""
Core functions
"""

def run_function(imports: str, function_to_run: str, obj: object, args: list, kwargs: dict, max_wait_secs: int, wait_after_run_secs: int, return_result: bool, exec_not_eval: bool):
    """
    Run the method given by function_to_run with the given arguments (args) and keyword arguments (kwargs).
    These two variables appear to not be used, however, they are used when evaluating the function_to_run
    since this is a string in the format function_signature(*args), function_signature(**kwargs) or
    function_signature(*args, **kwargs).
    """
    app.logger.info("Running function: %s", function_to_run[:100])
    # WARNING: potential security risk from exec and eval statements
    
    # if this option is True, function_to_run is a python script that we need to execute.
    # here we write it to a temporary module and prepare the command to execute it
    if exec_not_eval:
        with open(TEMP_EXEC_CODE_FILE, 'w') as f:
            f.write(function_to_run)
        exec_command = shlex.split("python3 out/code_file_tmp.py")

    # (0) start the cpu temperature measurement process
    sensors = start_sensors(print_server)
    # give sensors some time to gather initial measurements
    time.sleep(3)
    atexit.register(quit_process, sensors, "sensors")

    # (1) import relevant modules
    import_time = time.time_ns()
    exec(imports)
    app.logger.info("Imports value: %s", imports)

    # (2) continue only when CPU & GPU temperatures are below threshold and the system has reached a stable state of energy consumption

    # (2a) check that temperatures are below threshold, then quit the CPU temperature measurement process
    begin_temperature_check_time = time.time_ns()
    if not run_check_loop(True, max_wait_secs, WAIT_PER_STABLE_CHECK_LOOP_S, "low temperature", temperature_is_low_check, CHECK_LAST_N_POINTS, CPU_MAXIMUM_TEMPERATURE, GPU_MAXIMUM_TEMPERATURE):
        unregister_and_quit_process(sensors, "sensors")
        raise TimeoutError(f"CPU could not cool down to {CPU_MAXIMUM_TEMPERATURE} within {max_wait_secs} seconds")
    unregister_and_quit_process(sensors, "sensors")

    # (2b) check that the CPU, RAM and GPU energy consumption is stable
    begin_stable_check_time = time.time_ns()
    if not run_check_loop(False, max_wait_secs, WAIT_PER_STABLE_CHECK_LOOP_S, "stable state", server_is_stable_check, CHECK_LAST_N_POINTS, STABLE_CHECK_TOLERANCE):
        raise TimeoutError(f"Server could not reach a stable state within {max_wait_secs} seconds")

    # (3) evaluate the function. Get the start & end times from the files and also save their exact values.
    # TODO potentially correct here for the small time offset created by fetching the times for the files. We can use the server times for this.
    start_time_perf, start_time_nvidia = get_current_times(PERF_FILE, NVIDIA_SMI_FILE)
    start_time_server = time.time_ns()
    if exec_not_eval:
        subprocess.run(exec_command)
        func_return = None
    else:
        func_return = eval(function_to_run)
    end_time_server = time.time_ns()
    end_time_perf, end_time_nvidia = get_current_times(PERF_FILE, NVIDIA_SMI_FILE)

    # (4) Wait some specified amount of time to measure potentially elevated energy consumption after the function has terminated
    if DEBUG:
        print_server(f"waiting idle for {wait_after_run_secs} seconds after function execution")
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
        "begin_stable_check_time": begin_stable_check_time,
        "begin_temperature_check_time": begin_temperature_check_time
    }

    # (6) collect all relevant settings
    settings = {
        "max_wait_s": max_wait_secs,
        "wait_after_run_s": wait_after_run_secs,
        "wait_per_stable_check_loop_s": WAIT_PER_STABLE_CHECK_LOOP_S,
        "tolerance": STABLE_CHECK_TOLERANCE,
        "measurement_interval_s": MEASUREMENT_INTERVAL_S,
        "cpu_std_to_mean": CPU_STD_TO_MEAN,
        "ram_std_to_mean": RAM_STD_TO_MEAN,
        "gpu_std_to_mean": GPU_STD_TO_MEAN,
        "check_last_n_points": CHECK_LAST_N_POINTS,
        "cpu_max_temp": CPU_MAXIMUM_TEMPERATURE,
        "gpu_max_temp": GPU_MAXIMUM_TEMPERATURE,
        "cpu_temperature_interval_s": CPU_TEMPERATURE_INTERVAL_S
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
        print_server(f"Performed {function_to_run[:100]} on input and will now return energy data.")
    
    return return_dict


@app.route(API_PATH, methods=["POST"])
@auth.login_required
def run_function_and_return_result():
    # (1) deserialise request data
    pickle_load_time = time.time_ns()
    function_details = pickle.loads(request.data)
    
    if DEBUG:
        print_server(f"Received function details for function {function_details.function_to_run[:100]}")

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
        print_server("Waiting before running function for 10 seconds.")
        time.sleep(10)
        results = run_function(
            function_details.imports,
            function_details.function_to_run,
            function_details.method_object,
            function_details.args,
            function_details.kwargs,
            function_details.max_wait_secs,
            function_details.wait_after_run_secs,
            function_details.return_result,
            function_details.exec_not_eval
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

        if function_details.exec_not_eval:
            # if this option is enabled, we don't execute a single function, so use project-level as key instead
            # to avoid having a multiline code string as the dict key
            results = {ExperimentKinds.PROJECT_LEVEL.value: results}
            if os.path.isfile(TEMP_EXEC_CODE_FILE):
                os.remove(TEMP_EXEC_CODE_FILE)
            else:
                raise OSError("Could not remove temporary code file")
        else:
            # if the function executed is a method run on an object (first condition),
            # check if an object signature was sent with the request (second condition),
            # if yes, remove obj and prepend it to the call: e.g. obj.fit() becomes tf.keras.Sequential.fit()
            if (function_details.method_object is not None) and (function_details.object_signature is not None):
                results = {function_details.object_signature + function_details.function_to_run[3:]: results}
            else:
                results = {function_details.function_to_run: results}

    # (5) form the response to send to the client, stored in the response variable
    # if return_result is True, we need to serialise the response since we will return an object that is potentially not json-serialisable.
    # if status is not 200, there was an error which also needs to be serialised.
    if function_details.return_result or status != 200:
        if DEBUG:
            print_server("Pickling response to return result")
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