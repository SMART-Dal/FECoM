"""
Enable local execution of energy measurement experiments, removing the need
to serialise function details.
"""

import time
import pickle
import atexit
import json

from tool.server.utilities import custom_print
from tool.server.start_measurement import start_sensors, quit_process, unregister_and_quit_process
from tool.server.send_request import store_response

# TODO these methods should potentially be refactored into a new module
from tool.server.flask_server import run_check_loop, server_is_stable_check, temperature_is_low_check, get_current_times, get_energy_data, get_cpu_temperature_data

from tool.server.server_config import DEBUG
# stable state constants
from tool.server.server_config import CPU_STD_TO_MEAN, RAM_STD_TO_MEAN, GPU_STD_TO_MEAN, CPU_MAXIMUM_TEMPERATURE, GPU_MAXIMUM_TEMPERATURE
# stable state settings
from tool.server.server_config import WAIT_PER_STABLE_CHECK_LOOP_S, CHECK_LAST_N_POINTS, STABLE_CHECK_TOLERANCE, CPU_TEMPERATURE_INTERVAL_S, MEASUREMENT_INTERVAL_S
# file paths and separators
from tool.server.server_config import PERF_FILE, NVIDIA_SMI_FILE, EXECUTION_LOG_FILE, START_TIMES_FILE #, CPU_TEMPERATURE_FILE, CPU_FILE_SEPARATOR

from tool.experiment.experiments import ExperimentKinds

def print_local(message: str):
    custom_print("local", message)



def prepare_state(max_wait_secs: int):
    """
    Ensure the server is in the right state before starting execution
    """

    # (0) start the cpu temperature measurement process
    sensors = start_sensors(print_local)
    # give sensors some time to gather initial measurements
    time.sleep(3)
    atexit.register(quit_process, sensors, "sensors", print_local)


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
    
    # start execution
    return start_time_perf, start_time_nvidia, start_time_server, begin_stable_check_time, begin_temperature_check_time


def format_request(results, function_to_run, args, kwargs, method_object, object_signature, experiment_file_path, project_level: bool = False):
    # (4) The function has executed successfully. Now add size data and format the return dictionary
    # to be in the format {function_signature: results}
    results["input_sizes"] = {
        "args_size": len(pickle.dumps(args)) if args is not None else None,
        "kwargs_size": len(pickle.dumps(kwargs)) if kwargs is not None else None,
        "object_size": len(pickle.dumps(method_object)) if method_object is not None else None
    }

    if project_level:
        # if this option is enabled, we don't execute a single function, so use project-level as key instead
        # to avoid having a multiline code string as the dict key
        results = {ExperimentKinds.PROJECT_LEVEL.value: results}
    else:
        # if the function executed is a method run on an object (first condition),
        # check if an object signature was sent with the request (second condition),
        # if yes, remove obj and prepend it to the call: e.g. obj.fit() becomes tf.keras.Sequential.fit()
        if (method_object is not None) and (object_signature is not None):
            results = {object_signature + function_to_run[3:]: results}
        else:
            results = {function_to_run: results}
        
    # (6) Write the method details to the execution log file with a time stamp (to keep entries unique) and status
    # This triggers the reload of perf & nvidia-smi, clearing the energy data from the execution of this function
    # (see start_measurement.py for implementation of this process) 
    with open(EXECUTION_LOG_FILE, 'a') as f:
        f.write(f"{function_to_run};{time.time_ns()}")
    

    store_response(results, experiment_file_path)


"""
Core functions
"""

def before_execution(max_wait_secs, wait_after_run_secs):
    """
    Insert directly before the function call in the script.
    """
    # similar to send_request, re-try finding stable state in a loop
    while True:
        try:
            print_local("Waiting before running function for 10 seconds.")
            time.sleep(10)
            start_time_perf, start_time_nvidia, start_time_server, begin_stable_check_time, begin_temperature_check_time = prepare_state(max_wait_secs, wait_after_run_secs)
            print_local("Successfully reached stable state")
            break
        except TimeoutError as e:
            error_file = "timeout_energy_data.json"
            with open(error_file, 'w') as f:
                json.dump(get_energy_data()[0], f)
            time.sleep(30)
            continue  # retry reaching stable state
    
    return start_time_perf, start_time_nvidia, start_time_server, begin_stable_check_time, begin_temperature_check_time


# TODO add default values, and test this method
def after_execution(
        function_to_run, args, kwargs, max_wait_secs, wait_after_run_secs, method_object,
        object_signature, experiment_file_path, project_level,
        start_time_perf, start_time_nvidia, start_time_server,
        begin_stable_check_time, begin_temperature_check_time):
    """
    Insert directly after the function call in the script.

    For project-level experiments, project_level is True. The patcher should then
    simply prepend the before_execution call to the file, and append the after_execution
    call after the last line.

    The last 5 arguments (times) are provided by the return of before_execution, the rest
    by the patcher.
    """
    end_time_server = time.time_ns()
    end_time_perf, end_time_nvidia = get_current_times(PERF_FILE, NVIDIA_SMI_FILE)

    # (4) Wait some specified amount of time to measure potentially elevated energy consumption after the function has terminated
    if DEBUG:
        print_local(f"waiting idle for {wait_after_run_secs} seconds after function execution")
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


    if DEBUG:
        print_local(f"Performed {function_to_run[:100]} on input and will now return energy data.")
    
    format_request(return_dict, function_to_run, args, kwargs,
                   method_object, object_signature, experiment_file_path,
                   project_level)




    # TODO continue by figuring out file handling, then test with manual patching