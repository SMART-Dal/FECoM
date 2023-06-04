"""
Functions to run directly before and after a function call to measure energy consumption.
"""

import time
import pickle
import atexit
import json
from pathlib import Path

from tool.measurement.utilities import custom_print
from tool.measurement.start_measurement import start_sensors, quit_process, unregister_and_quit_process

from tool.measurement.stable_check import run_check_loop, server_is_stable_check, temperature_is_low_check
from tool.measurement.measurement_parse import get_current_times, get_energy_data, get_cpu_temperature_data

# TODO these two settings should be moved to measurement_config
from tool.patching.patching_config import MAX_WAIT_S, WAIT_AFTER_RUN_S

from tool.measurement.measurement_config import DEBUG
# stable state constants
from tool.measurement.measurement_config import CPU_STD_TO_MEAN, RAM_STD_TO_MEAN, GPU_STD_TO_MEAN, CPU_MAXIMUM_TEMPERATURE, GPU_MAXIMUM_TEMPERATURE
# stable state settings
from tool.measurement.measurement_config import WAIT_PER_STABLE_CHECK_LOOP_S, CHECK_LAST_N_POINTS, STABLE_CHECK_TOLERANCE, CPU_TEMPERATURE_INTERVAL_S, MEASUREMENT_INTERVAL_S
# file paths and separators
from tool.measurement.measurement_config import PERF_FILE, NVIDIA_SMI_FILE, EXECUTION_LOG_FILE, START_TIMES_FILE

from tool.experiment.experiments import ExperimentKinds

def print_exec(message: str):
    custom_print("execution", message)



def prepare_state():
    """
    Ensure the server is in the right state before starting execution
    """

    # (0) start the cpu temperature measurement process
    sensors = start_sensors(print_exec)
    # give sensors some time to gather initial measurements
    time.sleep(3)
    atexit.register(quit_process, sensors, "sensors", print_exec)


    # (2) continue only when CPU & GPU temperatures are below threshold and the system has reached a stable state of energy consumption

    # (2a) check that temperatures are below threshold, then quit the CPU temperature measurement process
    begin_temperature_check_time = time.time_ns()
    if not run_check_loop(True, MAX_WAIT_S, WAIT_PER_STABLE_CHECK_LOOP_S, "low temperature", temperature_is_low_check, CHECK_LAST_N_POINTS, CPU_MAXIMUM_TEMPERATURE, GPU_MAXIMUM_TEMPERATURE):
        unregister_and_quit_process(sensors, "sensors")
        raise TimeoutError(f"CPU could not cool down to {CPU_MAXIMUM_TEMPERATURE} within {MAX_WAIT_S} seconds")
    unregister_and_quit_process(sensors, "sensors")

    # (2b) check that the CPU, RAM and GPU energy consumption is stable
    begin_stable_check_time = time.time_ns()
    if not run_check_loop(False, MAX_WAIT_S, WAIT_PER_STABLE_CHECK_LOOP_S, "stable state", server_is_stable_check, CHECK_LAST_N_POINTS, STABLE_CHECK_TOLERANCE):
        raise TimeoutError(f"Server could not reach a stable state within {MAX_WAIT_S} seconds")

    # (3) evaluate the function. Get the start & end times from the files and also save their exact values.
    # TODO potentially correct here for the small time offset created by fetching the times for the files. We can use the server times for this.
    start_time_perf, start_time_nvidia = get_current_times(PERF_FILE, NVIDIA_SMI_FILE)
    start_time_execution = time.time_ns()
    
    # start execution
    return start_time_perf, start_time_nvidia, start_time_execution, begin_stable_check_time, begin_temperature_check_time


def store_data(data: dict, experiment_file_path: Path):
    """
    Create a new file and store the data as json, or append it to the existing data in this file.
    """
    
    if DEBUG:
        print(f"Result: {str(data)[:100]}")

    if experiment_file_path.is_file():
        with open(experiment_file_path, 'r') as f:
            file_content = f.read()
        if file_content.strip():
            existing_data = json.loads(file_content)
        else:
            existing_data = []
    else:
        existing_data = []

    existing_data.append(data)
    if DEBUG:
        print("Data loaded from response")
    with open(experiment_file_path, 'w') as f:
        json.dump(existing_data, f)
    if DEBUG:
        print(f"Data written to file {str(experiment_file_path)}")


"""
Core functions
"""

def before_execution():
    """
    Insert directly before the function call in the script.
    """
    # similar to send_request, re-try finding stable state in a loop
    while True:
        try:
            print_exec("Waiting before running function for 10 seconds.")
            time.sleep(10)
            start_time_perf, start_time_nvidia, start_time_execution, begin_stable_check_time, begin_temperature_check_time = prepare_state()
            print_exec("Successfully reached stable state")
            break
        except TimeoutError as e:
            error_file = "timeout_energy_data.json"
            with open(error_file, 'w') as f:
                json.dump(get_energy_data()[0], f)
            time.sleep(30)
            continue  # retry reaching stable state
    
    start_times = {
        "start_time_perf": start_time_perf,
        "start_time_nvidia": start_time_nvidia,
        "start_time_execution": start_time_execution,
        "begin_stable_check_time": begin_stable_check_time,
        "begin_temperature_check_time": begin_temperature_check_time

    }
    return start_times


# TODO remove object_signature when patching is refactored
def after_execution(
        start_times: dict, experiment_file_path: str, function_to_run: str,
        function_args: list = None, function_kwargs: dict = None,
        method_object: str = None, object_signature: str = None, project_level: bool = False):
    """
    Insert directly after the function call in the script.

    For project-level experiments, project_level is True. The patcher should then
    simply prepend the before_execution call to the file, and append the after_execution
    call after the last line.

    The last 5 arguments (times) are provided by the return of before_execution, the rest
    by the patcher.
    """
    end_time_execution = time.time_ns()
    end_time_perf, end_time_nvidia = get_current_times(PERF_FILE, NVIDIA_SMI_FILE)

    # (4) Wait some specified amount of time to measure potentially elevated energy consumption after the function has terminated
    if DEBUG:
        print_exec(f"waiting idle for {WAIT_AFTER_RUN_S} seconds after function execution")
    if WAIT_AFTER_RUN_S > 0:
        time.sleep(WAIT_AFTER_RUN_S)
    
    if DEBUG:
        print_exec(f"Performed {function_to_run[:100]} on input and will now save energy data.")

    # (5) get the energy data & gather all start and end times
    energy_data, df_gpu = get_energy_data()
    cpu_temperatures = get_cpu_temperature_data()

    # "normalise" nvidia-smi start/end times such that the first value in the gpu energy data has timestamp 0
    start_time_nvidia_normalised = start_times["start_time_nvidia"] - df_gpu["timestamp"].iloc[0]
    end_time_nvidia_normalised = end_time_nvidia - df_gpu["timestamp"].iloc[0]

    # get the start times generated by start_measurement.py
    with open(START_TIMES_FILE, 'r') as f:
        raw_times = f.readlines()
    
    # START_TIMES_FILE has format PERF_START <time_perf>\nNVIDIA_SMI_START <time_nvidia>
    sys_start_time_perf, sys_start_time_nvidia = [int(line.strip(' \n').split(" ")[1]) for line in raw_times]

    times = {
        "start_time_execution": start_times["start_time_execution"],
        "end_time_execution": end_time_execution,
        "start_time_perf": start_times["start_time_perf"], 
        "end_time_perf": end_time_perf,
        "sys_start_time_perf": sys_start_time_perf,
        "start_time_nvidia": start_time_nvidia_normalised,
        "end_time_nvidia": end_time_nvidia_normalised,
        "sys_start_time_nvidia": sys_start_time_nvidia,
        "begin_stable_check_time": start_times["begin_stable_check_time"],
        "begin_temperature_check_time": start_times["begin_temperature_check_time"]
    }

    # (6) collect all relevant settings
    settings = {
        "max_wait_s": MAX_WAIT_S,
        "wait_after_run_s": WAIT_AFTER_RUN_S,
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

    # (4) The function has executed successfully. Now add size data and format the return dictionary
    # to be in the format {function_signature: results}
    input_sizes = {
        "args_size": len(pickle.dumps(function_args)) if function_args is not None else None,
        "kwargs_size": len(pickle.dumps(function_kwargs)) if function_kwargs is not None else None,
        "object_size": len(pickle.dumps(method_object)) if method_object is not None else None
    }

    # (7) return the energy data, times, temperatures and settings
    results = {
        "energy_data": energy_data,
        "times": times,
        "cpu_temperatures": cpu_temperatures,
        "settings": settings,
        "input_sizes": input_sizes
    }
    
    if project_level:
        # if this option is enabled, we don't execute a single function, so use project-level as key instead
        # to avoid having a multiline code string as the dict key
        results = {ExperimentKinds.PROJECT_LEVEL.value: results}
    else:
        # if the function executed is a method run on an object (first condition),
        # check if an object signature is available (second condition),
        # if yes, remove obj and prepend it to the call: e.g. obj.fit() becomes tf.keras.Sequential.fit()
        if (method_object is not None) and (object_signature is not None):
            results = {object_signature + function_to_run[3:]: results}
        else:
            results = {function_to_run: results}
        
    # (6) Write the method details to the execution log file with a time stamp (to keep entries unique)
    # This triggers the reload of perf & nvidia-smi, clearing the energy data from the execution of this function
    # (see tool.server.start_measurement for the implementation of this process) 
    with open(EXECUTION_LOG_FILE, 'a') as f:
        f.write(f"{function_to_run};{time.time_ns()}")

    store_data(results, experiment_file_path)