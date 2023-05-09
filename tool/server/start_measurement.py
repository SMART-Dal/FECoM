"""
Start the server application together with perf & nvidia-smi
by running this python module.

This has the same effect as running bash energy_measurement.sh,
but is more useful for gathering precise start/end times of perf
& nvidia-smi, which helps to synchronise time measurements with
the server.
"""

from subprocess import Popen
import shlex
import atexit
import time
import os

from tool.server.server_config import MEASUREMENT_INTERVAL_MS, CPU_FILE_SEPARATOR
from tool.server.server_config import NVIDIA_SMI_FILE, PERF_FILE, SERVER_MODULE, START_TIMES_FILE, EXECUTION_LOG_FILE, CPU_TEMPERATURE_MODULE

def print_main(message: str):
    print("[MAIN] " + message)

def quit_process(process: Popen, message: str, print_func):
    process.terminate()
    print_func(f"Terminated {message}")
    del process

def unregister_and_quit_process(process: Popen, message: str):
    """
    This will unregister all instances of quit_process in the local interpreter, so use carefully.
    Used by flask_server.py to quit the CPU temperature process.
    """
    atexit.unregister(quit_process)
    quit_process(process, message, print_main)

# function registered atexit by start_measurements to terminate the measurement programs
def cleanup(perf_stat, nvidia_smi):
    quit_process(nvidia_smi, "nvidia smi", print_main)
    quit_process(perf_stat, "perf stat", print_main)


# start the flask server & make sure it is terminated when start_measurement is quit
def start_server():
    start_server = shlex.split(f"python3 {SERVER_MODULE}")

    server = Popen(start_server)

    server_start_time = time.time_ns()
    atexit.register(print_main, "Terminated the flask server")
    atexit.register(server.terminate)
    print_main(f"Server started at {server_start_time}")

    return server_start_time


# start nvidia-smi and return the process such that it can be registered by cleanup
def start_nvidia():
    # split bash command into a list, which is the required format for subprocess.Popen
    start_nvidia = shlex.split(f"nvidia-smi -i 0 --loop-ms={MEASUREMENT_INTERVAL_MS} --format=csv,noheader --query-gpu=timestamp,power.draw,temperature.gpu")

    # open the file for nvidia-smi output
    with open(NVIDIA_SMI_FILE, "w", encoding="utf-8") as nvidia_smi_file:
        # run nvidia-smi as a subprocess and continue execution of the python program
        nvidia_smi = Popen(start_nvidia, stdout=nvidia_smi_file)
    
    nvidia_smi_start_time = time.time_ns()
    print_main(f"Nvidia-smi started at {nvidia_smi_start_time}")
    
    return nvidia_smi, nvidia_smi_start_time


# start perf stat and return the process such that it can be registered by cleanup (similar to start_nvidia)
def start_perf():
    # equivalent procedure as with nvidia-smi for perf but perf writes to a file on its own
    start_perf = shlex.split(f"perf stat -I {MEASUREMENT_INTERVAL_MS} -e power/energy-pkg/,power/energy-ram/ -o {str(PERF_FILE)} -x \{CPU_FILE_SEPARATOR}")

    perf_stat = Popen(start_perf)

    perf_start_time = time.time_ns()
    print_main(f"Perf started at {perf_start_time}")

    return perf_stat, perf_start_time


# start sensors to track CPU temperature over time
def start_sensors(print_func):
    start_sensors = shlex.split(f"python3 {CPU_TEMPERATURE_MODULE}")

    sensors = Popen(start_sensors)

    sensors_start_time = time.time_ns()
    print_func(f"Sensors started at {sensors_start_time}")

    return sensors


# write start times to a file for further processing
def write_start_times(perf_start_time: int, nvidia_smi_start_time: int, server_start_time: int):
    """
    This method expects start times obtained from the time.time_ns() function such that
    the server can synchronise its timing with the perf & nvidia-smi tools.
    """
    with open(START_TIMES_FILE, "w") as f:
        f.writelines([
            f"PERF_START {perf_start_time}\n",
            f"NVIDIA_SMI_START {nvidia_smi_start_time}\n",
            f"SERVER_START {server_start_time}"
        ])


# called by the main program at initial startup or when restarting the energy measurement script through restart_measurements
def start_measurements(server_start_time):
    perf_stat, perf_start_time = start_perf()
    nvidia_smi, nvidia_smi_start_time = start_nvidia()
    atexit.register(cleanup, perf_stat=perf_stat, nvidia_smi=nvidia_smi)

    write_start_times(perf_start_time, nvidia_smi_start_time, server_start_time)
    return perf_stat, nvidia_smi


# quit, cleanup and restart all measurement programs in a way that avoids any file corruptions to the energy_measurement/out files
def restart_measurements(previous_perf_stat, previous_nvidia_smi, latest_execution, server_start_time):
    # unregister the previous cleanup function
    atexit.unregister(cleanup)
    # terminate the previous processes
    previous_nvidia_smi.terminate()
    del previous_nvidia_smi
    print_main(f"Quit nvidia-smi after executing {latest_execution}")
    previous_perf_stat.terminate()
    del previous_perf_stat
    print_main(f"Quit perf stat after executing {latest_execution}")

    # delete the perf & nvidia-smi files
    if PERF_FILE.is_file() and NVIDIA_SMI_FILE.is_file():
            os.remove(PERF_FILE)
            os.remove(NVIDIA_SMI_FILE)
    else:
        raise OSError("Could not find and remove perf & nvidia files")

    # restart the measurement programs
    perf_stat, nvidia_smi = start_measurements(server_start_time)

    return perf_stat, nvidia_smi

def print_experiment_settings():
    """
    Print the most important experiment settings such that the user
    can confirm they are correct when starting the server.
    """
    from tool.server.server_config import WAIT_PER_STABLE_CHECK_LOOP_S, STABLE_CHECK_TOLERANCE, MEASUREMENT_INTERVAL_S, CHECK_LAST_N_POINTS, CPU_MAXIMUM_TEMPERATURE, GPU_MAXIMUM_TEMPERATURE, CPU_TEMPERATURE_INTERVAL_S
    print_main(f"""### Experiment Settings ###
        "wait_per_stable_check_loop_s": {WAIT_PER_STABLE_CHECK_LOOP_S},
        "tolerance": {STABLE_CHECK_TOLERANCE},
        "measurement_interval_s": {MEASUREMENT_INTERVAL_S},
        "check_last_n_points": {CHECK_LAST_N_POINTS},
        "cpu_max_temp": {CPU_MAXIMUM_TEMPERATURE},
        "gpu_max_temp": {GPU_MAXIMUM_TEMPERATURE},
        "cpu_temperature_interval_s": {CPU_TEMPERATURE_INTERVAL_S}
        """
    )


if __name__ == "__main__":
    print_experiment_settings()
    atexit.register(print_main, "Successfully terminated the server application")
    
    # (1) Start the server & energy measurement programs (perf stat & nvidia-smi).
    # Keep a reference to perf stat & nvidia-smi such that they can be terminated by the program.
    server_start_time = start_server()
    perf_stat, nvidia_smi = start_measurements(server_start_time)

    # (2) Create the server execution log file which keeps track of the functions executed.
    # Initialise previous_execution with the initial contents of the file.
    previous_execution = f"START_SERVER;{server_start_time};Functions executed on the server are logged in this file\n"
    with open(EXECUTION_LOG_FILE, 'w') as f:
        f.write("function_executed;time_stamp;server_status_code\n")
        f.write(previous_execution)

    # (3) Start the main loop and quit when receiving keyboard interrupt (Control-C)
    try:
        while(True):
            # check the execution log: has the server added a new execution?
            with open(EXECUTION_LOG_FILE, 'r') as f:
                latest_execution = f.readlines()[-1]
            
            # When the server adds a new execution to the log file, we want to to restart perf & nvidia-smi to clear the energy measurement files
            if latest_execution != previous_execution:
                # restart all programs, and update the references to point at the new processes
                perf_stat, nvidia_smi = restart_measurements(perf_stat, nvidia_smi, latest_execution.split(";")[0], server_start_time)
                previous_execution = latest_execution
            
            # this is half the time the server waits after receiving a request, which gives the system enough time to restart in between method calls.
            time.sleep(5)
            continue
    except KeyboardInterrupt:
        print_main("\n\nKeyboardInterrupt by User. Shutting down the server application.\n")

    
