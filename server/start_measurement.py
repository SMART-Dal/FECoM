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

from config import NVIDIA_SMI_FILE, PERF_FILE, SERVER_MODULE, START_TIMES_FILE, EXECUTION_LOG_FILE, COUNT_INTERVAL_MS


# function registered atexit by start_measurements to terminate nvidia-smi and perf
def cleanup(perf_stat, nvidia_smi):
    print("Terminating the server application")
    nvidia_smi.terminate()
    print("Terminated nvidia smi")
    perf_stat.terminate()
    print("Terminated perf stat")


# start the flask server & make sure it is terminated when start_measurement is quit
def start_server():
    start_server = shlex.split(f"python3 {SERVER_MODULE}")

    server = Popen(start_server)

    server_start_time = time.time_ns()
    atexit.register(print, "Terminated the flask server")
    atexit.register(server.terminate)
    print(f"Server started at {server_start_time}")

    return server_start_time


# start nvidia-smi and return the process such that it can be registered by cleanup
def start_nvidia():
    # split bash command into a list, which is the required format for subprocess.Popen
    start_nvidia = shlex.split(f"nvidia-smi -i 0 --loop-ms={COUNT_INTERVAL_MS} --format=csv,noheader --query-gpu=timestamp,power.draw")

    # open the file for nvidia-smi output
    with open(NVIDIA_SMI_FILE, "w", encoding="utf-8") as nvidia_smi_file:
        # run nvidia-smi as a subprocess and continue execution of the python program
        nvidia_smi = Popen(start_nvidia, stdout=nvidia_smi_file)
    
    nvidia_smi_start_time = time.time_ns()
    print(f"Nvidia-smi started at {nvidia_smi_start_time}")
    
    return nvidia_smi, nvidia_smi_start_time


# start perf stat and return the process such that it can be registered by cleanup (similar to start_nvidia)
def start_perf():
    # equivalent procedure as with nvidia-smi for perf but perf writes to a file on its own
    start_perf = shlex.split(f"perf stat -I {COUNT_INTERVAL_MS} -e power/energy-pkg/,power/energy-ram/ -o {str(PERF_FILE)} -x \;")

    perf_stat = Popen(start_perf)

    perf_start_time = time.time_ns()
    print(f"Perf started at {perf_start_time}")

    return perf_stat, perf_start_time


# write start times to a file for further processing
def write_start_times(perf_start_time, nvidia_smi_start_time):
    with open(START_TIMES_FILE, "w") as f:
        f.writelines([
            f"PERF_START {perf_start_time}\n",
            f"NVIDIA_SMI_START {nvidia_smi_start_time}"
        ])


# called by the main program at initial startup or when restarting the energy measurement script through restart_measurements
def start_measurements():
    perf_stat, perf_start_time = start_perf()
    nvidia_smi, nvidia_smi_start_time = start_nvidia()
    atexit.register(cleanup, perf_stat=perf_stat, nvidia_smi=nvidia_smi)

    write_start_times(perf_start_time, nvidia_smi_start_time)
    return perf_stat, nvidia_smi


# quit, cleanup and restart perf & nvidia-smi in a way that avoids any file corruptions to the energy_measurement/out files
def restart_measurements(previous_perf_stat, previous_nvidia_smi, latest_execution):
    # unregister the previous cleanup function
    atexit.unregister(cleanup)
    # terminate the previous perf & nvidia-smi processes
    previous_nvidia_smi.terminate()
    print(f"Quit nvidia-smi after executing {latest_execution}")
    previous_perf_stat.terminate()
    print(f"Quit perf stat after executing {latest_execution}")

    # delete the perf & nvidia-smi files
    if os.path.isfile(PERF_FILE) and os.path.isfile(NVIDIA_SMI_FILE):
            os.remove(PERF_FILE)
            os.remove(NVIDIA_SMI_FILE)
    else:
        raise OSError("Could not find and remove perf & nvidia files")

    # restart the measurement programs
    perf_stat, nvidia_smi = start_measurements()

    return perf_stat, nvidia_smi


if __name__ == "__main__":
    atexit.register(print, "Successfully terminated the server application")
    
    # (1) Start the server & energy measurement programs (perf stat & nvidia-smi).
    # Keep a reference to perf stat & nvidia-smi such that they can be terminated by the program.
    server_start_time = start_server()
    perf_stat, nvidia_smi = start_measurements()

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
                # restart perf & nvidia-smi, then update the references perf_stat, nvidia_smi to point at the new processes
                perf_stat, nvidia_smi = restart_measurements(perf_stat, nvidia_smi, latest_execution.split(";")[0])
                previous_execution = latest_execution
            
            # TODO this is an arbitary number, does it work well?
            time.sleep(5)
            continue
    except KeyboardInterrupt:
        print("\n\nKeyboardInterrupt: User quit server application\n")

    
