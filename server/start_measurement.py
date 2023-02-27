"""
Start the server application together with perf & nvidia-smi
by running this python module.

This has the same effect as running bash energy_measurement.sh,
but is more useful for gathering precise start/end times of perf
& nvidia-smi, which helps to synchronise time measurements with
the server.
"""

import subprocess
import shlex
import atexit
import time
from pathlib import Path

from config import NVIDIA_SMI_FILE, PERF_FILE, SERVER_MODULE, START_TIMES_FILE

if __name__ == "__main__":
    atexit.register(print, "Successfully terminated the server application")

    # set count interval for perf & nvidia-smi here, must be a string
    count_interval_ms = "500"
    # split bash command into a list, which is the required format for subprocess.Popen
    start_nvidia = shlex.split(f"nvidia-smi -i 0 --loop-ms={count_interval_ms} --format=csv,noheader --query-gpu=timestamp,power.draw")
    
    # open the file for nvidia-smi output
    with open(NVIDIA_SMI_FILE, "w", encoding="utf-8") as nvidia_smi_file:
        # run nvidia-smi as a subprocess and continue execution of the python program
        nvidia_smi = subprocess.Popen(start_nvidia, stdout=nvidia_smi_file)
    nvidia_smi_start_time = time.time_ns()
    print(f"Nvidia-smi started at {nvidia_smi_start_time}")
    # at system exit, kill nvidia-smi
    atexit.register(nvidia_smi.terminate)

    # equivalent procedure for perf, but perf writes to a file on its own.
    start_perf = shlex.split(f"perf stat -I {count_interval_ms} -e power/energy-pkg/,power/energy-ram/ -o {str(PERF_FILE)} -x \; python3 {SERVER_MODULE} | ts '[%Y-%m-%d %H:%M:%.S]'")
    perf_stat = subprocess.Popen(start_perf)
    perf_start_time = time.time_ns()
    print(f"Perf & server started at {perf_start_time}")
    atexit.register(perf_stat.terminate)

    atexit.register(print, "Terminating the server application")

    # write start times to a file for further processing
    with open(START_TIMES_FILE, "w") as start_times_file:
        start_times_file.writelines([
            f"PERF_START {perf_start_time}\n",
            f"NVIDIA_SMI_START {nvidia_smi_start_time}"
        ])

    # wait until keyboard interrupt with Control-C
    # TODO implement graceful termination
    while(True):
        continue

    
