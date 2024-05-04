"""
Perform stable check for energy and temperature, run before every energy measurement experiment.
"""

import time
from statistics import mean, stdev
from pathlib import Path
from typing import List

from fecom.measurement.measurement_config import DEBUG, PERF_FILE, NVIDIA_SMI_FILE, CPU_TEMPERATURE_FILE, CPU_FILE_SEPARATOR
from fecom.measurement.utilities import custom_print

def print_stable(message: str):
    custom_print("stable check", message)

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
    Helper method for machine_is_stable_check to load the last n energy data points
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
        print_stable(f"Not stable: stdev/mean is {std_mean}, which is greater than {tolerated}")
    return is_stable


# compare the mean temperature to the maximum temperature we allow
def temperature_is_low(data: List[int], maximum_temperature: int):
    mean_temperature = mean(data)
    is_low = mean_temperature <= maximum_temperature
    if DEBUG and not is_low:
        print_stable(f"Temperature too high: mean is {mean_temperature}, which is greater than {maximum_temperature}")
    return is_low


def machine_is_stable_check(check_last_n_points: int, tolerance: float, cpu_std_to_mean: float, ram_std_to_mean: float, gpu_std_to_mean: float) -> bool:
    """
    Return True if all the energy data series are stable
    Settings that determine what "stable" means can be found in measurement_config.py.
    """
    cpu_energies, ram_energies, gpu_energies = load_last_n_cpu_ram_gpu_energies(check_last_n_points, PERF_FILE, NVIDIA_SMI_FILE)
    if (
        energy_is_stable(gpu_energies, tolerance, gpu_std_to_mean) and
        energy_is_stable(cpu_energies, tolerance, cpu_std_to_mean) and
        energy_is_stable(ram_energies, tolerance, ram_std_to_mean)
    ):
        print_stable("Success: Machine is stable.")
        return True
    else:
        print_stable("Machine is not stable yet.")
        return False


def temperature_is_low_check(check_last_n_points: int, cpu_max_temp: int, gpu_max_temp: int) -> bool:
    """
    Tet the latest CPU & GPU temperatures and check that they are below threshold.
    Return True if both are below threshold, else return False. Settings can be found in measurement_config.py.
    """
    cpu_temperatures = load_last_n_cpu_temperatures(check_last_n_points, CPU_TEMPERATURE_FILE)
    gpu_temperatures = load_last_n_gpu_temperatures(check_last_n_points, NVIDIA_SMI_FILE)
    if (
        temperature_is_low(gpu_temperatures, gpu_max_temp) and
        temperature_is_low(cpu_temperatures, cpu_max_temp)
    ):
        print_stable("Success: temperature is below threshold.")
        return True
    else:
        print_stable("Temperature is too high.")
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
        print_stable(f"Waiting {wait_per_loop_s} seconds to reach {check_name}.\n")
        time.sleep(wait_per_loop_s)
        if check_function(*args):
            return True
        
    return False