import time
import os

from fecom.measurement.measurement_config import CPU_TEMPERATURE_FILE, CPU_TEMPERATURE_INTERVAL_S, CPU_FILE_SEPARATOR


# the "sensors" command from lm_sensors gives statistics about the CPU temperature (https://wiki.archlinux.org/title/lm_sensors)
def run_sensors_once(time_zero = None):
    run_sensor = "sensors"
    time_before = time.time_ns()
    stream = os.popen(run_sensor)
    time_after = time.time_ns()
    
    # approximate sensor execution time by averaging the pre- and post-execution times
    sensor_execution_time = (time_before + time_after) / 2
    
    sensor_output = stream.readlines()

    # go through each line in the sensors output and check if it's the package CPU temperature reading,
    # which always starts with "Package" and has the following format:
    # Package id 0:  +49.0°C  (high = +92.0°C, crit = +102.0°C)
    # Note a key assumption: the machine has one CPU, so we can break after finding "Package id 0"
    package_temperature = None
    for line in sensor_output:
        line_items = line.strip().split()
        if len(line_items) > 0 and "Package"==line.strip().split()[0]:
            # line_items[3] is "+49.0°C", we want only the number
            package_temperature = line_items[3][1:-2]
            break
    
    if package_temperature is None:
        raise ValueError("Could not find Package temperature in sensors command output. To debug, check the output of the 'sensors' command.")
    
    # if time_zero is None, this is the first command
    if time_zero is not None:
        time_elapsed = (sensor_execution_time - time_zero)/1000000000
    else:
        time_elapsed = 0.0
    # open the file for sensor output
    with open(CPU_TEMPERATURE_FILE, "a", encoding="utf-8") as cpu_temperature_file:
        # write timestamp;package_temperature;time_elapsed
        cpu_temperature_file.write(f"{time_elapsed}{CPU_FILE_SEPARATOR}{package_temperature}{CPU_FILE_SEPARATOR}{sensor_execution_time}\n")

    return sensor_execution_time

if __name__ == "__main__":
    # clear cpu measurement file
    with open(CPU_TEMPERATURE_FILE, "w", encoding="utf-8") as cpu_temperature_file:
        pass
    # get time of initial reading
    time_zero = run_sensors_once()
    time.sleep(CPU_TEMPERATURE_INTERVAL_S)

    # run "sensors" in a regular interval
    try:
        while(True):
            run_sensors_once(time_zero)
            time.sleep(CPU_TEMPERATURE_INTERVAL_S)
    except KeyboardInterrupt:
        pass