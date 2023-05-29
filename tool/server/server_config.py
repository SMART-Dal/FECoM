from pathlib import Path
from tool.client.client_config import PROJECT_PATH

"""
SERVER URL, PORT & DEBUG CONFIG
"""
SERVER_MODULE = "flask_server.py"

### SET TO FALSE FOR DEV SETTINGS ON YOUR LOCAL MACHINE ###
PROD = True

# server settings for localhost environment
DEV_HOST = "localhost"
DEV_PORT = 54321

# server settings for production environment
PROD_HOST = "129.173.67.60" # can also set to "0.0.0.0" on the server
PROD_PORT = 8080

# configure server settings here
API_PATH = "/api/run_experiment"
SERVER_HOST = PROD_HOST if PROD else DEV_HOST
SERVER_PORT = PROD_PORT if PROD else DEV_PORT
URL = "https://"+SERVER_HOST+":"+str(SERVER_PORT)+API_PATH

# set this to True to get print outs as the server receives and processes requests
DEBUG = True

TEMP_EXEC_CODE_FILE = Path("out/code_file_tmp.py")

"""
STABLE STATE CONFIG
"""
# re-calculate statistics every x seconds when checking stable state
WAIT_PER_STABLE_CHECK_LOOP_S = 20

# only consider the last n points, with perf stat/nvidia-smi interval of 0.5secs this corresponds to the last 10 seconds
CHECK_LAST_N_POINTS = 20

# relative tolerance for difference between stable stdev/mean ratio and current ratios as measured by the server
# e.g. 0.1 would mean allowing a ratio that's 10% higher than the stable stdev/mean ratio
STABLE_CHECK_TOLERANCE = 0.5

# average (mean) standard deviations and means for stable energy data for 20 values in a row from, generated with replication/settings_calculation.stdev_mean_ratios()
# Do not change these between experiments!
CPU_STD_TO_MEAN = 0.06
RAM_STD_TO_MEAN = 0.03
GPU_STD_TO_MEAN = 0.01
# CPU_STD_TO_MEAN = 0.23
# RAM_STD_TO_MEAN = 0.15
# GPU_STD_TO_MEAN = 0.01

# used by tool.experiment.analysis
STABLE_CPU_ENERGY_STDEV = 1.125721
STABLE_CPU_ENERGY_MEAN = 17.326185
STABLE_RAM_ENERGY_STDEV = 0.119762
STABLE_RAM_ENERGY_MEAN = 3.559304
STABLE_GPU_POWER_STDEV = 0.153557
STABLE_GPU_POWER_MEAN = 19.654604
# STABLE_CPU_ENERGY_STDEV = 9.656240
# STABLE_CPU_ENERGY_MEAN = 42.266863
# STABLE_RAM_ENERGY_STDEV = 0.839687
# STABLE_RAM_ENERGY_MEAN = 5.500122
# STABLE_GPU_POWER_STDEV = 0.158472
# STABLE_GPU_POWER_MEAN = 18.096958

"""
ENERGY MEASUREMENT CONFIG
"""
# this is also used for the cpu temperature file
CPU_FILE_SEPARATOR = ';'
# set measurement interval for perf & nvidia-smi in milliseconds
MEASUREMENT_INTERVAL_MS = 500
# having the measurement interval in seconds is useful for converting power to energy and vice versa
MEASUREMENT_INTERVAL_S = MEASUREMENT_INTERVAL_MS / 1000
# path to find energy data relative to the server package
# if you want to use this in another package, you could
# - use os.path.dirname(os.path.abspath(__file__)) to get your current file's absolute path
# - use a relative path from your current file
# (!) Change this accordingly (!)
# if you want to use local execution, provide a path that starts from the PROJECT_PATH (from client_config)
energy_data_dir = PROJECT_PATH / "tool/server/out" # local execution
# energy_data_dir = Path("out") # server execution
PERF_FILE = energy_data_dir/"perf.txt"
NVIDIA_SMI_FILE = energy_data_dir/"nvidia_smi.txt"
# store start times here
START_TIMES_FILE = energy_data_dir/"start_times.txt"
# keep track of the functions executed by the server in this file
EXECUTION_LOG_FILE = energy_data_dir/"execution_log.txt"


"""
TEMPERATURE MEASUREMENT CONFIG
"""
CPU_TEMPERATURE_MODULE = "cpu_temperature.py"
# set cpu temperature measurement interval for sensors in seconds
# the actual interval will be a few milliseconds greater, due to processing time
CPU_TEMPERATURE_INTERVAL_S = 1
# store CPU temperatures in this file, populated by cpu_temperature.py
CPU_TEMPERATURE_FILE = energy_data_dir/"cpu_temperature.txt"
# the maximum average temperature in degrees Celsius that we allow the CPU & GPU to be before executing a method (to determine stable state)
CPU_MAXIMUM_TEMPERATURE = 55 # see replication package for how we arrived at this value
GPU_MAXIMUM_TEMPERATURE = 40


"""
AUTHENTICATION & SSL CONFIG
"""
# authentication
USERS = {
    "tim9220": "pbkdf2:sha256:260000$HCuie8IPjFLW3ZcA$57da68fec635caf7922a77f64781669c5427ace7eae1d8cb67e218f5e363956f",
    "saurabh3188": "pbkdf2:sha256:260000$NEviyP0Zrg5kobkO$7e3037a9053816e51e0e2b920c92b76ae0b9708f2ca755eb48fe76fd07cb5fa4"
}

# HTTPS certificate relative paths from server directory
CA_CERT_PATH = Path("certificates/cert.pem") # public key
CA_KEY_PATH = Path("certificates/key.pem") # private key