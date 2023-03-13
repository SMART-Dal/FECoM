from pathlib import Path

"""
SERVER URL, PORT & DEBUG CONFIG
"""
SERVER_MODULE = "server.py"

### SET TO FALSE FOR DEV SETTINGS ON YOUR LOCAL MACHINE ###
PROD = False

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

"""
STABLE STATE CONFIG
"""
# standard deviations and means for stable energy data, generated with plot_energy.combined_plot() for 2022-12-10 data
gpu_std = 0.211421
cpu_std = 0.507051
ram_std = 0.085571

gpu_mean = 21.061808
cpu_mean = 17.326540
ram_mean = 3.490937

# TODO remove old values
# CPU_STD_TO_MEAN = 0.0292644 # ~0.03
# RAM_STD_TO_MEAN = 0.0245123 # ~0.025
# GPU_STD_TO_MEAN = 0.0100381 # ~0.01

# average (mean) standard deviations and means for stable energy data for 20 values in a row from 2022-12-10 data, generated with plot_energy.calc_stats_for_split_data()
cpu_energy_stdv = 0.791626
cpu_energy_mean = 17.408177
ram_energy_stdv = 0.107354
ram_energy_mean = 3.503457
gpu_power_stdv = 0.190666
gpu_power_mean = 21.077714

CPU_STD_TO_MEAN = 0.05 # 0.0454744
RAM_STD_TO_MEAN = 0.03 # 0.0303726
GPU_STD_TO_MEAN = 0.01 # 0.0090459

"""
ENERGY MEASUREMENT CONFIG
"""
# set count interval for perf & nvidia-smi in milliseconds
COUNT_INTERVAL_MS = 500
# set cpu temperature measurement interval for sensors in seconds
# the actual interval will be a few milliseconds greater, due to processing time
CPU_TEMPERATURE_INTERVAL_S = 1

# path to find energy data (this is also manually written into the energy_measurement.sh script,
# but imported into the recommended way to start the program: start_measurement.py)
energy_data_dir = Path("energy_measurement/out")
PERF_FILE = energy_data_dir/"perf.txt"
NVIDIA_SMI_FILE = energy_data_dir/"nvidia_smi.txt"
# store start times here
START_TIMES_FILE = energy_data_dir/"start_times.txt"
# keep track of the functions executed by the server in this file
EXECUTION_LOG_FILE = energy_data_dir/"execution_log.txt"
# store CPU temperatures in this file, populated by cpu_temperature.py
CPU_TEMPERATURE_FILE = energy_data_dir/"cpu_temperature.txt"
CPU_TEMPERATURE_MODULE = "cpu_temperature.py"


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