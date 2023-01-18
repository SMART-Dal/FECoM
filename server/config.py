# server settings for localhost environment
SERVER_HOST= "localhost"
SERVER_PORT = 12345
API_PATH = "/api/run_experiment"
URL = "https://"+SERVER_HOST+":"+str(SERVER_PORT)+API_PATH

# set this to True to get print outs as the server receives and processes requests
DEBUG = True

# standard deviations and means for stable energy data, generated with plot_energy.combined_plot() for 2022-12-10 data
gpu_std = 0.211421
cpu_std = 0.507051
ram_std = 0.085571

gpu_mean = 21.061808
cpu_mean = 17.326540
ram_mean = 3.490937

GPU_STD_TO_MEAN = 0.0100381 # ~0.01
CPU_STD_TO_MEAN = 0.0292644 # ~0.03
RAM_STD_TO_MEAN = 0.0245123 # ~0.025

# authentication
USERS = {
    "tim9220": "pbkdf2:sha256:260000$HCuie8IPjFLW3ZcA$57da68fec635caf7922a77f64781669c5427ace7eae1d8cb67e218f5e363956f",
    "saurabh3188": "pbkdf2:sha256:260000$NEviyP0Zrg5kobkO$7e3037a9053816e51e0e2b920c92b76ae0b9708f2ca755eb48fe76fd07cb5fa4"
}

# HTTPS certificate relative paths from server directory
CA_CERT_PATH = "./certificates/cert.pem" # public key
CA_KEY_PATH = "./certificates/key.pem" # private key