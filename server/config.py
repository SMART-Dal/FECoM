SERVER_HOST= "localhost"
SERVER_PORT = 12345
API_PATH = "/api/run_experiment"

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