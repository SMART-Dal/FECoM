import pandas as pd
import numpy as np
from scipy.stats import pearsonr
# from executed_experiments import EXECUTED_RQ2_EXPERIMENTS
from tool.experiment.analysis import init_project_energy_data
from tool.experiment.experiment_kinds import ExperimentKinds

EXECUTED_RQ2_EXPERIMENTS = [
    "estimator/keras_model_to_estimator_train",
    "generative/autoencoder",
    "images/cnn_evaluate",
    "images/cnn_fit",
    "keras/classification_evaluate",
    "keras/classification_fit",
    "keras/regression_adapt",
    "keras/regression_predict",
    "load_data/numpy",
    "quickstart/beginner_fit"
]

# Initialize lists to store results
correlation_cpu_list = []
correlation_ram_list = []
correlation_gpu_list = []
p_value_cpu_list = []
p_value_ram_list = []
p_value_gpu_list = []

# Collect data from all projects
all_total_energies_cpu = []
all_args_sizes_cpu = []
all_total_energies_ram = []
all_args_sizes_ram = []
all_total_energies_gpu = []
all_args_sizes_gpu = []

for project_name in EXECUTED_RQ2_EXPERIMENTS:
    try:
        data = init_project_energy_data(project_name, ExperimentKinds.DATA_SIZE, first_experiment=1, last_experiment=10)
        
        for hardware in ["cpu", "ram", "gpu"]:
            function_energies = getattr(data, hardware)
            total_energies = []
            args_sizes = []
            for function_name, function_energy in function_energies.items():
                # assert len(set(function_energy.total_args_size)) == 1, "The argument size of the same function should be the same across experiments."
                args_sizes.append(int(function_energy.total_args_size[0]))
                total_energies.append(function_energy.mean_total_normalised)
                
            if hardware == "cpu":
                all_total_energies_cpu.extend(total_energies)
                all_args_sizes_cpu.extend(args_sizes)
            elif hardware == "ram":
                all_total_energies_ram.extend(total_energies)
                all_args_sizes_ram.extend(args_sizes)
            elif hardware == "gpu":
                all_total_energies_gpu.extend(total_energies)
                all_args_sizes_gpu.extend(args_sizes)
                
    except Exception as e:
        print("Exception in project: ", project_name)
        raise e

# Calculate Pearson correlation coefficient and p-value for CPU
correlation_cpu, p_value_cpu = pearsonr(all_args_sizes_cpu, all_total_energies_cpu)

# Calculate Pearson correlation coefficient and p-value for RAM
correlation_ram, p_value_ram = pearsonr(all_args_sizes_ram, all_total_energies_ram)

# Calculate Pearson correlation coefficient and p-value for GPU
correlation_gpu, p_value_gpu = pearsonr(all_args_sizes_gpu, all_total_energies_gpu)

# Print the results for each hardware component
print("Pearson correlation coefficient:")
print(f"CPU: {correlation_cpu}")
print(f"RAM: {correlation_ram}")
print(f"GPU: {correlation_gpu}")

print("P-values:")
print(f"CPU: {p_value_cpu}")
print(f"RAM: {p_value_ram}")
print(f"GPU: {p_value_gpu}")
