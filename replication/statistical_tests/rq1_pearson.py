import pandas as pd
import numpy as np
from scipy.stats import pearsonr, ttest_ind
from replication.executed_experiments import EXECUTED_EXPERIMENTS
from tool.experiment.analysis import init_project_energy_data
from tool.experiment.experiment_kinds import ExperimentKinds
from tool.experiment.analysis import prepare_total_energy_from_project

experiments_data = []
for project_name in EXECUTED_EXPERIMENTS:
    try:
        data = init_project_energy_data(project_name, ExperimentKinds.METHOD_LEVEL, first_experiment=1, last_experiment=10)
        experiments_data.append(data)
    except Exception as e:
        print("Exception in project: ", project_name)
        raise e
    
data_list = []
for method_level_energy in experiments_data:
    project_data_list, column_names = prepare_total_energy_from_project(method_level_energy)
    data_list.extend(project_data_list)

total_df = pd.DataFrame(data_list, columns=column_names)
print(total_df)
        

# Calculate Pearson correlation coefficients
correlation_cpu, p_value_cpu = pearsonr(total_df['run time'], total_df['CPU (mean)'])
correlation_ram, p_value_ram = pearsonr(total_df['run time'], total_df['RAM (mean)'])
correlation_gpu, p_value_gpu = pearsonr(total_df['run time'], total_df['GPU (mean)'])

# Effect size (Cohen's d) for CPU and RAM (assuming two groups: mean and median)
mean_cpu = total_df['CPU (mean)']
# median_cpu = total_df['CPU (median)']
median_cpu = total_df['run time']
effect_size_cpu = np.abs(np.mean(mean_cpu) - np.mean(median_cpu)) / np.sqrt((np.std(mean_cpu)**2 + np.std(median_cpu)**2) / 2)

mean_ram = total_df['RAM (mean)']
# median_ram = total_df['RAM (median)']
median_ram = total_df['run time']
effect_size_ram = np.abs(np.mean(mean_ram) - np.mean(median_ram)) / np.sqrt((np.std(mean_ram)**2 + np.std(median_ram)**2) / 2)

# Effect size for GPU (assuming two groups: mean and median)
mean_gpu = total_df['GPU (mean)']
# median_gpu = total_df['GPU (median)']
median_gpu = total_df['run time']
effect_size_gpu = np.abs(np.mean(mean_gpu) - np.mean(median_gpu)) / np.sqrt((np.std(mean_gpu)**2 + np.std(median_gpu)**2) / 2)

# Print the results
print("Pearson correlation coefficient and p-value:")
print(f"CPU: Correlation = {correlation_cpu}, p-value = {p_value_cpu}")
print(f"RAM: Correlation = {correlation_ram}, p-value = {p_value_ram}")
print(f"GPU: Correlation = {correlation_gpu}, p-value = {p_value_gpu}")

print("Effect size (Cohen's d):")
print(f"CPU: {effect_size_cpu}")
print(f"RAM: {effect_size_ram}")
print(f"GPU: {effect_size_gpu}")
