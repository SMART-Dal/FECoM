from tool.experiment.data import DataLoader
from tool.experiment.experiment_kinds import ExperimentKinds
from tool.experiment.analysis import init_project_energy_data, build_total_energy_df
from executed_experiments import EXECUTED_RQ1_EXPERIMENTS, EXECUTED_RQ2_EXPERIMENTS
import numpy as np

import matplotlib.pyplot as plt
import scipy.stats as stats

def check_normality(data, title):

    # Shapiro-Wilk Test
    statistic, p_value = stats.shapiro(data)
    print(f"Shapiro-Wilk Test for {title}:")
    print(f"Statistic: {statistic}, p-value: {p_value}")

    if p_value < 0.05:
        print(f"The data for {title} is not normally distributed.")
    else:
        print(f"The data for {title} is normally distributed.")


def check_significance_wilcoxon(project_level, method_level, title):
    zstatistic, p_value = stats.wilcoxon(project_level, method_level, alternative='greater')

    # Calculate the rank-biserial correlation coefficient (effect size)
    n = len(project_level)
    r = zstatistic / np.sqrt(n)
    
    print(f"Wilcoxon Signed-Rank Test for {title}, the r value is {r}, zstats is {zstatistic}")

    # Check the p-value
    if p_value < 0.05:
        print(f"The method-level {title} energy consumption is significantly less than the project-level {title} energy consumption.",p_value)
    else:
        print(f"There is no significant difference in {title} energy consumption between method level and project level.",p_value)


if __name__ == "__main__":
    total_energy_projects = {}
    for project_name in EXECUTED_RQ1_EXPERIMENTS:
        method_level_data = init_project_energy_data(project_name, ExperimentKinds.METHOD_LEVEL, first_experiment=1)
        project_level_data = init_project_energy_data(project_name, ExperimentKinds.PROJECT_LEVEL, first_experiment=1)
        total_energy_df = build_total_energy_df(method_level_data, project_level_data)
        total_energy_projects[project_name] = total_energy_df

    x = []
    y_cpu_method = []
    y_cpu_project = []
    y_gpu_method = []
    y_gpu_project = []
    y_ram_method = []
    y_ram_project = []

    for project_name, total_energy_df in total_energy_projects.items():
        project_data = total_energy_df[total_energy_df['function'].isin(['project-level', 'method-level (sum)'])]

        if 'method-level (sum)' not in project_data['function'].values:
            project_data = project_data.append({'function': 'method-level (sum)'}, ignore_index=True)
            project_data.fillna(0, inplace=True)

        x.append(project_name)
        y_cpu_method.append(project_data.loc[project_data['function'] == 'method-level (sum)', 'CPU (mean)'].tolist()[0])
        y_cpu_project.append(project_data.loc[project_data['function'] == 'project-level', 'CPU (mean)'].tolist()[0])
        y_gpu_method.append(project_data.loc[project_data['function'] == 'method-level (sum)', 'GPU (mean)'].tolist()[0])
        y_gpu_project.append(project_data.loc[project_data['function'] == 'project-level', 'GPU (mean)'].tolist()[0])
        y_ram_method.append(project_data.loc[project_data['function'] == 'method-level (sum)', 'RAM (mean)'].tolist()[0])
        y_ram_project.append(project_data.loc[project_data['function'] == 'project-level', 'RAM (mean)'].tolist()[0])

    check_normality(y_cpu_method, "CPU at Method Level")
    check_normality(y_cpu_project, "CPU at Project Level")
    check_normality(y_gpu_method, "GPU at Method Level")
    check_normality(y_gpu_project, "GPU at Project Level")
    check_normality(y_ram_method, "RAM at Method Level")
    check_normality(y_ram_project, "RAM at Project Level")

    check_significance_wilcoxon(y_cpu_project, y_cpu_method,"cpu")
    check_significance_wilcoxon(y_gpu_project, y_gpu_method,"gpu")
    check_significance_wilcoxon(y_ram_project, y_ram_method,"ram")

