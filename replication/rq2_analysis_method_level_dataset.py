"""
Create a CSV file of all methods that consume a significant amount of energy.
This is useful for finding methods that are worth investigating further with
data size experiments for RQ2.
"""

from tool.experiment.analysis import init_project_energy_data, build_total_energy_and_size_df
from tool.experiment.experiments import ExperimentKinds
from replication.executed_experiments import EXECUTED_EXPERIMENTS
import os

# This is for initial analysis of same method call across different projects, and compare the energy consumption and data size from method-level energy dataset.


if __name__ == '__main__':
    total_energy_dfs = []
    csv_file = './rq2_analysis/rq2_analysis.csv'
    if os.path.exists(csv_file):
        os.remove(csv_file)

    for project_name in EXECUTED_EXPERIMENTS:
        print(f"Project: {project_name}")
        method_level_data = init_project_energy_data(project_name, ExperimentKinds.METHOD_LEVEL, first_experiment=1)
        # create_summary(method_level_data)

        total_energy_df = build_total_energy_and_size_df(method_level_data)
        print(total_energy_df)

        # Add the project name as the first column in the DataFrame
        total_energy_df.insert(0, 'Project Name', project_name)

        # Append the data to rq2_analysis.csv if the file already exists, otherwise create a new file
        if os.path.exists(csv_file):
            total_energy_df.to_csv(csv_file, mode='a', index=False, header=False)
        else:
            total_energy_df.to_csv(csv_file, index=False)

