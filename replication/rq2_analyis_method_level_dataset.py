from tool.experiment.analysis import init_project_energy_data, build_total_energy_df, build_total_energy_and_size_df
from tool.experiment.experiments import ExperimentKinds
from replication.executed_experiments import EXECUTED_EXPERIMENTS
import os

import matplotlib.pyplot as plt

# This is for initial analysis of same method call across different projects, and compare the energy consumption and data size from method-level energy dataset.

def plot_args_size_vs_gpu_mean(total_energy_dfs):
    function_values = set()

    for df in total_energy_dfs:
        function_values.update(df['function'].unique())

    for function in function_values:
        fig, ax = plt.subplots(figsize=(8, 6))

        for df in total_energy_dfs:
            data = df[df['function'] == function]
            args_size_mean = data['Args Size (mean)']
            gpu_mean = data['GPU (mean)']
            ax.plot(args_size_mean, gpu_mean, marker='o', linestyle='-', label=f'Project: {df.iloc[0]["Project Name"]}')

        ax.set_xlabel('Args Size (mean)')
        ax.set_ylabel('GPU (mean)')
        ax.set_title(f'Function: {function}')
        ax.legend()

        plt.tight_layout()
        plt.savefig(f'./rq2_analysis/plot_args_size_vs_gpu_mean_{function}.png')
        plt.close()


if __name__ == '__main__':
    total_energy_dfs = []
    csv_file = 'rq2_analysis.csv'
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

        if not total_energy_df.empty:
            total_energy_dfs.append(total_energy_df)

    plot_args_size_vs_gpu_mean(total_energy_dfs)

