"""
Replication code for plots used in the paper. Currently only used for Tim's Final Year Project Report.
"""

from tool.experiment.data import DataLoader
from tool.experiment.experiments import ExperimentKinds
from tool.experiment.analysis import init_project_energy_data
from tool.client.client_config import EXPERIMENT_DIR
from tool.experiment.plot import plot_single_energy_with_times, plot_total_energy_vs_execution_time

def implementation_plot_GPU_energy_with_times():
    """
    create the GPU plot used in Design & Implementation, Processed Data Representation
    """
    dl = DataLoader("keras/classification", EXPERIMENT_DIR, ExperimentKinds.METHOD_LEVEL)
    energy_data_list = dl.load_single_file("experiment-6.json")
    for energy_data in energy_data_list:
        if energy_data.function_name == "tf.keras.Sequential.fit(*args, **kwargs)":
            plot_single_energy_with_times(energy_data, hardware_component="gpu")


def rq1_plot_total_energy_vs_time():
    project_name = "keras/classification"
    data_1 = init_project_energy_data(project_name, ExperimentKinds.METHOD_LEVEL, first_experiment=6, last_experiment=10)
    project_name = "images/cnn"
    data_2 = init_project_energy_data(project_name, ExperimentKinds.METHOD_LEVEL, first_experiment=6, last_experiment=10)

    plot_total_energy_vs_execution_time([data_1, data_2], title=False)

if __name__ == "__main__":
    rq1_plot_total_energy_vs_time()