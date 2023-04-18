"""
Replication code for plots used in the paper. Currently only used for Tim's Final Year Project Report.
"""

from tool.experiment.data import DataLoader
from tool.experiment.experiments import ExperimentKinds
from tool.client.client_config import EXPERIMENT_DIR
from tool.experiment.plot import plot_single_energy_with_times

if __name__ == "__main__":
    # create the GPU plot used in Design & Implementation, Processed Data Representation
    dl = DataLoader("keras/classification", EXPERIMENT_DIR, ExperimentKinds.METHOD_LEVEL)
    energy_data_list = dl.load_single_file("experiment-6.json")
    for energy_data in energy_data_list:
        if energy_data.function_name == "tf.keras.Sequential.fit(*args, **kwargs)":
            plot_single_energy_with_times(energy_data, hardware_component="gpu")