import copy

from tool.experiment.data import EnergyData, DataLoader
from tool.experiment.experiments import ExperimentKinds
from tool.client.client_config import EXPERIMENT_DIR

def analyse(project: str, experiment_kind: ExperimentKinds): 
    dl = DataLoader(project, EXPERIMENT_DIR, experiment_kind)
    # get all experiment data files for this project
    data_files = dl.get_all_data_files()
    print(data_files)

    # TODO continue here

    

    # all_method_data = []

    # for data_file in data_files:
    #     # list of EnergyData objects for this experiment
    #     exp_data = dl.load_single_file(data_file)
    #     for i, method_data in enumerate(exp_data):
    #         if method_data.function_name in all_method_data


    # print(example_data.function_name)
    # if example_data.has_energy_data:
    #     print(f"NVIDIA STABLE STATE: {example_data.begin_stable_check_time_nvidia}")
    #     print(f"Execution time: {example_data.execution_time_s}")
    #     print("CPU")
    #     print(example_data.total_cpu)
    #     print(example_data.total_cpu_normalised)
    #     print(example_data.cpu_lag_time)
    #     print(example_data.cpu_lag)
    #     print(example_data.cpu_lag_normalised)
    #     print(example_data.total_cpu_lag_normalised)
    #     print("RAM")
    #     print(example_data.total_ram)
    #     print(example_data.total_ram_normalised)
    #     print(example_data.ram_lag_time)
    #     print(example_data.ram_lag)
    #     print(example_data.ram_lag_normalised)
    #     print(example_data.total_ram_lag_normalised)
    #     print("GPU")
    #     print(example_data.total_gpu)
    #     print(example_data.total_gpu_normalised)
    #     print(example_data.gpu_lag_time)
    #     print(example_data.gpu_lag)
    #     print(example_data.gpu_lag_normalised)
    #     print(example_data.total_gpu_lag_normalised)
    #     plot_energy_with_times(example_data)



if __name__ == "__main__":
    project_name = "keras/classification"