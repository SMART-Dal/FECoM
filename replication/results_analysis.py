from tool.experiment.analysis import ExperimentKinds, DataLoader
from tool.experiment.plot import plot_energy_with_times, plot_combined

if __name__ == "__main__":
    # INITIAL TESTING
    from tool.client.client_config import EXPERIMENT_DIR
    dl = DataLoader("keras/classification", EXPERIMENT_DIR,
                    ExperimentKinds.METHOD_LEVEL)
    data_files = dl.get_all_data_files()
    print(data_files)
    example_data = dl.load_single_file(data_files[0])[2]
    print(example_data.function_name)
    if example_data.has_energy_data:
        print(f"Execution time: {example_data.execution_time_s}")
        print("CPU")
        print(example_data.total_cpu)
        print(example_data.total_cpu_normalised)
        print(example_data.cpu_lag_time)
        print(example_data.cpu_lag)
        print(example_data.cpu_lag_normalised)
        print(example_data.total_cpu_lag_normalised)
        print("RAM")
        print(example_data.total_ram)
        print(example_data.total_ram_normalised)
        print(example_data.ram_lag_time)
        print(example_data.ram_lag)
        print(example_data.ram_lag_normalised)
        print(example_data.total_ram_lag_normalised)
        print("GPU")
        print(example_data.total_gpu)
        print(example_data.total_gpu_normalised)
        print(example_data.gpu_lag_time)
        print(example_data.gpu_lag)
        print(example_data.gpu_lag_normalised)
        print(example_data.total_gpu_lag_normalised)
        plot_energy_with_times(example_data)