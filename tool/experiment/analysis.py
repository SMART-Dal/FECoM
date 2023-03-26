import copy

from tool.experiment.data import EnergyData, DataLoader, ProjectEnergyData
from tool.experiment.experiments import ExperimentKinds
from tool.client.client_config import EXPERIMENT_DIR

def init_project_energy_data(project: str, experiment_kind: ExperimentKinds, first_experiment: int = 1, last_experiment: int = 10) -> ProjectEnergyData: 
    """
    Initialise a ProjectEnergyData object containing 3 lists of FunctionEnergyData objects (CPU, RAM, GPU),
    each ob which holds lists of data for one function collected over multiple experiments.
    """
    dl = DataLoader(project, EXPERIMENT_DIR, experiment_kind)
    # get all experiment data files for this project
    experiment_files = dl.get_all_data_files()
    print(experiment_files)

    # initialse the project energy data holder object with the number of functions in the first experiment
    function_count = len(dl.load_single_file(experiment_files[0]))
    project_energy_data = ProjectEnergyData(function_count)

    for exp_file in experiment_files[first_experiment-1:last_experiment-1]:
        # exp_data is a list of EnergyData objects for this experiment
        exp_data = dl.load_single_file(exp_file)
        # the number of functions executed should be the same for every experiment, otherwise something went wrong
        assert len(exp_data) == function_count
        for function_number, function_data in enumerate(exp_data):
            # skip a function if it has no energy data, but keep track of it
            if not function_data.has_energy_data:
                project_energy_data.no_energy_functions.add(function_data.function_name)
                continue
            # add CPU data
            project_energy_data.cpu[function_number].total.append(function_data.total_cpu)
            project_energy_data.cpu[function_number].total_normalised.append(function_data.total_cpu_normalised)
            project_energy_data.cpu[function_number].lag_time.append(function_data.cpu_lag_time)
            project_energy_data.cpu[function_number].lag.append(function_data.cpu_lag)
            project_energy_data.cpu[function_number].total_lag_normalised.append(function_data.total_cpu_lag_normalised)
            # add RAM data
            project_energy_data.ram[function_number].total.append(function_data.total_ram)
            project_energy_data.ram[function_number].total_normalised.append(function_data.total_ram_normalised)
            project_energy_data.ram[function_number].lag_time.append(function_data.ram_lag_time)
            project_energy_data.ram[function_number].lag.append(function_data.ram_lag)
            project_energy_data.ram[function_number].total_lag_normalised.append(function_data.total_ram_lag_normalised)
            # add GPU data
            project_energy_data.gpu[function_number].total.append(function_data.total_gpu)
            project_energy_data.gpu[function_number].total_normalised.append(function_data.total_gpu_normalised)
            project_energy_data.gpu[function_number].lag_time.append(function_data.gpu_lag_time)
            project_energy_data.gpu[function_number].lag.append(function_data.gpu_lag)
            project_energy_data.gpu[function_number].total_lag_normalised.append(function_data.total_gpu_lag_normalised)
    
    return project_energy_data



if __name__ == "__main__":
    project_name = "keras/classification"
    data = init_project_energy_data(project_name, ExperimentKinds.METHOD_LEVEL)
    print(data.no_energy_functions)
    print(f"Number of functions: {len(data)}")