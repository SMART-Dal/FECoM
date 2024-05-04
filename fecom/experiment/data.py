"""
This module contains four classes for handling experiment data
1. FunctionEnergyData: a data structure for accumulating data for one function, but from multiple experiments for one hardware component (CPU, RAM or GPU)
2. ProjectEnergyData: a data structure for accumulating all data associated with one project
3. EnergyData: a data structure for storing all data associated with one single energy measurement (one function call in one experiment)
4. DataLoader: a helper class for loading the json data into EnergyData objects
"""
import os
import json
import re
from pathlib import Path
from typing import List, Dict
from statistics import mean, median, stdev

import pandas as pd

from fecom.experiment.experiments import format_full_output_dir, ExperimentKinds
from fecom.measurement.measurement_config import SKIP_CALLS_FILE_NAME

NS_CONVERSION = 1_000_000_000 # 1 second = 1,000,000,000 seconds
MB_CONVERSION = 1_048_576 # 1 Megabyte = 1,048,576 bytes


class FunctionEnergyData():
    """
    Each list contains the values collected over multiple experiments
    """
    def __init__(self):
        self.__name = None
        self.total = []
        self.total_normalised = []
        self.lag_time = []
        self.lag = []
        self.lag_normalised = []
        self.total_lag_normalised = []
        self.execution_time = []
        self.total_args_size = []
        self.total_input_size = []
        self.stdev_power = []
        self.mean_power = []
        self.median_power = []
    
    def __len__(self):
        return len(self.total)
    
    def __str__(self):
        return f"FunctionEnergyData of {self.name} \nTotal energy consumptions: {self.total}"
    
    @property
    def name(self):
        if self.__name is None:
            raise AttributeError("Function name was not initialised")
        return self.__name
    
    @name.setter
    def name(self, function_name: str):
        # the following line removes content enclosed by brackets, in this case "func(*args, **kwargs)"" becomes "func"
        # and has been adapted from https://www.geeksforgeeks.org/how-to-remove-text-inside-brackets-in-python/
        name_without_args = re.sub("\(.*?\)","", function_name)
        self.__name = name_without_args
    
    ### mean values
    @property
    def mean_total(self):
        return mean(self.total)
    
    @property
    def mean_total_normalised(self):
        return mean(self.total_normalised)
    
    @property
    def mean_lag_time(self):
        return mean(self.lag_time)
    
    @property
    def mean_lag(self):
        return mean(self.lag)
    
    @property
    def mean_lag_normalised(self):
        return mean(self.lag_normalised)
    
    @property
    def mean_total_lag_normalised(self):
        return mean(self.total_lag_normalised)
    
    @property
    def mean_execution_time(self):
        return mean(self.execution_time)
    
    @property
    def mean_total_args_size(self):
        return mean(self.total_args_size)
    
    @property
    def mean_mean_power(self):
        return mean(self.mean_power)
    
    @property
    def mean_stdev_power(self):
        return mean(self.stdev_power)
    
    ### median values
    @property
    def median_total(self):
        return median(self.total)
    
    @property
    def median_total_normalised(self):
        return median(self.total_normalised)
    
    @property
    def median_lag_time(self):
        return median(self.lag_time)
    
    @property
    def median_lag(self):
        return median(self.lag)
    
    @property
    def median_lag_normalised(self):
        return median(self.lag_normalised)
    
    @property
    def median_total_lag_normalised(self):
        return median(self.total_lag_normalised)
    
    @property
    def median_execution_time(self):
        return median(self.execution_time)
    
    @property
    def median_total_args_size(self):
        return median(self.total_args_size)
    
    @property
    def median_median_power(self):
        return median(self.median_power)
    
    @property
    def median_stdev_power(self):
        return median(self.stdev_power)
    
    # max values

    @property
    def max_total_normalised(self):
        return max(self.total_normalised)
    
    # min values

    @property
    def min_total_normalised(self):
        return min(self.total_normalised)
    

class ProjectEnergyData():
    """
    Contains three lists of FunctionEnergyData objects, one list each for CPU, RAM and GPU.
    The index of a function's data in the list corresponds to its index in the experiment file.
    """
    def __init__(self, project: str, experiment_kind: ExperimentKinds, experiment_count: int):
        self.name = project
        self.experiment_kind = experiment_kind
        self.experiment_count = experiment_count
        self.cpu = {} # dict of format {function_name: FunctionEnergyData}
        self.ram = {}
        self.gpu = {}
        # keep track of functions without energy 
        self.no_energy_functions = set()
        # dict of format {function_name: [execution_time_exp1, execution_time_exp2, ...]}
        self.execution_times = {}
        self.skip_calls = False # bool to indicate whether skip calls was used or not
    
    def __len__(self):
        return len(self.cpu_data)
    
    @property
    def cpu_data(self) -> List[FunctionEnergyData]:
        """
        The FunctionEnergyData objects where there is energy data for all experiments.
        Each object contains CPU data for one function.
        """
        return [data for data in self.cpu.values() if len(data) == self.experiment_count]
    
    @property
    def ram_data(self) -> List[FunctionEnergyData]:
        """
        The FunctionEnergyData objects where there is energy data for all experiments.
        Each object contains RAM data for one function.
        """
        return [data for data in self.ram.values() if len(data) == self.experiment_count]
    
    @property
    def gpu_data(self) -> List[FunctionEnergyData]:
        """
        The FunctionEnergyData objects where there is energy data for all experiments.
        Each object contains GPU data for one function.
        """
        return [data for data in self.gpu.values() if len(data) == self.experiment_count]
    
    @property
    def total_function_count(self) -> int:
        """
        The total number of functions in this project is the sum of the number of functions
        with energy data and the number of functions without energy data.
        """
        return len(self) + len(self.no_energy_functions)
    
    @property
    def no_energy_function_count(self) -> int:
        """
        The number of functions without energy data.
        """
        return len(self.no_energy_functions)
    
    @property
    def energy_function_count(self) -> int:
        """
        The number of functions with energy data.
        """
        return len(self)
    
    @property
    def no_energy_functions_execution_time_stats(self) -> list:
        """
        Calculate stats about the execution times of functions without energy data.
        Returns a list of tuples of the form (stdev_time, median_time, range_time) for each function without energy data.
        """
        if self.skip_calls:
            raise ValueError("Cannot calculate execution time stats for functions with skip calls")
        # only consider functions without energy data
        filtered_execution_times = {function_name: self.execution_times[function_name] for function_name in self.no_energy_functions}
    
        execution_times_stats = []
        for execution_times in filtered_execution_times.values():
            stdev_time = stdev(execution_times)
            median_time = median(execution_times)
            max_time = max(execution_times)
            min_time = min(execution_times)
            range_time = max_time - min_time
            execution_times_stats.append((stdev_time, median_time, range_time, max_time, min_time))
        
        return execution_times_stats


class EnergyData():
    """
    Initialised with the raw data for one sample, as obtained from one JSON data.
    This class has a range of useful properties that return statistics calculated from the raw data.
    """
    def __init__(self, function_name, project_name, cpu_energy: pd.DataFrame, ram_energy: pd.DataFrame, gpu_energy: pd.DataFrame, times: dict, input_sizes: dict, settings: dict):
        self.function_name = function_name
        self.project_name = project_name
        self.cpu_energy = cpu_energy
        self.ram_energy = ram_energy
        self.gpu_energy = gpu_energy
        self.times = times
        self.input_sizes = input_sizes
        self.settings = settings

        self.cpu_energy_in_execution = self.__energy_in_execution(
            cpu_energy, self.start_time_perf, self.end_time_perf)
        self.ram_energy_in_execution = self.__energy_in_execution(
            ram_energy, self.start_time_perf, self.end_time_perf)
        self.gpu_energy_in_execution = self.__energy_in_execution(
            gpu_energy, self.start_time_nvidia, self.end_time_nvidia)

        self.__cpu_lag_time = None
        self.__cpu_energy_lag_df = None
        self.__ram_lag_time = None
        self.__ram_energy_lag_df = None
        self.__gpu_lag_time = None
        self.__gpu_energy_lag_df = None

    def __str__(self):
        return f"EnergyData of {self.function_name}\nCPU ENERGY:\n{str(self.cpu_energy)}\nRAM ENERGY:\n{self.ram_energy}\nGPU ENERGY:\n{self.gpu_energy}\n TIMES:\n{self.times}\n INPUT SIZES:\n{self.input_sizes}"
    
    
    ### General properties
    @property
    def has_energy_data(self):
        return not self.__energy_in_execution_is_empty()

    @property
    def stable_check_waiting_time_s(self):
        """
        How long did the application check for stable state?
        """
        return (self.times["start_time_execution"]-self.times["begin_stable_check_time"]) / NS_CONVERSION
    
    @property
    def execution_time_s(self):
        return (self.times["end_time_execution"] - self.times["start_time_execution"]) / NS_CONVERSION

    @property
    def wait_per_stable_check_loop_s(self):
        return self.settings["wait_per_stable_check_loop_s"]
    
    @property
    def measurement_interval_s(self):
        # data gathered until 04 May 2023 does have the settings/measurement_interval_s attribute,
        # so use the previous measurement_config constant value of 0.5 if the attribute does not exist
        measurement_interval_s_old = 0.5
        return self.settings.get("measurement_interval_s", measurement_interval_s_old)
    
    @property
    def wait_after_run_s(self):
        return self.settings["wait_after_run_s"]
    

    ### perf-calibrated times
    @property
    def start_time_perf(self):
        return self.times["start_time_perf"]
    
    @property
    def end_time_perf(self):
        return self.times["end_time_perf"]

    @property
    def begin_stable_check_time_perf(self): 
        return (self.times["begin_stable_check_time"] - self.times["sys_start_time_perf"]) / NS_CONVERSION
    
    @property
    def lag_end_time_cpu(self):
        return self.end_time_perf + self.cpu_lag_time
    
    @property
    def lag_end_time_ram(self):
        return self.end_time_perf + self.ram_lag_time
    
    @property
    def begin_temperature_check_time_perf(self):
        return (self.times["begin_temperature_check_time"] - self.times["sys_start_time_perf"]) / NS_CONVERSION
    
    ### nvidia-calibrated times
    @property
    def start_time_nvidia(self):
        return self.times["start_time_nvidia"]
    
    @property
    def end_time_nvidia(self):
        return self.times["end_time_nvidia"]
    
    @property
    def begin_stable_check_time_nvidia(self):
        return (self.times["begin_stable_check_time"] - self.times["sys_start_time_nvidia"]) / NS_CONVERSION
    
    @property
    def lag_end_time_gpu(self):
        return self.end_time_nvidia + self.gpu_lag_time
    
    @property
    def begin_temperature_check_time_nvidia(self):
        return (self.times["begin_temperature_check_time"] - self.times["sys_start_time_nvidia"]) / NS_CONVERSION
    

    ### Mean stable state energy consumption
    # stable state is defined as the last <wait_per_stable_check_loop_s> seconds of energy data before execution,
    # i.e. the interval of energy data which was deemed "stable" by the stable check

    @property
    def stable_cpu_energy_mean(self):
        stable_mean = self.__stable_energy_pre_execution(self.cpu_energy, self.start_time_perf)["energy (J)"].mean()
        return stable_mean

    @property
    def stable_ram_energy_mean(self):
        stable_mean = self.__stable_energy_pre_execution(self.ram_energy, self.start_time_perf)["energy (J)"].mean()
        return stable_mean

    @property
    def stable_gpu_energy_mean(self):
        return self.stable_gpu_power_mean * self.measurement_interval_s # convert power (W) to energy (J)
    

    ### Mean stable state power draw
    @property
    def stable_cpu_power_mean(self):
        return self.stable_cpu_energy_mean / self.measurement_interval_s # convert energy (J) to power (W)
    
    @property
    def stable_ram_power_mean(self):
        return self.stable_ram_energy_mean / self.measurement_interval_s # convert energy (J) to power (W)
    
    @property
    def stable_gpu_power_mean(self):
        stable_mean = self.__stable_energy_pre_execution(self.gpu_energy, self.start_time_perf)["power_draw (W)"].mean()
        return stable_mean


    ### Total energy consumption during execution.
    @property
    def total_cpu(self):
        # make sure that the filtering works: first time stamp must be start_time_perf and last time stamp must be end_time_perf,
        # since these times are read from the perf file
        try:
            assert round(self.cpu_energy_in_execution["time_elapsed"].iloc[0], 6) >= round(self.start_time_perf, 6)
        except AssertionError:
            # for debugging purposes
            print("self.cpu_energy_in_execution['time_elapsed'].iloc[0]: ", self.cpu_energy_in_execution['time_elapsed'].iloc[0])
            print("self.start_time_perf: ", self.start_time_perf)
            print("### Energy in execution ###")
            print(self.cpu_energy_in_execution)
            print("### Energy ###")
            with pd.option_context('display.max_rows', None, 'display.max_columns', None, 'display.float_format', '{:.20f}'.format):
                print(self.cpu_energy)
            raise AssertionError
        assert self.cpu_energy_in_execution["time_elapsed"].iloc[-1] <= self.end_time_perf
        # return the total energy consumption
        return self.cpu_energy_in_execution["energy (J)"].sum()

    @property
    def total_ram(self):
        # make sure that the filtering works: first time stamp must be start_time_perf and last time stamp must be end_time_perf,
        # since these times are read from the perf file
        assert round(self.ram_energy_in_execution["time_elapsed"].iloc[0], 6) >= round(self.start_time_perf, 6)
        assert self.ram_energy_in_execution["time_elapsed"].iloc[-1] <= self.end_time_perf
        # return the total energy consumption
        return self.ram_energy_in_execution["energy (J)"].sum()
    
    @property
    def total_gpu(self):
        # for the gpu data we have slightly weaker assumptions than for the cpu data, since we "normalise"
        # the gpu time_elapsed & start/end times and limited floating point precision can lead to slight differences in the values
        assert round(self.gpu_energy_in_execution["time_elapsed"].iloc[0], 6) >= round(self.start_time_nvidia, 6)
        assert self.gpu_energy_in_execution["time_elapsed"].iloc[-1] <= self.end_time_nvidia
        return self.gpu_energy_in_execution["power_draw (W)"].sum() * self.measurement_interval_s # convert power to joules
    

    ### Statistics of the power during execution
    @property
    def stdev_power_cpu(self):
        return self.cpu_energy_in_execution.std(numeric_only=True)["energy (J)"]
    
    @property
    def stdev_power_ram(self):
        return self.ram_energy_in_execution.std(numeric_only=True)["energy (J)"]
    
    @property
    def stdev_power_gpu(self):
        return self.gpu_energy_in_execution.std(numeric_only=True)["power_draw (W)"]
    
    @property
    def mean_power_cpu(self):
        return self.cpu_energy_in_execution.mean(numeric_only=True)["energy (J)"]
    
    @property
    def mean_power_ram(self):
        return self.ram_energy_in_execution.mean(numeric_only=True)["energy (J)"]
    
    @property
    def mean_power_gpu(self):
        return self.gpu_energy_in_execution.mean(numeric_only=True)["power_draw (W)"]
    
    @property
    def median_power_cpu(self):
        return self.cpu_energy_in_execution.median(numeric_only=True)["energy (J)"]
    
    @property
    def median_power_ram(self):
        return self.ram_energy_in_execution.median(numeric_only=True)["energy (J)"]
    
    @property
    def median_power_gpu(self):
        return self.gpu_energy_in_execution.median(numeric_only=True)["power_draw (W)"]
    

    ### Normalised energy consumption during execution (subtracting baseline energy consumption)
    @property
    def total_cpu_normalised(self):
        return self.total_cpu - self.__baseline_consumption(self.cpu_energy_in_execution, self.stable_cpu_energy_mean)
    
    @property
    def total_ram_normalised(self):
        return self.total_ram - self.__baseline_consumption(self.ram_energy_in_execution, self.stable_ram_energy_mean)
    
    @property
    def total_gpu_normalised(self):
        return self.total_gpu - (self.__baseline_consumption(self.gpu_energy_in_execution, self.stable_gpu_power_mean) * self.measurement_interval_s) # convert power to joules
    

    ### Energy lag time
    @property
    def cpu_lag_time(self):
        """
        The time passed until energy consumption has returned close to baseline.
        """
        self.__cpu_lag_time_is_zero()
        return self.__cpu_lag_time
    
    @property
    def ram_lag_time(self):
        """
        The time passed until energy consumption has returned close to baseline.
        """
        self.__ram_lag_time_is_zero()
        return self.__ram_lag_time
    
    @property
    def gpu_lag_time(self):
        """
        The time passed until GPU energy consumption has returned close to baseline.
        """
        self.__gpu_lag_time_is_zero()
        return self.__gpu_lag_time
    
    def __cpu_lag_time_is_zero(self):
        """
        Check if cpu lag time and the cpu energy lag df have been initialised, if not, initialise them.
        Return true if cpu_lag_time is 0.
        """
        if self.__cpu_lag_time is None:
            self.__cpu_lag_time, self.__cpu_energy_lag_df = self.__energy_lag(
                self.cpu_energy, self.end_time_perf, self.stable_cpu_energy_mean, 0)
        return self.__cpu_lag_time == 0
   
    def __ram_lag_time_is_zero(self):
        """
        Check if RAM lag time and the RAM energy lag df have been initialised, if not, initialise them.
        Return true if ram_lag_time is 0.
        """
        if self.__ram_lag_time is None:
            self.__ram_lag_time, self.__ram_energy_lag_df = self.__energy_lag(
                self.ram_energy, self.end_time_perf, self.stable_ram_energy_mean, 0)
        return self.__ram_lag_time == 0

    def __gpu_lag_time_is_zero(self) -> bool:
        """
        Check if gpu lag time and the gpu energy lag df have been initialised, if not, initialise them.
        If lag time is zero, return true, else return false.
        GPU standard deviation during stable state is very low, since it is not used at all, therefore we use a reasonably low value 
        for the tolerance that is not too low to give false positives. GPU power will be significantly higher than in stable state
        if it is used by the method, so the lag is most noticeable here compared to CPU/RAM.
        """
        if self.__gpu_lag_time is None:
            self.__gpu_lag_time, self.__gpu_energy_lag_df = self.__energy_lag(
                self.gpu_energy, self.end_time_nvidia, self.stable_gpu_power_mean, 2)
        return self.__gpu_lag_time == 0
    
    ### Energy consumed during lag time
    @property
    def cpu_lag(self):
        """
        The sum of energy consumed during the lag time.
        """
        if self.__cpu_lag_time_is_zero():
            return 0
        else:
            return self.__cpu_energy_lag_df["energy (J)"].sum()
        
    @property
    def ram_lag(self):
        """
        The sum of energy consumed during the lag time.
        """
        if self.__ram_lag_time_is_zero():
            return 0
        else:
            return self.__ram_energy_lag_df["energy (J)"].sum()
        
    @property
    def gpu_lag(self):
        """
        The sum of energy consumed during the lag time.
        """
        if self.__gpu_lag_time_is_zero():
            return 0
        else:
            return self.__gpu_energy_lag_df["power_draw (W)"].sum() * self.measurement_interval_s # convert power to joules
        
    ### Normalised energy consumption during lag time (subtracting baseline energy consumption)
    @property
    def cpu_lag_normalised(self):
        """
        CPU lag, minus baseline energy consumption for that period.
        """
        if self.__cpu_lag_time_is_zero():
            return 0
        else:
            return self.cpu_lag - self.__baseline_consumption(self.__cpu_energy_lag_df, self.stable_cpu_energy_mean)
        
    @property
    def ram_lag_normalised(self):
        """
        RAM lag, minus baseline energy consumption for that period.
        """
        if self.__ram_lag_time_is_zero():
            return 0
        else:
            return self.ram_lag - self.__baseline_consumption(self.__ram_energy_lag_df, self.stable_ram_energy_mean)
        
    @property
    def gpu_lag_normalised(self):
        """
        The sum of energy consumed during the lag time, minus baseline energy consumption for that period.
        """
        if self.__gpu_lag_time_is_zero():
            return 0
        else:
            return self.gpu_lag - (self.__baseline_consumption(self.__gpu_energy_lag_df, self.stable_gpu_power_mean) * self.measurement_interval_s) # convert power to joules
        
    ### Normalised total energy consumption plus normalised energy consumed during lag time 
    @property
    def total_cpu_lag_normalised(self):
        """
        Total energy consumption above baseline including the energy consumed during the lag time.
        """
        return self.total_cpu_normalised + self.cpu_lag_normalised

    @property
    def total_ram_lag_normalised(self):
        """
        Total energy consumption above baseline including the energy consumed during the lag time.
        """
        return self.total_ram_normalised + self.ram_lag_normalised
    
    @property
    def total_gpu_lag_normalised(self):
        """
        The sum of total_gpu_normalised and gpu_lag_normalised: i.e. total energy consumption including the energy consumed during the lag time.
        """
        return self.total_gpu_normalised + self.gpu_lag_normalised

    ### Input sizes, and sums thereof in Megabytes
    # these parameters can be None, since not every function executed
    # will have args, kwargs and a method_object
    @property
    def args_size(self):
        if self.input_sizes["args_size"] is None:
            return 0
        else:
            return self.input_sizes["args_size"] / MB_CONVERSION
    
    @property
    def kwargs_size(self):
        if self.input_sizes["kwargs_size"] is None:
            return 0
        else:
            return self.input_sizes["kwargs_size"] / MB_CONVERSION
    
    @property
    def object_size(self):
        if self.input_sizes["object_size"] is None:
            return 0
        else:
            return self.input_sizes["object_size"] / MB_CONVERSION
    
    @property
    def total_args_size(self):
        return self.args_size + self.kwargs_size

    @property
    def total_input_size(self):
        return self.total_args_size + self.object_size
    

    ### Helper methods
    def __baseline_consumption(self, comparable_df: pd.DataFrame, stable_mean):
        return len(comparable_df) * stable_mean
    
    ## this was the previous implementation which had floating point errors
    # def __energy_in_execution(self, energy_df: pd.DataFrame, start_time: float, end_time: float):
    #     # get the energy data in between execution start and end time
    #     energy_in_execution = energy_df.loc[(energy_df["time_elapsed"] >= start_time) & (
    #         energy_df["time_elapsed"] < end_time)].copy()
    #     return energy_in_execution
    
    def __energy_in_execution(self, energy_df: pd.DataFrame, start_time: float, end_time: float):
        # Get the index of the closest value to start_time and end_time
        start_idx = (energy_df["time_elapsed"] - start_time).abs().idxmin()
        end_idx = (energy_df["time_elapsed"] - end_time).abs().idxmin()

        # Get the energy data in between execution start and end time (inclusive of start time, exclusive of end time)
        energy_in_execution = energy_df.loc[start_idx:end_idx-1].copy() if end_idx > start_idx else pd.DataFrame()
        
        return energy_in_execution
    
    def __energy_post_execution(self, energy_df: pd.DataFrame, end_time: float):
        return energy_df[(energy_df["time_elapsed"] >= end_time)].copy()

    def __stable_energy_pre_execution(self, energy_df: pd.DataFrame, start_time: float):
        """
        Return a slice of the given energy dataframe. This slice corresponds to the energy data
        from the last stable check interval before execution, which was thus deemed stable.
        """
        pre_execution_df = energy_df[(energy_df["time_elapsed"] < start_time)].copy()
        # number of values to consider for stable energy before execution
        stable_check_window_size = int(self.wait_per_stable_check_loop_s / self.measurement_interval_s)
        return pre_execution_df.iloc[-stable_check_window_size:,:]

    def __energy_lag(self, energy_df: pd.DataFrame, end_time: float, stable_energy_mean: float, abs_tolerance: float):
        """
        Returns a tuple
        - lag_time: calculated lag time
        - energy_lag_df: df containing the energy data during lag time. If lag time is 0, this is None.
        """
        # look at energy after function execution
        post_execution_df = self.__energy_post_execution(energy_df, end_time)

        # calculate a forward-looking moving average with window size 5 of the energy data column, using integer indexing since column headers are different for CPU/RAM and GPU
        post_execution_df["moving_average"] = post_execution_df.iloc[:, 1].rolling(
            window=5, min_periods=1).mean().shift(-4)

        # the energy_lag_df contains all data points where the energy consumption is still above stable state.
        # This makes use of a smart pandas trick by Christopher Tao (Jul 5, 2020) to make sure that the energy_lag_df
        # only contains a single consecutive group of data points from the beginning of the post-execution energy data
        # (https://towardsdatascience.com/pandas-dataframe-group-by-consecutive-certain-values-a6ed8e5d8cc)
        energy_lag_df_groups = post_execution_df[(post_execution_df["moving_average"] >= (stable_energy_mean + abs_tolerance))].groupby(
            ((post_execution_df["moving_average"] < (stable_energy_mean + abs_tolerance))).cumsum())

        # if there are no groups, the method in the line above could not detect any lag time.
        # if there are groups, but no group has key 0, then the method above recognised a lag time that does not directly precede end of execution, which is invalid.
        if (energy_lag_df_groups.ngroups == 0) or (0 not in energy_lag_df_groups.groups):
            return 0, None
        
        energy_lag_df = energy_lag_df_groups.get_group(0)

        # the first index of the energy lag df must be the same as that of the post execution df, otherwise there was no lag immediately preceding execution
        assert energy_lag_df.first_valid_index() == post_execution_df.first_valid_index()

        lag_time = energy_lag_df.iloc[-1]["time_elapsed"] - energy_lag_df.iloc[0]["time_elapsed"]

        return lag_time, energy_lag_df
    
    def __energy_in_execution_is_empty(self):
        if self.cpu_energy_in_execution.empty or self.ram_energy_in_execution.empty or self.gpu_energy_in_execution.empty:
            return True
        else:
            return False

class DataLoader():
    def __init__(self, project: str, output_dir: Path, experiment_kind: ExperimentKinds):
        self.experiment_kind = experiment_kind
        self.project_name = project
        self.__data_dir = format_full_output_dir(
            output_dir, experiment_kind.value, project)
        self.experiment_files = self.__get_all_data_files()
        self.skip_calls = self.__load_skip_calls()

    def __get_all_data_files(self) -> List[str]:
        """
        Get all non-empty data file names as a list of strings
        """
        all_data_files = []

        # get a list of only experiment file names (filter out the skip calls file)
        experiment_file_names = list(filter(lambda file_name: file_name != SKIP_CALLS_FILE_NAME, os.listdir(self.__data_dir)))

        # the custom sort key extracts the experiment number from the file name, such that
        # 'experiment-10.json' does not appear before 'experiment-2.json' in the sorted list
        for data_file in sorted(experiment_file_names, key=lambda file_name: int(file_name.split('-')[1].split('.')[0])):
            file_path = self.__data_dir / data_file
            try:
                with open(file_path, 'r') as f:
                    data = json.load(f)
            except json.decoder.JSONDecodeError:
                raise json.decoder.JSONDecodeError(f"Could not load file {data_file}. Is it empty?")
            if len(data) != 0:
                all_data_files.append(data_file)
        return all_data_files
    
    def __load_skip_calls(self):
        skip_calls_file_path = self.__data_dir / SKIP_CALLS_FILE_NAME
        if skip_calls_file_path.exists():
            with open(skip_calls_file_path, 'r') as f:
                skip_calls = json.load(f)
            return skip_calls
        else:
            return None


    def load_single_file(self, file_name: str) -> Dict[str, EnergyData]:
        """
         returns a dict of EnergyData objects where the key is
            - the function name (method-level experiment) or
            - "project-level" (project-level experiment), in this case there is only one EnergyData object in the dict
        """

        file_path = self.__data_dir / file_name
        with open(file_path, 'r') as f:
            raw_data_list = json.load(f)

        data_samples = {}

        # iterate through all samples in the file and create EnergyData objects for them
        for i, data_dict in enumerate(raw_data_list):
            # data_dict has only one key, the function name
            original_function_name = list(data_dict.keys())[0]
            energy_data = data_dict[original_function_name]["energy_data"]
            function_name = original_function_name

            # if it is a data size experiment, we want to treat each function (even though they have the same name)
            # as a separate function, such that we group by data size and not just function name
            # TODO: something similar could be done for method-level experiments, since there could be a call of the same
            # function but with different inputs. It depends on the use case whether this is desired or not.
            if self.experiment_kind == ExperimentKinds.DATA_SIZE:
                function_name = f"{original_function_name}_{i}"
            
            data_samples[function_name] = EnergyData(
                function_name,
                self.project_name,
                self.convert_json_dict_to_df(energy_data["cpu"]),
                self.convert_json_dict_to_df(energy_data["ram"]),
                self.convert_json_dict_to_df(energy_data["gpu"]),
                data_dict[original_function_name]["times"],
                data_dict[original_function_name]["input_sizes"],
                data_dict[original_function_name]["settings"]
            )

        return data_samples

    def convert_json_dict_to_df(self, energy_data: dict) -> pd.DataFrame:
        return pd.read_json(energy_data, orient="split")