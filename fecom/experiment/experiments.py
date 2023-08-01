"""
A concrete Experiment instance contains the logic needed to run
one kind of experiment for one specific project.
- PatchedExperiment: used for method-level and project-level experiments (RQ1)
- DataSizeExperiment: used for data-size experiments (RQ2)

"""

import subprocess
from abc import ABC, abstractmethod
from pathlib import Path

from fecom.measurement.execution import before_execution, after_execution
from fecom.measurement.utilities import custom_print
from fecom.experiment.experiment_kinds import ExperimentKinds


def print_exp(message: str):
    custom_print("experiment", message)


def format_full_output_dir(output_dir: Path, experiment_kind: str, project: str):
    """
    returns the path output_dir/experiment_kind/project
    """
    return output_dir / experiment_kind / project


def format_output_file(output_dir: Path, experiment_number: int):
    return output_dir / f"experiment-{experiment_number}.json"


# base class that any Experiment subclass must implement
# if there is shared code between experiments we can add it here as a method
class Experiment(ABC):
    def __init__(self, experiment_kind: ExperimentKinds, project: str, output_dir: Path):
        """
        args:
        - experiment_kind specifies the kind of experiment (e.g. ExperimentKinds.METHOD_LEVEL)
        - project is a string in the form "category/project_name"
        - output_dir should most likely be set to patching_config.EXPERIMENT_DIR
        """
        self.experiment_kind = experiment_kind
        self.number = None
        self.project = project
        self.__output_dir = format_full_output_dir(output_dir, experiment_kind.value, project)
    
    # the output files are always in the same format, so this general formatter should work for any Experiment
    @property
    def output_file(self) -> Path:
        if self.number is None:
            raise ValueError("Experiment number is None, but is expected to be a positive integer.")
        return format_output_file(self.__output_dir, self.number)
    
    # this method must update self.number to be equal to exp_number
    @abstractmethod
    def run(self, exp_number: int):
        pass


class PatchedExperiment(Experiment):
    def __init__(self, experiment_kind: ExperimentKinds, project: str, experiment_dir: Path, code_dir: Path):
        """
        See Experiment for more info on args.
        code_dir should most likely be set to patching_config.CODE_DIR
        """
        
        # only method-level or project-level experiments are PatchedExperiments
        assert (experiment_kind == ExperimentKinds.METHOD_LEVEL) or (experiment_kind == ExperimentKinds.PROJECT_LEVEL)
        
        super().__init__(experiment_kind, project, experiment_dir)
        self.__code_file = code_dir / f"{self.project}_{experiment_kind.value}.py"

    def run(self, exp_number):
        self.number = exp_number
        with subprocess.Popen(['python', self.__code_file, str(self.number), str(self.project)], stdout=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=1, universal_newlines=True) as p:
            for line in p.stdout:
                print(line, end='')
            for line in p.stderr:
                print(line, end='')  # Print error output to console
        return


class DataSizeExperiment(Experiment):
    def __init__(self, project: str, experiment_dir: Path, n_runs: int, prepare_experiment: callable,
                 function_to_run: str, function_signature: str, imports: str = None, start_at: int = 1):
        """
        args:
            - n_runs (int): the total number of runs per experiment (if start_at > 1, the actual number of runs is smaller)
            - prepare_experiments (callable): a function that takes a fraction (float) and returns function_args, function_kwarg and method_object with adjusted data size
            - function_to_run (str): a string such as obj.fit(*args, **kwargs) that can be executed with eval()
            - function_signature (str): the pretty name of the function_to_run, i.e. the full function signature without *args etc.
            - imports (str) (optional): a string with imports that have to be executed before the function_to_run (e.g. imports = "import tensorflow as tf")
            - start_at (int) (optional): if specified, this should be a number between 1 and n_runs, and the run() method will start at this number instead of at 1.
        """
        super().__init__(ExperimentKinds.DATA_SIZE, project, experiment_dir)
        assert start_at > 0 and start_at <= n_runs
        self.n_runs = n_runs
        self.start_at = start_at
        self.function_to_run = function_to_run
        self.function_signature = function_signature
        self.prepare_experiment = prepare_experiment
        self.imports = imports
    
    def run(self, exp_number):
        self.number = exp_number

        # start with run 1, such that the fraction is never 0
        for run_number in range(self.start_at, self.n_runs+1):
            fraction = run_number / self.n_runs
            assert fraction > 0 and fraction <= 1
            print_exp(f"Begin run [{run_number}] with data size {fraction} for {self.function_signature}")

            function_args, function_kwargs, method_object = self.prepare_experiment(fraction)

            self.execute_function(function_args, function_kwargs, method_object)
    
    def execute_function(self, args, kwargs, obj):
        # args, kwargs and obj appear unused but are used in the eval() call
        if self.imports is not None:
            exec(self.imports)

        start_times = before_execution(experiment_file_path=None, enable_skip_calls=False)

        eval(self.function_to_run)

        after_execution(start_times=start_times,
                        experiment_file_path=self.output_file,
                        function_to_run=self.function_signature, # function signature is the pretty form of the function_to_run
                        method_object=obj,
                        function_args=args,
                        function_kwargs=kwargs,
                        enable_skip_calls=False)