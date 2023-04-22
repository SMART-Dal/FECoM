"""
An Experiment contains the logic needed to run one kind of experiment for one specific project on the server.

"""

import subprocess, os
from abc import ABC, abstractmethod
from pathlib import Path
from enum import Enum

from tool.server.send_request import send_request_with_func_details as send_request
from tool.server.function_details import FunctionDetails, build_function_details


class ExperimentKinds(Enum):
    METHOD_LEVEL = "method-level"
    PROJECT_LEVEL = "project-level"
    DATA_SIZE = "data-size"


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
        - experiment_kind is a string that is hard-coded into the specific experiment implementation
        - project is a string in the form "category/project_name"
        - code_dir should most likely be set to client_config.CODE_DIR in TODO REMOVE THIS HERE
        - output_dir should most likely be set to client_config.EXPERIMENT_DIR
        """
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


class ProjectLevelExperiment(Experiment):
    def __init__(self, project: str, experiment_dir: Path, code_dir: Path, max_wait_secs: int, wait_after_run_secs: int):
        super().__init__(ExperimentKinds.PROJECT_LEVEL, project, experiment_dir)
        self.code_file = code_dir / f"{self.project}_original.py"
        self.max_wait_secs = max_wait_secs
        self.wait_after_run_secs = wait_after_run_secs
        self.__code_string = None

    def run(self, exp_number: int):
        self.number = exp_number
        # have we already read the code file?
        if self.__code_string is None:
            self.__code_string = self.read_project_file(self.code_file)

        function_details = build_function_details(
            imports = "",
            function_to_run = self.__code_string,
            max_wait_secs = self.max_wait_secs,
            wait_after_run_secs = self.wait_after_run_secs,
            exec_not_eval = True
        )
        send_request(function_details, experiment_file_path=self.output_file)
        return

    # read the code file into a string
    def read_project_file(self, code_file) -> str:
        with open(code_file, 'r') as f:
            code_string = f.read()
        return code_string


class MethodLevelExperiment(Experiment):
    def __init__(self, project: str, experiment_dir: Path, code_dir: Path):
        super().__init__(ExperimentKinds.METHOD_LEVEL, project, experiment_dir)
        self.__code_file = self.code_dir / f"{self.project}_patched.py"

    def run(self, exp_number):
        self.number = exp_number
        with subprocess.Popen(['python', self.__code_file, str(self.number), str(self.project)], stdout=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=1, universal_newlines=True) as p:
            for line in p.stdout:
                print(line, end='')
            for line in p.stderr:
                print(line, end='')  # Print error output to console
        return


class DataSizeExperiment(Experiment):
    def __init__(self, project: str, experiment_dir: Path, n_runs: int, function_details: FunctionDetails, vary_args: list):
        raise NotImplementedError("This experiment has not been tested yet")
        """
        args:
            - n_runs (int): the total number of runs per experiment
            - vary_size_of (list[str]): a list of attributes of a FunctionDetails object, which are numpy arrays and the size of which should be varied.
        """
        super().__init__(ExperimentKinds.DATA_SIZE, project, experiment_dir)
        self.n_runs = n_runs
        self.function_details = function_details
        self.vary_args = vary_args
    
    def run(self, exp_number):
        self.number = exp_number

        for run in range(self.n_runs):
            fraction = run / self.n_runs
            print(f"Begin run [{run}] with data size {fraction} of original")

            self.function_details.args = self.vary_arg_sizes(fraction)
            send_request(
                function_details = self.function_details,
                experiment_file_path= self.output_file
            )

    def vary_arg_sizes(self, fraction: float) -> list:
        """
        fraction must be between 0 and 1!
        E.g. if an arg in vary_args has shape (100,10,10) and fraction=0.5, return an array of shape (50,10,10).
        So this method only scales the first dimension of the array by the given fraction.
        """
        return [arg[:int(arg.shape[0]*fraction)] for arg in self.vary_args]


if __name__ == "__main__":
    """
    TODO move the following code to the replication package and run experiments.
    """
    import tensorflow as tf
    from tensorflow.keras import datasets, layers, models
    from tool.server.send_request import send_request


    #### begin copied code (from keras/classification)
    (train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()

    # Normalize pixel values to be between 0 and 1
    train_images, test_images = train_images / 255.0, test_images / 255.0

    model = models.Sequential()
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.Flatten())
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(10))

    model.compile(optimizer='adam',
                loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                metrics=['accuracy'])

    

    ## commented out relevant method call
    # history = model.fit(train_images, train_labels, epochs=10, 
    #                     validation_data=(test_images, test_labels))
    ## end comment

    #### end copied code

    # check data as described here https://www.tensorflow.org/api_docs/python/tf/keras/datasets/cifar10/load_data
    assert train_images.shape == (50000, 32, 32, 3)
    assert test_images.shape == (10000, 32, 32, 3)
    assert train_labels.shape == (50000, 1)
    assert test_labels.shape == (10000, 1)

    # testing the slicing used in vary_size
    assert train_images[:int(train_images.shape[0]*0.1)].shape == (int(50000*0.1), 32, 32, 3)
    print(train_images[:int(train_images.shape[0]*0.1)].shape)

    # build function details for function
    from tool.client.client_config import WAIT_AFTER_RUN_S, MAX_WAIT_S

    function_details = build_function_details(
        imports = "import tensorflow as tf\nfrom tensorflow.keras import datasets, layers, models",
        function_to_run = "obj.fit(*args,**kwargs)",
        kwargs = {
            "epochs": 10,
            "validation_data": (test_images, test_labels)
        },
        method_object = model,
        max_wait_secs = MAX_WAIT_S,
        wait_after_run_secs = WAIT_AFTER_RUN_S
    )

    vary_args = [train_images, train_labels]