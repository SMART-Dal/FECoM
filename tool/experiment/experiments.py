"""
An Experiment contains the logic needed to run one kind of experiment for one specific project on the server.

"""

import subprocess, os
from abc import ABC, abstractmethod
from pathlib import Path
from enum import Enum

from tool.server.send_request import send_request


class ExperimentKinds(Enum):
    METHOD_LEVEL = "method-level"
    PROJECT_LEVEL = "project-level"


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
    def __init__(self, experiment_kind: str, project: str, output_dir: Path, code_dir: Path, number: int):
        """
        args:
        - experiment_kind is a string that is hard-coded into the specific experiment implementation
        - project is a string in the form "category/project_name"
        - code_dir should most likely be set to client_config.CODE_DIR in 
        - output_dir should most likely be set to client_config.EXPERIMENT_DIR
        """
        self.number = number
        self.project = project
        self.code_dir = code_dir
        self.__output_dir = format_full_output_dir(output_dir, experiment_kind, project)
    
    # the output files are always in the same format, so this general formatter should work for any Experiment
    @property
    def output_file(self) -> Path:
        return format_output_file(self.__output_dir, self.number)
    
    # this method must increment number every time it is called
    @abstractmethod
    def run(self):
        pass

    # do we need this?
    @abstractmethod
    def stop(self):
        pass


class ProjectLevelExperiment(Experiment):
    def __init__(self, project: str, experiment_dir: Path, code_dir: Path, max_wait_secs: int, wait_after_run_secs: int, number: int):
        # raise NotImplementedError("This has not been tested properly yet. Test before using.")
        super().__init__(ExperimentKinds.PROJECT_LEVEL.value, project, experiment_dir, code_dir, number)
        self.code_file = self.code_dir / f"{self.project}_original.py"
        self.max_wait_secs = max_wait_secs
        self.wait_after_run_secs = wait_after_run_secs
        self.__code_string = None

    def run(self):
        # have we already read the code file?
        if self.__code_string is None:
            self.__code_string = self.read_project_file(self.code_file)

        send_request(
            imports="",
            function_to_run=self.__code_string,
            max_wait_secs=self.max_wait_secs,
            wait_after_run_secs=self.wait_after_run_secs,
            experiment_file_path=self.output_file,
            exec_not_eval=True
            )
        self.number += 1
        return

    def stop(self):
        pass

    # read the code file into a string
    def read_project_file(self, code_file) -> str:
        with open(code_file, 'r') as f:
            code_string = f.read()
        return code_string

class MethodLevelExperiment(Experiment):
    def __init__(self, project: str, experiment_dir: Path, code_dir: Path, number: int):
        # raise NotImplementedError("This has not been tested properly yet. Test before using.")
        super().__init__(ExperimentKinds.METHOD_LEVEL.value, project, experiment_dir, code_dir, number)
        self.__code_file = self.code_dir / f"{self.project}_patched.py"

    def run(self):
        try:
            with subprocess.Popen(['python', self.__code_file, str(self.number), str(self.project)], stdout=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=1, universal_newlines=True) as p:
                for line in p.stdout:
                    print(line, end='')
                for line in p.stderr:
                    print(line, end='')  # Print error output to console
            self.number += 1
            return
        except Exception as e:
            raise e

    def stop(self):
        pass

if __name__ == "__main__":
    pass