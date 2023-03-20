"""
An Experiment contains the logic needed to run one kind of experiment for one specific project on the server.

"""

from abc import ABC, abstractmethod
from pathlib import Path
from tool.server.send_request import send_request

# base class that any Experiment subclass must implement
# if there is shared code between experiments we can add it here as a method
class Experiment(ABC):
    def __init__(self, experiment_kind: str, project: str, output_dir: Path, code_dir: Path):
        """
        args:
        - experiment_kind is a string that is hard-coded into the specific experiment implementation
        - project is a string in the form "category/project_name"
        - code_dir should most likely be set to client_config.CODE_DIR in 
        - output_dir should most likely be set to client_config.EXPERIMENT_DIR
        """
        self.number = 0
        self.project = project
        self.code_dir = code_dir
        self.__output_dir = output_dir / experiment_kind / project
    
    # the output files are always in the same format, so this general formatter should work for any Experiment
    @property
    def output_file(self) -> Path:
        return self.__output_dir / f"experiment-{self.number}.json"

    @abstractmethod
    def run(self, max_wait_secs: int, wait_after_run_secs: int):
        pass

    # do we need this?
    @abstractmethod
    def stop(self):
        pass

class ProjectLevelExperiment(Experiment):
    def __init__(self, project: str, experiment_dir: Path, code_dir: Path):
        raise NotImplementedError("This has not been tested properly yet. Test before using.")
        super().__init__("project-level", project, experiment_dir, code_dir)
        self.__code_file = self.code_dir / f"{self.project}_original.py"
        self.__code_string = None

    def run(self, max_wait_secs: int, wait_after_run_secs: int):
        # have we already read the code file?
        if self.__code_string is None:
            self.__code_string = self.read_project_file(self.__code_file)

        send_request(
            imports="",
            function_to_run=self.__code_string,
            max_wait_secs=max_wait_secs,
            wait_after_run_secs=wait_after_run_secs,
            experiment_file_path=self.output_file
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
    def __init__(self, project: str, experiment_dir: Path, code_dir: Path):
        # raise NotImplementedError("This has not been tested properly yet. Test before using.").
        super().__init__("method-level", project, experiment_dir, code_dir)
        self.__code_file = self.code_dir / f"{self.project}_patched.py"

if __name__ == "__main__":
    pass