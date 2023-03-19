from abc import ABC, abstractmethod

# base class that any Experiment subclass must implement
# if there is shared code between experiments we can add it here as a method
class Experiment(ABC):
    def __init__(self, project):
        self.number = 0
        self.project = project

    @abstractmethod
    def run(self, number):
        pass

    # do we need this?
    @abstractmethod
    def stop(self):
        pass

    @property
    def number(self):
        return self.number

    @property
    def project(self):
        return self.project


# concrete Experiment classes implement the abstract methods run and stop
# this is an example to be implemented
class ProjectLevelExperiment(Experiment):
    def __init__(self, project):
        super().__init__(self, project)

    def run(self, number):
        self.number += 1
        return

    def stop(self):
        pass

# concrete Experiment classes implement the abstract methods run and stop
# this is an example to be implemented
class MethodLevelExperiment(Experiment):
    def __init__(self, project):
        super().__init__(self, project)

    def run(self, number):
        self.number += 1
        return

    def stop(self):
        pass