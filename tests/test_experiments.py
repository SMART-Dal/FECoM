from tool.experiment.experiments import ProjectLevelExperiment
from pathlib import Path

from tool.patching.patching_config import CODE_DIR, EXPERIMENT_DIR

def test_get_experiment_number():
    raise DeprecationWarning("Update to work with expanded Experiment class")
    exp_path = Path("keras/classification")
    exp = ProjectLevelExperiment(exp_path)
    assert exp.number==0
    assert exp.project==exp_path
    exp.run()
    assert exp.number==1
    exp.run()
    assert exp.number==2