
import sys
from tool.patching.patching_config import EXPERIMENT_DIR
from tool.measurement.execution import before_execution as before_execution_INSERTED_INTO_SCRIPT
from tool.measurement.execution import after_execution as after_execution_INSERTED_INTO_SCRIPT
from tool.experiment.experiments import ExperimentKinds

experiment_number = sys.argv[1]
experiment_project = sys.argv[2]

EXPERIMENT_FILE_PATH = EXPERIMENT_DIR / ExperimentKinds.METHOD_LEVEL.value / experiment_project / f'experiment-{experiment_number}.json'

