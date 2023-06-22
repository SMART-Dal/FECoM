"""
Replicate experiments for data gathered for RQ1. 
These are the method-level and project-level experiments.
"""

from tool.experiment.experiments import ExperimentKinds, PatchedExperiment
from tool.experiment.run import run_experiments
from tool.patching.patching_config import EXPERIMENT_DIR, CODE_DIR

### DO NOT CHANGE THESE VALUES 
# number of the first experiment
START = 1
# total number of experiments
COUNT = 7

def run_rq1_experiments(project, start=START, count=COUNT):
    # method-level
    # experiment = PatchedExperiment(ExperimentKinds.METHOD_LEVEL, project, EXPERIMENT_DIR, CODE_DIR)
    # run_experiments(experiment, count=count, start=start)
    # project-level
    experiment = PatchedExperiment(ExperimentKinds.PROJECT_LEVEL, project, EXPERIMENT_DIR, CODE_DIR)
    run_experiments(experiment, count=count, start=start)


if __name__ == "__main__":
    # run_rq1_experiments("estimator/keras_model_to_estimator")
    # run_rq1_experiments("generative/autoencoder")
    # run_rq1_experiments("audio/simple_audio")
    # run_rq1_experiments("keras/keras_tuner")
    # run_rq1_experiments("estimator/linear")
    # run_rq1_experiments("estimator/premade")
    # run_rq1_experiments("generative/adversarial_fgsm")
    # run_rq1_experiments("load_data/numpy")
    # run_rq1_experiments("keras/regression")
    run_rq1_experiments("quickstart/beginner")
    pass