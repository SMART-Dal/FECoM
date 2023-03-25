from tool.experiment.experiments import ProjectLevelExperiment
from tool.experiment.run import run_experiments
from tool.client.client_config import EXPERIMENT_DIR, CODE_DIR

# define experiment settings here
MAX_WAIT_S = 120
WAIT_AFTER_RUN_S = 30

if __name__ == "__main__":
    start_number = 1
    generative_cvae = ProjectLevelExperiment("generative/cvae", EXPERIMENT_DIR, CODE_DIR, MAX_WAIT_S, WAIT_AFTER_RUN_S, number=start_number)
    run_experiments(generative_cvae, count=10, start=start_number)