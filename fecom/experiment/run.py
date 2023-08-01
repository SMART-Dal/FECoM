from fecom.experiment.experiments import Experiment

def run_experiments(experiment: Experiment, count: int, start: int = 1): # experiments start with 1
    """
    Run several experiments in a row.

        args:
            experiment (Experiment): an initialised Experiment to run
            count (int): the number of experiments to run
            start (int): the number of the first experiment to run. Default is 1.
    """

    try:
        for n in range(start, start+count):
            print(f"Start running {experiment.experiment_kind.value} experiment ({experiment.project}) number {n}.")
            experiment.run(exp_number = n)
            print(f"Finished running {experiment.experiment_kind.value} experiment ({experiment.project}) number {n}.")
    except KeyboardInterrupt:
        print(f"Aborting experiment number {n}.")