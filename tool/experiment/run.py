from tool.experiment.experiments import Experiment

# skeleton method to run multiple experiments
def run_experiments(experiment: Experiment, count: int, start: int = 1): # experiments start with 1
    try:
        for n in range(start, start+count):
            print(f"Start running experiment ({experiment.project}) number {n}.")
            experiment.run()
            print(f"Finished running experiment ({experiment.project}) number {n}.")
    except KeyboardInterrupt:
        print("Aborting current experiment.")