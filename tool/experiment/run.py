from tool.experiment.experiments import Experiment

# skeleton method to run multiple experiments
def run_experiments(experiment: Experiment, count: int):
    try:
        # experiments start with 1
        for n in range(1, count+1):
            print(f"Start running experiment ({experiment.project}) number {n}.")
            experiment.run()
            print(f"Finished running experiment ({experiment.project}) number {n}.")
    except KeyboardInterrupt:
        print("Aborting current experiment.")