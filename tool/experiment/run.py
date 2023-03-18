from tool.experiment.experiments import Experiment

# skeleton method to run multiple experiments
def run_experiments(experiment: Experiment, number: int):
    try:
        for n in number:
            print(f"Start running experiment ({experiment.project}) number {number}.")
            experiment.run(n)
            print("Success")
    except KeyboardInterrupt:
        print("Aborting current experiment.")
        experiment.stop()
        print("Aborted.")