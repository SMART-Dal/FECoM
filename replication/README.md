# Replication Package
This package contains all code needed to replicate the results used in the study.

## Submodules
There are several submodules that cover different aspects of the results:
- **create_experiments_directory**: create the directory structures for storing experimental data
- **executed_experiments**: lists of executed experiments for RQ1 and RQ2
- **plots**: replicate plots drawn throughout the paper.
- **results_analysis**: create analysis tables from the experimental data, used throughout the paper.
- **rq1**: replicate method-level and project-level experiments for RQ1.
- **rq2**: replicate data-size experiments for RQ2.
- **rq3_error_log.md**: a markdown file providing detailed information regarding errors faced, used in RQ3.
- **settings_calculation**: calculate the settings used in the experiments & analysis, e.g. stable state stdev to mean ratios.
  
Each submodule has several functions defined, and each function replicates a certain part of the paper. These functions can then be called by inserting them into the `if __name__ == "__main__":` section, or uncommenting them if they are already there. 
  
Some functions write output, e.g. plots, to subdirectories:
- **rq1_analysis**: output used for answering RQ1
- **rq2_analysis**: output used for answering RQ2
- **settings**: calculated settings from the ```settings_analysis``` module
  
The **validation** subdirectory contains several further subdirectories, each of which contains a package of files to be reviewed as part of the patching algorithm's evaluation.

## Repository Used
The official [tensorflow-tutorials](https://github.com/tensorflow/docs/tree/master/site/en/tutorials) repository was used for all experiments. The version used is that of commit `e7f81c2` (parent: `39ff245`) from Mar 16, 2023 at 3:47 AM GMT.

## How to Run an RQ1 Experiment
These is the exact sequence of steps followed by the researchers to run all project-level and method-level experiments for one project in the repository as part of RQ1. It is written as a user manual and refers to a spreadsheet, which was used to keep track of executed experiments.
1. Add your name to the "Worked on by" column in the spreadsheet for your chosen project.
2. Activate the conda virtual environment installed using instructions in the top-level README: `conda activate <env_name>`
3. Check the GPU works by running   
```python3 -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"```  
The last output line should be  
```[PhysicalDevice(name='/physical_device:GPU:0', device_type='GPU')]```
4. Install the tool by running `pip install .` in the top-level directory of this repo.
5. Navigate to the relevant project directory in `data/code-dataset/Patched-Repositories`. Check the import statements to see if there are any non-standard libraries required by the project. Install these libraries and record them on the spreadsheet.
6. Run the original project file with `python3 <project_name>_original.py`. Make sure that the GPU is used by confirming that there is the following output printed in your terminal at some point:
    ```
    Created device /job:localhost/replica:0/task:0/device:GPU:0 with 6103 MB memory: 
    -> device: 0, name: NVIDIA GeForce RTX 3070 Ti, pci bus id: 0000:8a:00.0,
    compute capability: 8.6
    ```
7. If there are no errors, continue to the next step. If there are errors, try to fix them in the original jupyter notebook in the `data/code-dataset/Repositories` (not patched) directory. Record that you made fixes in the spreadsheet. Leave a comment in the original jupyter notebook file, e.g.  
    ```
    # dataset = dataset.map(lambda features, labels: ({'dense_2_input':features}, labels))
    # changed above line to below (01/06/23)
    dataset = dataset.map(lambda features, labels: ({'dense_input': features}, labels))
    ```
    To apply these fixes to the patched scripts, run `python3 repo-patching.py` in the `tool/patching` directory.
    If you cannot fix the errors, record them on the spreadsheet, mark the project as "experiencing errors" and start working on a new project.
8. If you reached this step you can run the original project as in step 5 with no errors. Mark the project on the spreadsheet as "Successfully tested, ready to run experiments" and record the approximate execution time.
9. Run the method-level & project-level experiments for RQ1 as follows. This might take a while, so make sure your terminal does not shut down in the middle of running experiments. If it does shut down, you can selectively run the missing experiments but this will be a little tedious.    
    1. Add the following function call (replacing the project name with the project you are running) at the bottom of `if __name__ == "__main__"` in `replication/rq1.py`:
        ```
        run_rq1_experiments("keras/classification")
        ```
        This example will run 10 method-level and project-level experiments (experiment 1 to 10) for the `keras/classification` project.
        Make sure this is the only function call that is uncommented. If you need to selectively run experiments, you can specify which experiments to run by modifying the `run_rq1_experiments` function.
    2. Make sure that the energy data `json` files in `data/energy-dataset` associated with the project are empty.
    3. Ensure that there are no background processes running that consume significant resources (it is important to close your VSCode Remote Connection if you previously used it to modify files). You can confirm this by running `ps aux` and checking the `%CPU` and `%MEM` columns: significant processes will have a non-zero value in at least one of these columns. You might need to kill these processes.
    4. Start the energy measurement processes by running `python3 start_measurement.py` in the `tool/measurement` directory. Wait for the specified number of seconds (default is 120) to check that the server is in a stable state. If the stdev/mean ratios are larger than the config ones, you might need to revisit the previous step.
    5. Mark the project on the spreadsheet as "running experiments". In a new terminal with activated conda environment, run `python3 rq1.py` in the `replication` directory and wait until the first experiment completes.
    6. Check that there were no run-time errors by inspecting the standard output & error. If there was an error, there is likely an issue in the `measurement.execution` module which needs to be reported. If there are no errors, wait for all experiments to complete.
10. Check that the GPU was used. For this it is recommended to use the VSCode "format document" function to format one of the json files in the `data/energy-dataset` directory into a more easily readible format. Then check that the GPU energy data (best checked for a training call) rises from about 20 to about 70 Watts. If this check fails, revisit steps 1 and 2 (environment setup). 
11. If the data looks good, quickly check that there is data for all 10 method-level and project-level experiments. Push the data & potential modifications to the source code (if you had to fix errors in step 7) to the GitHub Repository. 
12. Mark the project as "Completed 10 method & project-level experiments" on the spreadsheet. You may now start running the next project and you can start from step 5.