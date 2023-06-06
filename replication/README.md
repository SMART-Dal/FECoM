# Replication Package
This package contains all code needed to replicate the results used in the study.

## Submodules
There are several submodules that cover different aspects of the results:
- **data_size**: replicate data-size experiments for RQ2.
- **rq1**: replicate method-level and project-level experiments for RQ1.
- **plots**: replicate plots drawn throughout the paper.
- **results_analysis**: create analysis tables from the experimental data, used throughout the paper.
- **settings_calculation**: calculate the settings used in the experiments & analysis, e.g. stable state stdev to mean ratios.
  
Each submodule has several functions defined, and each function replicates a certain part of the paper. These functions can then be called by inserting them into the `if __name__ == "__main__":` section, or uncommenting them if they are already there. Some modules write their output to the `./out` directory, plots can sometimes be found in this directory.

## How to Run an RQ1 Experiment
Follow these steps to run an experiment for a new project.  
1. Add your name to the "Worked on by" column in the spreadsheet for your chosen project.
2. Activate the conda virtual environment installed using instructions in the top-level README: `conda activate <env_name>`
3. Check the GPU works by running   
```python3 -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"```  
The last output line should be  
```[PhysicalDevice(name='/physical_device:GPU:0', device_type='GPU')]```
4. Install the tool by running `pip install .` in the top-level directory of this repo.
5. Navigate to the relevant project directory in `data/code-dataset/Patched-Repositories`. Check the import statements to see if there are any non-standard libraries required by the project. Install these libraries and record them on the spreadsheet.
6. Run the original project file with `python3 <project_name>_original.py` 
7. If there are no errors, continue to the next step. If there are errors, try to fix them in the original jupyter notebook. Record that you made fixes in the spreadsheet. Leave a comment in the original jupyter notebook file, e.g.  
    ```
    # dataset = dataset.map(lambda features, labels: ({'dense_2_input':features}, labels))
    # changed above line to below (01/06/23)
    dataset = dataset.map(lambda features, labels: ({'dense_input': features}, labels))
    ```
    To apply these fixes to the patched scripts, run `python3 repo-patching.py` in the `tool/patching` directory.
    If you cannot fix the errors, record them on the spreadsheet, mark the project as "experiencing errors" and start working on a new project.
8. If you reached this step you can run the original project as in step 5 with no errors. Mark the project on the spreadsheet as "Successfully tested, ready to run experiments" and record the approximate execution time.
9. Ensure that there are no background processes running that consume significant resources (it is important to close your VSCode Remote Connection). You can confirm this by running `ps aux` and checking the `%CPU` and `%MEM` columns: significant processes will have a non-zero value in at least one of these columns. You might need to kill these processes.
10. Start the energy measurement processes by running `python3 start_measurement.py` in the `tool/measurement` directory. Wait for the specified number of seconds (currently 120) to check that the server is in a stable state. If the stdev/mean ratios are larger than the config ones, you might need to revisit step 8.
11. Run the method-level & project-level experiments for RQ1 as follows. This might take a while, so make sure your terminal does not shut down in the middle of running experiments. If it does shut down, you can selectively run the missing experiments but this will be a little tedious.
    1. Create a new function in the `replication/rq1.py` file. The following example will run 10 method-level and project-level experiments for the `keras/classification` project and you should use the same format for other projects:
        ```
        def keras_classification():
            project = "keras/classification"
            run_rq1_experiments(project)
        ```
        If you need to selectively run experiments, you can specify which experiments to run by modifying the `run_rq1_experiments` function.
    2. Add the new function call at the bottom of `if __name__ == "__main__"` in the same file and make sure it is the only function call that is uncommented.
    3. Make sure that the energy data `json` files in `data/energy-dataset` associated with the project are empty.
    4. Mark the project on the spreadsheet as "running experiments". In a new terminal, run `python3 rq1.py` in the `replication` directory and wait until the first method-level experiment completes.
    5. Check that there were no run-time errors by inspecting the standard output & error. If there was an error, there is likely an issue with the measurement/execution script.
    6. Check that the number of functions executed is the same as the number of patched functions in the project file. You can do this by inspecting the `tool/measurement/out/execution_log.txt` file which keeps track of the latest executions. If this check fails, revisit the previous step (11.5).
    7. Check that the GPU was used. For this it is recommended to use the VSCode "format document" function to format the json into a more easily readible format. Then check that the GPU energy data (best checked for a training call) rises from about 20 to about 70 Watts. If this check fails, revisit steps 1 and 2 (environment setup). 
    8. If the data looks fine, wait for the experiments to complete.
12. Once complete, quickly check that there is data for all 10 method-level and project-level experiments. Then mark the project as "Completed 10 method & project-level experiments" on the spreadsheet. You may now start running the next project and you can start from step 5.

## Repository Used
The official [tensorflow-tutorials](https://github.com/tensorflow/docs/tree/master/site/en/tutorials) repository was used for all experiments. The version used is that of commit `e7f81c2` (parent: `39ff245`) from Mar 16, 2023 at 3:47 AM GMT.