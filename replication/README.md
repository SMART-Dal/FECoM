# Replication Package
This package contains all code needed to replicate the results used in the study.

## Submodules
There are several submodules that cover different aspects of the results:
- **data_size**: replicate data-size experiments for RQ2.
- **method_level**: replicate method-level experiments for RQ1.
- **plots**: replicate plots drawn throughout the paper.
- **project_level**: replicate project-level experiments for RQ1.
- **results_analysis**: create analysis tables from the experimental data, used throughout the paper.
- **settings_calculation**: calculate the settings used in the experiments & analysis, e.g. stable state stdev to mean ratios.
  
Each submodule has several functions defined, and each function replicates a certain part of the paper. These functions can then be called by inserting them into the `if __name__ == "__main__":` section, or uncommenting them if they are already there. Some modules write their output to the `./out` directory, plots can sometimes be found in this directory.

## How to Run a project
Follow these steps to run a new project
1. Activate the conda virtual environment installed using instructions in the top-level README: `conda activate <env_name>`
2. Check the GPU works by running   
```python3 -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"```  
The last output line should be  
```[PhysicalDevice(name='/physical_device:GPU:0', device_type='GPU')]```
3. Install the tool by running `pip install .` in the top-level directory.
4. Run the original project file by navigating to the relevant directory in `data/code-dataset/Patched-Repositories` and running `python3 <project_name>_original.py`
5. You might get errors due to uninstalled libraries required by the project. Install these libraries and record them on the spreadsheet.
6. If there are no errors, continue. If there are errors, try to fix them. Record that you made fixes in the spreadsheet. Leave a comment in the original project file, e.g.  
    ```
    # dataset = dataset.map(lambda features, labels: ({'dense_2_input':features}, labels))
    # changed above line to below (01/06/23)
    dataset = dataset.map(lambda features, labels: ({'dense_input': features}, labels))
    ```
    Make the same fixes in the patched project file before you continue. If you cannot fix the errors, record them on the spreadsheet, mark the project as "experiencing errors" and start working on a new project.
7. Now you can run the original project as in step 4 with no errors. You can mark the project on the spreadsheet as "Successfully tested, ready to run experiments". 
8. Run the method-level experiments
    1. Create a new function in `replication/method_level.py`. The following example will run 10 method-level experiments for the `keras/classification` project:
        ```
        def run_keras_classification_method_level():
            experiment = MethodLevelExperiment("keras/classification", EXPERIMENT_DIR, CODE_DIR)
            run_experiments(experiment, count=10, start=1)
        ```
    2. Add the new function call at the bottom of `if __name__ == "__main__"` and make sure it is the only function call that is uncommented.
    3. Make sure that the energy data files in `data/energy-dataset` associated with the project are empty.
    4. Run `python3 method_level.py` in the `replication` directory.
    4. Wait until the first experiment completes. Then check that (1) there were no run-time errors (2) the number of functions in the energy data `json` file is the same as the number of patched functions in the project file. To check (2) it might suffice to check that the first and last functions are the same.
9. Run the project-level experiments TODO: continue after fixing.

## Repository Used
The official [tensorflow-tutorials](https://github.com/tensorflow/docs/tree/master/site/en/tutorials) repository was used for all experiments. The version used is that of commit `e7f81c2` (parent: `39ff245`) from Mar 16, 2023 at 3:47 AM GMT.