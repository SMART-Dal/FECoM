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

## Repository Used
The official [tensorflow-tutorials](https://github.com/tensorflow/docs/tree/master/site/en/tutorials) repository was used for all experiments. The version used is that of commit `e7f81c2` (parent: `39ff245`) from Mar 16, 2023 at 3:47 AM GMT.