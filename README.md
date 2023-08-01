# FECoM Tool

Our tool can calculate the energy consumption
- of an entire Tensorflow script (project-level experiments)
- of each individual TensorFlow function/method from a given script (method-level experiments)
- of different configurations of the same function/method (data-size experiments)

## Directory Structure
This repository has the following main directories:
- **data**: contains all input and output data used in the study and has three subdirectories:
    - `code-dataset` contains patched and original projects from the third-party source code repository
    - `energy-dataset` contains all experimental energy data
    - `other` contains failed experimental data or data used for calculating settings
- **replication**: replication package for the FECoM study.
- **tool**: source code for the FECoM tool. 

## Environment Setup
### Energy measurement libraries
First verify that [perf](https://perf.wiki.kernel.org/index.php/Main_Page), a library to measure CPU and RAM energy consumption, is already available in you Linux system, if not then install separately:
```bash
sudo apt install linux-tools-`uname -r`
```  
Also make sure the [lm-sensors](https://wiki.archlinux.org/title/lm_sensors) CPU temperature measurement library is installed:
```
sudo apt install lm-sensors
```
This tool also makes use of the [NVIDIA System Management Interface](https://developer.nvidia.com/nvidia-system-management-interface) or `nvidia-smi` to measure GPU energy consumption.  
   
### Python environment
Install [miniconda3](https://docs.conda.io/en/latest/miniconda.html). Then open the `environment.yml` file (provided in this directory) in a text editor and change the paths stored as `prefix:` and `variables:` to point at your miniconda installation. 
  
Finally, in this directory, run the following command to create the required TensorFlow environment from the specified `environment.yml` file:  
```conda env create -f environment.yml```   
Activate the environemnt:  
```conda activate tf2```  
Check if the GPU is setup correctly by running  
```python3 -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"```  
This might give some warnings about missing TensorRT libraries, but as long as the output is `[PhysicalDevice(name='/physical_device:GPU:0', device_type='GPU')]` there is a good chance that the GPU has been setup correctly. Despite this, an issue faced was an error message that `libdevice is required by this HLO module but was not found`. A fix for this is highlighted [here](https://discuss.tensorflow.org/t/cant-find-libdevice-directory-cuda-dir-nvvm-libdevice/11896/5).

### Install Tool
In this (top-level) directory, run  
```pip install .```  
to install the tool. If you make any changes, for example to the configuration files, you need to repeat this step such that all changes are loaded.

## Configuration
All constants and settings for the measurement script can be found in `tool/measurement/measurement_config.py`, and for the patching script in `tool/patching/patching_config.py`. These files are the single source of truth for all used constants.

`measurement_config.py` contains configurations regarding
- Stable state checking
- Energy measurements
- Temperature measurements

`patching_config.py` contains patching script file paths. 
  
**_You will need to add your own file paths in `patching_config`!_**

## Run Energy Consumption Experiments
Start the energy measurement processes by following the instructions below. Detailed instructions on how to run experiments can be found in `replication/README.md`.

With the activated conda environment, navigate to `tool/measurement` and run this command to start `perf` and `nvidia-smi` (energy measurement tools) as well as `sensors` (cpu temperature tool, which is run inside the wrapper `cpu_temperature.py`):  
```python3 start_measurement.py```  
  
The application can be terminated by pressing Control-C.  

## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.