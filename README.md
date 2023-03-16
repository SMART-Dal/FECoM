# GreenAI-extension

Our tool calculates the energy consumed by each individual ML method from a given script.

## Environment Setup
### Energy measurement tools
First verify that [perf](https://perf.wiki.kernel.org/index.php/Main_Page), a tool to measure CPU and RAM energy consumption, is already available in you Linux system, if not then install separately:
```bash
sudo apt install linux-tools-`uname -r`
```  
Also make sure the [lm-sensors](https://wiki.archlinux.org/title/lm_sensors) CPU temperature measurement tool is installed:
```
sudo apt install lm-sensors
```
This tool also makes use of the [NVIDIA System Management Interface](https://developer.nvidia.com/nvidia-system-management-interface) or `nvidia-smi` to measure GPU energy consumption.  
   
### Python environment
Install [miniconda3](https://docs.conda.io/en/latest/miniconda.html). Then open the `environment.yml` file in a text editor and change the paths stored as `prefix:` and `variables:` to point at your miniconda installation. If you keep the environment name the same (`tf2`), you will likely only need to replace `/home/tim`.  
  
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

### Testing
To test if you have setup everything correctly, go to the `tests` directory and follow the instructions in that directory's README file to run all tests. Make sure to do this in a new terminal with the activated environment. 

## Configuration
All constants and settings for the server can be found in `tool/server/server_config.py`, and for the client in `tool/client/client_config.py`. These files are the single source of truth for all used constants.

`server_config.py` contains configurations regarding
- Server URL and port
- Stable state checking
- Energy measurements
- Temperature measurements
- Authentication & SSL

`client_config.py` contains client file paths. 

Some of these constants are critical settings for the experiments. 

## Run Energy Consumption Experiments
Start the server by following the instructions below, wait a few seconds and then start sending requests from the client.  


With the activated venv, navigate to `tool/server` and run this command to start the server. This will also start `perf` and `nvidia-smi` (energy measurement tools) as well as `sensors` (cpu temperature tool, run inside the wrapper `cpu_temperature.py`):  
```python3 start_measurement.py```  
  
The application can be terminated by pressing Control-C.  
  
If you would just like to run the server, run:  
```python3 server.py```  

## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

Please make sure to update tests as appropriate.
