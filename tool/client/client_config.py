from pathlib import Path

"""
CLIENT CONFIG
"""

# (!) add your own data path here (!)
# change this to the absolute path on your machine before running anything in Patched-Repositories
data_path_saurabh = Path('/home/srajput/projects/def-tusharma/srajput/GreenAI-extension/data/')
data_path_tim = Path('/home/timw/GreenAI-extension/data/')

energy_dataset = 'energy-dataset/'

# energy dataset directories
energy_dir_saurabh = data_path_saurabh / energy_dataset
energy_dir_tim = data_path_tim / energy_dataset

# EXPERIMENT_TAG = 'experiment-1' (deprecated)

# directory where to store data, an Experiment will append to this the experiment kind
# (e.g. project-level) and after that the subdirectory structure will be equivalent to the code dataset
EXPERIMENT_DIR = energy_dir_saurabh

code_dataset = Path('code-dataset/Patched-Repositories')

# code dataset directories
code_dir_saurabh = data_path_saurabh / code_dataset
code_dir_tim = data_path_tim / code_dataset

# directory where to find patched code
CODE_DIR = code_dir_saurabh

PATCHING_SCRIPT_PATH = Path('script_patcher.py')
code_dataset_path = Path('../../data/code-dataset')
SOURCE_REPO_DIR = code_dataset_path / 'Repositories'
PATCHED_REPO_DIR =  code_dataset_path / 'Patched-Repositories'

# When you want to update these seconds, you have to re-run repo-patching.py
# such that the new settings are written to the patched python files
MAX_WAIT_S = 120
WAIT_AFTER_RUN_S = 25