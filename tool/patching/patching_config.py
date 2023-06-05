from pathlib import Path

"""
PATCHING CONFIG
"""

# (!) add your own data path here (!)
# change this to the absolute path to the top-level project directory on your machine
# before running anything in Patched-Repositories
# data_path_saurabh = Path('/home/srajput/projects/def-tusharma/srajput/GreenAI-extension/data/')
project_path_saurabh = Path('/home/saurabh/code-energy-consumption/')
project_path_tim = Path('/Users/tim.widmayer/UCL_local/GreenAI-extension/')
project_path_tim_compute_canada = Path('/home/timw/GreenAI-extension/')
project_path_tim_falcon = Path('/home/tim/GreenAI-extension/')

# (!) Change this to the relevant path variable (!)
PROJECT_PATH = project_path_saurabh

# directory where to store data, an Experiment will append to this the experiment kind
# (e.g. project-level) and after that the subdirectory structure will be equivalent to the code dataset
EXPERIMENT_DIR = PROJECT_PATH / 'data/energy-dataset/'

# directory where to find patched code
CODE_DIR = PROJECT_PATH / 'data/code-dataset/Patched-Repositories'
UNPATCHED_CODE_DIR = PROJECT_PATH / 'data/code-dataset/Repositories'

PATCHING_SCRIPT_PATH = Path('script_patcher.py')
code_dataset_path = Path('../../data/code-dataset')
SOURCE_REPO_DIR = code_dataset_path / 'Repositories'
PATCHED_REPO_DIR =  code_dataset_path / 'Patched-Repositories'