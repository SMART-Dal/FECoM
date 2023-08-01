from pathlib import Path

"""
PATCHING CONFIG
"""

# (!) add your own data path here (!)
# change this to the absolute path to the top-level project directory on your machine
# before running anything
raise NotImplementedError("Change this to the absolute path to the top-level project directory on your machine before running anything")
your_project_path = Path('/home/sample_user/sample_repo_name')

# (!) Change this to the relevant path variable (!)
PROJECT_PATH = your_project_path

# directory where to store data, an Experiment will append to this the experiment kind
# (e.g. project-level) and after that the subdirectory structure will be equivalent to the code dataset
EXPERIMENT_DIR = PROJECT_PATH / 'data/energy-dataset/'

# directory where to find patched code
CODE_DIR = PROJECT_PATH / 'data/code-dataset/Patched-Repositories'
UNPATCHED_CODE_DIR = PROJECT_PATH / 'data/code-dataset/Repositories'

METHOD_LEVEL_PATCHING_SCRIPT_PATH = Path('method_level_script_patcher.py')
PROJECT_LEVEL_PATCHING_SCRIPT_PATH = Path('project_level_script_patcher.py')
code_dataset_path = Path('../../data/code-dataset')
SOURCE_REPO_DIR = code_dataset_path / 'Repositories'
PATCHED_REPO_DIR =  code_dataset_path / 'Patched-Repositories'