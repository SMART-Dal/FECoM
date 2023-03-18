from pathlib import Path

"""
CLIENT CONFIG
"""
# experiment dataset path: change this to the absolute path on your machine before running anything in Patched-Repositories
experiment_dir_saurabh = Path('/home/srajput/projects/def-tusharma/srajput/GreenAI-extension/data/energy-dataset/')
experiment_dir_tim = Path('/home/timw/GreenAI-extension/data/energy-dataset/')

EXPERIMENT_TAG = 'experiment-1'
EXPERIMENT_DIR = experiment_dir_tim

# TODO are these two paths still needed?
# CLIENT_INPUT_DIR = Path('./code-dataset/input-code')
# CLIENT_OUTPUT_DIR = Path('./code-dataset/patched-code/')
PATCHING_SCRIPT_PATH = Path('script_patcher.py')
code_dataset_path = Path('../../data/code-dataset')
SOURCE_REPO_DIR = code_dataset_path / 'Repositories'
PATCHED_REPO_DIR =  code_dataset_path / 'Patched-Repositories'

# When you want to update these seconds, you have to re-run repo-patching.py
# such that the new settings are written to the patched python files
MAX_WAIT_S = 120
WAIT_AFTER_RUN_S = 25