from pathlib import Path

"""
CLIENT CONFIG
"""
# experiment dataset path
EXPERIMENT_TAG = 'experiment-1'
EXPERIMENT_DIR = Path('/home/srajput/projects/def-tusharma/srajput/GreenAI-extension/energy-dataset/')

# TODO are these two paths still needed?
# CLIENT_INPUT_DIR = Path('./code-dataset/input-code')
# CLIENT_OUTPUT_DIR = Path('./code-dataset/patched-code/')
PATCHING_SCRIPT_PATH = Path('script_patcher.py')
code_dataset_path = Path('../../data/code-dataset')
SOURCE_REPO_DIR = code_dataset_path / 'Repositories'
PATCHED_REPO_DIR =  code_dataset_path / 'Patched-Repositories'