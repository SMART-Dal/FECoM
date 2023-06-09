import os
import shutil
from tool.patching.patching_config import UNPATCHED_CODE_DIR, EXPERIMENT_DIR

def create_experiment_json_files(folder_path):
    for i in range(1, 11):
        file_name = f"experiment-{i}.json"
        file_path = os.path.join(folder_path, file_name)
        if not os.path.exists(file_path):
            with open(file_path, "w") as f:
                f.write("")

def scan_directory(source_dir, target_dir):
    for root, dirs, files in os.walk(source_dir):
        relative_path = os.path.relpath(root, source_dir)
        target_path = os.path.join(target_dir, relative_path)
        
        if not os.path.exists(target_path):
            os.makedirs(target_path)
        
        for file in files:
            if file.endswith(".ipynb") or file.endswith(".py"):
                folder_name = os.path.splitext(file)[0]
                subfolder_path = os.path.join(target_path, folder_name)
                if not os.path.exists(subfolder_path):
                    os.makedirs(subfolder_path)
                create_experiment_json_files(subfolder_path)

source_directory = UNPATCHED_CODE_DIR
# change this below path to 'method-level' and 'project-level' to create the respective directories
target_directory = EXPERIMENT_DIR/'method-level'

scan_directory(source_directory, target_directory)