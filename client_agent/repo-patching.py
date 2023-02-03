import os
import subprocess
import concurrent.futures
from pprint import pprint
import sys
import shutil
from clientconfig import *

os.environ['PYTHONPATH'] = '../server'

def get_python_scripts_path(directory):
    python_scripts_path = []
    for root, sub_dirs, files in os.walk(directory):
        for file in files:
            if file.endswith('.py'):
                python_scripts.append(os.path.join(root, file))
    return python_scripts_path

# delete all the files in the output directory before running the script
for filename in os.listdir(PATCHED_REPO_DIR):
    file_path = os.path.join(PATCHED_REPO_DIR, filename)
    try:
        if os.path.isfile(file_path) or os.path.islink(file_path):
            os.unlink(file_path)
    except Exception as e:
        print('Failed to delete %s ,for the Reason: %s' % (file_path, e))

#Create a copy of repository in the output directory
shutil.copytree(SOURCE_REPO_DIR, PATCHED_REPO_DIR)

# run the script for all the files in the input directory and save the patched files in the output directory
python_scripts = get_python_scripts_path(PATCHED_REPO_DIR)
for input_file_path in python_scripts:
        with open(input_file_path, 'w') as f:
            subprocess.run(['python3', PATCHING_SCRIPT_PATH, input_file_path], stdout=f)