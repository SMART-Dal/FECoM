import os
import subprocess
import concurrent.futures
from pprint import pprint
import sys
import shutil
from clientconfig import *
import nbformat
import importlib

sys.path.append("../server")
# from server.send_request import send_request, send_single_thread_request


def copy_directory_contents(src, dst):
    if os.path.exists(dst):
        shutil.rmtree(dst)
    shutil.copytree(src, dst)
    
def ipynb_to_py(ipynb_file):
    ipynb_file = os.path.abspath(ipynb_file)
    try:
        with open(ipynb_file) as f:
            nb = nbformat.read(f, as_version=4)
    except Exception as e:
        print(f"Error reading file {ipynb_file}: {e}")
        return None

    py_file = os.path.splitext(ipynb_file)[0] + '.py'
    try:
        with open(py_file, 'w') as f:
            for cell in nb['cells']:
                if cell['cell_type'] == 'code':
                    source_lines = cell['source'].split('\n')
                    for line in source_lines:
                        if not line.startswith('!'):
                            f.write(line + '\n')
    except Exception as e:
        print(f"Error writing file {py_file}: {e}")
        return None

    return py_file



def get_python_scripts_path(directory):
    python_scripts_path = []
    for root, sub_dirs, files in os.walk(directory):
        for file in files:
            if file.endswith('.py'):
                python_scripts_path.append(os.path.join(root, file))
            elif file.endswith('.ipynb'):
                py_file = ipynb_to_py(os.path.join(root, file))
                if py_file is not None:
                    python_scripts_path.append(py_file)
            else:
                continue

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
copy_directory_contents(SOURCE_REPO_DIR, PATCHED_REPO_DIR)

# run the script for all the files in the input directory and save the patched files in the output directory
python_scripts = get_python_scripts_path(PATCHED_REPO_DIR)
for input_file_path in python_scripts:

    result = subprocess.run(['python3', PATCHING_SCRIPT_PATH, input_file_path], stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    with open(input_file_path, 'w') as f:
        f.write(result.stdout.decode())
        f.write(result.stderr.decode())


print("done....")