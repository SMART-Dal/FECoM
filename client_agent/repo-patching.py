import os
import subprocess
import concurrent.futures
from pprint import pprint
import sys
import shutil
from clientconfig import *
# import nbformat

os.environ['PYTHONPATH'] = '../server'

def copy_directory_contents(src, dst):
    if os.path.exists(dst):
        shutil.rmtree(dst)
    shutil.copytree(src, dst)


# def ipynb_to_py(ipynb_file):
#     ipynb_file = os.path.abspath(ipynb_file)
#     try:
#         with open(ipynb_file) as f:
#             nb = nbformat.read(f, as_version=4)
#     except Exception as e:
#         print(f"Error reading file {ipynb_file}: {e}")
#         return None

#     py_file = os.path.splitext(ipynb_file)[0] + '.py'
#     try:
#         with open(py_file, 'w') as f:
#             for cell in nb['cells']:
#                 if cell['cell_type'] == 'code':
#                     f.write(''.join(cell['source']))
#     except Exception as e:
#         print(f"Error writing file {py_file}: {e}")
#         return None

#     return py_file

def get_python_scripts_path(directory):
    python_scripts_path = []
    for root, sub_dirs, files in os.walk(directory):
        for file in files:
            if file.endswith('.py'):
                python_scripts_path.append(os.path.join(root, file))
            # if file.endswith('.ipynb'):
                # ipynb_to_py(os.path.join(root, file))
                # py_file = os.path.splitext(os.path.join(root, file))[0] + '.py'
                # py_file = os.path.join(os.path.dirname(os.path.join(root, file)), py_file)
                # python_scripts_path.append(py_file)
    return python_scripts_path

# delete all the files in the output directory before running the script
for filename in os.listdir(INPUT_DIR):
    file_path = os.path.join(INPUT_DIR, filename)
    try:
        if os.path.isfile(file_path) or os.path.islink(file_path):
            os.unlink(file_path)
    except Exception as e:
        print('Failed to delete %s ,for the Reason: %s' % (file_path, e))

#Create a copy of repository in the output directory
copy_directory_contents(INPUT_DIR, OUTPUT_DIR)

# run the script for all the files in the input directory and save the patched files in the output directory
python_scripts = get_python_scripts_path(OUTPUT_DIR)
for input_file_path in python_scripts:
        with open(input_file_path, 'w') as f:
            subprocess.run(['python3', PATCHING_SCRIPT_PATH, input_file_path], stdout=f)