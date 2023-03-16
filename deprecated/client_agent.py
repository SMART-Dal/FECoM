import os
import subprocess
import concurrent.futures
from pprint import pprint
import sys
from ..config import CLIENT_INPUT_DIR, CLIENT_OUTPUT_DIR, PATCHING_SCRIPT_PATH

raise DeprecationWarning("Do we still have to modify PYTHONPATH after the repo restructuring?")
os.environ['PYTHONPATH'] = '../server'

input_dir = CLIENT_INPUT_DIR
output_dir = CLIENT_OUTPUT_DIR
script_path = PATCHING_SCRIPT_PATH

# run the patched files and save the output in a json file
def run_file(file_path: str):
    result = subprocess.run(['python3', file_path], stdout=subprocess.PIPE, stderr=subprocess.PIPE,env={'PYTHONPATH':os.environ['PYTHONPATH']})
    stderr = result.stderr.decode('utf-8')
    stderr = stderr.strip()
    print("Standard Error",stderr)
    return result

file_paths = [os.path.join(output_dir, file) for file in os.listdir(output_dir) if file.endswith('.py')]
print("File paths are:",file_paths)

with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
    results = [executor.submit(run_file, file_path) for file_path in file_paths]
    for f in concurrent.futures.as_completed(results):
        pprint(f.result())
        pprint(f)
