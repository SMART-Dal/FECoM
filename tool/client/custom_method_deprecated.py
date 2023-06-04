import os
# TODO do we need theses imports? @Saurabh
from pathlib import Path
import sys
import numpy as np
from tool.client.client_config import EXPERIMENT_DIR, MAX_WAIT_S, WAIT_AFTER_RUN_S
from tool.measurement.send_request import send_request
# TODO: do we need this import? @Saurabh
from tool.measurement.function_details import FunctionDetails
import json

# Get the path of the current file
current_path = os.path.abspath(__file__)

experiment_number = sys.argv[1]
experiment_project = sys.argv[2]

EXPERIMENT_FILE_PATH = EXPERIMENT_DIR /'method-level'/ experiment_project / f'experiment-{experiment_number}.json'

# List of calls that can be skipped if they consume negligible energy

skip_calls_file_path = EXPERIMENT_FILE_PATH.parent / "skip_calls.json"
if skip_calls_file_path.exists():
    with open(skip_calls_file_path, "r") as f:
        skip_calls = json.load(f)
else:
    skip_calls = []
    with open(skip_calls_file_path, "w") as f:
        json.dump(skip_calls, f)

def custom_method(imports: str, function_to_run: str, method_object=None,object_signature=None, function_args: list = None, function_kwargs: dict = None, custom_class=None):
   if skip_calls is not None and any(
        call['function_to_run'] == function_to_run and 
        np.array_equal(call['function_args'], function_args) and 
        call['function_kwargs'] == function_kwargs 
        for call in skip_calls
        ):
        print('skipping call: ', function_to_run)
        return
   
   result = send_request(imports=imports, function_to_run=function_to_run, function_args=function_args, function_kwargs=function_kwargs, max_wait_secs=MAX_WAIT_S, wait_after_run_secs=WAIT_AFTER_RUN_S, method_object=method_object,object_signature=object_signature, custom_class=custom_class,experiment_file_path=EXPERIMENT_FILE_PATH)
   if result is not None and isinstance(result, dict) and len(result) == 1:
    energy_data = next(iter(result.values()))
    if skip_calls is not None and 'start_time_perf' in energy_data['times'] and 'end_time_perf' in energy_data['times'] and 'start_time_nvidia' in energy_data['times'] and 'end_time_nvidia' in energy_data['times'] and energy_data['times']['start_time_perf'] == energy_data['times']['end_time_perf'] and energy_data['times']['start_time_nvidia'] == energy_data['times']['end_time_nvidia']:
        call_to_skip = {
            'function_to_run': function_to_run,
            'function_args': function_args,
            'function_kwargs': function_kwargs
        }
        try:
            json.dumps(call_to_skip)
            if call_to_skip not in skip_calls:
                skip_calls.append(call_to_skip)
                with open(skip_calls_file_path, 'w') as f:
                    json.dump(skip_calls, f)
                print('skipping call added, current list is: ', skip_calls)
            else:
                print('Skipping call already exists.')
        except TypeError:
            print('Ignore: Skipping call is not JSON serializable, skipping append and dump.')
#     print(energy_data)
   else:
    print("Invalid dictionary object or does not have one key-value pair.")
   

