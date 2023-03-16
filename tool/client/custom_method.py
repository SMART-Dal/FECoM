import os
from pathlib import Path
import dill as pickle

from tool.client.client_config import EXPERIMENT_DIR, EXPERIMENT_TAG
from tool.server.send_request import send_request
from tool.server.function_details import FunctionDetails

# Get the path of the current file
current_path = os.path.abspath(__file__)

# Get the immediate folder name and the file name
immediate_folder, file_name = os.path.split(current_path)
immediate_folder = os.path.basename(immediate_folder)

# Remove the file extension from the file name
experiment_file_name = os.path.splitext(file_name)[0]

EXPERIMENT_FILE_PATH = EXPERIMENT_DIR / immediate_folder / EXPERIMENT_TAG / (experiment_file_name + '-energy.json')

def custom_method(func,imports: str, function_to_run: str, method_object=None, function_args: list = None, function_kwargs: dict = None,max_wait_secs=0, custom_class=None):
   result = send_request(imports=imports, function_to_run=function_to_run, function_args=function_args, function_kwargs=function_kwargs, max_wait_secs=max_wait_secs, method_object=method_object, custom_class=custom_class,experiment_file_path=EXPERIMENT_FILE_PATH)
   return func

if __name__ == "__main__":
   print(EXPERIMENT_FILE_PATH)