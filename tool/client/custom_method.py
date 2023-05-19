import os
# TODO do we need theses imports? @Saurabh
from pathlib import Path
import dill as pickle
import sys

from tool.client.client_config import EXPERIMENT_DIR, MAX_WAIT_S, WAIT_AFTER_RUN_S
from tool.server.send_request import send_request
# TODO: do we need this import? @Saurabh
from tool.server.function_details import FunctionDetails

# Get the path of the current file
current_path = os.path.abspath(__file__)

experiment_number = sys.argv[1]
experiment_project = sys.argv[2]

EXPERIMENT_FILE_PATH = EXPERIMENT_DIR /'method-level'/ experiment_project / f'experiment-{experiment_number}.json'

def custom_method(imports: str, function_to_run: str, method_object=None,object_signature=None, function_args: list = None, function_kwargs: dict = None, custom_class=None):
   result = send_request(imports=imports, function_to_run=function_to_run, function_args=function_args, function_kwargs=function_kwargs, max_wait_secs=MAX_WAIT_S, wait_after_run_secs=WAIT_AFTER_RUN_S, method_object=method_object,object_signature=object_signature, custom_class=custom_class,experiment_file_path=EXPERIMENT_FILE_PATH)
