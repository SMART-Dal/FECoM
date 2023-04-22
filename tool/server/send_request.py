import dill
import requests
import os
import traceback
import time
import json
from pathlib import Path

from tool.server.server_config import URL, DEBUG
from tool.server.function_details import FunctionDetails, build_function_details

# TODO maybe requires some refactoring
def store_response(response, experiment_file_path: Path):
    # send_request give experiment_file_path a default value of None for testing purposes. But should be overwritten if want to save response.
    if experiment_file_path is None:
        raise FileNotFoundError("Experiment File Path is None, but was expected to be a valid Path.")
    
    if DEBUG:
        print(f"Result: {str(response)[:100]}")

    try:
        if experiment_file_path.is_file():
            with open(experiment_file_path, 'r') as f:
                file_content = f.read()
            if file_content.strip():
                existing_data = json.loads(file_content)
            else:
                existing_data = []
        else:
            existing_data = []
            with open(experiment_file_path, 'w') as f:
                json.dump(existing_data, f)

    except Exception as e:
        raise Exception(f"Error opening file: {e}")

    if response:
        try:
            data = json.loads(response)
            existing_data.append(data)
            if DEBUG:
                print("Data loaded from response")
            with open(experiment_file_path, 'w') as f:
                json.dump(existing_data, f)
            if DEBUG:
                print(f"Data written to file {str(experiment_file_path)}")
        except json.decoder.JSONDecodeError as e:
            raise Exception(f"Error decoding JSON: {e}")
        except Exception as e:
            raise Exception(f"Error writing JSON to file: {e}")
    else:
        print("Response content is empty")


# TODO how can we best pass the username and password to this function? Write a wrapper?
def send_request_with_func_details(function_details: FunctionDetails, experiment_file_path: Path = None, username: str = "tim9220", password: str = "qQ32XALjF9JqFh!vF3xY"):
    if DEBUG:
        print(f"Sending {function_details.function_to_run[:100]} request to {URL}")
        print("######SIZES######")
        print(f"Data size of function_args: {len(dill.dumps(function_details.args))}")
        print(f"Data size of function_kwargs: {len(dill.dumps(function_details.kwargs))}")
        print(f"Data size of method_object: {len(dill.dumps(function_details.method_object))}")

    # (1) Serialise data with dill 
    run_data = dill.dumps(function_details)

    # (2) Send the request to the server and wait for the response
    # verify = False because the server uses a self-signed certificate
    # TODO this setting throws a warning, we need to set verify to the trusted certificate path instead.
    # But this didn't work for a self-signed certificate, since a certificate authority (CA) bundle is required
    while True:
        run_resp = requests.post(URL, data=run_data, auth=(username, password), verify=False, headers={'Content-Type': 'application/octet-stream'})
        
        if DEBUG:
            print("RECEIVED RESPONSE")
        # (3) Check whether the server could execute the method successfully
        # if the HTTP status code is 500, the server could not reach a stable state.
        if run_resp.status_code == 500:
            deserialised_response = dill.loads(run_resp.content)
            error_file = "timeout_energy_data.json"
            with open(error_file, 'w') as f:
                json.dump(deserialised_response["energy_data"], f)
            time.sleep(30)
            continue  # retry the request
            # raise TimeoutError(str(deserialised_response["error"]) + "\nYou can find the energy data in ./" + error_file)
        # catch unauthorized error if authentication fails
        elif run_resp.status_code == 401:
            raise RuntimeError(run_resp.content)
        else:
            print("Successful Server response: " + str(run_resp.status_code))
            # Success, break out of the loop and continue with the rest of the code
            break

    # (4) Extract the relevant data from the response and return it
    if function_details.return_result:
        # when return_result is true, data is serialised (used for testing & debugging)
        return dill.loads(run_resp.content)
    else:
        # typically we expect json data
        store_response(run_resp.content, experiment_file_path)
        return run_resp.json()
    

# TODO refactor the client to use the build_function_details function together with the new send_request_with_func_details function,
# then remove this old function because we want to have default values in one place only.
def send_request(imports: str, function_to_run: str, function_args: list = None, function_kwargs: dict = None, max_wait_secs: int = 0, wait_after_run_secs: int = 0, return_result: bool = False, method_object = None, object_signature = None, custom_class: str = None, username: str = "tim9220", password: str = "qQ32XALjF9JqFh!vF3xY", experiment_file_path: Path = None, exec_not_eval: bool = False):
    """
    Send a request to execute any function to the server and return specified data.
    If return_result=False, the returned object is JSON of the format specified in this directory's README.md
    """ 

    function_details = build_function_details(
        imports,
        function_to_run,
        function_args,
        function_kwargs,
        max_wait_secs,
        wait_after_run_secs,
        return_result,
        method_object,
        object_signature,
        custom_class,
        exec_not_eval
    )
    return send_request_with_func_details(function_details, username, password, experiment_file_path)
