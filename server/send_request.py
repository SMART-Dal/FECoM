import dill as pickle
import requests
from config import URL, DEBUG
from function_details import FunctionDetails
import sys
sys.path.insert(0,'..')

import json
import concurrent.futures

# TODO how can we best pass the username and password to this function? Write a wrapper?
def send_request(imports: str, function_to_run: str, function_args: list = None, function_kwargs: dict = None, max_wait_secs: int = 0, wait_after_run_secs: int = 0, return_result: bool = False, method_object = None, custom_class: str = None, username: str = "tim9220", password: str = "qQ32XALjF9JqFh!vF3xY"):
    """
    Send a request to execute any function and show the result
    """
    raise DeprecationWarning("Do we still need this method? If yes, we should refactor it so that it gets the same functionality as send_single_thread_request. Let's discuss ~Tim")

    function_details = FunctionDetails(
        imports,
        function_to_run,
        function_args,
        function_kwargs,
        max_wait_secs,
        wait_after_run_secs,
        return_result,
        method_object,
        custom_class,
        method_object.__module__ if custom_class is not None else None
    )

    if DEBUG:
        print(f"sending {function_to_run} request to {URL}")

    run_data = pickle.dumps(function_details)

    # verify = False because the server uses a self-signed certificate
    # TODO this setting throws a warning, we need to set verify to the trusted certificate path instead.
    # But this didn't work for a self-signed certificate, since a certificate authority (CA) bundle is required
    
    # run_resp = requests.post(URL, data=run_data, auth=(username, password), verify=False, headers={'Content-Type': 'application/pickle'})
    with concurrent.futures.ThreadPoolExecutor() as executor:
                future_response = executor.submit(requests.post, URL, verify=False, data=run_data, auth=(username, password), headers={'Content-Type': 'application/octet-stream'})
                future_response.add_done_callback(store_response)


def store_response(future_response):
    response = future_response.result()
    # if HTTP status code is 500, the server could not reach a stable state.
    # now, simply raise an error. TODO: send a new request instead.
    if response.status_code == 500:
        raise TimeoutError(response.content)
    elif response.status_code == 401:
        raise RuntimeError(response.content)
    
    if DEBUG:
        print(f"Result: {response.content}")
    
    try:
        with open('methodcall-energy-dataset.json', 'r') as f:
            existing_data = json.load(f)
    except:
        existing_data = []

    if response.content:
        try:
            data = json.loads(response.content)
            with open('methodcall-energy-dataset.json', 'w') as f:
                existing_data.append(data)
                json.dump(existing_data, f)
        except json.decoder.JSONDecodeError as e:
            print(f"Error decoding JSON: {e}")
        except:
            print("Error writing JSON to file")
    else:
        print("Response content is empty")


# for testing purposes
# TODO how can we best pass the username and password to this function? Write a wrapper?
def send_single_thread_request(imports: str, function_to_run: str, function_args: list = None, function_kwargs: dict = None, max_wait_secs: int = 0, wait_after_run_secs: int = 0, return_result: bool = False, method_object = None, custom_class: str = None, username: str = "tim9220", password: str = "qQ32XALjF9JqFh!vF3xY"):
    """
    Send a request to execute any function to the server and return specified data.
    If return_result=False, the returned object is JSON of the following format:
    function_to_run: {
        "energy_data": {
            "cpu": df_cpu_json,
            "ram": df_ram_json,
            "gpu": df_gpu_json
        },
        "times": {
            "start_time_server": start_time_server,
            "end_time_server": end_time_server,
            "start_time_perf": start_time_perf, 
            "end_time_perf": end_time_perf,
            "start_time_nvidia": start_time_nvidia_normalised,
            "end_time_nvidia": end_time_nvidia_normalised,
        },
        "input_sizes" {
            "args_size": args_size_bit,
            "kwargs_size": kwargs_size_bit,
            "object_size": object_size_bit
        }
    }
    """
    # (1) Construct a FunctionDetails object containg all the function data & settings for the server
    function_details = FunctionDetails(
        imports,
        function_to_run,
        function_args,
        function_kwargs,
        max_wait_secs,
        wait_after_run_secs,
        return_result,
        method_object,
        custom_class,
        method_object.__module__ if custom_class is not None else None
    )

    if DEBUG:
        print(f"Sending {function_to_run} request to {URL}")
        print("######SIZES######")
        print(f"Data size of function_args: {len(pickle.dumps(function_args))}")
        print(f"Data size of function_kwargs: {len(pickle.dumps(function_kwargs))}")
        print(f"Data size of method_object: {len(pickle.dumps(method_object))}")

    # (2) Serialise data with pickle 
    run_data = pickle.dumps(function_details)

    # (3) Send the request to the server and wait for the response
    # verify = False because the server uses a self-signed certificate
    # TODO this setting throws a warning, we need to set verify to the trusted certificate path instead.
    # But this didn't work for a self-signed certificate, since a certificate authority (CA) bundle is required
    run_resp = requests.post(URL, data=run_data, auth=(username, password), verify=False, headers={'Content-Type': 'application/octet-stream'})
    
    if DEBUG:
        print("RECEIVED RESPONSE")
    # (4) Check whether the server could execute the method successfully
    # if the HTTP status code is 500, the server could not reach a stable state.
    # TODO: now, we simply raise an error and save the energy data. Should we send a new request instead?
    if run_resp.status_code == 500:
        deserialised_response = pickle.loads(run_resp.content)
        error_file = "timeout_energy_data.json"
        with open(error_file, 'w') as f:
            json.dump(deserialised_response["energy_data"], f)
        raise TimeoutError(str(deserialised_response["error"]) + "\nYou can find the energy data in ./" + error_file)
    # catch internal server errors
    elif run_resp.status_code == 401:
        raise RuntimeError(run_resp.content)

    # (5) Extract the relevant data from the response and return it
    if return_result:
        # when return_result is true, data is serialised (used for testing & debugging)
        return pickle.loads(run_resp.content)
    else:
        # typically we expect json data
        return run_resp.json()
    """
    """
    # TODO REMOVE UNUSED CODE
    """
    """
    # if run_resp.status_code == 500:
    #     raise TimeoutError(run_resp.content)
    # elif run_resp.status_code == 401:
    #     raise RuntimeError(run_resp.content)
    
    # if return_result:
    #     return pickle.loads(run_resp.content)
    # else:
    #     result = pickle.loads(result)

    # try:
    #     with open('methodcall-energy-dataset.json', 'r') as f:
    #         existing_data = json.load(f)
    # except:
    #     existing_data = []

    # if result:
    #     try:
    #         data = json.loads(result)
    #         with open('methodcall-energy-dataset.json', 'w') as f:
    #             existing_data.append(data)
    #             json.dump(existing_data, f)
    #     except json.decoder.JSONDecodeError as e:
    #         print(f"Error decoding JSON: {e}")
    #     except:
    #         print("Error writing JSON to file")
    # else:
    #     print("Response content is empty")
    
    # return run_resp.json()
