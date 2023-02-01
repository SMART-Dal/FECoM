import pickle
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
    # result = pickle.loads(run_resp.content)
    result = response.content
    # if HTTP status code is 500, the server could not reach a stable state.
    # now, simply raise an error. TODO: send a new request instead.
    if response.status_code == 500:
        raise TimeoutError(result)
    elif response.status_code == 401:
        raise RuntimeError(result)
    
    if DEBUG:
        print(f"Result: {result}")
    
    
    try:
        with open('data.json', 'r') as f:
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