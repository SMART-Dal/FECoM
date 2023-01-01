import pickle
import requests
# add parent directory to path
import sys
sys.path.insert(0,'..')
from config import URL, DEBUG

def send_request(imports: str, function_to_run: str, function_args: list = None, function_kwargs: dict = None, max_wait_secs=0, method_object=None):
    """
    Send a request to execute any function and show the result
    TODO continue here: test compatibility with other functions and modules
    """

    method_details = {
        "imports": imports,
        "function": function_to_run,
        "method_object": method_object,
        "args": function_args,
        "kwargs": function_kwargs,
        "max_wait_secs": max_wait_secs
    }

    if DEBUG:
        print(f"sending {function_to_run} request to {URL}")

    data = pickle.dumps(method_details)
    
    resp = requests.post(URL, data=data, headers={'Content-Type': 'application/octet-stream'})

    result = pickle.loads(resp.content)

    # if HTTP status code is 500, the server could not reach a stable state.
    # now, simply raise an error. TODO: send a new request instead.
    if resp.status_code == 500:
        raise TimeoutError(result)
    
    if DEBUG:
        print(f"Result: {result}")
    
    return result