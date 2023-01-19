import pickle
import requests
from config import URL, DEBUG
from function_details import FunctionDetails

# TODO how can we best pass the username and password to this function? Write a wrapper?
def send_request(imports: str, function_to_run: str, function_args: list = None, function_kwargs: dict = None, max_wait_secs=0, method_object=None, custom_class=None, username: str = "tim9220", password: str = "qQ32XALjF9JqFh!vF3xY"):
    """
    Send a request to execute any function and show the result
    TODO continue here: test compatibility with other functions and modules
    """

    function_details = FunctionDetails(
        imports,
        function_to_run,
        function_args,
        function_kwargs,
        max_wait_secs,
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
    run_resp = requests.post(URL, data=run_data, auth=(username, password), verify=False, headers={'Content-Type': 'application/octet-stream'})

    result = pickle.loads(run_resp.content)

    # if HTTP status code is 500, the server could not reach a stable state.
    # now, simply raise an error. TODO: send a new request instead.
    if run_resp.status_code == 500:
        raise TimeoutError(result)
    elif run_resp.status_code == 401:
        raise RuntimeError(result)
    
    if DEBUG:
        print(f"Result: {result}")
    
    return result