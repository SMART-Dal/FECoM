import pickle
import requests
# add parent directory to path
import sys
sys.path.insert(0,'..')
# from config import URL, DEBUG
from function_details import FunctionDetails
URL = ''
DEBUG = True

import asyncio
import aiohttp
import json

session = aiohttp.ClientSession()

async def post_and_store(url, data):
    async with session.post(url, json=data) as resp:
        data = await resp.json()
        with open('data.json', 'w') as f:
            json.dump(data, f)

# def run_post_and_store(url, data):
#     asyncio.run(post_and_store(url, data))

def send_request(imports: str, function_to_run: str, function_args: list = None, function_kwargs: dict = None, max_wait_secs=0, method_object=None, custom_class=None):
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

    # if DEBUG:
    #     print(f"sending {function_to_run} request to {URL}")

    run_data = pickle.dumps(function_details)
    asyncio.run(post_and_store(url, data))

    # run_resp = requests.post(URL, data=run_data, headers={'Content-Type': 'application/octet-stream'})

    result = pickle.loads(run_resp.content)

    # if HTTP status code is 500, the server could not reach a stable state.
    # now, simply raise an error. TODO: send a new request instead.
    # if run_resp.status_code == 500:
    #     raise TimeoutError(result)
    
    # if DEBUG:
    #     print(f"Result: {result}")
    
    return result