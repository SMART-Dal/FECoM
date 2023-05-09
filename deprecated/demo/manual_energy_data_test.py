import sys
import numpy as np
import pandas as pd

from tool.server.server_config import DEBUG
from tool.server.send_request import send_request


def test_matmul_request():
    arr1 = np.random.rand(100,100)
    arr2 = np.random.rand(100,100)

    imports = "import numpy as np"
    function_to_run = "np.matmul(*args)"
    function_args = [arr1, arr2]
    function_kwargs = None

    result = send_request(imports, function_to_run, function_args, function_kwargs, return_result=True)
    
    assert(type(result["return"])==np.ndarray)
    assert(result["return"].shape==(100,100))
    cpu_df = pd.read_json(result["energy_data"]["cpu"], orient="split")
    ram_df = pd.read_json(result["energy_data"]["ram"], orient="split")
    gpu_df = pd.read_json(result["energy_data"]["gpu"], orient="split")
    print(cpu_df)
    print(ram_df)
    print(gpu_df)

if __name__ == "__main__":
    test_matmul_request()