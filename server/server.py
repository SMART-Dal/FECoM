"""
Server to receive client requests to run ML methods and measure energy.
"""

import time
import statistics as stats
import pickle
from config import API_PATH, DEBUG, SERVER_HOST, SERVER_PORT, CPU_STD_TO_MEAN, RAM_STD_TO_MEAN, GPU_STD_TO_MEAN
from flask import Flask, Response, request


app = Flask(__name__)

def is_stable_state(max_wait_secs: int):
    """
    Return True only when the system's energy consumption is stable.
    """
    # For testing purposes
    if max_wait_secs == 0:
        return True

    # only consider the last n points
    n = 50
    # tolerance for difference between stable stdev/mean ratio and current ratio
    tolerance = 0.1

    filepath = "./energy_measurement/out/2022-11-23/"

    # in each loop iteration, load new data, calculate statistics and check if the energy is stable.
    # try this for the specified number of seconds
    for i in range(max_wait_secs*2):
        # load CPU & RAM data
        cpu_ram = []
        with open(f'{filepath}perf.txt', 'r') as f:
            cpu_ram = f.read().splitlines(True)
        
        # load GPU data
        gpu = []
        with open(f'{filepath}nvidia_smi.txt', 'r') as f:
            gpu = f.read().splitlines(True)

        # generate lists of data
        last_n_cpu_energies = [float(line.strip(' ').split(';')[1]) for line in cpu_ram[2::2][-n:]]
        last_n_ram_energies = [float(line.strip(' ').split(';')[1]) for line in cpu_ram[3::2][-n:]]
        last_n_gpu_energies = [float(line.split(' ')[2]) for line in gpu[-n:]]

        # stable means that stdv/mean ratios are smaller or equal to the stable ratios determined experimentally
        # the tolerance value set above specifies how much relative deviation we want to allow
        cpu_stable = (stats.stdev(last_n_cpu_energies) / stats.mean(last_n_cpu_energies)) <= ((1 + tolerance)*CPU_STD_TO_MEAN)
        ram_stable = (stats.stdev(last_n_ram_energies) / stats.mean(last_n_ram_energies)) <= ((1 + tolerance)*RAM_STD_TO_MEAN)
        gpu_stable = (stats.stdev(last_n_gpu_energies) / stats.mean(last_n_gpu_energies)) <= ((1 + tolerance)*GPU_STD_TO_MEAN)

        if DEBUG:
            print(f"CPU: {last_n_cpu_energies} \n stable: {cpu_stable} \n stdv: {stats.stdev(last_n_cpu_energies)} \n mean: {stats.mean(last_n_cpu_energies)}")
            print()
            print(f"RAM: {last_n_ram_energies} \n stable: {ram_stable} \nstdv: {stats.stdev(last_n_ram_energies)} \n mean: {stats.mean(last_n_ram_energies)}")
            print()
            print(f"GPU: {last_n_gpu_energies} \n stable: {gpu_stable} \nstdv: {stats.stdev(last_n_gpu_energies)} \n mean: {stats.mean(last_n_gpu_energies)}")

        if cpu_stable and ram_stable and gpu_stable:
            return True
        else:
            time.sleep(0.5)
        

    return False

def run_method(imports: str, function_to_run: str, method_object: object, args: list, kwargs: dict, max_wait_secs: int):
    """
    Run the method given by function_to_run with the given arguments (args) and keyword arguments (kwargs).
    These two variables appear to not be used, however, they are used when evaluating the function_to_run
    since this is a string in the format function_signature(*args), function_signature(**kwargs) or
    function_signature(*args, **kwargs).
    """
    # WARNING: potential security risk from exec and eval statements

    # (1) import relevant modules
    exec(imports)

    # # (2) continue only when the system has reached a stable state of energy consumption
    if not is_stable_state(max_wait_secs):
        raise TimeoutError(f"System could not reach a stable state within {max_wait_secs} seconds")

    # (3) evaluate the function return. This is where we should measure energy.
    # if this is a method, initialise obj to hold the given object
    if method_object is not None:
        obj = method_object
    
    func_return = eval(function_to_run)

    # if we run a method, also return the object
    if method_object is not None:
        func_return = {
            "return": func_return,
            "method_object": obj
        }


    if DEBUG:
        print(f"Performed {function_to_run} on input")
        print(f"Output: {func_return}")
    
    return func_return


@app.route(API_PATH, methods=["POST"])
def run_method_and_return_result():
    method_details = pickle.loads(request.data)
    
    if DEBUG:
        print(f"Received method details: {method_details}")
    
    try:
        output = run_method(
            method_details["imports"],
            method_details["function"],
            method_details["method_object"],
            method_details["args"],
            method_details["kwargs"],
            method_details["max_wait_secs"]
        )
        status = 200
    except TimeoutError as e:
        output = e
        status = 500

    data = pickle.dumps(output)

    response = Response(
        response=data,
        status=status,
        mimetype='application/octet_stream'
    )

    return response


# start flask app
if __name__ == "__main__":
    app.run(host=SERVER_HOST, port=SERVER_PORT, debug=True)