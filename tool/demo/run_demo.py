"""
Run different demos of the server program. They are based on a simple training call.
- demo_timeout(): 
    - if run=True, will send a request and if it times out, will plot a graph of the energy data received from the timeout
    - if run=False, will use the existing timeout_energy_data.json file to plot the graph
- demo_start_end_time_graphing(): 
    - if results=None, will send a request and plot the resulting energy data with start and end times
    - if results=<json object like the one returned by the server>, it will simply plot it and not send a request.
"""
import json
from pathlib import Path

import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt

from tool.server.send_request import send_request
from tool.demo.plot_energy import combined_plot, plot_energy_from_dfs

"""
HELPER METHODS to setup a simple model training task for the demos
"""
# set this here so it can be accessed by other methods than just run_mnist_model_train
FUNCTION_TO_RUN = "obj.fit(*args,**kwargs)"

def compile_model():
    model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(10)
    ])
    # configure & compile model
    model.compile(optimizer='adam',
                loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                metrics=['accuracy'])
    return model

def mnist():
    mnist = tf.keras.datasets.mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0
    return x_train, y_train, x_test, y_test

def run_mnist_model_train(max_wait_secs, wait_after_run_secs, epochs):
    x_train, y_train, _, _ = mnist()
    compiled_model = compile_model()

    # training
    imports = "import tensorflow as tf"
    function_args = x_train, y_train
    function_kwargs = {"epochs": epochs}
    method_object = compiled_model
    
    results = send_request(imports, FUNCTION_TO_RUN, function_args, function_kwargs, method_object=method_object, max_wait_secs=max_wait_secs, return_result=False, wait_after_run_secs=wait_after_run_secs)

    return results
    # "energy_data": {
    #     "cpu": df_cpu_json,
    #     "ram": df_ram_json,
    #     "gpu": df_gpu_json
    # },
    # "start_time": start_time,
    # "end_time": end_time

def convert_json_to_df(results):
    cpu_df = pd.read_json(results["energy_data"]["cpu"], orient="split")
    ram_df = pd.read_json(results["energy_data"]["ram"], orient="split")
    gpu_df = pd.read_json(results["energy_data"]["gpu"], orient="split")
    return cpu_df, ram_df, gpu_df

def convert_timeout_json_to_df(results):
    cpu_df = pd.read_json(results["cpu"], orient="split")
    ram_df = pd.read_json(results["ram"], orient="split")
    gpu_df = pd.read_json(results["gpu"], orient="split")
    return cpu_df, ram_df, gpu_df

"""
DEMO METHODS
"""
def demo_timeout():
    # uncomment to also send a request
    # results = run_mnist_model_train(max_wait_secs=30, wait_after_run_secs=20)
    with open('timeout_energy_data.json', 'r') as f:
        results = json.load(f)

    cpu_df, ram_df, gpu_df = convert_timeout_json_to_df(results)

"""
DEMO METHODS
"""
def demo_timeout(run=True):
    if run:
        resp = run_mnist_model_train(max_wait_secs=10, wait_after_run_secs=20)
    with open('timeout_energy_data.json', 'r') as f:
        results = json.load(f)

    cpu_df, ram_df, gpu_df = convert_timeout_json_to_df(results)
    cpu_stdv_mean = cpu_df["energy (J)"].std()/cpu_df["energy (J)"].mean()
    ram_stdv_mean = ram_df["energy (J)"].std()/ram_df["energy (J)"].mean()
    gpu_stdv_mean = gpu_df["power_draw (W)"].std()/gpu_df["power_draw (W)"].mean()
    print(f"CPU: {cpu_stdv_mean}")
    print(f"RAM: {ram_stdv_mean}")
    print(f"GPU: {gpu_stdv_mean}")

    df = combined_plot(cpu_df, ram_df, gpu_df)
    print("###COMBINED DF###")
    print(df)
    df.plot()

    figure = plt.gcf() # get current figure
    figure.set_size_inches(20, 6)
    plt.savefig('timout_energy_plot.png', dpi=200)

def demo_start_end_time_graphing(max_wait_secs, wait_after_run_secs, epochs, results=None):
    # allow passing results, for cases where only graphing of existing data is required
    if results is None:
        results = run_mnist_model_train(max_wait_secs, wait_after_run_secs, epochs)
        # save json response to file
        with open('methodcall_energy.json', 'w') as f:
            json.dump(results, f)
    
    results = results[FUNCTION_TO_RUN]

    cpu_df, ram_df, gpu_df = convert_json_to_df(results)

    start_time_perf = results["times"]["start_time_perf"]
    end_time_perf = results["times"]["end_time_perf"]
    start_time_nvidia = results["times"]["start_time_nvidia"]
    end_time_nvidia = results["times"]["end_time_nvidia"]

    print("###INPUT SIZES###")
    print(results["input_sizes"])
    print("###DATA FRAMES###")
    print(gpu_df)
    print(cpu_df)

    ns_conversion = 1000000000
    pickle_load_time_perf = (results["times"]["pickle_load_time"] - results["times"]["sys_start_time_perf"]) / ns_conversion
    import_time_perf = (results["times"]["import_time"] - results["times"]["sys_start_time_perf"]) / ns_conversion
    begin_stable_check_time_perf = (results["times"]["begin_stable_check_time"] - results["times"]["sys_start_time_perf"]) / ns_conversion
    
    # time, label, color
    perf_times = [
        (start_time_perf, "method_start", 'r'),
        (end_time_perf, "method_end", 'r'),
        (pickle_load_time_perf, "pickle_load", 'b'),
        (import_time_perf, "import", 'g'),
        (begin_stable_check_time_perf, "stable_check", 'y')
    ]

    pickle_load_time_nvidia = (results["times"]["pickle_load_time"] - results["times"]["sys_start_time_nvidia"]) / ns_conversion
    import_time_nvidia = (results["times"]["import_time"] - results["times"]["sys_start_time_nvidia"]) / ns_conversion
    begin_stable_check_time_nvidia = (results["times"]["begin_stable_check_time"] - results["times"]["sys_start_time_nvidia"]) / ns_conversion
    nvidia_times = [
        (start_time_nvidia, "method_start", 'r'),
        (end_time_nvidia, "method_end", 'r'),
        (pickle_load_time_nvidia, "pickle_load", 'b'),
        (import_time_nvidia, "import", 'g'),
        (begin_stable_check_time_nvidia, "stable_check", 'y')
    ]

    plot_energy_from_dfs(cpu_df, ram_df, gpu_df, perf_times, nvidia_times)

    ## uncomment for a different plot that shows all 3 time series on the same graph (does not show start/end times)
    # df = combined_plot(cpu_df, ram_df, gpu_df)
    # print("###COMBINED DF###")
    # print(df)
    # df.plot()

    figure = plt.gcf() # get current figure
    figure.set_size_inches(20, 6)
    plt.savefig('energy_plot.png', dpi=200)

if __name__ == "__main__":
    # with open('methodcall_energy.json', 'r') as f:
    #     results = json.load(f)
    demo_start_end_time_graphing(max_wait_secs=60, wait_after_run_secs=20, epochs=5) # results=results