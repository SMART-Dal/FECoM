"""
Server demo for the meeting on 3rd February 2023
"""
import json

import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt

import sys
sys.path.insert(0,'..')
from send_request import send_single_thread_request as send_request
sys.path.insert(0,'../energy_measurement')
from plot_energy import combined_plot, plot_energy_from_dfs

"""
HELPER METHODS to setup a simple model training task
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

def run_mnist_model_train(max_wait_secs, wait_after_run_secs):
    x_train, y_train, _, _ = mnist()
    compiled_model = compile_model()

    # training
    imports = "import tensorflow as tf"
    function_args = x_train, y_train
    function_kwargs = {"epochs": 5}
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

"""
DEMO METHODS
"""
def demo_timeout():
    results = run_mnist_model_train(max_wait_secs=30, wait_after_run_secs=20)
    with open('timeout_energy_data.json', 'r') as f:
        results = json.load(f)
    
    # TODO: remove this
    # results = {"energy_data": data}

    cpu_df, ram_df, gpu_df = convert_json_to_df(results)
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

def demo_start_end_time_graphing():
    results = run_mnist_model_train(max_wait_secs=30, wait_after_run_secs=20)
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

    plot_energy_from_dfs(cpu_df, ram_df, gpu_df, start_time_perf, end_time_perf, start_time_nvidia, end_time_nvidia)

    ## uncomment for a different plot that shows all 3 time series on the same graph (does not show start/end times)
    # df = combined_plot(cpu_df, ram_df, gpu_df)
    # print("###COMBINED DF###")
    # print(df)
    # df.plot()

    figure = plt.gcf() # get current figure
    figure.set_size_inches(20, 6)
    plt.savefig('energy_plot.png', dpi=200)

if __name__ == "__main__":
    demo_start_end_time_graphing()