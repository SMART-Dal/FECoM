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

def run_mnist_model_train():
    x_train, y_train, _, _ = mnist()
    compiled_model = compile_model()

    # training
    imports = "import tensorflow as tf"
    function_to_run = "obj.fit(*args,**kwargs)"
    function_args = x_train, y_train
    function_kwargs = {"epochs": 5}
    method_object = compiled_model
    
    results = send_request(imports, function_to_run, function_args, function_kwargs, method_object=method_object, max_wait_secs=0, return_result=False, wait_after_run_secs=10)

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

if __name__ == "__main__":
    results = run_mnist_model_train()
    with open('methodcall_energy.json', 'w') as f:
        json.dump(results, f)

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

    

    df = combined_plot(cpu_df, ram_df, gpu_df)
    print("###COMBINED DF###")
    print(df)
    df.plot()

    # ax = cpu_df["time_elapsed", "energy (J)"]
    # cmap = matplotlib.colors.ListedColormap(['grey', 'white'])
    # ax.pcolorfast(ax.get_xlim(), ax.get_ylim(),
    #           cpu_df['in_execution'].values[np.newaxis],
    #           cmap=cmap, alpha=0.3)
    figure = plt.gcf() # get current figure
    figure.set_size_inches(20, 6)
    plt.savefig('energy_plot.png', dpi=200)
