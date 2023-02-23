docker volume create my_dependencies

docker run --gpus 1 -it --rm --shm-size=10g --ulimit memlock=-1 --ulimit stack=67108864 -v /data/zoey/coco2017_maskrcnn_tf:/data -v /data/zoey/maskrcnn_tf_weights:/weights -v /data/zoey/maskrcnn_tf_results:/results -v /home/saurabh/code-energy-consumption:/app -p 8080:8080 -v my_dependencies:/app/dependencies -v ./mrcnn_tf2:/app/module/mrcnn_tf2 -v ~/code-energy-consumption/server:/app/server -e PYTHONPATH=/app/module:/app/server --network=host nvidia_mrcnn_tf2

docker exec <container_name_or_id> pip install -r /app/server/requirements.txt

Manual install : pip install -r requirements.txt

docker volume inspect my_dependencies | grep Mountpoint | awk '{print $2}' | tr -d ',"'
this will output : 
/data/var/lib/docker/volumes/my_dependencies/_data

add the required dependencies in the above path


# Client Agent Program for Code Energy Consumption

This is the client agent that takes a python program along with list of libraries/function they want to calculate energy consumption for, as inputs. The client agent then runs its analyzer script to extract the function calls using an Abstract Syntax Tree(AST) and creates Request Packets for these calls. Then it filters out the functions that the client/user wants the energy consumption to be calculated for. It will send these filtered request packets to the Server which will run the method,calculate the energy consumption and then send back the required energy data packet for the method calls, as a response packet.

## Usage

<!-- Read a bit about measuring Software Energy Consumption and how it is done in Linux using the Profiling Tool known as [Perf](https://perf.wiki.kernel.org/index.php/Main_Page). -->

```bash
python3 client_agent.py > testnote
```
this would give the following output(First Draft, work is going on so will update the sample output later):

```bash
['{"funcCall": "tf", "funcArgs": [], "funcKeywords": []}', '{"funcCall": "tf.placeholder", "funcArgs": ["tf.float32", ""], "funcKeywords": [""]}', '{"funcCall": "tf.placeholder", "funcArgs": ["tf.float32", ""], "funcKeywords": [""]}', '{"funcCall": "tf.Variable", "funcArgs": ["tf.float32.layer_1_neurons.tf.random_uniform"], "funcKeywords": []}', '{"funcCall": "tf.Variable", "funcArgs": ["tf.float32.layer_1_neurons.tf.zeros"], "funcKeywords": []}', '{"funcCall": "tf.nn.sigmoid", "funcArgs": ["b_h.w_h.X.tf.matmul"], "funcKeywords": []}', '{"funcCall": "tf.Variable", "funcArgs": ["tf.float32.layer_1_neurons.tf.random_uniform"], "funcKeywords": []}', '{"funcCall": "tf.Variable", "funcArgs": ["tf.float32.tf.zeros"], "funcKeywords": []}', '{"funcCall": "tf.train.AdamOptimizer.minimize", "funcArgs": ["Y.model.tf.nn.l2_loss"], "funcKeywords": []}', '{"funcCall": "tf.Session", "funcArgs": [], "funcKeywords": []}', '{"funcCall": "tf.random_uniform", "funcArgs": ["layer_1_neurons"], "funcKeywords": ["", "", "tf.float32"]}', '{"funcCall": "tf.zeros", "funcArgs": ["layer_1_neurons"], "funcKeywords": ["tf.float32"]}', '{"funcCall": "tf.random_uniform", "funcArgs": ["layer_1_neurons"], "funcKeywords": ["", "", "tf.float32"]}', '{"funcCall": "tf.zeros", "funcArgs": [""], "funcKeywords": ["tf.float32"]}', '{"funcCall": "tf.matmul", "funcArgs": ["h", "w_o"], "funcKeywords": []}', '{"funcCall": "tf.nn.l2_loss", "funcArgs": ["Y.model"], "funcKeywords": []}', '{"funcCall": "tf.initialize_all_variables", "funcArgs": [], "funcKeywords": []}', '{"funcCall": "tf.matmul", "funcArgs": ["X", "w_h"], "funcKeywords": []}', '{"funcCall": "tf.train.AdamOptimizer", "funcArgs": [], "funcKeywords": []}', '{"funcCall": "tf.nn.l2_loss", "funcArgs": ["y_validation.model"], "funcKeywords": []}']
```


## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

Please make sure to update tests as appropriate.
