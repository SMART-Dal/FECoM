import multiprocessing
import os
import random
import portpicker
import tensorflow as tf
import sys
from tool.client.client_config import EXPERIMENT_DIR
from tool.server.local_execution import before_execution as before_execution_INSERTED_INTO_SCRIPT
from tool.server.local_execution import after_execution as after_execution_INSERTED_INTO_SCRIPT
experiment_number = sys.argv[1]
experiment_project = sys.argv[2]
EXPERIMENT_FILE_PATH = EXPERIMENT_DIR / 'method-level' / experiment_project / f'experiment-{experiment_number}.json'

def create_in_process_cluster(num_workers, num_ps):
    """Creates and starts local servers and returns the cluster_resolver."""
    worker_ports = [portpicker.pick_unused_port() for _ in range(num_workers)]
    ps_ports = [portpicker.pick_unused_port() for _ in range(num_ps)]
    cluster_dict = {}
    cluster_dict['worker'] = ['localhost:%s' % port for port in worker_ports]
    if num_ps > 0:
        cluster_dict['ps'] = ['localhost:%s' % port for port in ps_ports]
    start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT()
    cluster_spec = tf.train.ClusterSpec(cluster_dict)
    after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tf.train.ClusterSpec(*args)', method_object=None, object_signature=None, function_args=[cluster_dict], function_kwargs={})
    start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT()
    worker_config = tf.compat.v1.ConfigProto()
    after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tf.compat.v1.ConfigProto()', method_object=None, object_signature=None, function_args=[], function_kwargs={})
    if multiprocessing.cpu_count() < num_workers + 1:
        worker_config.inter_op_parallelism_threads = num_workers + 1
    for i in range(num_workers):
        start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT()
        tf.distribute.Server(cluster_spec, job_name='worker', task_index=i, config=worker_config, protocol='grpc')
        after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tf.distribute.Server(*args, **kwargs)', method_object=None, object_signature=None, function_args=[cluster_spec], function_kwargs={'job_name': 'worker', 'task_index': i, 'config': worker_config, 'protocol': 'grpc'})
    for i in range(num_ps):
        start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT()
        tf.distribute.Server(cluster_spec, job_name='ps', task_index=i, protocol='grpc')
        after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tf.distribute.Server(*args, **kwargs)', method_object=None, object_signature=None, function_args=[cluster_spec], function_kwargs={'job_name': 'ps', 'task_index': i, 'protocol': 'grpc'})
    start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT()
    cluster_resolver = tf.distribute.cluster_resolver.SimpleClusterResolver(cluster_spec, rpc_layer='grpc')
    after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tf.distribute.cluster_resolver.SimpleClusterResolver(*args, **kwargs)', method_object=None, object_signature=None, function_args=[cluster_spec], function_kwargs={'rpc_layer': 'grpc'})
    return cluster_resolver
os.environ['GRPC_FAIL_FAST'] = 'use_caller'
NUM_WORKERS = 3
NUM_PS = 2
cluster_resolver = create_in_process_cluster(NUM_WORKERS, NUM_PS)
start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT()
variable_partitioner = tf.distribute.experimental.partitioners.MinSizePartitioner(min_shard_bytes=256 << 10, max_shards=NUM_PS)
after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tf.distribute.experimental.partitioners.MinSizePartitioner(**kwargs)', method_object=None, object_signature=None, function_args=[], function_kwargs={'min_shard_bytes': 256 << 10, 'max_shards': NUM_PS})
start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT()
strategy = tf.distribute.ParameterServerStrategy(cluster_resolver, variable_partitioner=variable_partitioner)
after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tf.distribute.ParameterServerStrategy(*args, **kwargs)', method_object=None, object_signature=None, function_args=[cluster_resolver], function_kwargs={'variable_partitioner': variable_partitioner})
global_batch_size = 64
start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT()
x = tf.random.uniform((10, 10))
after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tf.random.uniform(*args)', method_object=None, object_signature=None, function_args=[(10, 10)], function_kwargs={})
start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT()
y = tf.random.uniform((10,))
after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tf.random.uniform(*args)', method_object=None, object_signature=None, function_args=[(10,)], function_kwargs={})
start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT()
dataset = tf.data.Dataset.from_tensor_slices((x, y)).shuffle(10).repeat()
after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tf.data.Dataset.from_tensor_slices((x, y)).shuffle(10).repeat()', method_object=None, object_signature=None, function_args=[], function_kwargs={})
start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT()
dataset = dataset.batch(global_batch_size)
after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='obj.batch(*args)', method_object=dataset, object_signature=None, function_args=[global_batch_size], function_kwargs={})
start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT()
dataset = dataset.prefetch(2)
after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='obj.prefetch(*args)', method_object=dataset, object_signature=None, function_args=[2], function_kwargs={})
with strategy.scope():
    start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT()
    model = tf.keras.models.Sequential([tf.keras.layers.Dense(10)])
    after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tf.keras.models.Sequential(*args)', method_object=None, object_signature=None, function_args=[[tf.keras.layers.Dense(10)]], function_kwargs={})
    start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT()
    model.compile(tf.keras.optimizers.legacy.SGD(), loss='mse', steps_per_execution=10)
    after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='obj.compile(*args, **kwargs)', method_object=model, object_signature=None, function_args=[tf.keras.optimizers.legacy.SGD()], function_kwargs={'loss': 'mse', 'steps_per_execution': 10})
working_dir = '/tmp/my_working_dir'
log_dir = os.path.join(working_dir, 'log')
ckpt_filepath = os.path.join(working_dir, 'ckpt')
backup_dir = os.path.join(working_dir, 'backup')
callbacks = [tf.keras.callbacks.TensorBoard(log_dir=log_dir), tf.keras.callbacks.ModelCheckpoint(filepath=ckpt_filepath), tf.keras.callbacks.BackupAndRestore(backup_dir=backup_dir)]
start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT()
model.fit(dataset, epochs=5, steps_per_epoch=20, callbacks=callbacks)
after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='obj.fit(*args, **kwargs)', method_object=model, object_signature=None, function_args=[dataset], function_kwargs={'epochs': 5, 'steps_per_epoch': 20, 'callbacks': callbacks})
feature_vocab = ['avenger', 'ironman', 'batman', 'hulk', 'spiderman', 'kingkong', 'wonder_woman']
label_vocab = ['yes', 'no']
with strategy.scope():
    start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT()
    feature_lookup_layer = tf.keras.layers.StringLookup(vocabulary=feature_vocab, mask_token=None)
    after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tf.keras.layers.StringLookup(**kwargs)', method_object=None, object_signature=None, function_args=[], function_kwargs={'vocabulary': feature_vocab, 'mask_token': None})
    start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT()
    label_lookup_layer = tf.keras.layers.StringLookup(vocabulary=label_vocab, num_oov_indices=0, mask_token=None)
    after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tf.keras.layers.StringLookup(**kwargs)', method_object=None, object_signature=None, function_args=[], function_kwargs={'vocabulary': label_vocab, 'num_oov_indices': 0, 'mask_token': None})
    start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT()
    raw_feature_input = tf.keras.layers.Input(shape=(3,), dtype=tf.string, name='feature')
    after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tf.keras.layers.Input(**kwargs)', method_object=None, object_signature=None, function_args=[], function_kwargs={'shape': (3,), 'dtype': tf.string, 'name': 'feature'})
    start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT()
    feature_id_input = feature_lookup_layer(raw_feature_input)
    after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='obj(*args)', method_object=feature_lookup_layer, object_signature=None, function_args=[raw_feature_input], function_kwargs={})
    start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT()
    feature_preprocess_stage = tf.keras.Model({'features': raw_feature_input}, feature_id_input)
    after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tf.keras.Model(*args)', method_object=None, object_signature=None, function_args=[{'features': raw_feature_input}, feature_id_input], function_kwargs={})
    start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT()
    raw_label_input = tf.keras.layers.Input(shape=(1,), dtype=tf.string, name='label')
    after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tf.keras.layers.Input(**kwargs)', method_object=None, object_signature=None, function_args=[], function_kwargs={'shape': (1,), 'dtype': tf.string, 'name': 'label'})
    start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT()
    label_id_input = label_lookup_layer(raw_label_input)
    after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='obj(*args)', method_object=label_lookup_layer, object_signature=None, function_args=[raw_label_input], function_kwargs={})
    start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT()
    label_preprocess_stage = tf.keras.Model({'label': raw_label_input}, label_id_input)
    after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tf.keras.Model(*args)', method_object=None, object_signature=None, function_args=[{'label': raw_label_input}, label_id_input], function_kwargs={})

def feature_and_label_gen(num_examples=200):
    examples = {'features': [], 'label': []}
    for _ in range(num_examples):
        features = random.sample(feature_vocab, 3)
        label = ['yes'] if 'avenger' in features else ['no']
        examples['features'].append(features)
        examples['label'].append(label)
    return examples
examples = feature_and_label_gen()

def dataset_fn(_):
    start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT()
    raw_dataset = tf.data.Dataset.from_tensor_slices(examples)
    after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tf.data.Dataset.from_tensor_slices(*args)', method_object=None, object_signature=None, function_args=[examples], function_kwargs={})
    start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT()
    train_dataset = raw_dataset.map(lambda x: ({'features': feature_preprocess_stage(x['features'])}, label_preprocess_stage(x['label']))).shuffle(200).batch(32).repeat()
    after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run="obj.map(lambda x: ({'features': feature_preprocess_stage(x['features'])}, label_preprocess_stage(x['label']))).shuffle(200).batch(32).repeat()", method_object=raw_dataset, object_signature=None, function_args=[], function_kwargs={})
    return train_dataset
with strategy.scope():
    start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT()
    model_input = tf.keras.layers.Input(shape=(3,), dtype=tf.int64, name='model_input')
    after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tf.keras.layers.Input(**kwargs)', method_object=None, object_signature=None, function_args=[], function_kwargs={'shape': (3,), 'dtype': tf.int64, 'name': 'model_input'})
    start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT()
    emb_layer = tf.keras.layers.Embedding(input_dim=len(feature_lookup_layer.get_vocabulary()), output_dim=16384)
    after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tf.keras.layers.Embedding(**kwargs)', method_object=None, object_signature=None, function_args=[], function_kwargs={'input_dim': len(feature_lookup_layer.get_vocabulary()), 'output_dim': 16384})
    start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT()
    emb_output = tf.reduce_mean(emb_layer(model_input), axis=1)
    after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tf.reduce_mean(*args, **kwargs)', method_object=None, object_signature=None, function_args=[emb_layer(model_input)], function_kwargs={'axis': 1})
    start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT()
    dense_output = tf.keras.layers.Dense(units=1, activation='sigmoid')(emb_output)
    after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run="tf.keras.layers.Dense(units=1, activation='sigmoid')(*args)", method_object=None, object_signature=None, function_args=[emb_output], function_kwargs={})
    start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT()
    model = tf.keras.Model({'features': model_input}, dense_output)
    after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tf.keras.Model(*args)', method_object=None, object_signature=None, function_args=[{'features': model_input}, dense_output], function_kwargs={})
    start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT()
    optimizer = tf.keras.optimizers.legacy.RMSprop(learning_rate=0.1)
    after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tf.keras.optimizers.legacy.RMSprop(**kwargs)', method_object=None, object_signature=None, function_args=[], function_kwargs={'learning_rate': 0.1})
    start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT()
    accuracy = tf.keras.metrics.Accuracy()
    after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tf.keras.metrics.Accuracy()', method_object=None, object_signature=None, function_args=[], function_kwargs={})
assert len(emb_layer.weights) == 2
assert emb_layer.weights[0].shape == (4, 16384)
assert emb_layer.weights[1].shape == (4, 16384)
print(emb_layer.weights[0].device)
print(emb_layer.weights[1].device)

@tf.function
def step_fn(iterator):

    def replica_fn(batch_data, labels):
        with tf.GradientTape() as tape:
            start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT()
            pred = model(batch_data, training=True)
            after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='obj(*args, **kwargs)', method_object=model, object_signature=None, function_args=[batch_data], function_kwargs={'training': True})
            start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT()
            per_example_loss = tf.keras.losses.BinaryCrossentropy(reduction=tf.keras.losses.Reduction.NONE)(labels, pred)
            after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tf.keras.losses.BinaryCrossentropy(reduction=tf.keras.losses.Reduction.NONE)(*args)', method_object=None, object_signature=None, function_args=[labels, pred], function_kwargs={})
            start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT()
            loss = tf.nn.compute_average_loss(per_example_loss)
            after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tf.nn.compute_average_loss(*args)', method_object=None, object_signature=None, function_args=[per_example_loss], function_kwargs={})
            gradients = tape.gradient(loss, model.trainable_variables)
        start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT()
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='obj.apply_gradients(*args)', method_object=optimizer, object_signature=None, function_args=[zip(gradients, model.trainable_variables)], function_kwargs={})
        start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT()
        actual_pred = tf.cast(tf.greater(pred, 0.5), tf.int64)
        after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tf.cast(*args)', method_object=None, object_signature=None, function_args=[tf.greater(pred, 0.5), tf.int64], function_kwargs={})
        start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT()
        accuracy.update_state(labels, actual_pred)
        after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='obj.update_state(*args)', method_object=accuracy, object_signature=None, function_args=[labels, actual_pred], function_kwargs={})
        return loss
    (batch_data, labels) = next(iterator)
    start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT()
    losses = strategy.run(replica_fn, args=(batch_data, labels))
    after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='obj.run(*args, **kwargs)', method_object=strategy, object_signature=None, function_args=[replica_fn], function_kwargs={'args': (batch_data, labels)})
    return strategy.reduce(tf.distribute.ReduceOp.SUM, losses, axis=None)
start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT()
coordinator = tf.distribute.coordinator.ClusterCoordinator(strategy)
after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tf.distribute.coordinator.ClusterCoordinator(*args)', method_object=None, object_signature=None, function_args=[strategy], function_kwargs={})

@tf.function
def per_worker_dataset_fn():
    return strategy.distribute_datasets_from_function(dataset_fn)
start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT()
per_worker_dataset = coordinator.create_per_worker_dataset(per_worker_dataset_fn)
after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='obj.create_per_worker_dataset(*args)', method_object=coordinator, object_signature=None, function_args=[per_worker_dataset_fn], function_kwargs={})
per_worker_iterator = iter(per_worker_dataset)
num_epochs = 4
steps_per_epoch = 5
for i in range(num_epochs):
    start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT()
    accuracy.reset_states()
    after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='obj.reset_states()', method_object=accuracy, object_signature=None, function_args=[], function_kwargs={})
    for _ in range(steps_per_epoch):
        start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT()
        coordinator.schedule(step_fn, args=(per_worker_iterator,))
        after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='obj.schedule(*args, **kwargs)', method_object=coordinator, object_signature=None, function_args=[step_fn], function_kwargs={'args': (per_worker_iterator,)})
    start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT()
    coordinator.join()
    after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='obj.join()', method_object=coordinator, object_signature=None, function_args=[], function_kwargs={})
    print('Finished epoch %d, accuracy is %f.' % (i, accuracy.result().numpy()))
start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT()
loss = coordinator.schedule(step_fn, args=(per_worker_iterator,))
after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='obj.schedule(*args, **kwargs)', method_object=coordinator, object_signature=None, function_args=[step_fn], function_kwargs={'args': (per_worker_iterator,)})
print('Final loss is %f' % loss.fetch())
start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT()
eval_dataset = tf.data.Dataset.from_tensor_slices(feature_and_label_gen(num_examples=16)).map(lambda x: ({'features': feature_preprocess_stage(x['features'])}, label_preprocess_stage(x['label']))).batch(8)
after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run="tf.data.Dataset.from_tensor_slices(feature_and_label_gen(num_examples=16)).map(lambda x: ({'features': feature_preprocess_stage(x['features'])}, label_preprocess_stage(x['label']))).batch(*args)", method_object=None, object_signature=None, function_args=[8], function_kwargs={})
start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT()
eval_accuracy = tf.keras.metrics.Accuracy()
after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tf.keras.metrics.Accuracy()', method_object=None, object_signature=None, function_args=[], function_kwargs={})
for (batch_data, labels) in eval_dataset:
    start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT()
    pred = model(batch_data, training=False)
    after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='obj(*args, **kwargs)', method_object=model, object_signature=None, function_args=[batch_data], function_kwargs={'training': False})
    start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT()
    actual_pred = tf.cast(tf.greater(pred, 0.5), tf.int64)
    after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tf.cast(*args)', method_object=None, object_signature=None, function_args=[tf.greater(pred, 0.5), tf.int64], function_kwargs={})
    start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT()
    eval_accuracy.update_state(labels, actual_pred)
    after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='obj.update_state(*args)', method_object=eval_accuracy, object_signature=None, function_args=[labels, actual_pred], function_kwargs={})
print('Evaluation accuracy: %f' % eval_accuracy.result())
with strategy.scope():
    start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT()
    eval_accuracy = tf.keras.metrics.Accuracy()
    after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tf.keras.metrics.Accuracy()', method_object=None, object_signature=None, function_args=[], function_kwargs={})

@tf.function
def eval_step(iterator):

    def replica_fn(batch_data, labels):
        start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT()
        pred = model(batch_data, training=False)
        after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='obj(*args, **kwargs)', method_object=model, object_signature=None, function_args=[batch_data], function_kwargs={'training': False})
        start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT()
        actual_pred = tf.cast(tf.greater(pred, 0.5), tf.int64)
        after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tf.cast(*args)', method_object=None, object_signature=None, function_args=[tf.greater(pred, 0.5), tf.int64], function_kwargs={})
        start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT()
        eval_accuracy.update_state(labels, actual_pred)
        after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='obj.update_state(*args)', method_object=eval_accuracy, object_signature=None, function_args=[labels, actual_pred], function_kwargs={})
    (batch_data, labels) = next(iterator)
    start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT()
    strategy.run(replica_fn, args=(batch_data, labels))
    after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='obj.run(*args, **kwargs)', method_object=strategy, object_signature=None, function_args=[replica_fn], function_kwargs={'args': (batch_data, labels)})

def eval_dataset_fn():
    return tf.data.Dataset.from_tensor_slices(feature_and_label_gen(num_examples=16)).map(lambda x: ({'features': feature_preprocess_stage(x['features'])}, label_preprocess_stage(x['label']))).shuffle(16).repeat().batch(8)
start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT()
per_worker_eval_dataset = coordinator.create_per_worker_dataset(eval_dataset_fn)
after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='obj.create_per_worker_dataset(*args)', method_object=coordinator, object_signature=None, function_args=[eval_dataset_fn], function_kwargs={})
per_worker_eval_iterator = iter(per_worker_eval_dataset)
eval_steps_per_epoch = 2
for _ in range(eval_steps_per_epoch):
    start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT()
    coordinator.schedule(eval_step, args=(per_worker_eval_iterator,))
    after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='obj.schedule(*args, **kwargs)', method_object=coordinator, object_signature=None, function_args=[eval_step], function_kwargs={'args': (per_worker_eval_iterator,)})
start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT()
coordinator.join()
after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='obj.join()', method_object=coordinator, object_signature=None, function_args=[], function_kwargs={})
print('Evaluation accuracy: %f' % eval_accuracy.result())
