import multiprocessing
import os
import random
import portpicker
import tensorflow as tf
import os
from pathlib import Path
import dill as pickle
import sys
from tool.client.client_config import EXPERIMENT_DIR, MAX_WAIT_S, WAIT_AFTER_RUN_S
from tool.server.send_request import send_request
from tool.server.function_details import FunctionDetails
current_path = os.path.abspath(__file__)
(immediate_folder, file_name) = os.path.split(current_path)
immediate_folder = os.path.basename(immediate_folder)
experiment_number = int(sys.argv[0])
experiment_file_name = os.path.splitext(file_name)[0]
EXPERIMENT_FILE_PATH = EXPERIMENT_DIR / 'method-level' / immediate_folder / experiment_file_name / f'experiment-{experiment_number}.json'

def custom_method(func, imports: str, function_to_run: str, method_object=None, function_args: list=None, function_kwargs: dict=None, max_wait_secs=MAX_WAIT_S, custom_class=None, wait_after_run_secs=WAIT_AFTER_RUN_S):
    result = send_request(imports=imports, function_to_run=function_to_run, function_args=function_args, function_kwargs=function_kwargs, max_wait_secs=max_wait_secs, method_object=method_object, custom_class=custom_class, experiment_file_path=EXPERIMENT_FILE_PATH)
    return func
if __name__ == '__main__':
    print(EXPERIMENT_FILE_PATH)

def create_in_process_cluster(num_workers, num_ps):
    """Creates and starts local servers and returns the cluster_resolver."""
    worker_ports = [portpicker.pick_unused_port() for _ in range(num_workers)]
    ps_ports = [portpicker.pick_unused_port() for _ in range(num_ps)]
    cluster_dict = {}
    cluster_dict['worker'] = ['localhost:%s' % port for port in worker_ports]
    if num_ps > 0:
        cluster_dict['ps'] = ['localhost:%s' % port for port in ps_ports]
    cluster_spec = custom_method(
    tf.train.ClusterSpec(cluster_dict), imports='import portpicker;import tensorflow as tf;import random;import os;import multiprocessing', function_to_run='tf.train.ClusterSpec(*args)', method_object=None, function_args=[eval('cluster_dict')], function_kwargs={}, max_wait_secs=0)
    worker_config = custom_method(
    tf.compat.v1.ConfigProto(), imports='import portpicker;import tensorflow as tf;import random;import os;import multiprocessing', function_to_run='tf.compat.v1.ConfigProto()', method_object=None, function_args=[], function_kwargs={}, max_wait_secs=0)
    if multiprocessing.cpu_count() < num_workers + 1:
        worker_config.inter_op_parallelism_threads = num_workers + 1
    for i in range(num_workers):
        custom_method(
        tf.distribute.Server(cluster_spec, job_name='worker', task_index=i, config=worker_config, protocol='grpc'), imports='import portpicker;import tensorflow as tf;import random;import os;import multiprocessing', function_to_run='tf.distribute.Server(*args, **kwargs)', method_object=None, function_args=[eval('cluster_spec')], function_kwargs={'job_name': eval('"worker"'), 'task_index': eval('i'), 'config': eval('worker_config'), 'protocol': eval('"grpc"')}, max_wait_secs=0)
    for i in range(num_ps):
        custom_method(
        tf.distribute.Server(cluster_spec, job_name='ps', task_index=i, protocol='grpc'), imports='import portpicker;import tensorflow as tf;import random;import os;import multiprocessing', function_to_run='tf.distribute.Server(*args, **kwargs)', method_object=None, function_args=[eval('cluster_spec')], function_kwargs={'job_name': eval('"ps"'), 'task_index': eval('i'), 'protocol': eval('"grpc"')}, max_wait_secs=0)
    cluster_resolver = custom_method(
    tf.distribute.cluster_resolver.SimpleClusterResolver(cluster_spec, rpc_layer='grpc'), imports='import portpicker;import tensorflow as tf;import random;import os;import multiprocessing', function_to_run='tf.distribute.cluster_resolver.SimpleClusterResolver(*args, **kwargs)', method_object=None, function_args=[eval('cluster_spec')], function_kwargs={'rpc_layer': eval('"grpc"')}, max_wait_secs=0)
    return cluster_resolver
os.environ['GRPC_FAIL_FAST'] = 'use_caller'
NUM_WORKERS = 3
NUM_PS = 2
cluster_resolver = create_in_process_cluster(NUM_WORKERS, NUM_PS)
variable_partitioner = custom_method(
tf.distribute.experimental.partitioners.MinSizePartitioner(min_shard_bytes=256 << 10, max_shards=NUM_PS), imports='import portpicker;import tensorflow as tf;import random;import os;import multiprocessing', function_to_run='tf.distribute.experimental.partitioners.MinSizePartitioner(**kwargs)', method_object=None, function_args=[], function_kwargs={'min_shard_bytes': eval('256 << 10'), 'max_shards': eval('NUM_PS')}, max_wait_secs=0)
strategy = custom_method(
tf.distribute.ParameterServerStrategy(cluster_resolver, variable_partitioner=variable_partitioner), imports='import portpicker;import tensorflow as tf;import random;import os;import multiprocessing', function_to_run='tf.distribute.ParameterServerStrategy(*args, **kwargs)', method_object=None, function_args=[eval('cluster_resolver')], function_kwargs={'variable_partitioner': eval('variable_partitioner')}, max_wait_secs=0)
global_batch_size = 64
x = custom_method(
tf.random.uniform((10, 10)), imports='import portpicker;import tensorflow as tf;import random;import os;import multiprocessing', function_to_run='tf.random.uniform(*args)', method_object=None, function_args=[eval('(10, 10)')], function_kwargs={}, max_wait_secs=0)
y = custom_method(
tf.random.uniform((10,)), imports='import portpicker;import tensorflow as tf;import random;import os;import multiprocessing', function_to_run='tf.random.uniform(*args)', method_object=None, function_args=[eval('(10,)')], function_kwargs={}, max_wait_secs=0)
dataset = custom_method(
tf.data.Dataset.from_tensor_slices((x, y)).shuffle(10).repeat(), imports='import portpicker;import tensorflow as tf;import random;import os;import multiprocessing', function_to_run='tf.data.Dataset.from_tensor_slices((x, y)).shuffle(10).repeat()', method_object=None, function_args=[], function_kwargs={}, max_wait_secs=0)
dataset = custom_method(
dataset.batch(global_batch_size), imports='import portpicker;import tensorflow as tf;import random;import os;import multiprocessing', function_to_run='obj.batch(*args)', method_object=eval('dataset'), function_args=[eval('global_batch_size')], function_kwargs={}, max_wait_secs=0, custom_class=None)
dataset = custom_method(
dataset.prefetch(2), imports='import portpicker;import tensorflow as tf;import random;import os;import multiprocessing', function_to_run='obj.prefetch(*args)', method_object=eval('dataset'), function_args=[eval('2')], function_kwargs={}, max_wait_secs=0, custom_class=None)
with custom_method(
strategy.scope(), imports='import portpicker;import tensorflow as tf;import random;import os;import multiprocessing', function_to_run='obj.scope()', method_object=eval('strategy'), function_args=[], function_kwargs={}, max_wait_secs=0, custom_class=None):
    model = custom_method(
    tf.keras.models.Sequential([tf.keras.layers.Dense(10)]), imports='import portpicker;import tensorflow as tf;import random;import os;import multiprocessing', function_to_run='tf.keras.models.Sequential(*args)', method_object=None, function_args=[eval('[tf.keras.layers.Dense(10)]')], function_kwargs={}, max_wait_secs=0)
    custom_method(
    model.compile(tf.keras.optimizers.legacy.SGD(), loss='mse', steps_per_execution=10), imports='import portpicker;import tensorflow as tf;import random;import os;import multiprocessing', function_to_run='obj.compile(*args, **kwargs)', method_object=eval('model'), function_args=[eval('tf.keras.optimizers.legacy.SGD()')], function_kwargs={'loss': eval('"mse"'), 'steps_per_execution': eval('10')}, max_wait_secs=0, custom_class=None)
working_dir = '/tmp/my_working_dir'
log_dir = os.path.join(working_dir, 'log')
ckpt_filepath = os.path.join(working_dir, 'ckpt')
backup_dir = os.path.join(working_dir, 'backup')
callbacks = [custom_method(
tf.keras.callbacks.TensorBoard(log_dir=log_dir), imports='import portpicker;import tensorflow as tf;import random;import os;import multiprocessing', function_to_run='tf.keras.callbacks.TensorBoard(**kwargs)', method_object=None, function_args=[], function_kwargs={'log_dir': eval('log_dir')}, max_wait_secs=0), custom_method(
tf.keras.callbacks.ModelCheckpoint(filepath=ckpt_filepath), imports='import portpicker;import tensorflow as tf;import random;import os;import multiprocessing', function_to_run='tf.keras.callbacks.ModelCheckpoint(**kwargs)', method_object=None, function_args=[], function_kwargs={'filepath': eval('ckpt_filepath')}, max_wait_secs=0), custom_method(
tf.keras.callbacks.BackupAndRestore(backup_dir=backup_dir), imports='import portpicker;import tensorflow as tf;import random;import os;import multiprocessing', function_to_run='tf.keras.callbacks.BackupAndRestore(**kwargs)', method_object=None, function_args=[], function_kwargs={'backup_dir': eval('backup_dir')}, max_wait_secs=0)]
custom_method(
model.fit(dataset, epochs=5, steps_per_epoch=20, callbacks=callbacks), imports='import portpicker;import tensorflow as tf;import random;import os;import multiprocessing', function_to_run='obj.fit(*args, **kwargs)', method_object=eval('model'), function_args=[eval('dataset')], function_kwargs={'epochs': eval('5'), 'steps_per_epoch': eval('20'), 'callbacks': eval('callbacks')}, max_wait_secs=0, custom_class=None)
feature_vocab = ['avenger', 'ironman', 'batman', 'hulk', 'spiderman', 'kingkong', 'wonder_woman']
label_vocab = ['yes', 'no']
with custom_method(
strategy.scope(), imports='import portpicker;import tensorflow as tf;import random;import os;import multiprocessing', function_to_run='obj.scope()', method_object=eval('strategy'), function_args=[], function_kwargs={}, max_wait_secs=0, custom_class=None):
    feature_lookup_layer = custom_method(
    tf.keras.layers.StringLookup(vocabulary=feature_vocab, mask_token=None), imports='import portpicker;import tensorflow as tf;import random;import os;import multiprocessing', function_to_run='tf.keras.layers.StringLookup(**kwargs)', method_object=None, function_args=[], function_kwargs={'vocabulary': eval('feature_vocab'), 'mask_token': eval('None')}, max_wait_secs=0)
    label_lookup_layer = custom_method(
    tf.keras.layers.StringLookup(vocabulary=label_vocab, num_oov_indices=0, mask_token=None), imports='import portpicker;import tensorflow as tf;import random;import os;import multiprocessing', function_to_run='tf.keras.layers.StringLookup(**kwargs)', method_object=None, function_args=[], function_kwargs={'vocabulary': eval('label_vocab'), 'num_oov_indices': eval('0'), 'mask_token': eval('None')}, max_wait_secs=0)
    raw_feature_input = custom_method(
    tf.keras.layers.Input(shape=(3,), dtype=tf.string, name='feature'), imports='import portpicker;import tensorflow as tf;import random;import os;import multiprocessing', function_to_run='tf.keras.layers.Input(**kwargs)', method_object=None, function_args=[], function_kwargs={'shape': eval('(3,)'), 'dtype': eval('tf.string'), 'name': eval('"feature"')}, max_wait_secs=0)
    feature_id_input = custom_method(
    feature_lookup_layer(raw_feature_input), imports='import portpicker;import tensorflow as tf;import random;import os;import multiprocessing', function_to_run='obj(*args)', method_object=eval('feature_lookup_layer'), function_args=[eval('raw_feature_input')], function_kwargs={}, max_wait_secs=0, custom_class=None)
    feature_preprocess_stage = custom_method(
    tf.keras.Model({'features': raw_feature_input}, feature_id_input), imports='import portpicker;import tensorflow as tf;import random;import os;import multiprocessing', function_to_run='tf.keras.Model(*args)', method_object=None, function_args=[eval('{"features": raw_feature_input}'), eval('feature_id_input')], function_kwargs={}, max_wait_secs=0)
    raw_label_input = custom_method(
    tf.keras.layers.Input(shape=(1,), dtype=tf.string, name='label'), imports='import portpicker;import tensorflow as tf;import random;import os;import multiprocessing', function_to_run='tf.keras.layers.Input(**kwargs)', method_object=None, function_args=[], function_kwargs={'shape': eval('(1,)'), 'dtype': eval('tf.string'), 'name': eval('"label"')}, max_wait_secs=0)
    label_id_input = custom_method(
    label_lookup_layer(raw_label_input), imports='import portpicker;import tensorflow as tf;import random;import os;import multiprocessing', function_to_run='obj(*args)', method_object=eval('label_lookup_layer'), function_args=[eval('raw_label_input')], function_kwargs={}, max_wait_secs=0, custom_class=None)
    label_preprocess_stage = custom_method(
    tf.keras.Model({'label': raw_label_input}, label_id_input), imports='import portpicker;import tensorflow as tf;import random;import os;import multiprocessing', function_to_run='tf.keras.Model(*args)', method_object=None, function_args=[eval('{"label": raw_label_input}'), eval('label_id_input')], function_kwargs={}, max_wait_secs=0)

def feature_and_label_gen(num_examples=200):
    examples = {'features': [], 'label': []}
    for _ in range(num_examples):
        features = random.sample(feature_vocab, 3)
        label = ['yes'] if 'avenger' in features else ['no']
        custom_method(
        examples['features'].append(features), imports='import portpicker;import tensorflow as tf;import random;import os;import multiprocessing', function_to_run="obj['features'].append(*args)", method_object=eval('examples'), function_args=[eval('features')], function_kwargs={}, max_wait_secs=0, custom_class=None)
        custom_method(
        examples['label'].append(label), imports='import portpicker;import tensorflow as tf;import random;import os;import multiprocessing', function_to_run="obj['label'].append(*args)", method_object=eval('examples'), function_args=[eval('label')], function_kwargs={}, max_wait_secs=0, custom_class=None)
    return examples
examples = feature_and_label_gen()

def dataset_fn(_):
    raw_dataset = custom_method(
    tf.data.Dataset.from_tensor_slices(examples), imports='import portpicker;import tensorflow as tf;import random;import os;import multiprocessing', function_to_run='tf.data.Dataset.from_tensor_slices(*args)', method_object=None, function_args=[eval('examples')], function_kwargs={}, max_wait_secs=0)
    train_dataset = custom_method(
    raw_dataset.map(lambda x: ({'features': feature_preprocess_stage(x['features'])}, label_preprocess_stage(x['label']))).shuffle(200).batch(32).repeat(), imports='import portpicker;import tensorflow as tf;import random;import os;import multiprocessing', function_to_run="raw_dataset.map(lambda obj: ({'features': feature_preprocess_stage(x['features'])}, label_preprocess_stage(x['label']))).shuffle(200).batch(32).repeat()", method_object=eval('x'), function_args=[], function_kwargs={}, max_wait_secs=0, custom_class=None)
    return train_dataset
with custom_method(
strategy.scope(), imports='import portpicker;import tensorflow as tf;import random;import os;import multiprocessing', function_to_run='obj.scope()', method_object=eval('strategy'), function_args=[], function_kwargs={}, max_wait_secs=0, custom_class=None):
    model_input = custom_method(
    tf.keras.layers.Input(shape=(3,), dtype=tf.int64, name='model_input'), imports='import portpicker;import tensorflow as tf;import random;import os;import multiprocessing', function_to_run='tf.keras.layers.Input(**kwargs)', method_object=None, function_args=[], function_kwargs={'shape': eval('(3,)'), 'dtype': eval('tf.int64'), 'name': eval('"model_input"')}, max_wait_secs=0)
    emb_layer = custom_method(
    tf.keras.layers.Embedding(input_dim=len(feature_lookup_layer.get_vocabulary()), output_dim=16384), imports='import portpicker;import tensorflow as tf;import random;import os;import multiprocessing', function_to_run='tf.keras.layers.Embedding(**kwargs)', method_object=None, function_args=[], function_kwargs={'input_dim': eval('len(feature_lookup_layer.get_vocabulary())'), 'output_dim': eval('16384')}, max_wait_secs=0)
    emb_output = custom_method(
    tf.reduce_mean(emb_layer(model_input), axis=1), imports='import portpicker;import tensorflow as tf;import random;import os;import multiprocessing', function_to_run='tf.reduce_mean(*args, **kwargs)', method_object=None, function_args=[eval('emb_layer(model_input)')], function_kwargs={'axis': eval('1')}, max_wait_secs=0)
    dense_output = custom_method(
    tf.keras.layers.Dense(units=1, activation='sigmoid')(emb_output), imports='import portpicker;import tensorflow as tf;import random;import os;import multiprocessing', function_to_run="tf.keras.layers.Dense(units=1, activation='sigmoid')(*args)", method_object=None, function_args=[eval('emb_output')], function_kwargs={}, max_wait_secs=0)
    model = custom_method(
    tf.keras.Model({'features': model_input}, dense_output), imports='import portpicker;import tensorflow as tf;import random;import os;import multiprocessing', function_to_run='tf.keras.Model(*args)', method_object=None, function_args=[eval('{"features": model_input}'), eval('dense_output')], function_kwargs={}, max_wait_secs=0)
    optimizer = custom_method(
    tf.keras.optimizers.legacy.RMSprop(learning_rate=0.1), imports='import portpicker;import tensorflow as tf;import random;import os;import multiprocessing', function_to_run='tf.keras.optimizers.legacy.RMSprop(**kwargs)', method_object=None, function_args=[], function_kwargs={'learning_rate': eval('0.1')}, max_wait_secs=0)
    accuracy = custom_method(
    tf.keras.metrics.Accuracy(), imports='import portpicker;import tensorflow as tf;import random;import os;import multiprocessing', function_to_run='tf.keras.metrics.Accuracy()', method_object=None, function_args=[], function_kwargs={}, max_wait_secs=0)
assert len(emb_layer.weights) == 2
assert emb_layer.weights[0].shape == (4, 16384)
assert emb_layer.weights[1].shape == (4, 16384)
print(emb_layer.weights[0].device)
print(emb_layer.weights[1].device)

@tf.function
def step_fn(iterator):

    def replica_fn(batch_data, labels):
        with custom_method(
        tf.GradientTape(), imports='import portpicker;import tensorflow as tf;import random;import os;import multiprocessing', function_to_run='tf.GradientTape()', method_object=None, function_args=[], function_kwargs={}, max_wait_secs=0) as tape:
            pred = custom_method(
            model(batch_data, training=True), imports='import portpicker;import tensorflow as tf;import random;import os;import multiprocessing', function_to_run='obj(*args, **kwargs)', method_object=eval('model'), function_args=[eval('batch_data')], function_kwargs={'training': eval('True')}, max_wait_secs=0, custom_class=None)
            per_example_loss = custom_method(
            tf.keras.losses.BinaryCrossentropy(reduction=tf.keras.losses.Reduction.NONE)(labels, pred), imports='import portpicker;import tensorflow as tf;import random;import os;import multiprocessing', function_to_run='tf.keras.losses.BinaryCrossentropy(reduction=tf.keras.losses.Reduction.NONE)(*args)', method_object=None, function_args=[eval('labels'), eval('pred')], function_kwargs={}, max_wait_secs=0)
            loss = custom_method(
            tf.nn.compute_average_loss(per_example_loss), imports='import portpicker;import tensorflow as tf;import random;import os;import multiprocessing', function_to_run='tf.nn.compute_average_loss(*args)', method_object=None, function_args=[eval('per_example_loss')], function_kwargs={}, max_wait_secs=0)
            gradients = tape.gradient(loss, model.trainable_variables)
        custom_method(
        optimizer.apply_gradients(zip(gradients, model.trainable_variables)), imports='import portpicker;import tensorflow as tf;import random;import os;import multiprocessing', function_to_run='obj.apply_gradients(*args)', method_object=eval('optimizer'), function_args=[eval('zip(gradients, model.trainable_variables)')], function_kwargs={}, max_wait_secs=0, custom_class=None)
        actual_pred = custom_method(
        tf.cast(tf.greater(pred, 0.5), tf.int64), imports='import portpicker;import tensorflow as tf;import random;import os;import multiprocessing', function_to_run='tf.cast(*args)', method_object=None, function_args=[eval('tf.greater(pred, 0.5)'), eval('tf.int64')], function_kwargs={}, max_wait_secs=0)
        custom_method(
        accuracy.update_state(labels, actual_pred), imports='import portpicker;import tensorflow as tf;import random;import os;import multiprocessing', function_to_run='obj.update_state(*args)', method_object=eval('accuracy'), function_args=[eval('labels'), eval('actual_pred')], function_kwargs={}, max_wait_secs=0, custom_class=None)
        return loss
    (batch_data, labels) = custom_method(
    next(iterator), imports='import portpicker;import tensorflow as tf;import random;import os;import multiprocessing', function_to_run='obj(*args)', method_object=eval('next'), function_args=[eval('iterator')], function_kwargs={}, max_wait_secs=0, custom_class=None)
    losses = custom_method(
    strategy.run(replica_fn, args=(batch_data, labels)), imports='import portpicker;import tensorflow as tf;import random;import os;import multiprocessing', function_to_run='obj.run(*args, **kwargs)', method_object=eval('strategy'), function_args=[eval('replica_fn')], function_kwargs={'args': eval('(batch_data, labels)')}, max_wait_secs=0, custom_class=None)
    return custom_method(
    strategy.reduce(tf.distribute.ReduceOp.SUM, losses, axis=None), imports='import portpicker;import tensorflow as tf;import random;import os;import multiprocessing', function_to_run='obj.reduce(*args, **kwargs)', method_object=eval('strategy'), function_args=[eval('tf.distribute.ReduceOp.SUM'), eval('losses')], function_kwargs={'axis': eval('None')}, max_wait_secs=0, custom_class=None)
coordinator = custom_method(
tf.distribute.coordinator.ClusterCoordinator(strategy), imports='import portpicker;import tensorflow as tf;import random;import os;import multiprocessing', function_to_run='tf.distribute.coordinator.ClusterCoordinator(*args)', method_object=None, function_args=[eval('strategy')], function_kwargs={}, max_wait_secs=0)

@tf.function
def per_worker_dataset_fn():
    return custom_method(
    strategy.distribute_datasets_from_function(dataset_fn), imports='import portpicker;import tensorflow as tf;import random;import os;import multiprocessing', function_to_run='obj.distribute_datasets_from_function(*args)', method_object=eval('strategy'), function_args=[eval('dataset_fn')], function_kwargs={}, max_wait_secs=0, custom_class=None)
per_worker_dataset = custom_method(
coordinator.create_per_worker_dataset(per_worker_dataset_fn), imports='import portpicker;import tensorflow as tf;import random;import os;import multiprocessing', function_to_run='obj.create_per_worker_dataset(*args)', method_object=eval('coordinator'), function_args=[eval('per_worker_dataset_fn')], function_kwargs={}, max_wait_secs=0, custom_class=None)
per_worker_iterator = iter(per_worker_dataset)
num_epochs = 4
steps_per_epoch = 5
for i in range(num_epochs):
    custom_method(
    accuracy.reset_states(), imports='import portpicker;import tensorflow as tf;import random;import os;import multiprocessing', function_to_run='obj.reset_states()', method_object=eval('accuracy'), function_args=[], function_kwargs={}, max_wait_secs=0, custom_class=None)
    for _ in range(steps_per_epoch):
        custom_method(
        coordinator.schedule(step_fn, args=(per_worker_iterator,)), imports='import portpicker;import tensorflow as tf;import random;import os;import multiprocessing', function_to_run='obj.schedule(*args, **kwargs)', method_object=eval('coordinator'), function_args=[eval('step_fn')], function_kwargs={'args': eval('(per_worker_iterator,)')}, max_wait_secs=0, custom_class=None)
    custom_method(
    coordinator.join(), imports='import portpicker;import tensorflow as tf;import random;import os;import multiprocessing', function_to_run='obj.join()', method_object=eval('coordinator'), function_args=[], function_kwargs={}, max_wait_secs=0, custom_class=None)
    print('Finished epoch %d, accuracy is %f.' % (i, accuracy.result().numpy()))
loss = custom_method(
coordinator.schedule(step_fn, args=(per_worker_iterator,)), imports='import portpicker;import tensorflow as tf;import random;import os;import multiprocessing', function_to_run='obj.schedule(*args, **kwargs)', method_object=eval('coordinator'), function_args=[eval('step_fn')], function_kwargs={'args': eval('(per_worker_iterator,)')}, max_wait_secs=0, custom_class=None)
print('Final loss is %f' % loss.fetch())
eval_dataset = custom_method(
tf.data.Dataset.from_tensor_slices(feature_and_label_gen(num_examples=16)).map(lambda x: ({'features': feature_preprocess_stage(x['features'])}, label_preprocess_stage(x['label']))).batch(8), imports='import portpicker;import tensorflow as tf;import random;import os;import multiprocessing', function_to_run="tf.data.Dataset.from_tensor_slices(feature_and_label_gen(num_examples=16)).map(lambda x: ({'features': feature_preprocess_stage(x['features'])}, label_preprocess_stage(x['label']))).batch(*args)", method_object=None, function_args=[eval('8')], function_kwargs={}, max_wait_secs=0)
eval_accuracy = custom_method(
tf.keras.metrics.Accuracy(), imports='import portpicker;import tensorflow as tf;import random;import os;import multiprocessing', function_to_run='tf.keras.metrics.Accuracy()', method_object=None, function_args=[], function_kwargs={}, max_wait_secs=0)
for (batch_data, labels) in eval_dataset:
    pred = custom_method(
    model(batch_data, training=False), imports='import portpicker;import tensorflow as tf;import random;import os;import multiprocessing', function_to_run='obj(*args, **kwargs)', method_object=eval('model'), function_args=[eval('batch_data')], function_kwargs={'training': eval('False')}, max_wait_secs=0, custom_class=None)
    actual_pred = custom_method(
    tf.cast(tf.greater(pred, 0.5), tf.int64), imports='import portpicker;import tensorflow as tf;import random;import os;import multiprocessing', function_to_run='tf.cast(*args)', method_object=None, function_args=[eval('tf.greater(pred, 0.5)'), eval('tf.int64')], function_kwargs={}, max_wait_secs=0)
    custom_method(
    eval_accuracy.update_state(labels, actual_pred), imports='import portpicker;import tensorflow as tf;import random;import os;import multiprocessing', function_to_run='obj.update_state(*args)', method_object=eval('eval_accuracy'), function_args=[eval('labels'), eval('actual_pred')], function_kwargs={}, max_wait_secs=0, custom_class=None)
print('Evaluation accuracy: %f' % eval_accuracy.result())
with custom_method(
strategy.scope(), imports='import portpicker;import tensorflow as tf;import random;import os;import multiprocessing', function_to_run='obj.scope()', method_object=eval('strategy'), function_args=[], function_kwargs={}, max_wait_secs=0, custom_class=None):
    eval_accuracy = custom_method(
    tf.keras.metrics.Accuracy(), imports='import portpicker;import tensorflow as tf;import random;import os;import multiprocessing', function_to_run='tf.keras.metrics.Accuracy()', method_object=None, function_args=[], function_kwargs={}, max_wait_secs=0)

@tf.function
def eval_step(iterator):

    def replica_fn(batch_data, labels):
        pred = custom_method(
        model(batch_data, training=False), imports='import portpicker;import tensorflow as tf;import random;import os;import multiprocessing', function_to_run='obj(*args, **kwargs)', method_object=eval('model'), function_args=[eval('batch_data')], function_kwargs={'training': eval('False')}, max_wait_secs=0, custom_class=None)
        actual_pred = custom_method(
        tf.cast(tf.greater(pred, 0.5), tf.int64), imports='import portpicker;import tensorflow as tf;import random;import os;import multiprocessing', function_to_run='tf.cast(*args)', method_object=None, function_args=[eval('tf.greater(pred, 0.5)'), eval('tf.int64')], function_kwargs={}, max_wait_secs=0)
        custom_method(
        eval_accuracy.update_state(labels, actual_pred), imports='import portpicker;import tensorflow as tf;import random;import os;import multiprocessing', function_to_run='obj.update_state(*args)', method_object=eval('eval_accuracy'), function_args=[eval('labels'), eval('actual_pred')], function_kwargs={}, max_wait_secs=0, custom_class=None)
    (batch_data, labels) = custom_method(
    next(iterator), imports='import portpicker;import tensorflow as tf;import random;import os;import multiprocessing', function_to_run='obj(*args)', method_object=eval('next'), function_args=[eval('iterator')], function_kwargs={}, max_wait_secs=0, custom_class=None)
    custom_method(
    strategy.run(replica_fn, args=(batch_data, labels)), imports='import portpicker;import tensorflow as tf;import random;import os;import multiprocessing', function_to_run='obj.run(*args, **kwargs)', method_object=eval('strategy'), function_args=[eval('replica_fn')], function_kwargs={'args': eval('(batch_data, labels)')}, max_wait_secs=0, custom_class=None)

def eval_dataset_fn():
    return custom_method(
    tf.data.Dataset.from_tensor_slices(feature_and_label_gen(num_examples=16)).map(lambda x: ({'features': feature_preprocess_stage(x['features'])}, label_preprocess_stage(x['label']))).shuffle(16).repeat().batch(8), imports='import portpicker;import tensorflow as tf;import random;import os;import multiprocessing', function_to_run="tf.data.Dataset.from_tensor_slices(feature_and_label_gen(num_examples=16)).map(lambda x: ({'features': feature_preprocess_stage(x['features'])}, label_preprocess_stage(x['label']))).shuffle(16).repeat().batch(*args)", method_object=None, function_args=[eval('8')], function_kwargs={}, max_wait_secs=0)
per_worker_eval_dataset = custom_method(
coordinator.create_per_worker_dataset(eval_dataset_fn), imports='import portpicker;import tensorflow as tf;import random;import os;import multiprocessing', function_to_run='obj.create_per_worker_dataset(*args)', method_object=eval('coordinator'), function_args=[eval('eval_dataset_fn')], function_kwargs={}, max_wait_secs=0, custom_class=None)
per_worker_eval_iterator = iter(per_worker_eval_dataset)
eval_steps_per_epoch = 2
for _ in range(eval_steps_per_epoch):
    custom_method(
    coordinator.schedule(eval_step, args=(per_worker_eval_iterator,)), imports='import portpicker;import tensorflow as tf;import random;import os;import multiprocessing', function_to_run='obj.schedule(*args, **kwargs)', method_object=eval('coordinator'), function_args=[eval('eval_step')], function_kwargs={'args': eval('(per_worker_eval_iterator,)')}, max_wait_secs=0, custom_class=None)
custom_method(
coordinator.join(), imports='import portpicker;import tensorflow as tf;import random;import os;import multiprocessing', function_to_run='obj.join()', method_object=eval('coordinator'), function_args=[], function_kwargs={}, max_wait_secs=0, custom_class=None)
print('Evaluation accuracy: %f' % eval_accuracy.result())
