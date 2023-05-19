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
experiment_number = sys.argv[1]
experiment_project = sys.argv[2]
EXPERIMENT_FILE_PATH = EXPERIMENT_DIR / 'method-level' / experiment_project / f'experiment-{experiment_number}.json'

def custom_method(imports: str, function_to_run: str, method_object=None, object_signature=None, function_args: list=None, function_kwargs: dict=None, custom_class=None):
    result = send_request(imports=imports, function_to_run=function_to_run, function_args=function_args, function_kwargs=function_kwargs, max_wait_secs=MAX_WAIT_S, wait_after_run_secs=WAIT_AFTER_RUN_S, method_object=method_object, object_signature=object_signature, custom_class=custom_class, experiment_file_path=EXPERIMENT_FILE_PATH)

def create_in_process_cluster(num_workers, num_ps):
    """Creates and starts local servers and returns the cluster_resolver."""
    worker_ports = [portpicker.pick_unused_port() for _ in range(num_workers)]
    ps_ports = [portpicker.pick_unused_port() for _ in range(num_ps)]
    cluster_dict = {}
    cluster_dict['worker'] = ['localhost:%s' % port for port in worker_ports]
    if num_ps > 0:
        cluster_dict['ps'] = ['localhost:%s' % port for port in ps_ports]
    custom_method(imports='import os;import random;import portpicker;import multiprocessing;import tensorflow as tf', function_to_run='tf.train.ClusterSpec(*args)', method_object=None, object_signature=None, function_args=[eval('cluster_dict')], function_kwargs={})
    cluster_spec = tf.train.ClusterSpec(cluster_dict)
    custom_method(imports='import os;import random;import portpicker;import multiprocessing;import tensorflow as tf', function_to_run='tf.compat.v1.ConfigProto()', method_object=None, object_signature=None, function_args=[], function_kwargs={})
    worker_config = tf.compat.v1.ConfigProto()
    if multiprocessing.cpu_count() < num_workers + 1:
        worker_config.inter_op_parallelism_threads = num_workers + 1
    for i in range(num_workers):
        custom_method(imports='import os;import random;import portpicker;import multiprocessing;import tensorflow as tf', function_to_run='tf.distribute.Server(*args, **kwargs)', method_object=None, object_signature=None, function_args=[eval('cluster_spec')], function_kwargs={'job_name': eval('"worker"'), 'task_index': eval('i'), 'config': eval('worker_config'), 'protocol': eval('"grpc"')})
        tf.distribute.Server(cluster_spec, job_name='worker', task_index=i, config=worker_config, protocol='grpc')
    for i in range(num_ps):
        custom_method(imports='import os;import random;import portpicker;import multiprocessing;import tensorflow as tf', function_to_run='tf.distribute.Server(*args, **kwargs)', method_object=None, object_signature=None, function_args=[eval('cluster_spec')], function_kwargs={'job_name': eval('"ps"'), 'task_index': eval('i'), 'protocol': eval('"grpc"')})
        tf.distribute.Server(cluster_spec, job_name='ps', task_index=i, protocol='grpc')
    custom_method(imports='import os;import random;import portpicker;import multiprocessing;import tensorflow as tf', function_to_run='tf.distribute.cluster_resolver.SimpleClusterResolver(*args, **kwargs)', method_object=None, object_signature=None, function_args=[eval('cluster_spec')], function_kwargs={'rpc_layer': eval('"grpc"')})
    cluster_resolver = tf.distribute.cluster_resolver.SimpleClusterResolver(cluster_spec, rpc_layer='grpc')
    return cluster_resolver
os.environ['GRPC_FAIL_FAST'] = 'use_caller'
NUM_WORKERS = 3
NUM_PS = 2
cluster_resolver = create_in_process_cluster(NUM_WORKERS, NUM_PS)
custom_method(imports='import os;import random;import portpicker;import multiprocessing;import tensorflow as tf', function_to_run='tf.distribute.experimental.partitioners.MinSizePartitioner(**kwargs)', method_object=None, object_signature=None, function_args=[], function_kwargs={'min_shard_bytes': eval('256 << 10'), 'max_shards': eval('NUM_PS')})
variable_partitioner = tf.distribute.experimental.partitioners.MinSizePartitioner(min_shard_bytes=256 << 10, max_shards=NUM_PS)
custom_method(imports='import os;import random;import portpicker;import multiprocessing;import tensorflow as tf', function_to_run='tf.distribute.ParameterServerStrategy(*args, **kwargs)', method_object=None, object_signature=None, function_args=[eval('cluster_resolver')], function_kwargs={'variable_partitioner': eval('variable_partitioner')})
strategy = tf.distribute.ParameterServerStrategy(cluster_resolver, variable_partitioner=variable_partitioner)
global_batch_size = 64
custom_method(imports='import os;import random;import portpicker;import multiprocessing;import tensorflow as tf', function_to_run='tf.random.uniform(*args)', method_object=None, object_signature=None, function_args=[eval('(10, 10)')], function_kwargs={})
x = tf.random.uniform((10, 10))
custom_method(imports='import os;import random;import portpicker;import multiprocessing;import tensorflow as tf', function_to_run='tf.random.uniform(*args)', method_object=None, object_signature=None, function_args=[eval('(10,)')], function_kwargs={})
y = tf.random.uniform((10,))
custom_method(imports='import os;import random;import portpicker;import multiprocessing;import tensorflow as tf', function_to_run='tf.data.Dataset.from_tensor_slices((x, y)).shuffle(10).repeat()', method_object=None, object_signature=None, function_args=[], function_kwargs={})
dataset = tf.data.Dataset.from_tensor_slices((x, y)).shuffle(10).repeat()
custom_method(imports='import os;import random;import portpicker;import multiprocessing;import tensorflow as tf', function_to_run='obj.batch(*args)', method_object=eval('dataset'), object_signature=None, function_args=[eval('global_batch_size')], function_kwargs={}, custom_class=None)
dataset = dataset.batch(global_batch_size)
custom_method(imports='import os;import random;import portpicker;import multiprocessing;import tensorflow as tf', function_to_run='obj.prefetch(*args)', method_object=eval('dataset'), object_signature=None, function_args=[eval('2')], function_kwargs={}, custom_class=None)
dataset = dataset.prefetch(2)
with strategy.scope():
    custom_method(imports='import os;import random;import portpicker;import multiprocessing;import tensorflow as tf', function_to_run='tf.keras.models.Sequential(*args)', method_object=None, object_signature=None, function_args=[eval('[tf.keras.layers.Dense(10)]')], function_kwargs={})
    model = tf.keras.models.Sequential([tf.keras.layers.Dense(10)])
    custom_method(imports='import os;import random;import portpicker;import multiprocessing;import tensorflow as tf', function_to_run='obj.compile(*args, **kwargs)', method_object=eval('model'), object_signature=None, function_args=[eval('tf.keras.optimizers.legacy.SGD()')], function_kwargs={'loss': eval('"mse"'), 'steps_per_execution': eval('10')}, custom_class=None)
    model.compile(tf.keras.optimizers.legacy.SGD(), loss='mse', steps_per_execution=10)
working_dir = '/tmp/my_working_dir'
log_dir = os.path.join(working_dir, 'log')
ckpt_filepath = os.path.join(working_dir, 'ckpt')
backup_dir = os.path.join(working_dir, 'backup')
callbacks = [tf.keras.callbacks.TensorBoard(log_dir=log_dir), tf.keras.callbacks.ModelCheckpoint(filepath=ckpt_filepath), tf.keras.callbacks.BackupAndRestore(backup_dir=backup_dir)]
custom_method(imports='import os;import random;import portpicker;import multiprocessing;import tensorflow as tf', function_to_run='obj.fit(*args, **kwargs)', method_object=eval('model'), object_signature=None, function_args=[eval('dataset')], function_kwargs={'epochs': eval('5'), 'steps_per_epoch': eval('20'), 'callbacks': eval('callbacks')}, custom_class=None)
model.fit(dataset, epochs=5, steps_per_epoch=20, callbacks=callbacks)
feature_vocab = ['avenger', 'ironman', 'batman', 'hulk', 'spiderman', 'kingkong', 'wonder_woman']
label_vocab = ['yes', 'no']
with strategy.scope():
    custom_method(imports='import os;import random;import portpicker;import multiprocessing;import tensorflow as tf', function_to_run='tf.keras.layers.StringLookup(**kwargs)', method_object=None, object_signature=None, function_args=[], function_kwargs={'vocabulary': eval('feature_vocab'), 'mask_token': eval('None')})
    feature_lookup_layer = tf.keras.layers.StringLookup(vocabulary=feature_vocab, mask_token=None)
    custom_method(imports='import os;import random;import portpicker;import multiprocessing;import tensorflow as tf', function_to_run='tf.keras.layers.StringLookup(**kwargs)', method_object=None, object_signature=None, function_args=[], function_kwargs={'vocabulary': eval('label_vocab'), 'num_oov_indices': eval('0'), 'mask_token': eval('None')})
    label_lookup_layer = tf.keras.layers.StringLookup(vocabulary=label_vocab, num_oov_indices=0, mask_token=None)
    custom_method(imports='import os;import random;import portpicker;import multiprocessing;import tensorflow as tf', function_to_run='tf.keras.layers.Input(**kwargs)', method_object=None, object_signature=None, function_args=[], function_kwargs={'shape': eval('(3,)'), 'dtype': eval('tf.string'), 'name': eval('"feature"')})
    raw_feature_input = tf.keras.layers.Input(shape=(3,), dtype=tf.string, name='feature')
    custom_method(imports='import os;import random;import portpicker;import multiprocessing;import tensorflow as tf', function_to_run='obj(*args)', method_object=eval('feature_lookup_layer'), object_signature=None, function_args=[eval('raw_feature_input')], function_kwargs={}, custom_class=None)
    feature_id_input = feature_lookup_layer(raw_feature_input)
    custom_method(imports='import os;import random;import portpicker;import multiprocessing;import tensorflow as tf', function_to_run='tf.keras.Model(*args)', method_object=None, object_signature=None, function_args=[eval('{"features": raw_feature_input}'), eval('feature_id_input')], function_kwargs={})
    feature_preprocess_stage = tf.keras.Model({'features': raw_feature_input}, feature_id_input)
    custom_method(imports='import os;import random;import portpicker;import multiprocessing;import tensorflow as tf', function_to_run='tf.keras.layers.Input(**kwargs)', method_object=None, object_signature=None, function_args=[], function_kwargs={'shape': eval('(1,)'), 'dtype': eval('tf.string'), 'name': eval('"label"')})
    raw_label_input = tf.keras.layers.Input(shape=(1,), dtype=tf.string, name='label')
    custom_method(imports='import os;import random;import portpicker;import multiprocessing;import tensorflow as tf', function_to_run='obj(*args)', method_object=eval('label_lookup_layer'), object_signature=None, function_args=[eval('raw_label_input')], function_kwargs={}, custom_class=None)
    label_id_input = label_lookup_layer(raw_label_input)
    custom_method(imports='import os;import random;import portpicker;import multiprocessing;import tensorflow as tf', function_to_run='tf.keras.Model(*args)', method_object=None, object_signature=None, function_args=[eval('{"label": raw_label_input}'), eval('label_id_input')], function_kwargs={})
    label_preprocess_stage = tf.keras.Model({'label': raw_label_input}, label_id_input)

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
    custom_method(imports='import os;import random;import portpicker;import multiprocessing;import tensorflow as tf', function_to_run='tf.data.Dataset.from_tensor_slices(*args)', method_object=None, object_signature=None, function_args=[eval('examples')], function_kwargs={})
    raw_dataset = tf.data.Dataset.from_tensor_slices(examples)
    custom_method(imports='import os;import random;import portpicker;import multiprocessing;import tensorflow as tf', function_to_run="obj.map(lambda x: ({'features': feature_preprocess_stage(x['features'])}, label_preprocess_stage(x['label']))).shuffle(200).batch(32).repeat()", method_object=eval('raw_dataset'), object_signature=None, function_args=[], function_kwargs={}, custom_class=None)
    train_dataset = raw_dataset.map(lambda x: ({'features': feature_preprocess_stage(x['features'])}, label_preprocess_stage(x['label']))).shuffle(200).batch(32).repeat()
    return train_dataset
with strategy.scope():
    custom_method(imports='import os;import random;import portpicker;import multiprocessing;import tensorflow as tf', function_to_run='tf.keras.layers.Input(**kwargs)', method_object=None, object_signature=None, function_args=[], function_kwargs={'shape': eval('(3,)'), 'dtype': eval('tf.int64'), 'name': eval('"model_input"')})
    model_input = tf.keras.layers.Input(shape=(3,), dtype=tf.int64, name='model_input')
    custom_method(imports='import os;import random;import portpicker;import multiprocessing;import tensorflow as tf', function_to_run='tf.keras.layers.Embedding(**kwargs)', method_object=None, object_signature=None, function_args=[], function_kwargs={'input_dim': eval('len(feature_lookup_layer.get_vocabulary())'), 'output_dim': eval('16384')})
    emb_layer = tf.keras.layers.Embedding(input_dim=len(feature_lookup_layer.get_vocabulary()), output_dim=16384)
    custom_method(imports='import os;import random;import portpicker;import multiprocessing;import tensorflow as tf', function_to_run='tf.reduce_mean(*args, **kwargs)', method_object=None, object_signature=None, function_args=[eval('emb_layer(model_input)')], function_kwargs={'axis': eval('1')})
    emb_output = tf.reduce_mean(emb_layer(model_input), axis=1)
    custom_method(imports='import os;import random;import portpicker;import multiprocessing;import tensorflow as tf', function_to_run="tf.keras.layers.Dense(units=1, activation='sigmoid')(*args)", method_object=None, object_signature=None, function_args=[eval('emb_output')], function_kwargs={})
    dense_output = tf.keras.layers.Dense(units=1, activation='sigmoid')(emb_output)
    custom_method(imports='import os;import random;import portpicker;import multiprocessing;import tensorflow as tf', function_to_run='tf.keras.Model(*args)', method_object=None, object_signature=None, function_args=[eval('{"features": model_input}'), eval('dense_output')], function_kwargs={})
    model = tf.keras.Model({'features': model_input}, dense_output)
    custom_method(imports='import os;import random;import portpicker;import multiprocessing;import tensorflow as tf', function_to_run='tf.keras.optimizers.legacy.RMSprop(**kwargs)', method_object=None, object_signature=None, function_args=[], function_kwargs={'learning_rate': eval('0.1')})
    optimizer = tf.keras.optimizers.legacy.RMSprop(learning_rate=0.1)
    custom_method(imports='import os;import random;import portpicker;import multiprocessing;import tensorflow as tf', function_to_run='tf.keras.metrics.Accuracy()', method_object=None, object_signature=None, function_args=[], function_kwargs={})
    accuracy = tf.keras.metrics.Accuracy()
assert len(emb_layer.weights) == 2
assert emb_layer.weights[0].shape == (4, 16384)
assert emb_layer.weights[1].shape == (4, 16384)
print(emb_layer.weights[0].device)
print(emb_layer.weights[1].device)

@tf.function
def step_fn(iterator):

    def replica_fn(batch_data, labels):
        with tf.GradientTape() as tape:
            custom_method(imports='import os;import random;import portpicker;import multiprocessing;import tensorflow as tf', function_to_run='obj(*args, **kwargs)', method_object=eval('model'), object_signature=None, function_args=[eval('batch_data')], function_kwargs={'training': eval('True')}, custom_class=None)
            pred = model(batch_data, training=True)
            custom_method(imports='import os;import random;import portpicker;import multiprocessing;import tensorflow as tf', function_to_run='tf.keras.losses.BinaryCrossentropy(reduction=tf.keras.losses.Reduction.NONE)(*args)', method_object=None, object_signature=None, function_args=[eval('labels'), eval('pred')], function_kwargs={})
            per_example_loss = tf.keras.losses.BinaryCrossentropy(reduction=tf.keras.losses.Reduction.NONE)(labels, pred)
            custom_method(imports='import os;import random;import portpicker;import multiprocessing;import tensorflow as tf', function_to_run='tf.nn.compute_average_loss(*args)', method_object=None, object_signature=None, function_args=[eval('per_example_loss')], function_kwargs={})
            loss = tf.nn.compute_average_loss(per_example_loss)
            gradients = tape.gradient(loss, model.trainable_variables)
        custom_method(imports='import os;import random;import portpicker;import multiprocessing;import tensorflow as tf', function_to_run='obj.apply_gradients(*args)', method_object=eval('optimizer'), object_signature=None, function_args=[eval('zip(gradients, model.trainable_variables)')], function_kwargs={}, custom_class=None)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        custom_method(imports='import os;import random;import portpicker;import multiprocessing;import tensorflow as tf', function_to_run='tf.cast(*args)', method_object=None, object_signature=None, function_args=[eval('tf.greater(pred, 0.5)'), eval('tf.int64')], function_kwargs={})
        actual_pred = tf.cast(tf.greater(pred, 0.5), tf.int64)
        custom_method(imports='import os;import random;import portpicker;import multiprocessing;import tensorflow as tf', function_to_run='obj.update_state(*args)', method_object=eval('accuracy'), object_signature=None, function_args=[eval('labels'), eval('actual_pred')], function_kwargs={}, custom_class=None)
        accuracy.update_state(labels, actual_pred)
        return loss
    (batch_data, labels) = next(iterator)
    custom_method(imports='import os;import random;import portpicker;import multiprocessing;import tensorflow as tf', function_to_run='obj.run(*args, **kwargs)', method_object=eval('strategy'), object_signature=None, function_args=[eval('replica_fn')], function_kwargs={'args': eval('(batch_data, labels)')}, custom_class=None)
    losses = strategy.run(replica_fn, args=(batch_data, labels))
    return strategy.reduce(tf.distribute.ReduceOp.SUM, losses, axis=None)
custom_method(imports='import os;import random;import portpicker;import multiprocessing;import tensorflow as tf', function_to_run='tf.distribute.coordinator.ClusterCoordinator(*args)', method_object=None, object_signature=None, function_args=[eval('strategy')], function_kwargs={})
coordinator = tf.distribute.coordinator.ClusterCoordinator(strategy)

@tf.function
def per_worker_dataset_fn():
    return strategy.distribute_datasets_from_function(dataset_fn)
custom_method(imports='import os;import random;import portpicker;import multiprocessing;import tensorflow as tf', function_to_run='obj.create_per_worker_dataset(*args)', method_object=eval('coordinator'), object_signature=None, function_args=[eval('per_worker_dataset_fn')], function_kwargs={}, custom_class=None)
per_worker_dataset = coordinator.create_per_worker_dataset(per_worker_dataset_fn)
per_worker_iterator = iter(per_worker_dataset)
num_epochs = 4
steps_per_epoch = 5
for i in range(num_epochs):
    custom_method(imports='import os;import random;import portpicker;import multiprocessing;import tensorflow as tf', function_to_run='obj.reset_states()', method_object=eval('accuracy'), object_signature=None, function_args=[], function_kwargs={}, custom_class=None)
    accuracy.reset_states()
    for _ in range(steps_per_epoch):
        custom_method(imports='import os;import random;import portpicker;import multiprocessing;import tensorflow as tf', function_to_run='obj.schedule(*args, **kwargs)', method_object=eval('coordinator'), object_signature=None, function_args=[eval('step_fn')], function_kwargs={'args': eval('(per_worker_iterator,)')}, custom_class=None)
        coordinator.schedule(step_fn, args=(per_worker_iterator,))
    custom_method(imports='import os;import random;import portpicker;import multiprocessing;import tensorflow as tf', function_to_run='obj.join()', method_object=eval('coordinator'), object_signature=None, function_args=[], function_kwargs={}, custom_class=None)
    coordinator.join()
    print('Finished epoch %d, accuracy is %f.' % (i, accuracy.result().numpy()))
custom_method(imports='import os;import random;import portpicker;import multiprocessing;import tensorflow as tf', function_to_run='obj.schedule(*args, **kwargs)', method_object=eval('coordinator'), object_signature=None, function_args=[eval('step_fn')], function_kwargs={'args': eval('(per_worker_iterator,)')}, custom_class=None)
loss = coordinator.schedule(step_fn, args=(per_worker_iterator,))
print('Final loss is %f' % loss.fetch())
custom_method(imports='import os;import random;import portpicker;import multiprocessing;import tensorflow as tf', function_to_run="tf.data.Dataset.from_tensor_slices(feature_and_label_gen(num_examples=16)).map(lambda x: ({'features': feature_preprocess_stage(x['features'])}, label_preprocess_stage(x['label']))).batch(*args)", method_object=None, object_signature=None, function_args=[eval('8')], function_kwargs={})
eval_dataset = tf.data.Dataset.from_tensor_slices(feature_and_label_gen(num_examples=16)).map(lambda x: ({'features': feature_preprocess_stage(x['features'])}, label_preprocess_stage(x['label']))).batch(8)
custom_method(imports='import os;import random;import portpicker;import multiprocessing;import tensorflow as tf', function_to_run='tf.keras.metrics.Accuracy()', method_object=None, object_signature=None, function_args=[], function_kwargs={})
eval_accuracy = tf.keras.metrics.Accuracy()
for (batch_data, labels) in eval_dataset:
    custom_method(imports='import os;import random;import portpicker;import multiprocessing;import tensorflow as tf', function_to_run='obj(*args, **kwargs)', method_object=eval('model'), object_signature=None, function_args=[eval('batch_data')], function_kwargs={'training': eval('False')}, custom_class=None)
    pred = model(batch_data, training=False)
    custom_method(imports='import os;import random;import portpicker;import multiprocessing;import tensorflow as tf', function_to_run='tf.cast(*args)', method_object=None, object_signature=None, function_args=[eval('tf.greater(pred, 0.5)'), eval('tf.int64')], function_kwargs={})
    actual_pred = tf.cast(tf.greater(pred, 0.5), tf.int64)
    custom_method(imports='import os;import random;import portpicker;import multiprocessing;import tensorflow as tf', function_to_run='obj.update_state(*args)', method_object=eval('eval_accuracy'), object_signature=None, function_args=[eval('labels'), eval('actual_pred')], function_kwargs={}, custom_class=None)
    eval_accuracy.update_state(labels, actual_pred)
print('Evaluation accuracy: %f' % eval_accuracy.result())
with strategy.scope():
    custom_method(imports='import os;import random;import portpicker;import multiprocessing;import tensorflow as tf', function_to_run='tf.keras.metrics.Accuracy()', method_object=None, object_signature=None, function_args=[], function_kwargs={})
    eval_accuracy = tf.keras.metrics.Accuracy()

@tf.function
def eval_step(iterator):

    def replica_fn(batch_data, labels):
        custom_method(imports='import os;import random;import portpicker;import multiprocessing;import tensorflow as tf', function_to_run='obj(*args, **kwargs)', method_object=eval('model'), object_signature=None, function_args=[eval('batch_data')], function_kwargs={'training': eval('False')}, custom_class=None)
        pred = model(batch_data, training=False)
        custom_method(imports='import os;import random;import portpicker;import multiprocessing;import tensorflow as tf', function_to_run='tf.cast(*args)', method_object=None, object_signature=None, function_args=[eval('tf.greater(pred, 0.5)'), eval('tf.int64')], function_kwargs={})
        actual_pred = tf.cast(tf.greater(pred, 0.5), tf.int64)
        custom_method(imports='import os;import random;import portpicker;import multiprocessing;import tensorflow as tf', function_to_run='obj.update_state(*args)', method_object=eval('eval_accuracy'), object_signature=None, function_args=[eval('labels'), eval('actual_pred')], function_kwargs={}, custom_class=None)
        eval_accuracy.update_state(labels, actual_pred)
    (batch_data, labels) = next(iterator)
    custom_method(imports='import os;import random;import portpicker;import multiprocessing;import tensorflow as tf', function_to_run='obj.run(*args, **kwargs)', method_object=eval('strategy'), object_signature=None, function_args=[eval('replica_fn')], function_kwargs={'args': eval('(batch_data, labels)')}, custom_class=None)
    strategy.run(replica_fn, args=(batch_data, labels))

def eval_dataset_fn():
    return tf.data.Dataset.from_tensor_slices(feature_and_label_gen(num_examples=16)).map(lambda x: ({'features': feature_preprocess_stage(x['features'])}, label_preprocess_stage(x['label']))).shuffle(16).repeat().batch(8)
custom_method(imports='import os;import random;import portpicker;import multiprocessing;import tensorflow as tf', function_to_run='obj.create_per_worker_dataset(*args)', method_object=eval('coordinator'), object_signature=None, function_args=[eval('eval_dataset_fn')], function_kwargs={}, custom_class=None)
per_worker_eval_dataset = coordinator.create_per_worker_dataset(eval_dataset_fn)
per_worker_eval_iterator = iter(per_worker_eval_dataset)
eval_steps_per_epoch = 2
for _ in range(eval_steps_per_epoch):
    custom_method(imports='import os;import random;import portpicker;import multiprocessing;import tensorflow as tf', function_to_run='obj.schedule(*args, **kwargs)', method_object=eval('coordinator'), object_signature=None, function_args=[eval('eval_step')], function_kwargs={'args': eval('(per_worker_eval_iterator,)')}, custom_class=None)
    coordinator.schedule(eval_step, args=(per_worker_eval_iterator,))
custom_method(imports='import os;import random;import portpicker;import multiprocessing;import tensorflow as tf', function_to_run='obj.join()', method_object=eval('coordinator'), object_signature=None, function_args=[], function_kwargs={}, custom_class=None)
coordinator.join()
print('Evaluation accuracy: %f' % eval_accuracy.result())
