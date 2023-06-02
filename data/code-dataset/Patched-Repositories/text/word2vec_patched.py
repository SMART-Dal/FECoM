import io
import re
import string
import tqdm
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
import sys
from tool.client.client_config import EXPERIMENT_DIR
from tool.server.local_execution import before_execution as before_execution_INSERTED_INTO_SCRIPT
from tool.server.local_execution import after_execution as after_execution_INSERTED_INTO_SCRIPT
experiment_number = sys.argv[1]
experiment_project = sys.argv[2]
EXPERIMENT_FILE_PATH = EXPERIMENT_DIR / 'local-execution' / experiment_project / f'experiment-{experiment_number}.json'
SEED = 42
AUTOTUNE = tf.data.AUTOTUNE
sentence = 'The wide road shimmered in the hot sun'
tokens = list(sentence.lower().split())
print(len(tokens))
(vocab, index) = ({}, 1)
vocab['<pad>'] = 0
for token in tokens:
    if token not in vocab:
        vocab[token] = index
        index += 1
vocab_size = len(vocab)
print(vocab)
inverse_vocab = {index: token for (token, index) in vocab.items()}
print(inverse_vocab)
example_sequence = [vocab[word] for word in tokens]
print(example_sequence)
window_size = 2
start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT()
(positive_skip_grams, _) = tf.keras.preprocessing.sequence.skipgrams(example_sequence, vocabulary_size=vocab_size, window_size=window_size, negative_samples=0)
after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tf.keras.preprocessing.sequence.skipgrams()', method_object=None, function_args=[example_sequence], function_kwargs={'vocabulary_size': vocab_size, 'window_size': window_size, 'negative_samples': 0})
print(len(positive_skip_grams))
for (target, context) in positive_skip_grams[:5]:
    print(f'({target}, {context}): ({inverse_vocab[target]}, {inverse_vocab[context]})')
(target_word, context_word) = positive_skip_grams[0]
num_ns = 4
start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT()
context_class = tf.reshape(tf.constant(context_word, dtype='int64'), (1, 1))
after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tf.reshape()', method_object=None, function_args=[tf.constant(context_word, dtype='int64'), (1, 1)], function_kwargs=None)
start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT()
(negative_sampling_candidates, _, _) = tf.random.log_uniform_candidate_sampler(true_classes=context_class, num_true=1, num_sampled=num_ns, unique=True, range_max=vocab_size, seed=SEED, name='negative_sampling')
after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tf.random.log_uniform_candidate_sampler()', method_object=None, function_args=None, function_kwargs={'true_classes': context_class, 'num_true': 1, 'num_sampled': num_ns, 'unique': True, 'range_max': vocab_size, 'seed': SEED, 'name': 'negative_sampling'})
print(negative_sampling_candidates)
print([inverse_vocab[index.numpy()] for index in negative_sampling_candidates])
start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT()
squeezed_context_class = tf.squeeze(context_class, 1)
after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tf.squeeze()', method_object=None, function_args=[context_class, 1], function_kwargs=None)
start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT()
context = tf.concat([squeezed_context_class, negative_sampling_candidates], 0)
after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tf.concat()', method_object=None, function_args=[[squeezed_context_class, negative_sampling_candidates], 0], function_kwargs=None)
start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT()
label = tf.constant([1] + [0] * num_ns, dtype='int64')
after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tf.constant()', method_object=None, function_args=[[1] + [0] * num_ns], function_kwargs={'dtype': 'int64'})
target = target_word
print(f'target_index    : {target}')
print(f'target_word     : {inverse_vocab[target_word]}')
print(f'context_indices : {context}')
print(f'context_words   : {[inverse_vocab[c.numpy()] for c in context]}')
print(f'label           : {label}')
print('target  :', target)
print('context :', context)
print('label   :', label)
start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT()
sampling_table = tf.keras.preprocessing.sequence.make_sampling_table(size=10)
after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tf.keras.preprocessing.sequence.make_sampling_table()', method_object=None, function_args=None, function_kwargs={'size': 10})
print(sampling_table)

def generate_training_data(sequences, window_size, num_ns, vocab_size, seed):
    (targets, contexts, labels) = ([], [], [])
    start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT()
    sampling_table = tf.keras.preprocessing.sequence.make_sampling_table(vocab_size)
    after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tf.keras.preprocessing.sequence.make_sampling_table()', method_object=None, function_args=[vocab_size], function_kwargs=None)
    for sequence in tqdm.tqdm(sequences):
        start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT()
        (positive_skip_grams, _) = tf.keras.preprocessing.sequence.skipgrams(sequence, vocabulary_size=vocab_size, sampling_table=sampling_table, window_size=window_size, negative_samples=0)
        after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tf.keras.preprocessing.sequence.skipgrams()', method_object=None, function_args=[sequence], function_kwargs={'vocabulary_size': vocab_size, 'sampling_table': sampling_table, 'window_size': window_size, 'negative_samples': 0})
        for (target_word, context_word) in positive_skip_grams:
            start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT()
            context_class = tf.expand_dims(tf.constant([context_word], dtype='int64'), 1)
            after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tf.expand_dims()', method_object=None, function_args=[tf.constant([context_word], dtype='int64'), 1], function_kwargs=None)
            start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT()
            (negative_sampling_candidates, _, _) = tf.random.log_uniform_candidate_sampler(true_classes=context_class, num_true=1, num_sampled=num_ns, unique=True, range_max=vocab_size, seed=seed, name='negative_sampling')
            after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tf.random.log_uniform_candidate_sampler()', method_object=None, function_args=None, function_kwargs={'true_classes': context_class, 'num_true': 1, 'num_sampled': num_ns, 'unique': True, 'range_max': vocab_size, 'seed': seed, 'name': 'negative_sampling'})
            start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT()
            context = tf.concat([tf.squeeze(context_class, 1), negative_sampling_candidates], 0)
            after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tf.concat()', method_object=None, function_args=[[tf.squeeze(context_class, 1), negative_sampling_candidates], 0], function_kwargs=None)
            start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT()
            label = tf.constant([1] + [0] * num_ns, dtype='int64')
            after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tf.constant()', method_object=None, function_args=[[1] + [0] * num_ns], function_kwargs={'dtype': 'int64'})
            targets.append(target_word)
            contexts.append(context)
            labels.append(label)
    return (targets, contexts, labels)
start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT()
path_to_file = tf.keras.utils.get_file('shakespeare.txt', 'https://storage.googleapis.com/download.tensorflow.org/data/shakespeare.txt')
after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tf.keras.utils.get_file()', method_object=None, function_args=['shakespeare.txt', 'https://storage.googleapis.com/download.tensorflow.org/data/shakespeare.txt'], function_kwargs=None)
with open(path_to_file) as f:
    lines = f.read().splitlines()
for line in lines[:20]:
    print(line)
start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT()
text_ds = tf.data.TextLineDataset(path_to_file).filter(lambda x: tf.cast(tf.strings.length(x), bool))
after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tf.data.TextLineDataset(path_to_file).filter()', method_object=None, function_args=[lambda x: tf.cast(tf.strings.length(x), bool)], function_kwargs=None)

def custom_standardization(input_data):
    start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT()
    lowercase = tf.strings.lower(input_data)
    after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tf.strings.lower()', method_object=None, function_args=[input_data], function_kwargs=None)
    return tf.strings.regex_replace(lowercase, '[%s]' % re.escape(string.punctuation), '')
vocab_size = 4096
sequence_length = 10
start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT()
vectorize_layer = layers.TextVectorization(standardize=custom_standardization, max_tokens=vocab_size, output_mode='int', output_sequence_length=sequence_length)
after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='layers.TextVectorization()', method_object=None, function_args=None, function_kwargs={'standardize': custom_standardization, 'max_tokens': vocab_size, 'output_mode': 'int', 'output_sequence_length': sequence_length})
start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT()
vectorize_layer.adapt(text_ds.batch(1024))
after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='layers.TextVectorization.adapt()', method_object=vectorize_layer, function_args=[text_ds.batch(1024)], function_kwargs=None)
start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT()
inverse_vocab = vectorize_layer.get_vocabulary()
after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='layers.TextVectorization.get_vocabulary()', method_object=vectorize_layer, function_args=None, function_kwargs=None)
print(inverse_vocab[:20])
start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT()
text_vector_ds = text_ds.batch(1024).prefetch(AUTOTUNE).map(vectorize_layer).unbatch()
after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tf.data.TextLineDataset(path_to_file).filter.batch(1024).prefetch(AUTOTUNE).map(vectorize_layer).unbatch()', method_object=text_ds, function_args=None, function_kwargs=None)
sequences = list(text_vector_ds.as_numpy_iterator())
print(len(sequences))
for seq in sequences[:5]:
    print(f'{seq} => {[inverse_vocab[i] for i in seq]}')
(targets, contexts, labels) = generate_training_data(sequences=sequences, window_size=2, num_ns=4, vocab_size=vocab_size, seed=SEED)
targets = np.array(targets)
contexts = np.array(contexts)
labels = np.array(labels)
print('\n')
print(f'targets.shape: {targets.shape}')
print(f'contexts.shape: {contexts.shape}')
print(f'labels.shape: {labels.shape}')
BATCH_SIZE = 1024
BUFFER_SIZE = 10000
start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT()
dataset = tf.data.Dataset.from_tensor_slices(((targets, contexts), labels))
after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tf.data.Dataset.from_tensor_slices()', method_object=None, function_args=[((targets, contexts), labels)], function_kwargs=None)
start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT()
dataset = dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE, drop_remainder=True)
after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='dataset.shuffle(BUFFER_SIZE).batch.shuffle(BUFFER_SIZE).batch()', method_object=dataset, function_args=[BATCH_SIZE], function_kwargs={'drop_remainder': True})
print(dataset)
start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT()
dataset = dataset.cache().prefetch(buffer_size=AUTOTUNE)
after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='dataset.cache().prefetch.cache().prefetch()', method_object=dataset, function_args=None, function_kwargs={'buffer_size': AUTOTUNE})
print(dataset)

class Word2Vec(tf.keras.Model):

    def __init__(self, vocab_size, embedding_dim):
        super(Word2Vec, self).__init__()
        start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT()
        self.target_embedding = layers.Embedding(vocab_size, embedding_dim, input_length=1, name='w2v_embedding')
        after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='layers.Embedding()', method_object=None, function_args=[vocab_size, embedding_dim], function_kwargs={'input_length': 1, 'name': 'w2v_embedding'})
        start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT()
        self.context_embedding = layers.Embedding(vocab_size, embedding_dim, input_length=num_ns + 1)
        after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='layers.Embedding()', method_object=None, function_args=[vocab_size, embedding_dim], function_kwargs={'input_length': num_ns + 1})

    def call(self, pair):
        (target, context) = pair
        if len(target.shape) == 2:
            start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT()
            target = tf.squeeze(target, axis=1)
            after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tf.squeeze()', method_object=None, function_args=[target], function_kwargs={'axis': 1})
        word_emb = self.target_embedding(target)
        context_emb = self.context_embedding(context)
        start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT()
        dots = tf.einsum('be,bce->bc', word_emb, context_emb)
        after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tf.einsum()', method_object=None, function_args=['be,bce->bc', word_emb, context_emb], function_kwargs=None)
        return dots
embedding_dim = 128
word2vec = Word2Vec(vocab_size, embedding_dim)
start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT()
word2vec.compile(optimizer='adam', loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True), metrics=['accuracy'])
after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='Word2Vec.compile()', method_object=word2vec, function_args=None, function_kwargs={'optimizer': 'adam', 'loss': tf.keras.losses.CategoricalCrossentropy(from_logits=True), 'metrics': ['accuracy']})
start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT()
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir='logs')
after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tf.keras.callbacks.TensorBoard()', method_object=None, function_args=None, function_kwargs={'log_dir': 'logs'})
start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT()
word2vec.fit(dataset, epochs=20, callbacks=[tensorboard_callback])
after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='Word2Vec.fit()', method_object=word2vec, function_args=[dataset], function_kwargs={'epochs': 20, 'callbacks': [tensorboard_callback]})
weights = word2vec.get_layer('w2v_embedding').get_weights()[0]
start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT()
vocab = vectorize_layer.get_vocabulary()
after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='layers.TextVectorization.get_vocabulary()', method_object=vectorize_layer, function_args=None, function_kwargs=None)
out_v = io.open('vectors.tsv', 'w', encoding='utf-8')
out_m = io.open('metadata.tsv', 'w', encoding='utf-8')
for (index, word) in enumerate(vocab):
    if index == 0:
        continue
    vec = weights[index]
    out_v.write('\t'.join([str(x) for x in vec]) + '\n')
    out_m.write(word + '\n')
out_v.close()
out_m.close()
