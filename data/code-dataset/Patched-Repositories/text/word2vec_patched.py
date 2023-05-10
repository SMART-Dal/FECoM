import io
import re
import string
import tqdm
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
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

def custom_method(func, imports: str, function_to_run: str, method_object=None, object_signature=None, function_args: list=None, function_kwargs: dict=None, custom_class=None):
    result = send_request(imports=imports, function_to_run=function_to_run, function_args=function_args, function_kwargs=function_kwargs, max_wait_secs=MAX_WAIT_S, wait_after_run_secs=WAIT_AFTER_RUN_S, method_object=method_object, object_signature=object_signature, custom_class=custom_class, experiment_file_path=EXPERIMENT_FILE_PATH)
    return func
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
(positive_skip_grams, _) = custom_method(
tf.keras.preprocessing.sequence.skipgrams(example_sequence, vocabulary_size=vocab_size, window_size=window_size, negative_samples=0), imports='from tensorflow.keras import layers;import tqdm;import numpy as np;import re;import tensorflow as tf;from google.colab import files;import string;import io', function_to_run='tf.keras.preprocessing.sequence.skipgrams(*args, **kwargs)', method_object=None, object_signature=None, function_args=[eval('example_sequence')], function_kwargs={'vocabulary_size': eval('vocab_size'), 'window_size': eval('window_size'), 'negative_samples': eval('0')})
print(len(positive_skip_grams))
for (target, context) in positive_skip_grams[:5]:
    print(f'({target}, {context}): ({inverse_vocab[target]}, {inverse_vocab[context]})')
(target_word, context_word) = positive_skip_grams[0]
num_ns = 4
context_class = custom_method(
tf.reshape(tf.constant(context_word, dtype='int64'), (1, 1)), imports='from tensorflow.keras import layers;import tqdm;import numpy as np;import re;import tensorflow as tf;from google.colab import files;import string;import io', function_to_run='tf.reshape(*args)', method_object=None, object_signature=None, function_args=[eval('tf.constant(context_word, dtype="int64")'), eval('(1, 1)')], function_kwargs={})
(negative_sampling_candidates, _, _) = custom_method(
tf.random.log_uniform_candidate_sampler(true_classes=context_class, num_true=1, num_sampled=num_ns, unique=True, range_max=vocab_size, seed=SEED, name='negative_sampling'), imports='from tensorflow.keras import layers;import tqdm;import numpy as np;import re;import tensorflow as tf;from google.colab import files;import string;import io', function_to_run='tf.random.log_uniform_candidate_sampler(**kwargs)', method_object=None, object_signature=None, function_args=[], function_kwargs={'true_classes': eval('context_class'), 'num_true': eval('1'), 'num_sampled': eval('num_ns'), 'unique': eval('True'), 'range_max': eval('vocab_size'), 'seed': eval('SEED'), 'name': eval('"negative_sampling"')})
print(negative_sampling_candidates)
print([inverse_vocab[index.numpy()] for index in negative_sampling_candidates])
squeezed_context_class = custom_method(
tf.squeeze(context_class, 1), imports='from tensorflow.keras import layers;import tqdm;import numpy as np;import re;import tensorflow as tf;from google.colab import files;import string;import io', function_to_run='tf.squeeze(*args)', method_object=None, object_signature=None, function_args=[eval('context_class'), eval('1')], function_kwargs={})
context = custom_method(
tf.concat([squeezed_context_class, negative_sampling_candidates], 0), imports='from tensorflow.keras import layers;import tqdm;import numpy as np;import re;import tensorflow as tf;from google.colab import files;import string;import io', function_to_run='tf.concat(*args)', method_object=None, object_signature=None, function_args=[eval('[squeezed_context_class, negative_sampling_candidates]'), eval('0')], function_kwargs={})
label = custom_method(
tf.constant([1] + [0] * num_ns, dtype='int64'), imports='from tensorflow.keras import layers;import tqdm;import numpy as np;import re;import tensorflow as tf;from google.colab import files;import string;import io', function_to_run='tf.constant(*args, **kwargs)', method_object=None, object_signature=None, function_args=[eval('[1] + [0]*num_ns')], function_kwargs={'dtype': eval('"int64"')})
target = target_word
print(f'target_index    : {target}')
print(f'target_word     : {inverse_vocab[target_word]}')
print(f'context_indices : {context}')
print(f'context_words   : {[inverse_vocab[c.numpy()] for c in context]}')
print(f'label           : {label}')
print('target  :', target)
print('context :', context)
print('label   :', label)
sampling_table = custom_method(
tf.keras.preprocessing.sequence.make_sampling_table(size=10), imports='from tensorflow.keras import layers;import tqdm;import numpy as np;import re;import tensorflow as tf;from google.colab import files;import string;import io', function_to_run='tf.keras.preprocessing.sequence.make_sampling_table(**kwargs)', method_object=None, object_signature=None, function_args=[], function_kwargs={'size': eval('10')})
print(sampling_table)

def generate_training_data(sequences, window_size, num_ns, vocab_size, seed):
    (targets, contexts, labels) = ([], [], [])
    sampling_table = custom_method(
    tf.keras.preprocessing.sequence.make_sampling_table(vocab_size), imports='from tensorflow.keras import layers;import tqdm;import numpy as np;import re;import tensorflow as tf;from google.colab import files;import string;import io', function_to_run='tf.keras.preprocessing.sequence.make_sampling_table(*args)', method_object=None, object_signature=None, function_args=[eval('vocab_size')], function_kwargs={})
    for sequence in tqdm.tqdm(sequences):
        (positive_skip_grams, _) = custom_method(
        tf.keras.preprocessing.sequence.skipgrams(sequence, vocabulary_size=vocab_size, sampling_table=sampling_table, window_size=window_size, negative_samples=0), imports='from tensorflow.keras import layers;import tqdm;import numpy as np;import re;import tensorflow as tf;from google.colab import files;import string;import io', function_to_run='tf.keras.preprocessing.sequence.skipgrams(*args, **kwargs)', method_object=None, object_signature=None, function_args=[eval('sequence')], function_kwargs={'vocabulary_size': eval('vocab_size'), 'sampling_table': eval('sampling_table'), 'window_size': eval('window_size'), 'negative_samples': eval('0')})
        for (target_word, context_word) in positive_skip_grams:
            context_class = custom_method(
            tf.expand_dims(tf.constant([context_word], dtype='int64'), 1), imports='from tensorflow.keras import layers;import tqdm;import numpy as np;import re;import tensorflow as tf;from google.colab import files;import string;import io', function_to_run='tf.expand_dims(*args)', method_object=None, object_signature=None, function_args=[eval('tf.constant([context_word], dtype="int64")'), eval('1')], function_kwargs={})
            (negative_sampling_candidates, _, _) = custom_method(
            tf.random.log_uniform_candidate_sampler(true_classes=context_class, num_true=1, num_sampled=num_ns, unique=True, range_max=vocab_size, seed=seed, name='negative_sampling'), imports='from tensorflow.keras import layers;import tqdm;import numpy as np;import re;import tensorflow as tf;from google.colab import files;import string;import io', function_to_run='tf.random.log_uniform_candidate_sampler(**kwargs)', method_object=None, object_signature=None, function_args=[], function_kwargs={'true_classes': eval('context_class'), 'num_true': eval('1'), 'num_sampled': eval('num_ns'), 'unique': eval('True'), 'range_max': eval('vocab_size'), 'seed': eval('seed'), 'name': eval('"negative_sampling"')})
            context = custom_method(
            tf.concat([tf.squeeze(context_class, 1), negative_sampling_candidates], 0), imports='from tensorflow.keras import layers;import tqdm;import numpy as np;import re;import tensorflow as tf;from google.colab import files;import string;import io', function_to_run='tf.concat(*args)', method_object=None, object_signature=None, function_args=[eval('[tf.squeeze(context_class,1), negative_sampling_candidates]'), eval('0')], function_kwargs={})
            label = custom_method(
            tf.constant([1] + [0] * num_ns, dtype='int64'), imports='from tensorflow.keras import layers;import tqdm;import numpy as np;import re;import tensorflow as tf;from google.colab import files;import string;import io', function_to_run='tf.constant(*args, **kwargs)', method_object=None, object_signature=None, function_args=[eval('[1] + [0]*num_ns')], function_kwargs={'dtype': eval('"int64"')})
            targets.append(target_word)
            contexts.append(context)
            labels.append(label)
    return (targets, contexts, labels)
path_to_file = custom_method(
tf.keras.utils.get_file('shakespeare.txt', 'https://storage.googleapis.com/download.tensorflow.org/data/shakespeare.txt'), imports='from tensorflow.keras import layers;import tqdm;import numpy as np;import re;import tensorflow as tf;from google.colab import files;import string;import io', function_to_run='tf.keras.utils.get_file(*args)', method_object=None, object_signature=None, function_args=[eval("'shakespeare.txt'"), eval("'https://storage.googleapis.com/download.tensorflow.org/data/shakespeare.txt'")], function_kwargs={})
with open(path_to_file) as f:
    lines = f.read().splitlines()
for line in lines[:20]:
    print(line)
text_ds = custom_method(
tf.data.TextLineDataset(path_to_file).filter(lambda x: tf.cast(tf.strings.length(x), bool)), imports='from tensorflow.keras import layers;import tqdm;import numpy as np;import re;import tensorflow as tf;from google.colab import files;import string;import io', function_to_run='tf.data.TextLineDataset(path_to_file).filter(*args)', method_object=None, object_signature=None, function_args=[eval('lambda x: tf.cast(tf.strings.length(x), bool)')], function_kwargs={})

def custom_standardization(input_data):
    lowercase = custom_method(
    tf.strings.lower(input_data), imports='from tensorflow.keras import layers;import tqdm;import numpy as np;import re;import tensorflow as tf;from google.colab import files;import string;import io', function_to_run='tf.strings.lower(*args)', method_object=None, object_signature=None, function_args=[eval('input_data')], function_kwargs={})
    return custom_method(
    tf.strings.regex_replace(lowercase, '[%s]' % re.escape(string.punctuation), ''), imports='from tensorflow.keras import layers;import tqdm;import numpy as np;import re;import tensorflow as tf;from google.colab import files;import string;import io', function_to_run='tf.strings.regex_replace(*args)', method_object=None, object_signature=None, function_args=[eval('lowercase'), eval("'[%s]' % re.escape(string.punctuation)"), eval("''")], function_kwargs={})
vocab_size = 4096
sequence_length = 10
vectorize_layer = custom_method(
layers.TextVectorization(standardize=custom_standardization, max_tokens=vocab_size, output_mode='int', output_sequence_length=sequence_length), imports='from tensorflow.keras import layers;import tqdm;import numpy as np;import re;import tensorflow as tf;from google.colab import files;import string;import io', function_to_run='layers.TextVectorization(**kwargs)', method_object=None, object_signature=None, function_args=[], function_kwargs={'standardize': eval('custom_standardization'), 'max_tokens': eval('vocab_size'), 'output_mode': eval("'int'"), 'output_sequence_length': eval('sequence_length')})
custom_method(
vectorize_layer.adapt(text_ds.batch(1024)), imports='from tensorflow.keras import layers;import tqdm;import numpy as np;import re;import tensorflow as tf;from google.colab import files;import string;import io', function_to_run='obj.adapt(*args)', method_object=eval('vectorize_layer'), object_signature=None, function_args=[eval('text_ds.batch(1024)')], function_kwargs={}, custom_class=None)
inverse_vocab = custom_method(
vectorize_layer.get_vocabulary(), imports='from tensorflow.keras import layers;import tqdm;import numpy as np;import re;import tensorflow as tf;from google.colab import files;import string;import io', function_to_run='obj.get_vocabulary()', method_object=eval('vectorize_layer'), object_signature=None, function_args=[], function_kwargs={}, custom_class=None)
print(inverse_vocab[:20])
text_vector_ds = custom_method(
text_ds.batch(1024).prefetch(AUTOTUNE).map(vectorize_layer).unbatch(), imports='from tensorflow.keras import layers;import tqdm;import numpy as np;import re;import tensorflow as tf;from google.colab import files;import string;import io', function_to_run='obj.batch(1024).prefetch(AUTOTUNE).map(vectorize_layer).unbatch()', method_object=eval('text_ds'), object_signature=None, function_args=[], function_kwargs={}, custom_class=None)
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
dataset = custom_method(
tf.data.Dataset.from_tensor_slices(((targets, contexts), labels)), imports='from tensorflow.keras import layers;import tqdm;import numpy as np;import re;import tensorflow as tf;from google.colab import files;import string;import io', function_to_run='tf.data.Dataset.from_tensor_slices(*args)', method_object=None, object_signature=None, function_args=[eval('((targets, contexts), labels)')], function_kwargs={})
dataset = custom_method(
dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE, drop_remainder=True), imports='from tensorflow.keras import layers;import tqdm;import numpy as np;import re;import tensorflow as tf;from google.colab import files;import string;import io', function_to_run='obj.shuffle(BUFFER_SIZE).batch(*args, **kwargs)', method_object=eval('dataset'), object_signature=None, function_args=[eval('BATCH_SIZE')], function_kwargs={'drop_remainder': eval('True')}, custom_class=None)
print(dataset)
dataset = custom_method(
dataset.cache().prefetch(buffer_size=AUTOTUNE), imports='from tensorflow.keras import layers;import tqdm;import numpy as np;import re;import tensorflow as tf;from google.colab import files;import string;import io', function_to_run='obj.cache().prefetch(**kwargs)', method_object=eval('dataset'), object_signature=None, function_args=[], function_kwargs={'buffer_size': eval('AUTOTUNE')}, custom_class=None)
print(dataset)

class Word2Vec(tf.keras.Model):

    def __init__(self, vocab_size, embedding_dim):
        super(Word2Vec, self).__init__()
        self.target_embedding = custom_method(
        layers.Embedding(vocab_size, embedding_dim, input_length=1, name='w2v_embedding'), imports='from tensorflow.keras import layers;import tqdm;import numpy as np;import re;import tensorflow as tf;from google.colab import files;import string;import io', function_to_run='layers.Embedding(*args, **kwargs)', method_object=None, object_signature=None, function_args=[eval('vocab_size'), eval('embedding_dim')], function_kwargs={'input_length': eval('1'), 'name': eval('"w2v_embedding"')})
        self.context_embedding = custom_method(
        layers.Embedding(vocab_size, embedding_dim, input_length=num_ns + 1), imports='from tensorflow.keras import layers;import tqdm;import numpy as np;import re;import tensorflow as tf;from google.colab import files;import string;import io', function_to_run='layers.Embedding(*args, **kwargs)', method_object=None, object_signature=None, function_args=[eval('vocab_size'), eval('embedding_dim')], function_kwargs={'input_length': eval('num_ns+1')})

    def call(self, pair):
        (target, context) = pair
        if len(target.shape) == 2:
            target = custom_method(
            tf.squeeze(target, axis=1), imports='from tensorflow.keras import layers;import tqdm;import numpy as np;import re;import tensorflow as tf;from google.colab import files;import string;import io', function_to_run='tf.squeeze(*args, **kwargs)', method_object=None, object_signature=None, function_args=[eval('target')], function_kwargs={'axis': eval('1')})
        word_emb = self.target_embedding(target)
        context_emb = self.context_embedding(context)
        dots = custom_method(
        tf.einsum('be,bce->bc', word_emb, context_emb), imports='from tensorflow.keras import layers;import tqdm;import numpy as np;import re;import tensorflow as tf;from google.colab import files;import string;import io', function_to_run='tf.einsum(*args)', method_object=None, object_signature=None, function_args=[eval("'be,bce->bc'"), eval('word_emb'), eval('context_emb')], function_kwargs={})
        return dots
embedding_dim = 128
word2vec = Word2Vec(vocab_size, embedding_dim)
custom_method(
word2vec.compile(optimizer='adam', loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True), metrics=['accuracy']), imports='from tensorflow.keras import layers;import tqdm;import numpy as np;import re;import tensorflow as tf;from google.colab import files;import string;import io', function_to_run='obj.compile(**kwargs)', method_object=eval('word2vec'), object_signature='Word2Vec(vocab_size, embedding_dim)', function_args=[], function_kwargs={'optimizer': eval("'adam'"), 'loss': eval('tf.keras.losses.CategoricalCrossentropy(from_logits=True)'), 'metrics': eval("['accuracy']")}, custom_class='class Word2Vec(tf.keras.Model):\n  def __init__(self, vocab_size, embedding_dim):\n    super(Word2Vec, self).__init__()\n    self.target_embedding = layers.Embedding(vocab_size,\n                                      embedding_dim,\n                                      input_length=1,\n                                      name="w2v_embedding")\n    self.context_embedding = layers.Embedding(vocab_size,\n                                       embedding_dim,\n                                       input_length=num_ns+1)\n\n  def call(self, pair):\n    target, context = pair\n    if len(target.shape) == 2:\n      target = tf.squeeze(target, axis=1)\n    word_emb = self.target_embedding(target)\n    context_emb = self.context_embedding(context)\n    dots = tf.einsum(\'be,bce->bc\', word_emb, context_emb)\n    return dots')
tensorboard_callback = custom_method(
tf.keras.callbacks.TensorBoard(log_dir='logs'), imports='from tensorflow.keras import layers;import tqdm;import numpy as np;import re;import tensorflow as tf;from google.colab import files;import string;import io', function_to_run='tf.keras.callbacks.TensorBoard(**kwargs)', method_object=None, object_signature=None, function_args=[], function_kwargs={'log_dir': eval('"logs"')})
custom_method(
word2vec.fit(dataset, epochs=20, callbacks=[tensorboard_callback]), imports='from tensorflow.keras import layers;import tqdm;import numpy as np;import re;import tensorflow as tf;from google.colab import files;import string;import io', function_to_run='obj.fit(*args, **kwargs)', method_object=eval('word2vec'), object_signature=None, function_args=[eval('dataset')], function_kwargs={'epochs': eval('20'), 'callbacks': eval('[tensorboard_callback]')}, custom_class='class Word2Vec(tf.keras.Model):\n  def __init__(self, vocab_size, embedding_dim):\n    super(Word2Vec, self).__init__()\n    self.target_embedding = layers.Embedding(vocab_size,\n                                      embedding_dim,\n                                      input_length=1,\n                                      name="w2v_embedding")\n    self.context_embedding = layers.Embedding(vocab_size,\n                                       embedding_dim,\n                                       input_length=num_ns+1)\n\n  def call(self, pair):\n    target, context = pair\n    if len(target.shape) == 2:\n      target = tf.squeeze(target, axis=1)\n    word_emb = self.target_embedding(target)\n    context_emb = self.context_embedding(context)\n    dots = tf.einsum(\'be,bce->bc\', word_emb, context_emb)\n    return dots')
weights = word2vec.get_layer('w2v_embedding').get_weights()[0]
vocab = custom_method(
vectorize_layer.get_vocabulary(), imports='from tensorflow.keras import layers;import tqdm;import numpy as np;import re;import tensorflow as tf;from google.colab import files;import string;import io', function_to_run='obj.get_vocabulary()', method_object=eval('vectorize_layer'), object_signature=None, function_args=[], function_kwargs={}, custom_class=None)
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
try:
    from google.colab import files
    files.download('vectors.tsv')
    files.download('metadata.tsv')
except Exception:
    pass
