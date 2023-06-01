import concurrent.futures
import collections
import dataclasses
import hashlib
import itertools
import json
import math
import os
import pathlib
import random
import re
import string
import time
import urllib.request
import einops
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
import requests
import tqdm
import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_text as text
import tensorflow_datasets as tfds
import sys
from tool.client.client_config import EXPERIMENT_DIR
from tool.server.local_execution import before_execution as before_execution_INSERTED_INTO_SCRIPT
from tool.server.local_execution import after_execution as after_execution_INSERTED_INTO_SCRIPT
experiment_number = sys.argv[1]
experiment_project = sys.argv[2]
EXPERIMENT_FILE_PATH = EXPERIMENT_DIR / 'method-level' / experiment_project / f'experiment-{experiment_number}.json'

def flickr8k(path='flickr8k'):
    path = pathlib.Path(path)
    if len(list(path.rglob('*'))) < 16197:
        start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT()
        tf.keras.utils.get_file(origin='https://github.com/jbrownlee/Datasets/releases/download/Flickr8k/Flickr8k_Dataset.zip', cache_dir='.', cache_subdir=path, extract=True)
        after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tf.keras.utils.get_file(**kwargs)', method_object=None, object_signature=None, function_args=[], function_kwargs={'origin': 'https://github.com/jbrownlee/Datasets/releases/download/Flickr8k/Flickr8k_Dataset.zip', 'cache_dir': '.', 'cache_subdir': path, 'extract': True})
        start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT()
        tf.keras.utils.get_file(origin='https://github.com/jbrownlee/Datasets/releases/download/Flickr8k/Flickr8k_text.zip', cache_dir='.', cache_subdir=path, extract=True)
        after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tf.keras.utils.get_file(**kwargs)', method_object=None, object_signature=None, function_args=[], function_kwargs={'origin': 'https://github.com/jbrownlee/Datasets/releases/download/Flickr8k/Flickr8k_text.zip', 'cache_dir': '.', 'cache_subdir': path, 'extract': True})
    captions = (path / 'Flickr8k.token.txt').read_text().splitlines()
    captions = (line.split('\t') for line in captions)
    captions = ((fname.split('#')[0], caption) for (fname, caption) in captions)
    cap_dict = collections.defaultdict(list)
    for (fname, cap) in captions:
        cap_dict[fname].append(cap)
    train_files = (path / 'Flickr_8k.trainImages.txt').read_text().splitlines()
    train_captions = [(str(path / 'Flicker8k_Dataset' / fname), cap_dict[fname]) for fname in train_files]
    test_files = (path / 'Flickr_8k.testImages.txt').read_text().splitlines()
    test_captions = [(str(path / 'Flicker8k_Dataset' / fname), cap_dict[fname]) for fname in test_files]
    start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT()
    train_ds = tf.data.experimental.from_list(train_captions)
    after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tf.data.experimental.from_list(*args)', method_object=None, object_signature=None, function_args=[train_captions], function_kwargs={})
    start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT()
    test_ds = tf.data.experimental.from_list(test_captions)
    after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tf.data.experimental.from_list(*args)', method_object=None, object_signature=None, function_args=[test_captions], function_kwargs={})
    return (train_ds, test_ds)

def conceptual_captions(*, data_dir='conceptual_captions', num_train, num_val):

    def iter_index(index_path):
        with open(index_path) as f:
            for line in f:
                (caption, url) = line.strip().split('\t')
                yield (caption, url)

    def download_image_urls(data_dir, urls):
        ex = concurrent.futures.ThreadPoolExecutor(max_workers=100)

        def save_image(url):
            hash = hashlib.sha1(url.encode())
            file_path = data_dir / f'{hash.hexdigest()}.jpeg'
            if file_path.exists():
                return file_path
            try:
                result = requests.get(url, timeout=5)
            except Exception:
                file_path = None
            else:
                file_path.write_bytes(result.content)
            return file_path
        result = []
        out_paths = ex.map(save_image, urls)
        for file_path in tqdm.tqdm(out_paths, total=len(urls)):
            start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT()
            result.append(file_path)
            after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='obj.append(*args)', method_object=result, object_signature=None, function_args=[file_path], function_kwargs={})
        return result

    def ds_from_index_file(index_path, data_dir, count):
        data_dir.mkdir(exist_ok=True)
        index = list(itertools.islice(iter_index(index_path), count))
        captions = [caption for (caption, url) in index]
        urls = [url for (caption, url) in index]
        paths = download_image_urls(data_dir, urls)
        new_captions = []
        new_paths = []
        for (cap, path) in zip(captions, paths):
            if path is None:
                continue
            new_captions.append(cap)
            new_paths.append(path)
        new_paths = [str(p) for p in new_paths]
        start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT()
        ds = tf.data.Dataset.from_tensor_slices((new_paths, new_captions))
        after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tf.data.Dataset.from_tensor_slices(*args)', method_object=None, object_signature=None, function_args=[(new_paths, new_captions)], function_kwargs={})
        start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT()
        ds = ds.map(lambda path, cap: (path, cap[tf.newaxis]))
        after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='obj.map(*args)', method_object=ds, object_signature=None, function_args=[lambda path, cap: (path, cap[tf.newaxis])], function_kwargs={})
        return ds
    data_dir = pathlib.Path(data_dir)
    start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT()
    train_index_path = tf.keras.utils.get_file(origin='https://storage.googleapis.com/gcc-data/Train/GCC-training.tsv', cache_subdir=data_dir, cache_dir='.')
    after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tf.keras.utils.get_file(**kwargs)', method_object=None, object_signature=None, function_args=[], function_kwargs={'origin': 'https://storage.googleapis.com/gcc-data/Train/GCC-training.tsv', 'cache_subdir': data_dir, 'cache_dir': '.'})
    start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT()
    val_index_path = tf.keras.utils.get_file(origin='https://storage.googleapis.com/gcc-data/Validation/GCC-1.1.0-Validation.tsv', cache_subdir=data_dir, cache_dir='.')
    after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tf.keras.utils.get_file(**kwargs)', method_object=None, object_signature=None, function_args=[], function_kwargs={'origin': 'https://storage.googleapis.com/gcc-data/Validation/GCC-1.1.0-Validation.tsv', 'cache_subdir': data_dir, 'cache_dir': '.'})
    train_raw = ds_from_index_file(train_index_path, data_dir=data_dir / 'train', count=num_train)
    test_raw = ds_from_index_file(val_index_path, data_dir=data_dir / 'val', count=num_val)
    return (train_raw, test_raw)
choose = 'flickr8k'
if choose == 'flickr8k':
    (train_raw, test_raw) = flickr8k()
else:
    (train_raw, test_raw) = conceptual_captions(num_train=10000, num_val=5000)
train_raw.element_spec
for (ex_path, ex_captions) in train_raw.take(1):
    print(ex_path)
    print(ex_captions)
IMAGE_SHAPE = (224, 224, 3)
start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT()
mobilenet = tf.keras.applications.MobileNetV3Small(input_shape=IMAGE_SHAPE, include_top=False, include_preprocessing=True)
after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tf.keras.applications.MobileNetV3Small(**kwargs)', method_object=None, object_signature=None, function_args=[], function_kwargs={'input_shape': IMAGE_SHAPE, 'include_top': False, 'include_preprocessing': True})
mobilenet.trainable = False

def load_image(image_path):
    start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT()
    img = tf.io.read_file(image_path)
    after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tf.io.read_file(*args)', method_object=None, object_signature=None, function_args=[image_path], function_kwargs={})
    start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT()
    img = tf.io.decode_jpeg(img, channels=3)
    after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tf.io.decode_jpeg(*args, **kwargs)', method_object=None, object_signature=None, function_args=[img], function_kwargs={'channels': 3})
    start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT()
    img = tf.image.resize(img, IMAGE_SHAPE[:-1])
    after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tf.image.resize(*args)', method_object=None, object_signature=None, function_args=[img, IMAGE_SHAPE[:-1]], function_kwargs={})
    return img
test_img_batch = load_image(ex_path)[tf.newaxis, :]
print(test_img_batch.shape)
print(mobilenet(test_img_batch).shape)

def standardize(s):
    start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT()
    s = tf.strings.lower(s)
    after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tf.strings.lower(*args)', method_object=None, object_signature=None, function_args=[s], function_kwargs={})
    start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT()
    s = tf.strings.regex_replace(s, f'[{re.escape(string.punctuation)}]', '')
    after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tf.strings.regex_replace(*args)', method_object=None, object_signature=None, function_args=[s, f'[{re.escape(string.punctuation)}]', ''], function_kwargs={})
    start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT()
    s = tf.strings.join(['[START]', s, '[END]'], separator=' ')
    after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tf.strings.join(*args, **kwargs)', method_object=None, object_signature=None, function_args=[['[START]', s, '[END]']], function_kwargs={'separator': ' '})
    return s
vocabulary_size = 5000
start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT()
tokenizer = tf.keras.layers.TextVectorization(max_tokens=vocabulary_size, standardize=standardize, ragged=True)
after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tf.keras.layers.TextVectorization(**kwargs)', method_object=None, object_signature=None, function_args=[], function_kwargs={'max_tokens': vocabulary_size, 'standardize': standardize, 'ragged': True})
start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT()
tokenizer.adapt(train_raw.map(lambda fp, txt: txt).unbatch().batch(1024))
after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='obj.adapt(*args)', method_object=tokenizer, object_signature=None, function_args=[train_raw.map(lambda fp, txt: txt).unbatch().batch(1024)], function_kwargs={})
tokenizer.get_vocabulary()[:10]
start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT()
t = tokenizer([['a cat in a hat'], ['a robot dog']])
after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='obj(*args)', method_object=tokenizer, object_signature=None, function_args=[[['a cat in a hat'], ['a robot dog']]], function_kwargs={})
t
start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT()
word_to_index = tf.keras.layers.StringLookup(mask_token='', vocabulary=tokenizer.get_vocabulary())
after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tf.keras.layers.StringLookup(**kwargs)', method_object=None, object_signature=None, function_args=[], function_kwargs={'mask_token': '', 'vocabulary': tokenizer.get_vocabulary()})
start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT()
index_to_word = tf.keras.layers.StringLookup(mask_token='', vocabulary=tokenizer.get_vocabulary(), invert=True)
after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tf.keras.layers.StringLookup(**kwargs)', method_object=None, object_signature=None, function_args=[], function_kwargs={'mask_token': '', 'vocabulary': tokenizer.get_vocabulary(), 'invert': True})
start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT()
w = index_to_word(t)
after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='obj(*args)', method_object=index_to_word, object_signature=None, function_args=[t], function_kwargs={})
w.to_list()
start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT()
tf.strings.reduce_join(w, separator=' ', axis=-1).numpy()
after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run="tf.strings.reduce_join(w, separator=' ', axis=-1).numpy()", method_object=None, object_signature=None, function_args=[], function_kwargs={})

def match_shapes(images, captions):
    caption_shape = einops.parse_shape(captions, 'b c')
    captions = einops.rearrange(captions, 'b c -> (b c)')
    images = einops.repeat(images, 'b ... -> (b c) ...', c=caption_shape['c'])
    return (images, captions)
for (ex_paths, ex_captions) in train_raw.batch(32).take(1):
    break
print('image paths:', ex_paths.shape)
print('captions:', ex_captions.shape)
print()
(ex_paths, ex_captions) = match_shapes(images=ex_paths, captions=ex_captions)
print('image_paths:', ex_paths.shape)
print('captions:', ex_captions.shape)

def prepare_txt(imgs, txts):
    start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT()
    tokens = tokenizer(txts)
    after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='obj(*args)', method_object=tokenizer, object_signature=None, function_args=[txts], function_kwargs={})
    input_tokens = tokens[..., :-1]
    label_tokens = tokens[..., 1:]
    return ((imgs, input_tokens), label_tokens)

def prepare_dataset(ds, tokenizer, batch_size=32, shuffle_buffer=1000):
    start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT()
    ds = ds.shuffle(10000).map(lambda path, caption: (load_image(path), caption)).apply(tf.data.experimental.ignore_errors()).batch(batch_size)
    after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='obj.shuffle(10000).map(lambda path, caption: (load_image(path), caption)).apply(tf.data.experimental.ignore_errors()).batch(*args)', method_object=ds, object_signature=None, function_args=[batch_size], function_kwargs={})

    def to_tensor(inputs, labels):
        ((images, in_tok), out_tok) = (inputs, labels)
        return ((images, in_tok.to_tensor()), out_tok.to_tensor())
    return ds.map(match_shapes, tf.data.AUTOTUNE).unbatch().shuffle(shuffle_buffer).batch(batch_size).map(prepare_txt, tf.data.AUTOTUNE).map(to_tensor, tf.data.AUTOTUNE)
train_ds = prepare_dataset(train_raw, tokenizer)
train_ds.element_spec
test_ds = prepare_dataset(test_raw, tokenizer)
test_ds.element_spec

def save_dataset(ds, save_path, image_model, tokenizer, shards=10, batch_size=32):
    start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT()
    ds = ds.map(lambda path, caption: (load_image(path), caption)).apply(tf.data.experimental.ignore_errors()).batch(batch_size)
    after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='obj.map(lambda path, caption: (load_image(path), caption)).apply(tf.data.experimental.ignore_errors()).batch(*args)', method_object=ds, object_signature=None, function_args=[batch_size], function_kwargs={})

    def gen():
        for (images, captions) in tqdm.tqdm(ds):
            feature_maps = image_model(images)
            (feature_maps, captions) = match_shapes(feature_maps, captions)
            yield (feature_maps, captions)
    start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT()
    new_ds = tf.data.Dataset.from_generator(gen, output_signature=(tf.TensorSpec(shape=image_model.output_shape), tf.TensorSpec(shape=(None,), dtype=tf.string)))
    after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tf.data.Dataset.from_generator(*args, **kwargs)', method_object=None, object_signature=None, function_args=[gen], function_kwargs={'output_signature': (tf.TensorSpec(shape=image_model.output_shape), tf.TensorSpec(shape=(None,), dtype=tf.string))})
    start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT()
    new_ds = new_ds.map(prepare_txt, tf.data.AUTOTUNE).unbatch().shuffle(1000)
    after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='obj.map(prepare_txt, tf.data.AUTOTUNE).unbatch().shuffle(*args)', method_object=new_ds, object_signature=None, function_args=[1000], function_kwargs={})

    def shard_func(i, item):
        return i % shards
    start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT()
    new_ds.enumerate().save(save_path, shard_func=shard_func)
    after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='obj.enumerate().save(*args, **kwargs)', method_object=new_ds, object_signature=None, function_args=[save_path], function_kwargs={'shard_func': shard_func})

def load_dataset(save_path, batch_size=32, shuffle=1000, cycle_length=2):

    def custom_reader_func(datasets):
        datasets = datasets.shuffle(1000)
        return datasets.interleave(lambda x: x, cycle_length=cycle_length)
    start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT()
    ds = tf.data.Dataset.load(save_path, reader_func=custom_reader_func)
    after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tf.data.Dataset.load(*args, **kwargs)', method_object=None, object_signature=None, function_args=[save_path], function_kwargs={'reader_func': custom_reader_func})

    def drop_index(i, x):
        return x
    start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT()
    ds = ds.map(drop_index, tf.data.AUTOTUNE).shuffle(shuffle).padded_batch(batch_size).prefetch(tf.data.AUTOTUNE)
    after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='obj.map(drop_index, tf.data.AUTOTUNE).shuffle(shuffle).padded_batch(batch_size).prefetch(*args)', method_object=ds, object_signature=None, function_args=[tf.data.AUTOTUNE], function_kwargs={})
    return ds
save_dataset(train_raw, 'train_cache', mobilenet, tokenizer)
save_dataset(test_raw, 'test_cache', mobilenet, tokenizer)
train_ds = load_dataset('train_cache')
test_ds = load_dataset('test_cache')
train_ds.element_spec
for (inputs, ex_labels) in train_ds.take(1):
    (ex_img, ex_in_tok) = inputs
print(ex_img.shape)
print(ex_in_tok.shape)
print(ex_labels.shape)
print(ex_in_tok[0].numpy())
print(ex_labels[0].numpy())

class SeqEmbedding(tf.keras.layers.Layer):

    def __init__(self, vocab_size, max_length, depth):
        super().__init__()
        start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT()
        self.pos_embedding = tf.keras.layers.Embedding(input_dim=max_length, output_dim=depth)
        after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tf.keras.layers.Embedding(**kwargs)', method_object=None, object_signature=None, function_args=[], function_kwargs={'input_dim': max_length, 'output_dim': depth})
        start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT()
        self.token_embedding = tf.keras.layers.Embedding(input_dim=vocab_size, output_dim=depth, mask_zero=True)
        after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tf.keras.layers.Embedding(**kwargs)', method_object=None, object_signature=None, function_args=[], function_kwargs={'input_dim': vocab_size, 'output_dim': depth, 'mask_zero': True})
        start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT()
        self.add = tf.keras.layers.Add()
        after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tf.keras.layers.Add()', method_object=None, object_signature=None, function_args=[], function_kwargs={})

    def call(self, seq):
        seq = self.token_embedding(seq)
        start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT()
        x = tf.range(tf.shape(seq)[1])
        after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tf.range(*args)', method_object=None, object_signature=None, function_args=[tf.shape(seq)[1]], function_kwargs={})
        x = x[tf.newaxis, :]
        x = self.pos_embedding(x)
        return self.add([seq, x])

class CausalSelfAttention(tf.keras.layers.Layer):

    def __init__(self, **kwargs):
        super().__init__()
        start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT()
        self.mha = tf.keras.layers.MultiHeadAttention(**kwargs)
        after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tf.keras.layers.MultiHeadAttention(**kwargs)', method_object=None, object_signature=None, function_args=[], function_kwargs={None: kwargs})
        start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT()
        self.add = tf.keras.layers.Add()
        after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tf.keras.layers.Add()', method_object=None, object_signature=None, function_args=[], function_kwargs={})
        start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT()
        self.layernorm = tf.keras.layers.LayerNormalization()
        after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tf.keras.layers.LayerNormalization()', method_object=None, object_signature=None, function_args=[], function_kwargs={})

    def call(self, x):
        attn = self.mha(query=x, value=x, use_causal_mask=True)
        x = self.add([x, attn])
        return self.layernorm(x)

class CrossAttention(tf.keras.layers.Layer):

    def __init__(self, **kwargs):
        super().__init__()
        start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT()
        self.mha = tf.keras.layers.MultiHeadAttention(**kwargs)
        after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tf.keras.layers.MultiHeadAttention(**kwargs)', method_object=None, object_signature=None, function_args=[], function_kwargs={None: kwargs})
        start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT()
        self.add = tf.keras.layers.Add()
        after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tf.keras.layers.Add()', method_object=None, object_signature=None, function_args=[], function_kwargs={})
        start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT()
        self.layernorm = tf.keras.layers.LayerNormalization()
        after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tf.keras.layers.LayerNormalization()', method_object=None, object_signature=None, function_args=[], function_kwargs={})

    def call(self, x, y, **kwargs):
        (attn, attention_scores) = self.mha(query=x, value=y, return_attention_scores=True)
        self.last_attention_scores = attention_scores
        x = self.add([x, attn])
        return self.layernorm(x)

class FeedForward(tf.keras.layers.Layer):

    def __init__(self, units, dropout_rate=0.1):
        super().__init__()
        start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT()
        self.seq = tf.keras.Sequential([tf.keras.layers.Dense(units=2 * units, activation='relu'), tf.keras.layers.Dense(units=units), tf.keras.layers.Dropout(rate=dropout_rate)])
        after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tf.keras.Sequential(*args)', method_object=None, object_signature=None, function_args=[[tf.keras.layers.Dense(units=2 * units, activation='relu'), tf.keras.layers.Dense(units=units), tf.keras.layers.Dropout(rate=dropout_rate)]], function_kwargs={})
        start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT()
        self.layernorm = tf.keras.layers.LayerNormalization()
        after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tf.keras.layers.LayerNormalization()', method_object=None, object_signature=None, function_args=[], function_kwargs={})

    def call(self, x):
        x = x + self.seq(x)
        return self.layernorm(x)

class DecoderLayer(tf.keras.layers.Layer):

    def __init__(self, units, num_heads=1, dropout_rate=0.1):
        super().__init__()
        self.self_attention = CausalSelfAttention(num_heads=num_heads, key_dim=units, dropout=dropout_rate)
        self.cross_attention = CrossAttention(num_heads=num_heads, key_dim=units, dropout=dropout_rate)
        self.ff = FeedForward(units=units, dropout_rate=dropout_rate)

    def call(self, inputs, training=False):
        (in_seq, out_seq) = inputs
        out_seq = self.self_attention(out_seq)
        out_seq = self.cross_attention(out_seq, in_seq)
        self.last_attention_scores = self.cross_attention.last_attention_scores
        out_seq = self.ff(out_seq)
        return out_seq

class TokenOutput(tf.keras.layers.Layer):

    def __init__(self, tokenizer, banned_tokens=('', '[UNK]', '[START]'), **kwargs):
        super().__init__()
        start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT()
        self.dense = tf.keras.layers.Dense(units=tokenizer.vocabulary_size(), **kwargs)
        after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tf.keras.layers.Dense(**kwargs)', method_object=None, object_signature=None, function_args=[], function_kwargs={'units': tokenizer.vocabulary_size(), None: kwargs})
        self.tokenizer = tokenizer
        self.banned_tokens = banned_tokens
        self.bias = None

    def adapt(self, ds):
        counts = collections.Counter()
        vocab_dict = {name: id for (id, name) in enumerate(self.tokenizer.get_vocabulary())}
        for tokens in tqdm.tqdm(ds):
            counts.update(tokens.numpy().flatten())
        counts_arr = np.zeros(shape=(self.tokenizer.vocabulary_size(),))
        counts_arr[np.array(list(counts.keys()), dtype=np.int32)] = list(counts.values())
        counts_arr = counts_arr[:]
        for token in self.banned_tokens:
            counts_arr[vocab_dict[token]] = 0
        total = counts_arr.sum()
        p = counts_arr / total
        p[counts_arr == 0] = 1.0
        log_p = np.log(p)
        entropy = -(log_p * p).sum()
        print()
        print(f'Uniform entropy: {np.log(self.tokenizer.vocabulary_size()):0.2f}')
        print(f'Marginal entropy: {entropy:0.2f}')
        self.bias = log_p
        self.bias[counts_arr == 0] = -1000000000.0

    def call(self, x):
        x = self.dense(x)
        return x + self.bias
output_layer = TokenOutput(tokenizer, banned_tokens=('', '[UNK]', '[START]'))
start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT()
output_layer.adapt(train_ds.map(lambda inputs, labels: labels))
after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='obj.adapt(*args)', method_object='output_layer', object_signature="TokenOutput(tokenizer, banned_tokens=('', '[UNK]', '[START]'))", function_args=[train_ds.map(lambda inputs, labels: labels)], function_kwargs={})

class Captioner(tf.keras.Model):

    @classmethod
    def add_method(cls, fun):
        setattr(cls, fun.__name__, fun)
        return fun

    def __init__(self, tokenizer, feature_extractor, output_layer, num_layers=1, units=256, max_length=50, num_heads=1, dropout_rate=0.1):
        super().__init__()
        self.feature_extractor = feature_extractor
        self.tokenizer = tokenizer
        start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT()
        self.word_to_index = tf.keras.layers.StringLookup(mask_token='', vocabulary=tokenizer.get_vocabulary())
        after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tf.keras.layers.StringLookup(**kwargs)', method_object=None, object_signature=None, function_args=[], function_kwargs={'mask_token': '', 'vocabulary': tokenizer.get_vocabulary()})
        start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT()
        self.index_to_word = tf.keras.layers.StringLookup(mask_token='', vocabulary=tokenizer.get_vocabulary(), invert=True)
        after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tf.keras.layers.StringLookup(**kwargs)', method_object=None, object_signature=None, function_args=[], function_kwargs={'mask_token': '', 'vocabulary': tokenizer.get_vocabulary(), 'invert': True})
        self.seq_embedding = SeqEmbedding(vocab_size=tokenizer.vocabulary_size(), depth=units, max_length=max_length)
        self.decoder_layers = [DecoderLayer(units, num_heads=num_heads, dropout_rate=dropout_rate) for n in range(num_layers)]
        self.output_layer = output_layer

    @Captioner.add_method
    def call(self, inputs):
        (image, txt) = inputs
        if image.shape[-1] == 3:
            image = self.feature_extractor(image)
        image = einops.rearrange(image, 'b h w c -> b (h w) c')
        if txt.dtype == tf.string:
            start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT()
            txt = tokenizer(txt)
            after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='obj(*args)', method_object=tokenizer, object_signature=None, function_args=[txt], function_kwargs={})
        txt = self.seq_embedding(txt)
        for dec_layer in self.decoder_layers:
            txt = dec_layer(inputs=(image, txt))
        txt = self.output_layer(txt)
        return txt
model = Captioner(tokenizer, feature_extractor=mobilenet, output_layer=output_layer, units=256, dropout_rate=0.5, num_layers=2, num_heads=2)
image_url = 'https://tensorflow.org/images/surf.jpg'
start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT()
image_path = tf.keras.utils.get_file('surf.jpg', origin=image_url)
after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tf.keras.utils.get_file(*args, **kwargs)', method_object=None, object_signature=None, function_args=['surf.jpg'], function_kwargs={'origin': image_url})
image = load_image(image_path)

@Captioner.add_method
def simple_gen(self, image, temperature=1):
    initial = self.word_to_index([['[START]']])
    img_features = self.feature_extractor(image[tf.newaxis, ...])
    tokens = initial
    for n in range(50):
        preds = self((img_features, tokens)).numpy()
        preds = preds[:, -1, :]
        if temperature == 0:
            next = tf.argmax(preds, axis=-1)[:, tf.newaxis]
        else:
            start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT()
            next = tf.random.categorical(preds / temperature, num_samples=1)
            after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tf.random.categorical(*args, **kwargs)', method_object=None, object_signature=None, function_args=[preds / temperature], function_kwargs={'num_samples': 1})
        start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT()
        tokens = tf.concat([tokens, next], axis=1)
        after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tf.concat(*args, **kwargs)', method_object=None, object_signature=None, function_args=[[tokens, next]], function_kwargs={'axis': 1})
        if next[0] == self.word_to_index('[END]'):
            break
    start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT()
    words = index_to_word(tokens[0, 1:-1])
    after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='obj(*args)', method_object=index_to_word, object_signature=None, function_args=[tokens[0, 1:-1]], function_kwargs={})
    start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT()
    result = tf.strings.reduce_join(words, axis=-1, separator=' ')
    after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tf.strings.reduce_join(*args, **kwargs)', method_object=None, object_signature=None, function_args=[words], function_kwargs={'axis': -1, 'separator': ' '})
    return result.numpy().decode()
for t in (0.0, 0.5, 1.0):
    start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT()
    result = model.simple_gen(image, temperature=t)
    after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='obj.simple_gen(*args, **kwargs)', method_object='model', object_signature=None, function_args=[image], function_kwargs={'temperature': t})
    print(result)

def masked_loss(labels, preds):
    start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT()
    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels, preds)
    after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tf.nn.sparse_softmax_cross_entropy_with_logits(*args)', method_object=None, object_signature=None, function_args=[labels, preds], function_kwargs={})
    mask = (labels != 0) & (loss < 100000000.0)
    start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT()
    mask = tf.cast(mask, loss.dtype)
    after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tf.cast(*args)', method_object=None, object_signature=None, function_args=[mask, loss.dtype], function_kwargs={})
    loss = loss * mask
    loss = tf.reduce_sum(loss) / tf.reduce_sum(mask)
    return loss

def masked_acc(labels, preds):
    start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT()
    mask = tf.cast(labels != 0, tf.float32)
    after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tf.cast(*args)', method_object=None, object_signature=None, function_args=[labels != 0, tf.float32], function_kwargs={})
    start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT()
    preds = tf.argmax(preds, axis=-1)
    after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tf.argmax(*args, **kwargs)', method_object=None, object_signature=None, function_args=[preds], function_kwargs={'axis': -1})
    start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT()
    labels = tf.cast(labels, tf.int64)
    after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tf.cast(*args)', method_object=None, object_signature=None, function_args=[labels, tf.int64], function_kwargs={})
    start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT()
    match = tf.cast(preds == labels, mask.dtype)
    after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tf.cast(*args)', method_object=None, object_signature=None, function_args=[preds == labels, mask.dtype], function_kwargs={})
    acc = tf.reduce_sum(match * mask) / tf.reduce_sum(mask)
    return acc

class GenerateText(tf.keras.callbacks.Callback):

    def __init__(self):
        image_url = 'https://tensorflow.org/images/surf.jpg'
        start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT()
        image_path = tf.keras.utils.get_file('surf.jpg', origin=image_url)
        after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tf.keras.utils.get_file(*args, **kwargs)', method_object=None, object_signature=None, function_args=['surf.jpg'], function_kwargs={'origin': image_url})
        self.image = load_image(image_path)

    def on_epoch_end(self, epochs=None, logs=None):
        print()
        print()
        for t in (0.0, 0.5, 1.0):
            result = self.model.simple_gen(self.image, temperature=t)
            print(result)
        print()
g = GenerateText()
g.model = model
start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT()
g.on_epoch_end(0)
after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='obj.on_epoch_end(*args)', method_object='g', object_signature='Captioner(tokenizer, feature_extractor=mobilenet, output_layer=output_layer,\n                  units=256, dropout_rate=0.5, num_layers=2, num_heads=2)', function_args=[0], function_kwargs={})
callbacks = [GenerateText(), tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True)]
start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT()
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001), loss=masked_loss, metrics=[masked_acc])
after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='obj.compile(**kwargs)', method_object='model', object_signature=None, function_args=[], function_kwargs={'optimizer': tf.keras.optimizers.Adam(learning_rate=0.0001), 'loss': masked_loss, 'metrics': [masked_acc]})
start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT()
history = model.fit(train_ds.repeat(), steps_per_epoch=100, validation_data=test_ds.repeat(), validation_steps=20, epochs=100, callbacks=callbacks)
after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='obj.fit(*args, **kwargs)', method_object='model', object_signature=None, function_args=[train_ds.repeat()], function_kwargs={'steps_per_epoch': 100, 'validation_data': test_ds.repeat(), 'validation_steps': 20, 'epochs': 100, 'callbacks': callbacks})
plt.plot(history.history['loss'], label='loss')
plt.plot(history.history['val_loss'], label='val_loss')
plt.ylim([0, max(plt.ylim())])
plt.xlabel('Epoch #')
plt.ylabel('CE/token')
plt.legend()
plt.plot(history.history['masked_acc'], label='accuracy')
plt.plot(history.history['val_masked_acc'], label='val_accuracy')
plt.ylim([0, max(plt.ylim())])
plt.xlabel('Epoch #')
plt.ylabel('CE/token')
plt.legend()
start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT()
result = model.simple_gen(image, temperature=0.0)
after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='obj.simple_gen(*args, **kwargs)', method_object='model', object_signature=None, function_args=[image], function_kwargs={'temperature': 0.0})
result
start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT()
str_tokens = result.split()
after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='obj.split()', method_object=result, object_signature=None, function_args=[], function_kwargs={})
str_tokens.append('[END]')
attn_maps = [layer.last_attention_scores for layer in model.decoder_layers]
[map.shape for map in attn_maps]
start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT()
attention_maps = tf.concat(attn_maps, axis=0)
after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tf.concat(*args, **kwargs)', method_object=None, object_signature=None, function_args=[attn_maps], function_kwargs={'axis': 0})
attention_maps = einops.reduce(attention_maps, 'batch heads sequence (height width) -> sequence height width', height=7, width=7, reduction='mean')
einops.reduce(attention_maps, 'sequence height width -> sequence', reduction='sum')

def plot_attention_maps(image, str_tokens, attention_map):
    fig = plt.figure(figsize=(16, 9))
    len_result = len(str_tokens)
    titles = []
    for i in range(len_result):
        map = attention_map[i]
        grid_size = max(int(np.ceil(len_result / 2)), 2)
        ax = fig.add_subplot(3, grid_size, i + 1)
        titles.append(ax.set_title(str_tokens[i]))
        img = ax.imshow(image)
        ax.imshow(map, cmap='gray', alpha=0.6, extent=img.get_extent(), clim=[0.0, np.max(map)])
    plt.tight_layout()
plot_attention_maps(image / 255, str_tokens, attention_maps)

@Captioner.add_method
def run_and_show_attention(self, image, temperature=0.0):
    result_txt = self.simple_gen(image, temperature)
    str_tokens = result_txt.split()
    str_tokens.append('[END]')
    attention_maps = [layer.last_attention_scores for layer in self.decoder_layers]
    start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT()
    attention_maps = tf.concat(attention_maps, axis=0)
    after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tf.concat(*args, **kwargs)', method_object=None, object_signature=None, function_args=[attention_maps], function_kwargs={'axis': 0})
    attention_maps = einops.reduce(attention_maps, 'batch heads sequence (height width) -> sequence height width', height=7, width=7, reduction='mean')
    plot_attention_maps(image / 255, str_tokens, attention_maps)
    t = plt.suptitle(result_txt)
    t.set_y(1.05)
run_and_show_attention(model, image)
image_url = 'https://tensorflow.org/images/bedroom_hrnet_tutorial.jpg'
start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT()
image_path = tf.keras.utils.get_file(origin=image_url)
after_execution_INSERTED_INTO_SCRIPT(start_times=start_times_INSERTED_INTO_SCRIPT, experiment_file_path=EXPERIMENT_FILE_PATH, function_to_run='tf.keras.utils.get_file(**kwargs)', method_object=None, object_signature=None, function_args=[], function_kwargs={'origin': image_url})
image = load_image(image_path)
run_and_show_attention(model, image)
