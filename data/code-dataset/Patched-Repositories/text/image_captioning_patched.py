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

def custom_method(func, imports: str, function_to_run: str, method_object=None, object_signature=None, function_args: list=None, function_kwargs: dict=None, max_wait_secs=MAX_WAIT_S, custom_class=None, wait_after_run_secs=WAIT_AFTER_RUN_S):
    result = send_request(imports=imports, function_to_run=function_to_run, function_args=function_args, function_kwargs=function_kwargs, max_wait_secs=max_wait_secs, wait_after_run_secs=wait_after_run_secs, method_object=method_object, object_signature=object_signature, custom_class=custom_class, experiment_file_path=EXPERIMENT_FILE_PATH)
    return func

def flickr8k(path='flickr8k'):
    path = pathlib.Path(path)
    if len(list(path.rglob('*'))) < 16197:
        custom_method(
        tf.keras.utils.get_file(origin='https://github.com/jbrownlee/Datasets/releases/download/Flickr8k/Flickr8k_Dataset.zip', cache_dir='.', cache_subdir=path, extract=True), imports='import concurrent.futures;import collections;import hashlib;import itertools;import urllib.request;import tensorflow_hub as hub;import os;import pandas as pd;import einops;import tqdm;import numpy as np;import math;import requests;import pathlib;import tensorflow_text as text;import tensorflow as tf;import random;import tensorflow_datasets as tfds;from PIL import Image;import dataclasses;import string;import matplotlib.pyplot as plt;import json;import time;import re', function_to_run='tf.keras.utils.get_file(**kwargs)', method_object=None, object_signature=None, function_args=[], function_kwargs={'origin': eval("'https://github.com/jbrownlee/Datasets/releases/download/Flickr8k/Flickr8k_Dataset.zip'"), 'cache_dir': eval("'.'"), 'cache_subdir': eval('path'), 'extract': eval('True')}, max_wait_secs=0)
        custom_method(
        tf.keras.utils.get_file(origin='https://github.com/jbrownlee/Datasets/releases/download/Flickr8k/Flickr8k_text.zip', cache_dir='.', cache_subdir=path, extract=True), imports='import concurrent.futures;import collections;import hashlib;import itertools;import urllib.request;import tensorflow_hub as hub;import os;import pandas as pd;import einops;import tqdm;import numpy as np;import math;import requests;import pathlib;import tensorflow_text as text;import tensorflow as tf;import random;import tensorflow_datasets as tfds;from PIL import Image;import dataclasses;import string;import matplotlib.pyplot as plt;import json;import time;import re', function_to_run='tf.keras.utils.get_file(**kwargs)', method_object=None, object_signature=None, function_args=[], function_kwargs={'origin': eval("'https://github.com/jbrownlee/Datasets/releases/download/Flickr8k/Flickr8k_text.zip'"), 'cache_dir': eval("'.'"), 'cache_subdir': eval('path'), 'extract': eval('True')}, max_wait_secs=0)
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
    train_ds = custom_method(
    tf.data.experimental.from_list(train_captions), imports='import concurrent.futures;import collections;import hashlib;import itertools;import urllib.request;import tensorflow_hub as hub;import os;import pandas as pd;import einops;import tqdm;import numpy as np;import math;import requests;import pathlib;import tensorflow_text as text;import tensorflow as tf;import random;import tensorflow_datasets as tfds;from PIL import Image;import dataclasses;import string;import matplotlib.pyplot as plt;import json;import time;import re', function_to_run='tf.data.experimental.from_list(*args)', method_object=None, object_signature=None, function_args=[eval('train_captions')], function_kwargs={}, max_wait_secs=0)
    test_ds = custom_method(
    tf.data.experimental.from_list(test_captions), imports='import concurrent.futures;import collections;import hashlib;import itertools;import urllib.request;import tensorflow_hub as hub;import os;import pandas as pd;import einops;import tqdm;import numpy as np;import math;import requests;import pathlib;import tensorflow_text as text;import tensorflow as tf;import random;import tensorflow_datasets as tfds;from PIL import Image;import dataclasses;import string;import matplotlib.pyplot as plt;import json;import time;import re', function_to_run='tf.data.experimental.from_list(*args)', method_object=None, object_signature=None, function_args=[eval('test_captions')], function_kwargs={}, max_wait_secs=0)
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
            custom_method(
            result.append(file_path), imports='import concurrent.futures;import collections;import hashlib;import itertools;import urllib.request;import tensorflow_hub as hub;import os;import pandas as pd;import einops;import tqdm;import numpy as np;import math;import requests;import pathlib;import tensorflow_text as text;import tensorflow as tf;import random;import tensorflow_datasets as tfds;from PIL import Image;import dataclasses;import string;import matplotlib.pyplot as plt;import json;import time;import re', function_to_run='obj.append(*args)', method_object=eval('result'), object_signature='tf.strings.reduce_join', function_args=[eval('file_path')], function_kwargs={}, max_wait_secs=0, custom_class=None)
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
        ds = custom_method(
        tf.data.Dataset.from_tensor_slices((new_paths, new_captions)), imports='import concurrent.futures;import collections;import hashlib;import itertools;import urllib.request;import tensorflow_hub as hub;import os;import pandas as pd;import einops;import tqdm;import numpy as np;import math;import requests;import pathlib;import tensorflow_text as text;import tensorflow as tf;import random;import tensorflow_datasets as tfds;from PIL import Image;import dataclasses;import string;import matplotlib.pyplot as plt;import json;import time;import re', function_to_run='tf.data.Dataset.from_tensor_slices(*args)', method_object=None, object_signature=None, function_args=[eval('(new_paths, new_captions)')], function_kwargs={}, max_wait_secs=0)
        ds = custom_method(
        ds.map(lambda path, cap: (path, cap[tf.newaxis])), imports='import concurrent.futures;import collections;import hashlib;import itertools;import urllib.request;import tensorflow_hub as hub;import os;import pandas as pd;import einops;import tqdm;import numpy as np;import math;import requests;import pathlib;import tensorflow_text as text;import tensorflow as tf;import random;import tensorflow_datasets as tfds;from PIL import Image;import dataclasses;import string;import matplotlib.pyplot as plt;import json;import time;import re', function_to_run='obj.map(*args)', method_object=eval('ds'), object_signature='tf.data.Dataset.from_tensor_slices', function_args=[eval('lambda path,cap: (path, cap[tf.newaxis])')], function_kwargs={}, max_wait_secs=0, custom_class=None)
        return ds
    data_dir = pathlib.Path(data_dir)
    train_index_path = custom_method(
    tf.keras.utils.get_file(origin='https://storage.googleapis.com/gcc-data/Train/GCC-training.tsv', cache_subdir=data_dir, cache_dir='.'), imports='import concurrent.futures;import collections;import hashlib;import itertools;import urllib.request;import tensorflow_hub as hub;import os;import pandas as pd;import einops;import tqdm;import numpy as np;import math;import requests;import pathlib;import tensorflow_text as text;import tensorflow as tf;import random;import tensorflow_datasets as tfds;from PIL import Image;import dataclasses;import string;import matplotlib.pyplot as plt;import json;import time;import re', function_to_run='tf.keras.utils.get_file(**kwargs)', method_object=None, object_signature=None, function_args=[], function_kwargs={'origin': eval("'https://storage.googleapis.com/gcc-data/Train/GCC-training.tsv'"), 'cache_subdir': eval('data_dir'), 'cache_dir': eval("'.'")}, max_wait_secs=0)
    val_index_path = custom_method(
    tf.keras.utils.get_file(origin='https://storage.googleapis.com/gcc-data/Validation/GCC-1.1.0-Validation.tsv', cache_subdir=data_dir, cache_dir='.'), imports='import concurrent.futures;import collections;import hashlib;import itertools;import urllib.request;import tensorflow_hub as hub;import os;import pandas as pd;import einops;import tqdm;import numpy as np;import math;import requests;import pathlib;import tensorflow_text as text;import tensorflow as tf;import random;import tensorflow_datasets as tfds;from PIL import Image;import dataclasses;import string;import matplotlib.pyplot as plt;import json;import time;import re', function_to_run='tf.keras.utils.get_file(**kwargs)', method_object=None, object_signature=None, function_args=[], function_kwargs={'origin': eval("'https://storage.googleapis.com/gcc-data/Validation/GCC-1.1.0-Validation.tsv'"), 'cache_subdir': eval('data_dir'), 'cache_dir': eval("'.'")}, max_wait_secs=0)
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
mobilenet = custom_method(
tf.keras.applications.MobileNetV3Small(input_shape=IMAGE_SHAPE, include_top=False, include_preprocessing=True), imports='import concurrent.futures;import collections;import hashlib;import itertools;import urllib.request;import tensorflow_hub as hub;import os;import pandas as pd;import einops;import tqdm;import numpy as np;import math;import requests;import pathlib;import tensorflow_text as text;import tensorflow as tf;import random;import tensorflow_datasets as tfds;from PIL import Image;import dataclasses;import string;import matplotlib.pyplot as plt;import json;import time;import re', function_to_run='tf.keras.applications.MobileNetV3Small(**kwargs)', method_object=None, object_signature=None, function_args=[], function_kwargs={'input_shape': eval('IMAGE_SHAPE'), 'include_top': eval('False'), 'include_preprocessing': eval('True')}, max_wait_secs=0)
mobilenet.trainable = False

def load_image(image_path):
    img = custom_method(
    tf.io.read_file(image_path), imports='import concurrent.futures;import collections;import hashlib;import itertools;import urllib.request;import tensorflow_hub as hub;import os;import pandas as pd;import einops;import tqdm;import numpy as np;import math;import requests;import pathlib;import tensorflow_text as text;import tensorflow as tf;import random;import tensorflow_datasets as tfds;from PIL import Image;import dataclasses;import string;import matplotlib.pyplot as plt;import json;import time;import re', function_to_run='tf.io.read_file(*args)', method_object=None, object_signature=None, function_args=[eval('image_path')], function_kwargs={}, max_wait_secs=0)
    img = custom_method(
    tf.io.decode_jpeg(img, channels=3), imports='import concurrent.futures;import collections;import hashlib;import itertools;import urllib.request;import tensorflow_hub as hub;import os;import pandas as pd;import einops;import tqdm;import numpy as np;import math;import requests;import pathlib;import tensorflow_text as text;import tensorflow as tf;import random;import tensorflow_datasets as tfds;from PIL import Image;import dataclasses;import string;import matplotlib.pyplot as plt;import json;import time;import re', function_to_run='tf.io.decode_jpeg(*args, **kwargs)', method_object=None, object_signature=None, function_args=[eval('img')], function_kwargs={'channels': eval('3')}, max_wait_secs=0)
    img = custom_method(
    tf.image.resize(img, IMAGE_SHAPE[:-1]), imports='import concurrent.futures;import collections;import hashlib;import itertools;import urllib.request;import tensorflow_hub as hub;import os;import pandas as pd;import einops;import tqdm;import numpy as np;import math;import requests;import pathlib;import tensorflow_text as text;import tensorflow as tf;import random;import tensorflow_datasets as tfds;from PIL import Image;import dataclasses;import string;import matplotlib.pyplot as plt;import json;import time;import re', function_to_run='tf.image.resize(*args)', method_object=None, object_signature=None, function_args=[eval('img'), eval('IMAGE_SHAPE[:-1]')], function_kwargs={}, max_wait_secs=0)
    return img
test_img_batch = load_image(ex_path)[tf.newaxis, :]
print(test_img_batch.shape)
print(mobilenet(test_img_batch).shape)

def standardize(s):
    s = custom_method(
    tf.strings.lower(s), imports='import concurrent.futures;import collections;import hashlib;import itertools;import urllib.request;import tensorflow_hub as hub;import os;import pandas as pd;import einops;import tqdm;import numpy as np;import math;import requests;import pathlib;import tensorflow_text as text;import tensorflow as tf;import random;import tensorflow_datasets as tfds;from PIL import Image;import dataclasses;import string;import matplotlib.pyplot as plt;import json;import time;import re', function_to_run='tf.strings.lower(*args)', method_object=None, object_signature=None, function_args=[eval('s')], function_kwargs={}, max_wait_secs=0)
    s = custom_method(
    tf.strings.regex_replace(s, f'[{re.escape(string.punctuation)}]', ''), imports='import concurrent.futures;import collections;import hashlib;import itertools;import urllib.request;import tensorflow_hub as hub;import os;import pandas as pd;import einops;import tqdm;import numpy as np;import math;import requests;import pathlib;import tensorflow_text as text;import tensorflow as tf;import random;import tensorflow_datasets as tfds;from PIL import Image;import dataclasses;import string;import matplotlib.pyplot as plt;import json;import time;import re', function_to_run='tf.strings.regex_replace(*args)', method_object=None, object_signature=None, function_args=[eval('s'), eval("f'[{re.escape(string.punctuation)}]'"), eval("''")], function_kwargs={}, max_wait_secs=0)
    s = custom_method(
    tf.strings.join(['[START]', s, '[END]'], separator=' '), imports='import concurrent.futures;import collections;import hashlib;import itertools;import urllib.request;import tensorflow_hub as hub;import os;import pandas as pd;import einops;import tqdm;import numpy as np;import math;import requests;import pathlib;import tensorflow_text as text;import tensorflow as tf;import random;import tensorflow_datasets as tfds;from PIL import Image;import dataclasses;import string;import matplotlib.pyplot as plt;import json;import time;import re', function_to_run='tf.strings.join(*args, **kwargs)', method_object=None, object_signature=None, function_args=[eval("['[START]', s, '[END]']")], function_kwargs={'separator': eval("' '")}, max_wait_secs=0)
    return s
vocabulary_size = 5000
tokenizer = custom_method(
tf.keras.layers.TextVectorization(max_tokens=vocabulary_size, standardize=standardize, ragged=True), imports='import concurrent.futures;import collections;import hashlib;import itertools;import urllib.request;import tensorflow_hub as hub;import os;import pandas as pd;import einops;import tqdm;import numpy as np;import math;import requests;import pathlib;import tensorflow_text as text;import tensorflow as tf;import random;import tensorflow_datasets as tfds;from PIL import Image;import dataclasses;import string;import matplotlib.pyplot as plt;import json;import time;import re', function_to_run='tf.keras.layers.TextVectorization(**kwargs)', method_object=None, object_signature=None, function_args=[], function_kwargs={'max_tokens': eval('vocabulary_size'), 'standardize': eval('standardize'), 'ragged': eval('True')}, max_wait_secs=0)
custom_method(
tokenizer.adapt(train_raw.map(lambda fp, txt: txt).unbatch().batch(1024)), imports='import concurrent.futures;import collections;import hashlib;import itertools;import urllib.request;import tensorflow_hub as hub;import os;import pandas as pd;import einops;import tqdm;import numpy as np;import math;import requests;import pathlib;import tensorflow_text as text;import tensorflow as tf;import random;import tensorflow_datasets as tfds;from PIL import Image;import dataclasses;import string;import matplotlib.pyplot as plt;import json;import time;import re', function_to_run='obj.adapt(*args)', method_object=eval('tokenizer'), object_signature='tf.keras.layers.TextVectorization', function_args=[eval('train_raw.map(lambda fp,txt: txt).unbatch().batch(1024)')], function_kwargs={}, max_wait_secs=0, custom_class=None)
custom_method(
tokenizer.get_vocabulary(), imports='import concurrent.futures;import collections;import hashlib;import itertools;import urllib.request;import tensorflow_hub as hub;import os;import pandas as pd;import einops;import tqdm;import numpy as np;import math;import requests;import pathlib;import tensorflow_text as text;import tensorflow as tf;import random;import tensorflow_datasets as tfds;from PIL import Image;import dataclasses;import string;import matplotlib.pyplot as plt;import json;import time;import re', function_to_run='obj.get_vocabulary()', method_object=eval('tokenizer'), object_signature='tf.keras.layers.TextVectorization', function_args=[], function_kwargs={}, max_wait_secs=0, custom_class=None)[:10]
t = custom_method(
tokenizer([['a cat in a hat'], ['a robot dog']]), imports='import concurrent.futures;import collections;import hashlib;import itertools;import urllib.request;import tensorflow_hub as hub;import os;import pandas as pd;import einops;import tqdm;import numpy as np;import math;import requests;import pathlib;import tensorflow_text as text;import tensorflow as tf;import random;import tensorflow_datasets as tfds;from PIL import Image;import dataclasses;import string;import matplotlib.pyplot as plt;import json;import time;import re', function_to_run='obj(*args)', method_object=eval('tokenizer'), object_signature='tf.keras.layers.TextVectorization', function_args=[eval("[['a cat in a hat'], ['a robot dog']]")], function_kwargs={}, max_wait_secs=0, custom_class=None)
t
word_to_index = custom_method(
tf.keras.layers.StringLookup(mask_token='', vocabulary=tokenizer.get_vocabulary()), imports='import concurrent.futures;import collections;import hashlib;import itertools;import urllib.request;import tensorflow_hub as hub;import os;import pandas as pd;import einops;import tqdm;import numpy as np;import math;import requests;import pathlib;import tensorflow_text as text;import tensorflow as tf;import random;import tensorflow_datasets as tfds;from PIL import Image;import dataclasses;import string;import matplotlib.pyplot as plt;import json;import time;import re', function_to_run='tf.keras.layers.StringLookup(**kwargs)', method_object=None, object_signature=None, function_args=[], function_kwargs={'mask_token': eval('""'), 'vocabulary': eval('tokenizer.get_vocabulary()')}, max_wait_secs=0)
index_to_word = custom_method(
tf.keras.layers.StringLookup(mask_token='', vocabulary=tokenizer.get_vocabulary(), invert=True), imports='import concurrent.futures;import collections;import hashlib;import itertools;import urllib.request;import tensorflow_hub as hub;import os;import pandas as pd;import einops;import tqdm;import numpy as np;import math;import requests;import pathlib;import tensorflow_text as text;import tensorflow as tf;import random;import tensorflow_datasets as tfds;from PIL import Image;import dataclasses;import string;import matplotlib.pyplot as plt;import json;import time;import re', function_to_run='tf.keras.layers.StringLookup(**kwargs)', method_object=None, object_signature=None, function_args=[], function_kwargs={'mask_token': eval('""'), 'vocabulary': eval('tokenizer.get_vocabulary()'), 'invert': eval('True')}, max_wait_secs=0)
w = custom_method(
index_to_word(t), imports='import concurrent.futures;import collections;import hashlib;import itertools;import urllib.request;import tensorflow_hub as hub;import os;import pandas as pd;import einops;import tqdm;import numpy as np;import math;import requests;import pathlib;import tensorflow_text as text;import tensorflow as tf;import random;import tensorflow_datasets as tfds;from PIL import Image;import dataclasses;import string;import matplotlib.pyplot as plt;import json;import time;import re', function_to_run='obj(*args)', method_object=eval('index_to_word'), object_signature='tf.keras.layers.StringLookup', function_args=[eval('t')], function_kwargs={}, max_wait_secs=0, custom_class=None)
w.to_list()
custom_method(
tf.strings.reduce_join(w, separator=' ', axis=-1).numpy(), imports='import concurrent.futures;import collections;import hashlib;import itertools;import urllib.request;import tensorflow_hub as hub;import os;import pandas as pd;import einops;import tqdm;import numpy as np;import math;import requests;import pathlib;import tensorflow_text as text;import tensorflow as tf;import random;import tensorflow_datasets as tfds;from PIL import Image;import dataclasses;import string;import matplotlib.pyplot as plt;import json;import time;import re', function_to_run="tf.strings.reduce_join(w, separator=' ', axis=-1).numpy()", method_object=None, object_signature=None, function_args=[], function_kwargs={}, max_wait_secs=0)

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
    tokens = custom_method(
    tokenizer(txts), imports='import concurrent.futures;import collections;import hashlib;import itertools;import urllib.request;import tensorflow_hub as hub;import os;import pandas as pd;import einops;import tqdm;import numpy as np;import math;import requests;import pathlib;import tensorflow_text as text;import tensorflow as tf;import random;import tensorflow_datasets as tfds;from PIL import Image;import dataclasses;import string;import matplotlib.pyplot as plt;import json;import time;import re', function_to_run='obj(*args)', method_object=eval('tokenizer'), object_signature='tf.keras.layers.TextVectorization', function_args=[eval('txts')], function_kwargs={}, max_wait_secs=0, custom_class=None)
    input_tokens = tokens[..., :-1]
    label_tokens = tokens[..., 1:]
    return ((imgs, input_tokens), label_tokens)

def prepare_dataset(ds, tokenizer, batch_size=32, shuffle_buffer=1000):
    ds = custom_method(
    ds.shuffle(10000).map(lambda path, caption: (load_image(path), caption)).apply(tf.data.experimental.ignore_errors()).batch(batch_size), imports='import concurrent.futures;import collections;import hashlib;import itertools;import urllib.request;import tensorflow_hub as hub;import os;import pandas as pd;import einops;import tqdm;import numpy as np;import math;import requests;import pathlib;import tensorflow_text as text;import tensorflow as tf;import random;import tensorflow_datasets as tfds;from PIL import Image;import dataclasses;import string;import matplotlib.pyplot as plt;import json;import time;import re', function_to_run='ds.shuffle(10000).map(lambda path, caption: (load_image(path), caption)).apply(tf.data.experimental.ignore_errors()).batch(*args)', method_object=None, object_signature=None, function_args=[eval('batch_size')], function_kwargs={}, max_wait_secs=0)

    def to_tensor(inputs, labels):
        ((images, in_tok), out_tok) = (inputs, labels)
        return ((images, in_tok.to_tensor()), out_tok.to_tensor())
    return custom_method(
    ds.map(match_shapes, tf.data.AUTOTUNE).unbatch().shuffle(shuffle_buffer).batch(batch_size).map(prepare_txt, tf.data.AUTOTUNE).map(to_tensor, tf.data.AUTOTUNE), imports='import concurrent.futures;import collections;import hashlib;import itertools;import urllib.request;import tensorflow_hub as hub;import os;import pandas as pd;import einops;import tqdm;import numpy as np;import math;import requests;import pathlib;import tensorflow_text as text;import tensorflow as tf;import random;import tensorflow_datasets as tfds;from PIL import Image;import dataclasses;import string;import matplotlib.pyplot as plt;import json;import time;import re', function_to_run='ds.map(match_shapes, tf.data.AUTOTUNE).unbatch().shuffle(shuffle_buffer).batch(batch_size).map(prepare_txt, tf.data.AUTOTUNE).map(*args)', method_object=None, object_signature=None, function_args=[eval('to_tensor'), eval('tf.data.AUTOTUNE')], function_kwargs={}, max_wait_secs=0)
train_ds = prepare_dataset(train_raw, tokenizer)
train_ds.element_spec
test_ds = prepare_dataset(test_raw, tokenizer)
test_ds.element_spec

def save_dataset(ds, save_path, image_model, tokenizer, shards=10, batch_size=32):
    ds = custom_method(
    ds.map(lambda path, caption: (load_image(path), caption)).apply(tf.data.experimental.ignore_errors()).batch(batch_size), imports='import concurrent.futures;import collections;import hashlib;import itertools;import urllib.request;import tensorflow_hub as hub;import os;import pandas as pd;import einops;import tqdm;import numpy as np;import math;import requests;import pathlib;import tensorflow_text as text;import tensorflow as tf;import random;import tensorflow_datasets as tfds;from PIL import Image;import dataclasses;import string;import matplotlib.pyplot as plt;import json;import time;import re', function_to_run='ds.map(lambda path, caption: (load_image(path), caption)).apply(tf.data.experimental.ignore_errors()).batch(*args)', method_object=None, object_signature=None, function_args=[eval('batch_size')], function_kwargs={}, max_wait_secs=0)

    def gen():
        for (images, captions) in tqdm.tqdm(ds):
            feature_maps = image_model(images)
            (feature_maps, captions) = match_shapes(feature_maps, captions)
            yield (feature_maps, captions)
    new_ds = custom_method(
    tf.data.Dataset.from_generator(gen, output_signature=(tf.TensorSpec(shape=image_model.output_shape), tf.TensorSpec(shape=(None,), dtype=tf.string))), imports='import concurrent.futures;import collections;import hashlib;import itertools;import urllib.request;import tensorflow_hub as hub;import os;import pandas as pd;import einops;import tqdm;import numpy as np;import math;import requests;import pathlib;import tensorflow_text as text;import tensorflow as tf;import random;import tensorflow_datasets as tfds;from PIL import Image;import dataclasses;import string;import matplotlib.pyplot as plt;import json;import time;import re', function_to_run='tf.data.Dataset.from_generator(*args, **kwargs)', method_object=None, object_signature=None, function_args=[eval('gen')], function_kwargs={'output_signature': eval('(\n          tf.TensorSpec(shape=image_model.output_shape),\n          tf.TensorSpec(shape=(None,), dtype=tf.string))')}, max_wait_secs=0)
    new_ds = custom_method(
    new_ds.map(prepare_txt, tf.data.AUTOTUNE).unbatch().shuffle(1000), imports='import concurrent.futures;import collections;import hashlib;import itertools;import urllib.request;import tensorflow_hub as hub;import os;import pandas as pd;import einops;import tqdm;import numpy as np;import math;import requests;import pathlib;import tensorflow_text as text;import tensorflow as tf;import random;import tensorflow_datasets as tfds;from PIL import Image;import dataclasses;import string;import matplotlib.pyplot as plt;import json;import time;import re', function_to_run='new_ds.map(prepare_txt, tf.data.AUTOTUNE).unbatch().shuffle(*args)', method_object=None, object_signature=None, function_args=[eval('1000')], function_kwargs={}, max_wait_secs=0)

    def shard_func(i, item):
        return i % shards
    custom_method(
    new_ds.enumerate().save(save_path, shard_func=shard_func), imports='import concurrent.futures;import collections;import hashlib;import itertools;import urllib.request;import tensorflow_hub as hub;import os;import pandas as pd;import einops;import tqdm;import numpy as np;import math;import requests;import pathlib;import tensorflow_text as text;import tensorflow as tf;import random;import tensorflow_datasets as tfds;from PIL import Image;import dataclasses;import string;import matplotlib.pyplot as plt;import json;import time;import re', function_to_run='obj.enumerate().save(*args, **kwargs)', method_object=eval('new_ds'), object_signature='tf.data.Dataset.from_generator', function_args=[eval('save_path')], function_kwargs={'shard_func': eval('shard_func')}, max_wait_secs=0, custom_class=None)

def load_dataset(save_path, batch_size=32, shuffle=1000, cycle_length=2):

    def custom_reader_func(datasets):
        datasets = datasets.shuffle(1000)
        return datasets.interleave(lambda x: x, cycle_length=cycle_length)
    ds = custom_method(
    tf.data.Dataset.load(save_path, reader_func=custom_reader_func), imports='import concurrent.futures;import collections;import hashlib;import itertools;import urllib.request;import tensorflow_hub as hub;import os;import pandas as pd;import einops;import tqdm;import numpy as np;import math;import requests;import pathlib;import tensorflow_text as text;import tensorflow as tf;import random;import tensorflow_datasets as tfds;from PIL import Image;import dataclasses;import string;import matplotlib.pyplot as plt;import json;import time;import re', function_to_run='tf.data.Dataset.load(*args, **kwargs)', method_object=None, object_signature=None, function_args=[eval('save_path')], function_kwargs={'reader_func': eval('custom_reader_func')}, max_wait_secs=0)

    def drop_index(i, x):
        return x
    ds = custom_method(
    ds.map(drop_index, tf.data.AUTOTUNE).shuffle(shuffle).padded_batch(batch_size).prefetch(tf.data.AUTOTUNE), imports='import concurrent.futures;import collections;import hashlib;import itertools;import urllib.request;import tensorflow_hub as hub;import os;import pandas as pd;import einops;import tqdm;import numpy as np;import math;import requests;import pathlib;import tensorflow_text as text;import tensorflow as tf;import random;import tensorflow_datasets as tfds;from PIL import Image;import dataclasses;import string;import matplotlib.pyplot as plt;import json;import time;import re', function_to_run='ds.map(drop_index, tf.data.AUTOTUNE).shuffle(shuffle).padded_batch(batch_size).prefetch(*args)', method_object=None, object_signature=None, function_args=[eval('tf.data.AUTOTUNE')], function_kwargs={}, max_wait_secs=0)
    return ds
save_dataset(train_raw, 'train_cache', mobilenet, tokenizer)
save_dataset(test_raw, 'test_cache', mobilenet, tokenizer)
train_ds = load_dataset('train_cache')
test_ds = load_dataset('test_cache')
train_ds.element_spec
for (inputs, ex_labels) in custom_method(
train_ds.take(1), imports='import concurrent.futures;import collections;import hashlib;import itertools;import urllib.request;import tensorflow_hub as hub;import os;import pandas as pd;import einops;import tqdm;import numpy as np;import math;import requests;import pathlib;import tensorflow_text as text;import tensorflow as tf;import random;import tensorflow_datasets as tfds;from PIL import Image;import dataclasses;import string;import matplotlib.pyplot as plt;import json;import time;import re', function_to_run='obj.take(*args)', method_object=eval('train_ds'), object_signature='tf.data.experimental.from_list', function_args=[eval('1')], function_kwargs={}, max_wait_secs=0, custom_class=None):
    (ex_img, ex_in_tok) = inputs
print(ex_img.shape)
print(ex_in_tok.shape)
print(ex_labels.shape)
print(ex_in_tok[0].numpy())
print(ex_labels[0].numpy())

class SeqEmbedding(tf.keras.layers.Layer):

    def __init__(self, vocab_size, max_length, depth):
        super().__init__()
        self.pos_embedding = custom_method(
        tf.keras.layers.Embedding(input_dim=max_length, output_dim=depth), imports='import concurrent.futures;import collections;import hashlib;import itertools;import urllib.request;import tensorflow_hub as hub;import os;import pandas as pd;import einops;import tqdm;import numpy as np;import math;import requests;import pathlib;import tensorflow_text as text;import tensorflow as tf;import random;import tensorflow_datasets as tfds;from PIL import Image;import dataclasses;import string;import matplotlib.pyplot as plt;import json;import time;import re', function_to_run='tf.keras.layers.Embedding(**kwargs)', method_object=None, object_signature=None, function_args=[], function_kwargs={'input_dim': eval('max_length'), 'output_dim': eval('depth')}, max_wait_secs=0)
        self.token_embedding = custom_method(
        tf.keras.layers.Embedding(input_dim=vocab_size, output_dim=depth, mask_zero=True), imports='import concurrent.futures;import collections;import hashlib;import itertools;import urllib.request;import tensorflow_hub as hub;import os;import pandas as pd;import einops;import tqdm;import numpy as np;import math;import requests;import pathlib;import tensorflow_text as text;import tensorflow as tf;import random;import tensorflow_datasets as tfds;from PIL import Image;import dataclasses;import string;import matplotlib.pyplot as plt;import json;import time;import re', function_to_run='tf.keras.layers.Embedding(**kwargs)', method_object=None, object_signature=None, function_args=[], function_kwargs={'input_dim': eval('vocab_size'), 'output_dim': eval('depth'), 'mask_zero': eval('True')}, max_wait_secs=0)
        self.add = custom_method(
        tf.keras.layers.Add(), imports='import concurrent.futures;import collections;import hashlib;import itertools;import urllib.request;import tensorflow_hub as hub;import os;import pandas as pd;import einops;import tqdm;import numpy as np;import math;import requests;import pathlib;import tensorflow_text as text;import tensorflow as tf;import random;import tensorflow_datasets as tfds;from PIL import Image;import dataclasses;import string;import matplotlib.pyplot as plt;import json;import time;import re', function_to_run='tf.keras.layers.Add()', method_object=None, object_signature=None, function_args=[], function_kwargs={}, max_wait_secs=0)

    def call(self, seq):
        seq = self.token_embedding(seq)
        x = custom_method(
        tf.range(tf.shape(seq)[1]), imports='import concurrent.futures;import collections;import hashlib;import itertools;import urllib.request;import tensorflow_hub as hub;import os;import pandas as pd;import einops;import tqdm;import numpy as np;import math;import requests;import pathlib;import tensorflow_text as text;import tensorflow as tf;import random;import tensorflow_datasets as tfds;from PIL import Image;import dataclasses;import string;import matplotlib.pyplot as plt;import json;import time;import re', function_to_run='tf.range(*args)', method_object=None, object_signature=None, function_args=[eval('tf.shape(seq)[1]')], function_kwargs={}, max_wait_secs=0)
        x = x[tf.newaxis, :]
        x = self.pos_embedding(x)
        return self.add([seq, x])

class CausalSelfAttention(tf.keras.layers.Layer):

    def __init__(self, **kwargs):
        super().__init__()
        self.mha = custom_method(
        tf.keras.layers.MultiHeadAttention(**kwargs), imports='import concurrent.futures;import collections;import hashlib;import itertools;import urllib.request;import tensorflow_hub as hub;import os;import pandas as pd;import einops;import tqdm;import numpy as np;import math;import requests;import pathlib;import tensorflow_text as text;import tensorflow as tf;import random;import tensorflow_datasets as tfds;from PIL import Image;import dataclasses;import string;import matplotlib.pyplot as plt;import json;import time;import re', function_to_run='tf.keras.layers.MultiHeadAttention(**kwargs)', method_object=None, object_signature=None, function_args=[], function_kwargs={None: eval('kwargs')}, max_wait_secs=0)
        self.add = custom_method(
        tf.keras.layers.Add(), imports='import concurrent.futures;import collections;import hashlib;import itertools;import urllib.request;import tensorflow_hub as hub;import os;import pandas as pd;import einops;import tqdm;import numpy as np;import math;import requests;import pathlib;import tensorflow_text as text;import tensorflow as tf;import random;import tensorflow_datasets as tfds;from PIL import Image;import dataclasses;import string;import matplotlib.pyplot as plt;import json;import time;import re', function_to_run='tf.keras.layers.Add()', method_object=None, object_signature=None, function_args=[], function_kwargs={}, max_wait_secs=0)
        self.layernorm = custom_method(
        tf.keras.layers.LayerNormalization(), imports='import concurrent.futures;import collections;import hashlib;import itertools;import urllib.request;import tensorflow_hub as hub;import os;import pandas as pd;import einops;import tqdm;import numpy as np;import math;import requests;import pathlib;import tensorflow_text as text;import tensorflow as tf;import random;import tensorflow_datasets as tfds;from PIL import Image;import dataclasses;import string;import matplotlib.pyplot as plt;import json;import time;import re', function_to_run='tf.keras.layers.LayerNormalization()', method_object=None, object_signature=None, function_args=[], function_kwargs={}, max_wait_secs=0)

    def call(self, x):
        attn = self.mha(query=x, value=x, use_causal_mask=True)
        x = self.add([x, attn])
        return self.layernorm(x)

class CrossAttention(tf.keras.layers.Layer):

    def __init__(self, **kwargs):
        super().__init__()
        self.mha = custom_method(
        tf.keras.layers.MultiHeadAttention(**kwargs), imports='import concurrent.futures;import collections;import hashlib;import itertools;import urllib.request;import tensorflow_hub as hub;import os;import pandas as pd;import einops;import tqdm;import numpy as np;import math;import requests;import pathlib;import tensorflow_text as text;import tensorflow as tf;import random;import tensorflow_datasets as tfds;from PIL import Image;import dataclasses;import string;import matplotlib.pyplot as plt;import json;import time;import re', function_to_run='tf.keras.layers.MultiHeadAttention(**kwargs)', method_object=None, object_signature=None, function_args=[], function_kwargs={None: eval('kwargs')}, max_wait_secs=0)
        self.add = custom_method(
        tf.keras.layers.Add(), imports='import concurrent.futures;import collections;import hashlib;import itertools;import urllib.request;import tensorflow_hub as hub;import os;import pandas as pd;import einops;import tqdm;import numpy as np;import math;import requests;import pathlib;import tensorflow_text as text;import tensorflow as tf;import random;import tensorflow_datasets as tfds;from PIL import Image;import dataclasses;import string;import matplotlib.pyplot as plt;import json;import time;import re', function_to_run='tf.keras.layers.Add()', method_object=None, object_signature=None, function_args=[], function_kwargs={}, max_wait_secs=0)
        self.layernorm = custom_method(
        tf.keras.layers.LayerNormalization(), imports='import concurrent.futures;import collections;import hashlib;import itertools;import urllib.request;import tensorflow_hub as hub;import os;import pandas as pd;import einops;import tqdm;import numpy as np;import math;import requests;import pathlib;import tensorflow_text as text;import tensorflow as tf;import random;import tensorflow_datasets as tfds;from PIL import Image;import dataclasses;import string;import matplotlib.pyplot as plt;import json;import time;import re', function_to_run='tf.keras.layers.LayerNormalization()', method_object=None, object_signature=None, function_args=[], function_kwargs={}, max_wait_secs=0)

    def call(self, x, y, **kwargs):
        (attn, attention_scores) = self.mha(query=x, value=y, return_attention_scores=True)
        self.last_attention_scores = attention_scores
        x = self.add([x, attn])
        return self.layernorm(x)

class FeedForward(tf.keras.layers.Layer):

    def __init__(self, units, dropout_rate=0.1):
        super().__init__()
        self.seq = custom_method(
        tf.keras.Sequential([tf.keras.layers.Dense(units=2 * units, activation='relu'), tf.keras.layers.Dense(units=units), tf.keras.layers.Dropout(rate=dropout_rate)]), imports='import concurrent.futures;import collections;import hashlib;import itertools;import urllib.request;import tensorflow_hub as hub;import os;import pandas as pd;import einops;import tqdm;import numpy as np;import math;import requests;import pathlib;import tensorflow_text as text;import tensorflow as tf;import random;import tensorflow_datasets as tfds;from PIL import Image;import dataclasses;import string;import matplotlib.pyplot as plt;import json;import time;import re', function_to_run='tf.keras.Sequential(*args)', method_object=None, object_signature=None, function_args=[eval("[\n        tf.keras.layers.Dense(units=2*units, activation='relu'),\n        tf.keras.layers.Dense(units=units),\n        tf.keras.layers.Dropout(rate=dropout_rate),\n    ]")], function_kwargs={}, max_wait_secs=0)
        self.layernorm = custom_method(
        tf.keras.layers.LayerNormalization(), imports='import concurrent.futures;import collections;import hashlib;import itertools;import urllib.request;import tensorflow_hub as hub;import os;import pandas as pd;import einops;import tqdm;import numpy as np;import math;import requests;import pathlib;import tensorflow_text as text;import tensorflow as tf;import random;import tensorflow_datasets as tfds;from PIL import Image;import dataclasses;import string;import matplotlib.pyplot as plt;import json;import time;import re', function_to_run='tf.keras.layers.LayerNormalization()', method_object=None, object_signature=None, function_args=[], function_kwargs={}, max_wait_secs=0)

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
        self.dense = custom_method(
        tf.keras.layers.Dense(units=tokenizer.vocabulary_size(), **kwargs), imports='import concurrent.futures;import collections;import hashlib;import itertools;import urllib.request;import tensorflow_hub as hub;import os;import pandas as pd;import einops;import tqdm;import numpy as np;import math;import requests;import pathlib;import tensorflow_text as text;import tensorflow as tf;import random;import tensorflow_datasets as tfds;from PIL import Image;import dataclasses;import string;import matplotlib.pyplot as plt;import json;import time;import re', function_to_run='tf.keras.layers.Dense(**kwargs)', method_object=None, object_signature=None, function_args=[], function_kwargs={'units': eval('tokenizer.vocabulary_size()'), None: eval('kwargs')}, max_wait_secs=0)
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
custom_method(
output_layer.adapt(train_ds.map(lambda inputs, labels: labels)), imports='import concurrent.futures;import collections;import hashlib;import itertools;import urllib.request;import tensorflow_hub as hub;import os;import pandas as pd;import einops;import tqdm;import numpy as np;import math;import requests;import pathlib;import tensorflow_text as text;import tensorflow as tf;import random;import tensorflow_datasets as tfds;from PIL import Image;import dataclasses;import string;import matplotlib.pyplot as plt;import json;import time;import re', function_to_run='obj.adapt(*args)', method_object=eval('output_layer'), object_signature='TokenOutput', function_args=[eval('train_ds.map(lambda inputs, labels: labels)')], function_kwargs={}, max_wait_secs=0, custom_class='class TokenOutput(tf.keras.layers.Layer):\n  def __init__(self, tokenizer, banned_tokens=(\'\', \'[UNK]\', \'[START]\'), **kwargs):\n    super().__init__()\n    \n    self.dense = tf.keras.layers.Dense(\n        units=tokenizer.vocabulary_size(), **kwargs)\n    self.tokenizer = tokenizer\n    self.banned_tokens = banned_tokens\n\n    self.bias = None\n\n  def adapt(self, ds):\n    counts = collections.Counter()\n    vocab_dict = {name: id \n                  for id, name in enumerate(self.tokenizer.get_vocabulary())}\n\n    for tokens in tqdm.tqdm(ds):\n      counts.update(tokens.numpy().flatten())\n\n    counts_arr = np.zeros(shape=(self.tokenizer.vocabulary_size(),))\n    counts_arr[np.array(list(counts.keys()), dtype=np.int32)] = list(counts.values())\n\n    counts_arr = counts_arr[:]\n    for token in self.banned_tokens:\n      counts_arr[vocab_dict[token]] = 0\n\n    total = counts_arr.sum()\n    p = counts_arr/total\n    p[counts_arr==0] = 1.0\n    log_p = np.log(p)  # log(1) == 0\n\n    entropy = -(log_p*p).sum()\n\n    print()\n    print(f"Uniform entropy: {np.log(self.tokenizer.vocabulary_size()):0.2f}")\n    print(f"Marginal entropy: {entropy:0.2f}")\n\n    self.bias = log_p\n    self.bias[counts_arr==0] = -1e9\n\n  def call(self, x):\n    x = self.dense(x)\n    return x + self.bias')

class Captioner(tf.keras.Model):

    @classmethod
    def add_method(cls, fun):
        setattr(cls, fun.__name__, fun)
        return fun

    def __init__(self, tokenizer, feature_extractor, output_layer, num_layers=1, units=256, max_length=50, num_heads=1, dropout_rate=0.1):
        super().__init__()
        self.feature_extractor = feature_extractor
        self.tokenizer = tokenizer
        self.word_to_index = custom_method(
        tf.keras.layers.StringLookup(mask_token='', vocabulary=tokenizer.get_vocabulary()), imports='import concurrent.futures;import collections;import hashlib;import itertools;import urllib.request;import tensorflow_hub as hub;import os;import pandas as pd;import einops;import tqdm;import numpy as np;import math;import requests;import pathlib;import tensorflow_text as text;import tensorflow as tf;import random;import tensorflow_datasets as tfds;from PIL import Image;import dataclasses;import string;import matplotlib.pyplot as plt;import json;import time;import re', function_to_run='tf.keras.layers.StringLookup(**kwargs)', method_object=None, object_signature=None, function_args=[], function_kwargs={'mask_token': eval('""'), 'vocabulary': eval('tokenizer.get_vocabulary()')}, max_wait_secs=0)
        self.index_to_word = custom_method(
        tf.keras.layers.StringLookup(mask_token='', vocabulary=tokenizer.get_vocabulary(), invert=True), imports='import concurrent.futures;import collections;import hashlib;import itertools;import urllib.request;import tensorflow_hub as hub;import os;import pandas as pd;import einops;import tqdm;import numpy as np;import math;import requests;import pathlib;import tensorflow_text as text;import tensorflow as tf;import random;import tensorflow_datasets as tfds;from PIL import Image;import dataclasses;import string;import matplotlib.pyplot as plt;import json;import time;import re', function_to_run='tf.keras.layers.StringLookup(**kwargs)', method_object=None, object_signature=None, function_args=[], function_kwargs={'mask_token': eval('""'), 'vocabulary': eval('tokenizer.get_vocabulary()'), 'invert': eval('True')}, max_wait_secs=0)
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
            txt = custom_method(
            tokenizer(txt), imports='import concurrent.futures;import collections;import hashlib;import itertools;import urllib.request;import tensorflow_hub as hub;import os;import pandas as pd;import einops;import tqdm;import numpy as np;import math;import requests;import pathlib;import tensorflow_text as text;import tensorflow as tf;import random;import tensorflow_datasets as tfds;from PIL import Image;import dataclasses;import string;import matplotlib.pyplot as plt;import json;import time;import re', function_to_run='obj(*args)', method_object=eval('tokenizer'), object_signature='tf.keras.layers.TextVectorization', function_args=[eval('txt')], function_kwargs={}, max_wait_secs=0, custom_class=None)
        txt = self.seq_embedding(txt)
        for dec_layer in self.decoder_layers:
            txt = dec_layer(inputs=(image, txt))
        txt = self.output_layer(txt)
        return txt
model = Captioner(tokenizer, feature_extractor=mobilenet, output_layer=output_layer, units=256, dropout_rate=0.5, num_layers=2, num_heads=2)
image_url = 'https://tensorflow.org/images/surf.jpg'
image_path = custom_method(
tf.keras.utils.get_file('surf.jpg', origin=image_url), imports='import concurrent.futures;import collections;import hashlib;import itertools;import urllib.request;import tensorflow_hub as hub;import os;import pandas as pd;import einops;import tqdm;import numpy as np;import math;import requests;import pathlib;import tensorflow_text as text;import tensorflow as tf;import random;import tensorflow_datasets as tfds;from PIL import Image;import dataclasses;import string;import matplotlib.pyplot as plt;import json;import time;import re', function_to_run='tf.keras.utils.get_file(*args, **kwargs)', method_object=None, object_signature=None, function_args=[eval("'surf.jpg'")], function_kwargs={'origin': eval('image_url')}, max_wait_secs=0)
image = load_image(image_path)

@Captioner.add_method
def simple_gen(self, image, temperature=1):
    initial = self.word_to_index([['[START]']])
    img_features = self.feature_extractor(image[tf.newaxis, ...])
    tokens = initial
    for n in range(50):
        preds = custom_method(
        self((img_features, tokens)).numpy(), imports='import concurrent.futures;import collections;import hashlib;import itertools;import urllib.request;import tensorflow_hub as hub;import os;import pandas as pd;import einops;import tqdm;import numpy as np;import math;import requests;import pathlib;import tensorflow_text as text;import tensorflow as tf;import random;import tensorflow_datasets as tfds;from PIL import Image;import dataclasses;import string;import matplotlib.pyplot as plt;import json;import time;import re', function_to_run='self((img_features, obj)).numpy()', method_object=eval('tokens'), object_signature='tf.concat', function_args=[], function_kwargs={}, max_wait_secs=0, custom_class=None)
        preds = preds[:, -1, :]
        if temperature == 0:
            next = custom_method(
            tf.argmax(preds, axis=-1), imports='import concurrent.futures;import collections;import hashlib;import itertools;import urllib.request;import tensorflow_hub as hub;import os;import pandas as pd;import einops;import tqdm;import numpy as np;import math;import requests;import pathlib;import tensorflow_text as text;import tensorflow as tf;import random;import tensorflow_datasets as tfds;from PIL import Image;import dataclasses;import string;import matplotlib.pyplot as plt;import json;import time;import re', function_to_run='tf.argmax(*args, **kwargs)', method_object=None, object_signature=None, function_args=[eval('preds')], function_kwargs={'axis': eval('-1')}, max_wait_secs=0)[:, tf.newaxis]
        else:
            next = custom_method(
            tf.random.categorical(preds / temperature, num_samples=1), imports='import concurrent.futures;import collections;import hashlib;import itertools;import urllib.request;import tensorflow_hub as hub;import os;import pandas as pd;import einops;import tqdm;import numpy as np;import math;import requests;import pathlib;import tensorflow_text as text;import tensorflow as tf;import random;import tensorflow_datasets as tfds;from PIL import Image;import dataclasses;import string;import matplotlib.pyplot as plt;import json;import time;import re', function_to_run='tf.random.categorical(*args, **kwargs)', method_object=None, object_signature=None, function_args=[eval('preds/temperature')], function_kwargs={'num_samples': eval('1')}, max_wait_secs=0)
        tokens = custom_method(
        tf.concat([tokens, next], axis=1), imports='import concurrent.futures;import collections;import hashlib;import itertools;import urllib.request;import tensorflow_hub as hub;import os;import pandas as pd;import einops;import tqdm;import numpy as np;import math;import requests;import pathlib;import tensorflow_text as text;import tensorflow as tf;import random;import tensorflow_datasets as tfds;from PIL import Image;import dataclasses;import string;import matplotlib.pyplot as plt;import json;import time;import re', function_to_run='tf.concat(*args, **kwargs)', method_object=None, object_signature=None, function_args=[eval('[tokens, next]')], function_kwargs={'axis': eval('1')}, max_wait_secs=0)
        if next[0] == self.word_to_index('[END]'):
            break
    words = custom_method(
    index_to_word(tokens[0, 1:-1]), imports='import concurrent.futures;import collections;import hashlib;import itertools;import urllib.request;import tensorflow_hub as hub;import os;import pandas as pd;import einops;import tqdm;import numpy as np;import math;import requests;import pathlib;import tensorflow_text as text;import tensorflow as tf;import random;import tensorflow_datasets as tfds;from PIL import Image;import dataclasses;import string;import matplotlib.pyplot as plt;import json;import time;import re', function_to_run='obj(*args)', method_object=eval('index_to_word'), object_signature='tf.keras.layers.StringLookup', function_args=[eval('tokens[0, 1:-1]')], function_kwargs={}, max_wait_secs=0, custom_class=None)
    result = custom_method(
    tf.strings.reduce_join(words, axis=-1, separator=' '), imports='import concurrent.futures;import collections;import hashlib;import itertools;import urllib.request;import tensorflow_hub as hub;import os;import pandas as pd;import einops;import tqdm;import numpy as np;import math;import requests;import pathlib;import tensorflow_text as text;import tensorflow as tf;import random;import tensorflow_datasets as tfds;from PIL import Image;import dataclasses;import string;import matplotlib.pyplot as plt;import json;import time;import re', function_to_run='tf.strings.reduce_join(*args, **kwargs)', method_object=None, object_signature=None, function_args=[eval('words')], function_kwargs={'axis': eval('-1'), 'separator': eval("' '")}, max_wait_secs=0)
    return custom_method(
    result.numpy().decode(), imports='import concurrent.futures;import collections;import hashlib;import itertools;import urllib.request;import tensorflow_hub as hub;import os;import pandas as pd;import einops;import tqdm;import numpy as np;import math;import requests;import pathlib;import tensorflow_text as text;import tensorflow as tf;import random;import tensorflow_datasets as tfds;from PIL import Image;import dataclasses;import string;import matplotlib.pyplot as plt;import json;import time;import re', function_to_run='obj.numpy().decode()', method_object=eval('result'), object_signature='tf.strings.reduce_join', function_args=[], function_kwargs={}, max_wait_secs=0, custom_class=None)
for t in (0.0, 0.5, 1.0):
    result = custom_method(
    model.simple_gen(image, temperature=t), imports='import concurrent.futures;import collections;import hashlib;import itertools;import urllib.request;import tensorflow_hub as hub;import os;import pandas as pd;import einops;import tqdm;import numpy as np;import math;import requests;import pathlib;import tensorflow_text as text;import tensorflow as tf;import random;import tensorflow_datasets as tfds;from PIL import Image;import dataclasses;import string;import matplotlib.pyplot as plt;import json;import time;import re', function_to_run='obj.simple_gen(*args, **kwargs)', method_object=eval('model'), object_signature='Captioner', function_args=[eval('image')], function_kwargs={'temperature': eval('t')}, max_wait_secs=0, custom_class='class Captioner(tf.keras.Model):\n  @classmethod\n  def add_method(cls, fun):\n    setattr(cls, fun.__name__, fun)\n    return fun\n\n  def __init__(self, tokenizer, feature_extractor, output_layer, num_layers=1,\n               units=256, max_length=50, num_heads=1, dropout_rate=0.1):\n    super().__init__()\n    self.feature_extractor = feature_extractor\n    self.tokenizer = tokenizer\n    self.word_to_index = tf.keras.layers.StringLookup(\n        mask_token="",\n        vocabulary=tokenizer.get_vocabulary())\n    self.index_to_word = tf.keras.layers.StringLookup(\n        mask_token="",\n        vocabulary=tokenizer.get_vocabulary(),\n        invert=True) \n\n    self.seq_embedding = SeqEmbedding(\n        vocab_size=tokenizer.vocabulary_size(),\n        depth=units,\n        max_length=max_length)\n\n    self.decoder_layers = [\n        DecoderLayer(units, num_heads=num_heads, dropout_rate=dropout_rate)\n        for n in range(num_layers)]\n\n    self.output_layer = output_layer\n  @Captioner.add_method\n  def call(self, inputs):\n    image, txt = inputs\n\n    if image.shape[-1] == 3:\n      image = self.feature_extractor(image)\n    \n    image = einops.rearrange(image, \'b h w c -> b (h w) c\')\n\n\n    if txt.dtype == tf.string:\n      txt = tokenizer(txt)\n\n    txt = self.seq_embedding(txt)\n\n    for dec_layer in self.decoder_layers:\n      txt = dec_layer(inputs=(image, txt))\n      \n    txt = self.output_layer(txt)\n\n    return txt')
    print(result)

def masked_loss(labels, preds):
    loss = custom_method(
    tf.nn.sparse_softmax_cross_entropy_with_logits(labels, preds), imports='import concurrent.futures;import collections;import hashlib;import itertools;import urllib.request;import tensorflow_hub as hub;import os;import pandas as pd;import einops;import tqdm;import numpy as np;import math;import requests;import pathlib;import tensorflow_text as text;import tensorflow as tf;import random;import tensorflow_datasets as tfds;from PIL import Image;import dataclasses;import string;import matplotlib.pyplot as plt;import json;import time;import re', function_to_run='tf.nn.sparse_softmax_cross_entropy_with_logits(*args)', method_object=None, object_signature=None, function_args=[eval('labels'), eval('preds')], function_kwargs={}, max_wait_secs=0)
    mask = (labels != 0) & (loss < 100000000.0)
    mask = custom_method(
    tf.cast(mask, loss.dtype), imports='import concurrent.futures;import collections;import hashlib;import itertools;import urllib.request;import tensorflow_hub as hub;import os;import pandas as pd;import einops;import tqdm;import numpy as np;import math;import requests;import pathlib;import tensorflow_text as text;import tensorflow as tf;import random;import tensorflow_datasets as tfds;from PIL import Image;import dataclasses;import string;import matplotlib.pyplot as plt;import json;import time;import re', function_to_run='tf.cast(*args)', method_object=None, object_signature=None, function_args=[eval('mask'), eval('loss.dtype')], function_kwargs={}, max_wait_secs=0)
    loss = loss * mask
    loss = custom_method(
    tf.reduce_sum(loss), imports='import concurrent.futures;import collections;import hashlib;import itertools;import urllib.request;import tensorflow_hub as hub;import os;import pandas as pd;import einops;import tqdm;import numpy as np;import math;import requests;import pathlib;import tensorflow_text as text;import tensorflow as tf;import random;import tensorflow_datasets as tfds;from PIL import Image;import dataclasses;import string;import matplotlib.pyplot as plt;import json;import time;import re', function_to_run='tf.reduce_sum(*args)', method_object=None, object_signature=None, function_args=[eval('loss')], function_kwargs={}, max_wait_secs=0) / custom_method(
    tf.reduce_sum(mask), imports='import concurrent.futures;import collections;import hashlib;import itertools;import urllib.request;import tensorflow_hub as hub;import os;import pandas as pd;import einops;import tqdm;import numpy as np;import math;import requests;import pathlib;import tensorflow_text as text;import tensorflow as tf;import random;import tensorflow_datasets as tfds;from PIL import Image;import dataclasses;import string;import matplotlib.pyplot as plt;import json;import time;import re', function_to_run='tf.reduce_sum(*args)', method_object=None, object_signature=None, function_args=[eval('mask')], function_kwargs={}, max_wait_secs=0)
    return loss

def masked_acc(labels, preds):
    mask = custom_method(
    tf.cast(labels != 0, tf.float32), imports='import concurrent.futures;import collections;import hashlib;import itertools;import urllib.request;import tensorflow_hub as hub;import os;import pandas as pd;import einops;import tqdm;import numpy as np;import math;import requests;import pathlib;import tensorflow_text as text;import tensorflow as tf;import random;import tensorflow_datasets as tfds;from PIL import Image;import dataclasses;import string;import matplotlib.pyplot as plt;import json;import time;import re', function_to_run='tf.cast(*args)', method_object=None, object_signature=None, function_args=[eval('labels!=0'), eval('tf.float32')], function_kwargs={}, max_wait_secs=0)
    preds = custom_method(
    tf.argmax(preds, axis=-1), imports='import concurrent.futures;import collections;import hashlib;import itertools;import urllib.request;import tensorflow_hub as hub;import os;import pandas as pd;import einops;import tqdm;import numpy as np;import math;import requests;import pathlib;import tensorflow_text as text;import tensorflow as tf;import random;import tensorflow_datasets as tfds;from PIL import Image;import dataclasses;import string;import matplotlib.pyplot as plt;import json;import time;import re', function_to_run='tf.argmax(*args, **kwargs)', method_object=None, object_signature=None, function_args=[eval('preds')], function_kwargs={'axis': eval('-1')}, max_wait_secs=0)
    labels = custom_method(
    tf.cast(labels, tf.int64), imports='import concurrent.futures;import collections;import hashlib;import itertools;import urllib.request;import tensorflow_hub as hub;import os;import pandas as pd;import einops;import tqdm;import numpy as np;import math;import requests;import pathlib;import tensorflow_text as text;import tensorflow as tf;import random;import tensorflow_datasets as tfds;from PIL import Image;import dataclasses;import string;import matplotlib.pyplot as plt;import json;import time;import re', function_to_run='tf.cast(*args)', method_object=None, object_signature=None, function_args=[eval('labels'), eval('tf.int64')], function_kwargs={}, max_wait_secs=0)
    match = custom_method(
    tf.cast(preds == labels, mask.dtype), imports='import concurrent.futures;import collections;import hashlib;import itertools;import urllib.request;import tensorflow_hub as hub;import os;import pandas as pd;import einops;import tqdm;import numpy as np;import math;import requests;import pathlib;import tensorflow_text as text;import tensorflow as tf;import random;import tensorflow_datasets as tfds;from PIL import Image;import dataclasses;import string;import matplotlib.pyplot as plt;import json;import time;import re', function_to_run='tf.cast(*args)', method_object=None, object_signature=None, function_args=[eval('preds == labels'), eval('mask.dtype')], function_kwargs={}, max_wait_secs=0)
    acc = custom_method(
    tf.reduce_sum(match * mask), imports='import concurrent.futures;import collections;import hashlib;import itertools;import urllib.request;import tensorflow_hub as hub;import os;import pandas as pd;import einops;import tqdm;import numpy as np;import math;import requests;import pathlib;import tensorflow_text as text;import tensorflow as tf;import random;import tensorflow_datasets as tfds;from PIL import Image;import dataclasses;import string;import matplotlib.pyplot as plt;import json;import time;import re', function_to_run='tf.reduce_sum(*args)', method_object=None, object_signature=None, function_args=[eval('match*mask')], function_kwargs={}, max_wait_secs=0) / custom_method(
    tf.reduce_sum(mask), imports='import concurrent.futures;import collections;import hashlib;import itertools;import urllib.request;import tensorflow_hub as hub;import os;import pandas as pd;import einops;import tqdm;import numpy as np;import math;import requests;import pathlib;import tensorflow_text as text;import tensorflow as tf;import random;import tensorflow_datasets as tfds;from PIL import Image;import dataclasses;import string;import matplotlib.pyplot as plt;import json;import time;import re', function_to_run='tf.reduce_sum(*args)', method_object=None, object_signature=None, function_args=[eval('mask')], function_kwargs={}, max_wait_secs=0)
    return acc

class GenerateText(tf.keras.callbacks.Callback):

    def __init__(self):
        image_url = 'https://tensorflow.org/images/surf.jpg'
        image_path = custom_method(
        tf.keras.utils.get_file('surf.jpg', origin=image_url), imports='import concurrent.futures;import collections;import hashlib;import itertools;import urllib.request;import tensorflow_hub as hub;import os;import pandas as pd;import einops;import tqdm;import numpy as np;import math;import requests;import pathlib;import tensorflow_text as text;import tensorflow as tf;import random;import tensorflow_datasets as tfds;from PIL import Image;import dataclasses;import string;import matplotlib.pyplot as plt;import json;import time;import re', function_to_run='tf.keras.utils.get_file(*args, **kwargs)', method_object=None, object_signature=None, function_args=[eval("'surf.jpg'")], function_kwargs={'origin': eval('image_url')}, max_wait_secs=0)
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
custom_method(
g.on_epoch_end(0), imports='import concurrent.futures;import collections;import hashlib;import itertools;import urllib.request;import tensorflow_hub as hub;import os;import pandas as pd;import einops;import tqdm;import numpy as np;import math;import requests;import pathlib;import tensorflow_text as text;import tensorflow as tf;import random;import tensorflow_datasets as tfds;from PIL import Image;import dataclasses;import string;import matplotlib.pyplot as plt;import json;import time;import re', function_to_run='obj.on_epoch_end(*args)', method_object=eval('g'), object_signature='GenerateText', function_args=[eval('0')], function_kwargs={}, max_wait_secs=0, custom_class="class GenerateText(tf.keras.callbacks.Callback):\n  def __init__(self):\n    image_url = 'https://tensorflow.org/images/surf.jpg'\n    image_path = tf.keras.utils.get_file('surf.jpg', origin=image_url)\n    self.image = load_image(image_path)\n\n  def on_epoch_end(self, epochs=None, logs=None):\n    print()\n    print()\n    for t in (0.0, 0.5, 1.0):\n      result = self.model.simple_gen(self.image, temperature=t)\n      print(result)\n    print()")
callbacks = [GenerateText(), custom_method(
tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True), imports='import concurrent.futures;import collections;import hashlib;import itertools;import urllib.request;import tensorflow_hub as hub;import os;import pandas as pd;import einops;import tqdm;import numpy as np;import math;import requests;import pathlib;import tensorflow_text as text;import tensorflow as tf;import random;import tensorflow_datasets as tfds;from PIL import Image;import dataclasses;import string;import matplotlib.pyplot as plt;import json;import time;import re', function_to_run='tf.keras.callbacks.EarlyStopping(**kwargs)', method_object=None, object_signature=None, function_args=[], function_kwargs={'patience': eval('5'), 'restore_best_weights': eval('True')}, max_wait_secs=0)]
custom_method(
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001), loss=masked_loss, metrics=[masked_acc]), imports='import concurrent.futures;import collections;import hashlib;import itertools;import urllib.request;import tensorflow_hub as hub;import os;import pandas as pd;import einops;import tqdm;import numpy as np;import math;import requests;import pathlib;import tensorflow_text as text;import tensorflow as tf;import random;import tensorflow_datasets as tfds;from PIL import Image;import dataclasses;import string;import matplotlib.pyplot as plt;import json;import time;import re', function_to_run='obj.compile(**kwargs)', method_object=eval('model'), object_signature='Captioner', function_args=[], function_kwargs={'optimizer': eval('tf.keras.optimizers.Adam(learning_rate=1e-4)'), 'loss': eval('masked_loss'), 'metrics': eval('[masked_acc]')}, max_wait_secs=0, custom_class='class Captioner(tf.keras.Model):\n  @classmethod\n  def add_method(cls, fun):\n    setattr(cls, fun.__name__, fun)\n    return fun\n\n  def __init__(self, tokenizer, feature_extractor, output_layer, num_layers=1,\n               units=256, max_length=50, num_heads=1, dropout_rate=0.1):\n    super().__init__()\n    self.feature_extractor = feature_extractor\n    self.tokenizer = tokenizer\n    self.word_to_index = tf.keras.layers.StringLookup(\n        mask_token="",\n        vocabulary=tokenizer.get_vocabulary())\n    self.index_to_word = tf.keras.layers.StringLookup(\n        mask_token="",\n        vocabulary=tokenizer.get_vocabulary(),\n        invert=True) \n\n    self.seq_embedding = SeqEmbedding(\n        vocab_size=tokenizer.vocabulary_size(),\n        depth=units,\n        max_length=max_length)\n\n    self.decoder_layers = [\n        DecoderLayer(units, num_heads=num_heads, dropout_rate=dropout_rate)\n        for n in range(num_layers)]\n\n    self.output_layer = output_layer\n  @Captioner.add_method\n  def call(self, inputs):\n    image, txt = inputs\n\n    if image.shape[-1] == 3:\n      image = self.feature_extractor(image)\n    \n    image = einops.rearrange(image, \'b h w c -> b (h w) c\')\n\n\n    if txt.dtype == tf.string:\n      txt = tokenizer(txt)\n\n    txt = self.seq_embedding(txt)\n\n    for dec_layer in self.decoder_layers:\n      txt = dec_layer(inputs=(image, txt))\n      \n    txt = self.output_layer(txt)\n\n    return txt')
history = custom_method(
model.fit(train_ds.repeat(), steps_per_epoch=100, validation_data=test_ds.repeat(), validation_steps=20, epochs=100, callbacks=callbacks), imports='import concurrent.futures;import collections;import hashlib;import itertools;import urllib.request;import tensorflow_hub as hub;import os;import pandas as pd;import einops;import tqdm;import numpy as np;import math;import requests;import pathlib;import tensorflow_text as text;import tensorflow as tf;import random;import tensorflow_datasets as tfds;from PIL import Image;import dataclasses;import string;import matplotlib.pyplot as plt;import json;import time;import re', function_to_run='obj.fit(*args, **kwargs)', method_object=eval('model'), object_signature='Captioner', function_args=[eval('train_ds.repeat()')], function_kwargs={'steps_per_epoch': eval('100'), 'validation_data': eval('test_ds.repeat()'), 'validation_steps': eval('20'), 'epochs': eval('100'), 'callbacks': eval('callbacks')}, max_wait_secs=0, custom_class='class Captioner(tf.keras.Model):\n  @classmethod\n  def add_method(cls, fun):\n    setattr(cls, fun.__name__, fun)\n    return fun\n\n  def __init__(self, tokenizer, feature_extractor, output_layer, num_layers=1,\n               units=256, max_length=50, num_heads=1, dropout_rate=0.1):\n    super().__init__()\n    self.feature_extractor = feature_extractor\n    self.tokenizer = tokenizer\n    self.word_to_index = tf.keras.layers.StringLookup(\n        mask_token="",\n        vocabulary=tokenizer.get_vocabulary())\n    self.index_to_word = tf.keras.layers.StringLookup(\n        mask_token="",\n        vocabulary=tokenizer.get_vocabulary(),\n        invert=True) \n\n    self.seq_embedding = SeqEmbedding(\n        vocab_size=tokenizer.vocabulary_size(),\n        depth=units,\n        max_length=max_length)\n\n    self.decoder_layers = [\n        DecoderLayer(units, num_heads=num_heads, dropout_rate=dropout_rate)\n        for n in range(num_layers)]\n\n    self.output_layer = output_layer\n  @Captioner.add_method\n  def call(self, inputs):\n    image, txt = inputs\n\n    if image.shape[-1] == 3:\n      image = self.feature_extractor(image)\n    \n    image = einops.rearrange(image, \'b h w c -> b (h w) c\')\n\n\n    if txt.dtype == tf.string:\n      txt = tokenizer(txt)\n\n    txt = self.seq_embedding(txt)\n\n    for dec_layer in self.decoder_layers:\n      txt = dec_layer(inputs=(image, txt))\n      \n    txt = self.output_layer(txt)\n\n    return txt')
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
result = custom_method(
model.simple_gen(image, temperature=0.0), imports='import concurrent.futures;import collections;import hashlib;import itertools;import urllib.request;import tensorflow_hub as hub;import os;import pandas as pd;import einops;import tqdm;import numpy as np;import math;import requests;import pathlib;import tensorflow_text as text;import tensorflow as tf;import random;import tensorflow_datasets as tfds;from PIL import Image;import dataclasses;import string;import matplotlib.pyplot as plt;import json;import time;import re', function_to_run='obj.simple_gen(*args, **kwargs)', method_object=eval('model'), object_signature='Captioner', function_args=[eval('image')], function_kwargs={'temperature': eval('0.0')}, max_wait_secs=0, custom_class='class Captioner(tf.keras.Model):\n  @classmethod\n  def add_method(cls, fun):\n    setattr(cls, fun.__name__, fun)\n    return fun\n\n  def __init__(self, tokenizer, feature_extractor, output_layer, num_layers=1,\n               units=256, max_length=50, num_heads=1, dropout_rate=0.1):\n    super().__init__()\n    self.feature_extractor = feature_extractor\n    self.tokenizer = tokenizer\n    self.word_to_index = tf.keras.layers.StringLookup(\n        mask_token="",\n        vocabulary=tokenizer.get_vocabulary())\n    self.index_to_word = tf.keras.layers.StringLookup(\n        mask_token="",\n        vocabulary=tokenizer.get_vocabulary(),\n        invert=True) \n\n    self.seq_embedding = SeqEmbedding(\n        vocab_size=tokenizer.vocabulary_size(),\n        depth=units,\n        max_length=max_length)\n\n    self.decoder_layers = [\n        DecoderLayer(units, num_heads=num_heads, dropout_rate=dropout_rate)\n        for n in range(num_layers)]\n\n    self.output_layer = output_layer\n  @Captioner.add_method\n  def call(self, inputs):\n    image, txt = inputs\n\n    if image.shape[-1] == 3:\n      image = self.feature_extractor(image)\n    \n    image = einops.rearrange(image, \'b h w c -> b (h w) c\')\n\n\n    if txt.dtype == tf.string:\n      txt = tokenizer(txt)\n\n    txt = self.seq_embedding(txt)\n\n    for dec_layer in self.decoder_layers:\n      txt = dec_layer(inputs=(image, txt))\n      \n    txt = self.output_layer(txt)\n\n    return txt')
result
str_tokens = custom_method(
result.split(), imports='import concurrent.futures;import collections;import hashlib;import itertools;import urllib.request;import tensorflow_hub as hub;import os;import pandas as pd;import einops;import tqdm;import numpy as np;import math;import requests;import pathlib;import tensorflow_text as text;import tensorflow as tf;import random;import tensorflow_datasets as tfds;from PIL import Image;import dataclasses;import string;import matplotlib.pyplot as plt;import json;import time;import re', function_to_run='obj.split()', method_object=eval('result'), object_signature='tf.strings.reduce_join', function_args=[], function_kwargs={}, max_wait_secs=0, custom_class=None)
str_tokens.append('[END]')
attn_maps = [layer.last_attention_scores for layer in model.decoder_layers]
[map.shape for map in attn_maps]
attention_maps = custom_method(
tf.concat(attn_maps, axis=0), imports='import concurrent.futures;import collections;import hashlib;import itertools;import urllib.request;import tensorflow_hub as hub;import os;import pandas as pd;import einops;import tqdm;import numpy as np;import math;import requests;import pathlib;import tensorflow_text as text;import tensorflow as tf;import random;import tensorflow_datasets as tfds;from PIL import Image;import dataclasses;import string;import matplotlib.pyplot as plt;import json;import time;import re', function_to_run='tf.concat(*args, **kwargs)', method_object=None, object_signature=None, function_args=[eval('attn_maps')], function_kwargs={'axis': eval('0')}, max_wait_secs=0)
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
    attention_maps = custom_method(
    tf.concat(attention_maps, axis=0), imports='import concurrent.futures;import collections;import hashlib;import itertools;import urllib.request;import tensorflow_hub as hub;import os;import pandas as pd;import einops;import tqdm;import numpy as np;import math;import requests;import pathlib;import tensorflow_text as text;import tensorflow as tf;import random;import tensorflow_datasets as tfds;from PIL import Image;import dataclasses;import string;import matplotlib.pyplot as plt;import json;import time;import re', function_to_run='tf.concat(*args, **kwargs)', method_object=None, object_signature=None, function_args=[eval('attention_maps')], function_kwargs={'axis': eval('0')}, max_wait_secs=0)
    attention_maps = einops.reduce(attention_maps, 'batch heads sequence (height width) -> sequence height width', height=7, width=7, reduction='mean')
    plot_attention_maps(image / 255, str_tokens, attention_maps)
    t = plt.suptitle(result_txt)
    t.set_y(1.05)
run_and_show_attention(model, image)
image_url = 'https://tensorflow.org/images/bedroom_hrnet_tutorial.jpg'
image_path = custom_method(
tf.keras.utils.get_file(origin=image_url), imports='import concurrent.futures;import collections;import hashlib;import itertools;import urllib.request;import tensorflow_hub as hub;import os;import pandas as pd;import einops;import tqdm;import numpy as np;import math;import requests;import pathlib;import tensorflow_text as text;import tensorflow as tf;import random;import tensorflow_datasets as tfds;from PIL import Image;import dataclasses;import string;import matplotlib.pyplot as plt;import json;import time;import re', function_to_run='tf.keras.utils.get_file(**kwargs)', method_object=None, object_signature=None, function_args=[], function_kwargs={'origin': eval('image_url')}, max_wait_secs=0)
image = load_image(image_path)
run_and_show_attention(model, image)
