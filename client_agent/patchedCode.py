node module: tensorflow.keras.layers
required Alias: ['tf', 'math', 'torch', 'nn', 'F', 'optim', 'tf', 'keras', 'layers', 'Dense', 'drop', 'K', 'L', 'tf', 'Dense', 'Dropout', 'Conv2D', 'clear_session', 'set_session', 'dense_layer', 'drop', 'conv']
++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
Module(
    body=[
        Module(
            body=[
                Import(
                    names=[
                        alias(name='pickle')]),
                Import(
                    names=[
                        alias(name='requests')]),
                ImportFrom(
                    module='server.send_request',
                    names=[
                        alias(name='send_request')],
                    level=0),
                FunctionDef(
                    name='custom_method',
                    args=arguments(
                        posonlyargs=[],
                        args=[
                            arg(arg='func'),
                            arg(arg='imports'),
                            arg(arg='function_to_run'),
                            arg(arg='method_object'),
                            arg(arg='function_args'),
                            arg(arg='function_kwargs'),
                            arg(arg='max_wait_secs')],
                        kwonlyargs=[],
                        kw_defaults=[],
                        defaults=[]),
                    body=[
                        Assign(
                            targets=[
                                Name(id='result', ctx=Store())],
                            value=Call(
                                func=Name(id='send_request', ctx=Load()),
                                args=[
                                    Name(id='imports', ctx=Load()),
                                    Name(id='function_to_run', ctx=Load()),
                                    Name(id='function_args', ctx=Load()),
                                    Name(id='function_kwargs', ctx=Load()),
                                    Name(id='max_wait_secs', ctx=Load()),
                                    Name(id='method_object', ctx=Load())],
                                keywords=[])),
                        Return(
                            value=Name(id='func', ctx=Load()))],
                    decorator_list=[])],
            type_ignores=[]),
        Import(
            names=[
                alias(name='tensorflow', asname='tf')]),
        Import(
            names=[
                alias(name='numpy', asname='np')]),
        Import(
            names=[
                alias(name='math'),
                alias(name='random')]),
        Import(
            names=[
                alias(name='matplotlib.pyplot', asname='plt')]),
        ImportFrom(
            module='pprint.xyz.abc',
            names=[
                alias(name='pprint', asname='ppr')],
            level=0),
        Import(
            names=[
                alias(name='torch')]),
        Import(
            names=[
                alias(name='torchvision')]),
        Import(
            names=[
                alias(name='torchvision.transforms', asname='transforms')]),
        ImportFrom(
            module='tf',
            names=[
                alias(name='trans', asname='tf1')],
            level=0),
        ImportFrom(
            module='pt',
            names=[
                alias(name='trans', asname='tf2')],
            level=0),
        Import(
            names=[
                alias(name='torch.nn', asname='nn')]),
        Import(
            names=[
                alias(name='torch.nn.functional', asname='F')]),
        Import(
            names=[
                alias(name='torch.optim', asname='optim')]),
        ImportFrom(
            module='os.path',
            names=[
                alias(name='exists')],
            level=0),
        Import(
            names=[
                alias(name='sys')]),
        Import(
            names=[
                alias(name='tensorflow', asname='tf')]),
        ImportFrom(
            module='tensorflow',
            names=[
                alias(name='keras')],
            level=0),
        ImportFrom(
            module='tensorflow.keras',
            names=[
                alias(name='layers')],
            level=0),
        ImportFrom(
            module='tensorflow.keras.layers',
            names=[
                alias(name='Dense'),
                alias(name='Dropout', asname='drop')],
            level=0),
        ImportFrom(
            module='tensorflow.keras.layers',
            names=[
                alias(name='*')],
            level=0),
        Import(
            names=[
                alias(name='tensorflow.keras.backend', asname='K')]),
        Import(
            names=[
                alias(name='tensorflow.keras.layers', asname='L')]),
        Import(
            names=[
                alias(name='tensorflow.compat.v1', asname='tf')]),
        ImportFrom(
            module='tensorflow.keras.layers',
            names=[
                alias(name='Dense'),
                alias(name='Dropout'),
                alias(name='Conv2D')],
            level=0),
        ImportFrom(
            module='tensorflow.keras.backend',
            names=[
                alias(name='clear_session'),
                alias(name='set_session')],
            level=0),
        ImportFrom(
            module='tensorflow.keras.layers',
            names=[
                alias(name='Dense', asname='dense_layer'),
                alias(name='Dropout', asname='drop'),
                alias(name='Conv2D', asname='conv')],
            level=0),
        Expr(
            value=Call(
                func=Name(id='custom_method', ctx=Load()),
                args=[
                    Expr(
                        value=Call(
                            func=Name(id='tf', ctx=Load()),
                            args=[],
                            keywords=[]))],
                keywords=[
                    keyword(
                        arg='imports',
                        value=Constant(value='from tensorflow.keras.layers import *;from os.path import exists;import math, random;from tensorflow.keras import layers;import tensorflow as tf;import torch;import sys;import matplotlib.pyplot as plt;from tensorflow import keras;import tensorflow.keras.backend as K;from tensorflow.keras.layers import Dense, Dropout, Conv2D;from pprint.xyz.abc import pprint as ppr;from tensorflow.keras.backend import clear_session, set_session;from tf import trans as tf1;import torch.nn.functional as F;import tensorflow.compat.v1 as tf;import torch.nn as nn;from tensorflow.keras.layers import Dense, Dropout as drop;import numpy as np;from tensorflow.keras.layers import Dense as dense_layer, Dropout as drop, Conv2D as conv;import tensorflow.keras.layers as L;import torch.optim as optim;import torchvision.transforms as transforms;from pt import trans as tf2;import torchvision')),
                    keyword(
                        arg='function_to_run',
                        value=Constant(value='tf()')),
                    keyword(
                        arg='method_object',
                        value=Constant(value=None)),
                    keyword(
                        arg='function_args',
                        value=List(elts=[], ctx=Load())),
                    keyword(
                        arg='function_kwargs',
                        value=Dict(keys=[], values=[])),
                    keyword(
                        arg='max_wait_secs',
                        value=Constant(value=30))])),
        Assign(
            targets=[
                Name(id='testlist', ctx=Store())],
            value=List(
                elts=[
                    Constant(value='a'),
                    Constant(value='b'),
                    Constant(value='c')],
                ctx=Load())),
        Assign(
            targets=[
                Name(id='X', ctx=Store())],
            value=Call(
                func=Name(id='custom_method', ctx=Load()),
                args=[
                    Expr(
                        value=Call(
                            func=Attribute(
                                value=Name(id='tf', ctx=Load()),
                                attr='placeholder',
                                ctx=Load()),
                            args=[
                                Attribute(
                                    value=Name(id='tf', ctx=Load()),
                                    attr='float32',
                                    ctx=Load()),
                                List(
                                    elts=[
                                        Constant(value=None),
                                        Constant(value=1)],
                                    ctx=Load())],
                            keywords=[
                                keyword(
                                    arg='name',
                                    value=Constant(value='X'))]))],
                keywords=[
                    keyword(
                        arg='imports',
                        value=Constant(value='from tensorflow.keras.layers import *;from os.path import exists;import math, random;from tensorflow.keras import layers;import tensorflow as tf;import torch;import sys;import matplotlib.pyplot as plt;from tensorflow import keras;import tensorflow.keras.backend as K;from tensorflow.keras.layers import Dense, Dropout, Conv2D;from pprint.xyz.abc import pprint as ppr;from tensorflow.keras.backend import clear_session, set_session;from tf import trans as tf1;import torch.nn.functional as F;import tensorflow.compat.v1 as tf;import torch.nn as nn;from tensorflow.keras.layers import Dense, Dropout as drop;import numpy as np;from tensorflow.keras.layers import Dense as dense_layer, Dropout as drop, Conv2D as conv;import tensorflow.keras.layers as L;import torch.optim as optim;import torchvision.transforms as transforms;from pt import trans as tf2;import torchvision')),
                    keyword(
                        arg='function_to_run',
                        value=Constant(value='tf.placeholder(*args, **kwargs)')),
                    keyword(
                        arg='method_object',
                        value=Constant(value=None)),
                    keyword(
                        arg='function_args',
                        value=List(
                            elts=[
                                Call(
                                    func=Name(id='eval', ctx=Load()),
                                    args=[
                                        Constant(value='tf.float32')],
                                    keywords=[]),
                                Call(
                                    func=Name(id='eval', ctx=Load()),
                                    args=[
                                        Constant(value='[None, 1]')],
                                    keywords=[])],
                            ctx=Load())),
                    keyword(
                        arg='function_kwargs',
                        value=Dict(
                            keys=[
                                Constant(value='name')],
                            values=[
                                Call(
                                    func=Name(id='eval', ctx=Load()),
                                    args=[
                                        Constant(value='"X"')],
                                    keywords=[])])),
                    keyword(
                        arg='max_wait_secs',
                        value=Constant(value=30))])),
        Assign(
            targets=[
                Name(id='Y', ctx=Store())],
            value=Call(
                func=Name(id='custom_method', ctx=Load()),
                args=[
                    Expr(
                        value=Call(
                            func=Attribute(
                                value=Name(id='tf', ctx=Load()),
                                attr='placeholder',
                                ctx=Load()),
                            args=[
                                Attribute(
                                    value=Name(id='tf', ctx=Load()),
                                    attr='float32',
                                    ctx=Load()),
                                List(
                                    elts=[
                                        Constant(value=None),
                                        Constant(value=1)],
                                    ctx=Load())],
                            keywords=[
                                keyword(
                                    arg='name',
                                    value=Constant(value='Y'))]))],
                keywords=[
                    keyword(
                        arg='imports',
                        value=Constant(value='from tensorflow.keras.layers import *;from os.path import exists;import math, random;from tensorflow.keras import layers;import tensorflow as tf;import torch;import sys;import matplotlib.pyplot as plt;from tensorflow import keras;import tensorflow.keras.backend as K;from tensorflow.keras.layers import Dense, Dropout, Conv2D;from pprint.xyz.abc import pprint as ppr;from tensorflow.keras.backend import clear_session, set_session;from tf import trans as tf1;import torch.nn.functional as F;import tensorflow.compat.v1 as tf;import torch.nn as nn;from tensorflow.keras.layers import Dense, Dropout as drop;import numpy as np;from tensorflow.keras.layers import Dense as dense_layer, Dropout as drop, Conv2D as conv;import tensorflow.keras.layers as L;import torch.optim as optim;import torchvision.transforms as transforms;from pt import trans as tf2;import torchvision')),
                    keyword(
                        arg='function_to_run',
                        value=Constant(value='tf.placeholder(*args, **kwargs)')),
                    keyword(
                        arg='method_object',
                        value=Constant(value=None)),
                    keyword(
                        arg='function_args',
                        value=List(
                            elts=[
                                Call(
                                    func=Name(id='eval', ctx=Load()),
                                    args=[
                                        Constant(value='tf.float32')],
                                    keywords=[]),
                                Call(
                                    func=Name(id='eval', ctx=Load()),
                                    args=[
                                        Constant(value='[None, 1]')],
                                    keywords=[])],
                            ctx=Load())),
                    keyword(
                        arg='function_kwargs',
                        value=Dict(
                            keys=[
                                Constant(value='name')],
                            values=[
                                Call(
                                    func=Name(id='eval', ctx=Load()),
                                    args=[
                                        Constant(value='"Y"')],
                                    keywords=[])])),
                    keyword(
                        arg='max_wait_secs',
                        value=Constant(value=30))])),
        Assign(
            targets=[
                Name(id='z', ctx=Store())],
            value=Call(
                func=Name(id='custom_method', ctx=Load()),
                args=[
                    Expr(
                        value=Call(
                            func=Attribute(
                                value=Name(id='tf', ctx=Load()),
                                attr='value',
                                ctx=Load()),
                            args=[
                                Constant(value='aname'),
                                Name(id='bname', ctx=Load())],
                            keywords=[
                                keyword(
                                    arg='name',
                                    value=Constant(value='tname'))]))],
                keywords=[
                    keyword(
                        arg='imports',
                        value=Constant(value='from tensorflow.keras.layers import *;from os.path import exists;import math, random;from tensorflow.keras import layers;import tensorflow as tf;import torch;import sys;import matplotlib.pyplot as plt;from tensorflow import keras;import tensorflow.keras.backend as K;from tensorflow.keras.layers import Dense, Dropout, Conv2D;from pprint.xyz.abc import pprint as ppr;from tensorflow.keras.backend import clear_session, set_session;from tf import trans as tf1;import torch.nn.functional as F;import tensorflow.compat.v1 as tf;import torch.nn as nn;from tensorflow.keras.layers import Dense, Dropout as drop;import numpy as np;from tensorflow.keras.layers import Dense as dense_layer, Dropout as drop, Conv2D as conv;import tensorflow.keras.layers as L;import torch.optim as optim;import torchvision.transforms as transforms;from pt import trans as tf2;import torchvision')),
                    keyword(
                        arg='function_to_run',
                        value=Constant(value='tf.value(*args, **kwargs)')),
                    keyword(
                        arg='method_object',
                        value=Constant(value=None)),
                    keyword(
                        arg='function_args',
                        value=List(
                            elts=[
                                Call(
                                    func=Name(id='eval', ctx=Load()),
                                    args=[
                                        Constant(value='"aname"')],
                                    keywords=[]),
                                Call(
                                    func=Name(id='eval', ctx=Load()),
                                    args=[
                                        Constant(value='bname')],
                                    keywords=[])],
                            ctx=Load())),
                    keyword(
                        arg='function_kwargs',
                        value=Dict(
                            keys=[
                                Constant(value='name')],
                            values=[
                                Call(
                                    func=Name(id='eval', ctx=Load()),
                                    args=[
                                        Constant(value='"tname"')],
                                    keywords=[])])),
                    keyword(
                        arg='max_wait_secs',
                        value=Constant(value=30))])),
        Assign(
            targets=[
                Name(id='w_o', ctx=Store())],
            value=Call(
                func=Name(id='custom_method', ctx=Load()),
                args=[
                    Expr(
                        value=Call(
                            func=Attribute(
                                value=Name(id='tf', ctx=Load()),
                                attr='Variable',
                                ctx=Load()),
                            args=[
                                Call(
                                    func=Attribute(
                                        value=Name(id='tf', ctx=Load()),
                                        attr='random_uniform',
                                        ctx=Load()),
                                    args=[
                                        List(
                                            elts=[
                                                Name(id='layer_1_neurons', ctx=Load()),
                                                Constant(value=1)],
                                            ctx=Load())],
                                    keywords=[
                                        keyword(
                                            arg='minval',
                                            value=UnaryOp(
                                                op=USub(),
                                                operand=Constant(value=1))),
                                        keyword(
                                            arg='maxval',
                                            value=Constant(value=1)),
                                        keyword(
                                            arg='dtype',
                                            value=Attribute(
                                                value=Name(id='tf', ctx=Load()),
                                                attr='float32',
                                                ctx=Load()))]),
                                Name(id='test1', ctx=Load()),
                                Name(id='test2', ctx=Load())],
                            keywords=[
                                keyword(
                                    arg='test3',
                                    value=Constant(value='test3'))]))],
                keywords=[
                    keyword(
                        arg='imports',
                        value=Constant(value='from tensorflow.keras.layers import *;from os.path import exists;import math, random;from tensorflow.keras import layers;import tensorflow as tf;import torch;import sys;import matplotlib.pyplot as plt;from tensorflow import keras;import tensorflow.keras.backend as K;from tensorflow.keras.layers import Dense, Dropout, Conv2D;from pprint.xyz.abc import pprint as ppr;from tensorflow.keras.backend import clear_session, set_session;from tf import trans as tf1;import torch.nn.functional as F;import tensorflow.compat.v1 as tf;import torch.nn as nn;from tensorflow.keras.layers import Dense, Dropout as drop;import numpy as np;from tensorflow.keras.layers import Dense as dense_layer, Dropout as drop, Conv2D as conv;import tensorflow.keras.layers as L;import torch.optim as optim;import torchvision.transforms as transforms;from pt import trans as tf2;import torchvision')),
                    keyword(
                        arg='function_to_run',
                        value=Constant(value='tf.Variable(*args, **kwargs)')),
                    keyword(
                        arg='method_object',
                        value=Constant(value=None)),
                    keyword(
                        arg='function_args',
                        value=List(
                            elts=[
                                Call(
                                    func=Name(id='eval', ctx=Load()),
                                    args=[
                                        Constant(value='tf.random_uniform([layer_1_neurons, 1], minval = -1, maxval = 1, dtype = tf.float32)')],
                                    keywords=[]),
                                Call(
                                    func=Name(id='eval', ctx=Load()),
                                    args=[
                                        Constant(value='test1')],
                                    keywords=[]),
                                Call(
                                    func=Name(id='eval', ctx=Load()),
                                    args=[
                                        Constant(value='test2')],
                                    keywords=[])],
                            ctx=Load())),
                    keyword(
                        arg='function_kwargs',
                        value=Dict(
                            keys=[
                                Constant(value='test3')],
                            values=[
                                Call(
                                    func=Name(id='eval', ctx=Load()),
                                    args=[
                                        Constant(value='"test3"')],
                                    keywords=[])])),
                    keyword(
                        arg='max_wait_secs',
                        value=Constant(value=30))])),
        Assign(
            targets=[
                Name(id='b_o', ctx=Store())],
            value=Call(
                func=Name(id='custom_method', ctx=Load()),
                args=[
                    Expr(
                        value=Call(
                            func=Attribute(
                                value=Name(id='tf', ctx=Load()),
                                attr='Variable',
                                ctx=Load()),
                            args=[
                                Call(
                                    func=Attribute(
                                        value=Name(id='tf', ctx=Load()),
                                        attr='zeros',
                                        ctx=Load()),
                                    args=[
                                        List(
                                            elts=[
                                                Constant(value=1),
                                                Constant(value=1)],
                                            ctx=Load())],
                                    keywords=[
                                        keyword(
                                            arg='dtype',
                                            value=Attribute(
                                                value=Name(id='tf', ctx=Load()),
                                                attr='float32',
                                                ctx=Load()))])],
                            keywords=[]))],
                keywords=[
                    keyword(
                        arg='imports',
                        value=Constant(value='from tensorflow.keras.layers import *;from os.path import exists;import math, random;from tensorflow.keras import layers;import tensorflow as tf;import torch;import sys;import matplotlib.pyplot as plt;from tensorflow import keras;import tensorflow.keras.backend as K;from tensorflow.keras.layers import Dense, Dropout, Conv2D;from pprint.xyz.abc import pprint as ppr;from tensorflow.keras.backend import clear_session, set_session;from tf import trans as tf1;import torch.nn.functional as F;import tensorflow.compat.v1 as tf;import torch.nn as nn;from tensorflow.keras.layers import Dense, Dropout as drop;import numpy as np;from tensorflow.keras.layers import Dense as dense_layer, Dropout as drop, Conv2D as conv;import tensorflow.keras.layers as L;import torch.optim as optim;import torchvision.transforms as transforms;from pt import trans as tf2;import torchvision')),
                    keyword(
                        arg='function_to_run',
                        value=Constant(value='tf.Variable(*args)')),
                    keyword(
                        arg='method_object',
                        value=Constant(value=None)),
                    keyword(
                        arg='function_args',
                        value=List(
                            elts=[
                                Call(
                                    func=Name(id='eval', ctx=Load()),
                                    args=[
                                        Constant(value='tf.zeros([1, 1], dtype = tf.float32)')],
                                    keywords=[])],
                            ctx=Load())),
                    keyword(
                        arg='function_kwargs',
                        value=Dict(keys=[], values=[])),
                    keyword(
                        arg='max_wait_secs',
                        value=Constant(value=30))])),
        Assign(
            targets=[
                Name(id='cd', ctx=Store())],
            value=Call(
                func=Attribute(
                    value=Name(id='ab', ctx=Load()),
                    attr='test',
                    ctx=Load()),
                args=[
                    Name(id='a', ctx=Load()),
                    Name(id='b', ctx=Load())],
                keywords=[])),
        Assign(
            targets=[
                Name(id='c_o', ctx=Store())],
            value=Call(
                func=Attribute(
                    value=Name(id='xf', ctx=Load()),
                    attr='test',
                    ctx=Load()),
                args=[],
                keywords=[])),
        Assign(
            targets=[
                Name(id='d_o', ctx=Store())],
            value=Call(
                func=Attribute(
                    value=Name(id='xy', ctx=Load()),
                    attr='test',
                    ctx=Load()),
                args=[],
                keywords=[])),
        Assign(
            targets=[
                Name(id='model', ctx=Store())],
            value=Call(
                func=Name(id='custom_method', ctx=Load()),
                args=[
                    Expr(
                        value=Call(
                            func=Attribute(
                                value=Attribute(
                                    value=Attribute(
                                        value=Name(id='tf', ctx=Load()),
                                        attr='keras',
                                        ctx=Load()),
                                    attr='models',
                                    ctx=Load()),
                                attr='Sequential',
                                ctx=Load()),
                            args=[
                                List(
                                    elts=[
                                        Call(
                                            func=Attribute(
                                                value=Attribute(
                                                    value=Attribute(
                                                        value=Name(id='tf', ctx=Load()),
                                                        attr='keras',
                                                        ctx=Load()),
                                                    attr='layers',
                                                    ctx=Load()),
                                                attr='Flatten',
                                                ctx=Load()),
                                            args=[],
                                            keywords=[
                                                keyword(
                                                    arg='input_shape',
                                                    value=Tuple(
                                                        elts=[
                                                            Constant(value=28),
                                                            Constant(value=28)],
                                                        ctx=Load()))]),
                                        Call(
                                            func=Attribute(
                                                value=Attribute(
                                                    value=Attribute(
                                                        value=Name(id='tf', ctx=Load()),
                                                        attr='keras',
                                                        ctx=Load()),
                                                    attr='layers',
                                                    ctx=Load()),
                                                attr='Dense',
                                                ctx=Load()),
                                            args=[
                                                Constant(value=128)],
                                            keywords=[
                                                keyword(
                                                    arg='activation',
                                                    value=Constant(value='relu'))]),
                                        Call(
                                            func=Attribute(
                                                value=Attribute(
                                                    value=Attribute(
                                                        value=Name(id='tf', ctx=Load()),
                                                        attr='keras',
                                                        ctx=Load()),
                                                    attr='layers',
                                                    ctx=Load()),
                                                attr='Dropout',
                                                ctx=Load()),
                                            args=[
                                                Constant(value=0.2)],
                                            keywords=[]),
                                        Call(
                                            func=Attribute(
                                                value=Attribute(
                                                    value=Attribute(
                                                        value=Name(id='tf', ctx=Load()),
                                                        attr='keras',
                                                        ctx=Load()),
                                                    attr='layers',
                                                    ctx=Load()),
                                                attr='Dense',
                                                ctx=Load()),
                                            args=[
                                                Constant(value=10)],
                                            keywords=[])],
                                    ctx=Load())],
                            keywords=[]))],
                keywords=[
                    keyword(
                        arg='imports',
                        value=Constant(value='from tensorflow.keras.layers import *;from os.path import exists;import math, random;from tensorflow.keras import layers;import tensorflow as tf;import torch;import sys;import matplotlib.pyplot as plt;from tensorflow import keras;import tensorflow.keras.backend as K;from tensorflow.keras.layers import Dense, Dropout, Conv2D;from pprint.xyz.abc import pprint as ppr;from tensorflow.keras.backend import clear_session, set_session;from tf import trans as tf1;import torch.nn.functional as F;import tensorflow.compat.v1 as tf;import torch.nn as nn;from tensorflow.keras.layers import Dense, Dropout as drop;import numpy as np;from tensorflow.keras.layers import Dense as dense_layer, Dropout as drop, Conv2D as conv;import tensorflow.keras.layers as L;import torch.optim as optim;import torchvision.transforms as transforms;from pt import trans as tf2;import torchvision')),
                    keyword(
                        arg='function_to_run',
                        value=Constant(value='tf.keras.models.Sequential(*args)')),
                    keyword(
                        arg='method_object',
                        value=Constant(value=None)),
                    keyword(
                        arg='function_args',
                        value=List(
                            elts=[
                                Call(
                                    func=Name(id='eval', ctx=Load()),
                                    args=[
                                        Constant(value="[\n    tf.keras.layers.Flatten(input_shape=(28, 28)),\n    tf.keras.layers.Dense(128, activation='relu'),\n    tf.keras.layers.Dropout(0.2),\n    tf.keras.layers.Dense(10)\n    ]")],
                                    keywords=[])],
                            ctx=Load())),
                    keyword(
                        arg='function_kwargs',
                        value=Dict(keys=[], values=[])),
                    keyword(
                        arg='max_wait_secs',
                        value=Constant(value=30))])),
        Expr(
            value=Call(
                func=Name(id='custom_method', ctx=Load()),
                args=[
                    Expr(
                        value=Call(
                            func=Attribute(
                                value=Name(id='model', ctx=Load()),
                                attr='compile',
                                ctx=Load()),
                            args=[],
                            keywords=[]))],
                keywords=[
                    keyword(
                        arg='imports',
                        value=Constant(value='from tensorflow.keras.layers import *;from os.path import exists;import math, random;from tensorflow.keras import layers;import tensorflow as tf;import torch;import sys;import matplotlib.pyplot as plt;from tensorflow import keras;import tensorflow.keras.backend as K;from tensorflow.keras.layers import Dense, Dropout, Conv2D;from pprint.xyz.abc import pprint as ppr;from tensorflow.keras.backend import clear_session, set_session;from tf import trans as tf1;import torch.nn.functional as F;import tensorflow.compat.v1 as tf;import torch.nn as nn;from tensorflow.keras.layers import Dense, Dropout as drop;import numpy as np;from tensorflow.keras.layers import Dense as dense_layer, Dropout as drop, Conv2D as conv;import tensorflow.keras.layers as L;import torch.optim as optim;import torchvision.transforms as transforms;from pt import trans as tf2;import torchvision')),
                    keyword(
                        arg='function_to_run',
                        value=Constant(value='obj.compile()')),
                    keyword(
                        arg='method_object',
                        value=Constant(value='model')),
                    keyword(
                        arg='function_args',
                        value=List(elts=[], ctx=Load())),
                    keyword(
                        arg='function_kwargs',
                        value=Dict(keys=[], values=[])),
                    keyword(
                        arg='max_wait_secs',
                        value=Constant(value=30))])),
        Assign(
            targets=[
                Name(id='f_o', ctx=Store())],
            value=Call(
                func=Name(id='custom_method', ctx=Load()),
                args=[
                    Expr(
                        value=Call(
                            func=Attribute(
                                value=Attribute(
                                    value=Name(id='b_o', ctx=Load()),
                                    attr='test',
                                    ctx=Load()),
                                attr='compile',
                                ctx=Load()),
                            args=[],
                            keywords=[]))],
                keywords=[
                    keyword(
                        arg='imports',
                        value=Constant(value='from tensorflow.keras.layers import *;from os.path import exists;import math, random;from tensorflow.keras import layers;import tensorflow as tf;import torch;import sys;import matplotlib.pyplot as plt;from tensorflow import keras;import tensorflow.keras.backend as K;from tensorflow.keras.layers import Dense, Dropout, Conv2D;from pprint.xyz.abc import pprint as ppr;from tensorflow.keras.backend import clear_session, set_session;from tf import trans as tf1;import torch.nn.functional as F;import tensorflow.compat.v1 as tf;import torch.nn as nn;from tensorflow.keras.layers import Dense, Dropout as drop;import numpy as np;from tensorflow.keras.layers import Dense as dense_layer, Dropout as drop, Conv2D as conv;import tensorflow.keras.layers as L;import torch.optim as optim;import torchvision.transforms as transforms;from pt import trans as tf2;import torchvision')),
                    keyword(
                        arg='function_to_run',
                        value=Constant(value='obj.test.compile()')),
                    keyword(
                        arg='method_object',
                        value=Constant(value='b_o')),
                    keyword(
                        arg='function_args',
                        value=List(elts=[], ctx=Load())),
                    keyword(
                        arg='function_kwargs',
                        value=Dict(keys=[], values=[])),
                    keyword(
                        arg='max_wait_secs',
                        value=Constant(value=30))])),
        Assign(
            targets=[
                Name(id='z_o', ctx=Store())],
            value=Call(
                func=Attribute(
                    value=Name(id='c_o', ctx=Load()),
                    attr='compile',
                    ctx=Load()),
                args=[],
                keywords=[]))],
    type_ignores=[])
____________________________________________________________________________________________________
import pickle
import requests
from server.send_request import send_request

def custom_method(func, imports, function_to_run, method_object, function_args, function_kwargs, max_wait_secs):
    result = send_request(imports, function_to_run, function_args, function_kwargs, max_wait_secs, method_object)
    return func
import tensorflow as tf
import numpy as np
import math, random
import matplotlib.pyplot as plt
from pprint.xyz.abc import pprint as ppr
import torch
import torchvision
import torchvision.transforms as transforms
from tf import trans as tf1
from pt import trans as tf2
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from os.path import exists
import sys
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers import Dense, Dropout as drop
from tensorflow.keras.layers import *
import tensorflow.keras.backend as K
import tensorflow.keras.layers as L
import tensorflow.compat.v1 as tf
from tensorflow.keras.layers import Dense, Dropout, Conv2D
from tensorflow.keras.backend import clear_session, set_session
from tensorflow.keras.layers import Dense as dense_layer, Dropout as drop, Conv2D as conv
custom_method(
tf(), imports='from tensorflow.keras.layers import *;from os.path import exists;import math, random;from tensorflow.keras import layers;import tensorflow as tf;import torch;import sys;import matplotlib.pyplot as plt;from tensorflow import keras;import tensorflow.keras.backend as K;from tensorflow.keras.layers import Dense, Dropout, Conv2D;from pprint.xyz.abc import pprint as ppr;from tensorflow.keras.backend import clear_session, set_session;from tf import trans as tf1;import torch.nn.functional as F;import tensorflow.compat.v1 as tf;import torch.nn as nn;from tensorflow.keras.layers import Dense, Dropout as drop;import numpy as np;from tensorflow.keras.layers import Dense as dense_layer, Dropout as drop, Conv2D as conv;import tensorflow.keras.layers as L;import torch.optim as optim;import torchvision.transforms as transforms;from pt import trans as tf2;import torchvision', function_to_run='tf()', method_object=None, function_args=[], function_kwargs={}, max_wait_secs=30)
testlist = ['a', 'b', 'c']
X = custom_method(
tf.placeholder(tf.float32, [None, 1], name='X'), imports='from tensorflow.keras.layers import *;from os.path import exists;import math, random;from tensorflow.keras import layers;import tensorflow as tf;import torch;import sys;import matplotlib.pyplot as plt;from tensorflow import keras;import tensorflow.keras.backend as K;from tensorflow.keras.layers import Dense, Dropout, Conv2D;from pprint.xyz.abc import pprint as ppr;from tensorflow.keras.backend import clear_session, set_session;from tf import trans as tf1;import torch.nn.functional as F;import tensorflow.compat.v1 as tf;import torch.nn as nn;from tensorflow.keras.layers import Dense, Dropout as drop;import numpy as np;from tensorflow.keras.layers import Dense as dense_layer, Dropout as drop, Conv2D as conv;import tensorflow.keras.layers as L;import torch.optim as optim;import torchvision.transforms as transforms;from pt import trans as tf2;import torchvision', function_to_run='tf.placeholder(*args, **kwargs)', method_object=None, function_args=[eval('tf.float32'), eval('[None, 1]')], function_kwargs={'name': eval('"X"')}, max_wait_secs=30)
Y = custom_method(
tf.placeholder(tf.float32, [None, 1], name='Y'), imports='from tensorflow.keras.layers import *;from os.path import exists;import math, random;from tensorflow.keras import layers;import tensorflow as tf;import torch;import sys;import matplotlib.pyplot as plt;from tensorflow import keras;import tensorflow.keras.backend as K;from tensorflow.keras.layers import Dense, Dropout, Conv2D;from pprint.xyz.abc import pprint as ppr;from tensorflow.keras.backend import clear_session, set_session;from tf import trans as tf1;import torch.nn.functional as F;import tensorflow.compat.v1 as tf;import torch.nn as nn;from tensorflow.keras.layers import Dense, Dropout as drop;import numpy as np;from tensorflow.keras.layers import Dense as dense_layer, Dropout as drop, Conv2D as conv;import tensorflow.keras.layers as L;import torch.optim as optim;import torchvision.transforms as transforms;from pt import trans as tf2;import torchvision', function_to_run='tf.placeholder(*args, **kwargs)', method_object=None, function_args=[eval('tf.float32'), eval('[None, 1]')], function_kwargs={'name': eval('"Y"')}, max_wait_secs=30)
z = custom_method(
tf.value('aname', bname, name='tname'), imports='from tensorflow.keras.layers import *;from os.path import exists;import math, random;from tensorflow.keras import layers;import tensorflow as tf;import torch;import sys;import matplotlib.pyplot as plt;from tensorflow import keras;import tensorflow.keras.backend as K;from tensorflow.keras.layers import Dense, Dropout, Conv2D;from pprint.xyz.abc import pprint as ppr;from tensorflow.keras.backend import clear_session, set_session;from tf import trans as tf1;import torch.nn.functional as F;import tensorflow.compat.v1 as tf;import torch.nn as nn;from tensorflow.keras.layers import Dense, Dropout as drop;import numpy as np;from tensorflow.keras.layers import Dense as dense_layer, Dropout as drop, Conv2D as conv;import tensorflow.keras.layers as L;import torch.optim as optim;import torchvision.transforms as transforms;from pt import trans as tf2;import torchvision', function_to_run='tf.value(*args, **kwargs)', method_object=None, function_args=[eval('"aname"'), eval('bname')], function_kwargs={'name': eval('"tname"')}, max_wait_secs=30)
w_o = custom_method(
tf.Variable(tf.random_uniform([layer_1_neurons, 1], minval=-1, maxval=1, dtype=tf.float32), test1, test2, test3='test3'), imports='from tensorflow.keras.layers import *;from os.path import exists;import math, random;from tensorflow.keras import layers;import tensorflow as tf;import torch;import sys;import matplotlib.pyplot as plt;from tensorflow import keras;import tensorflow.keras.backend as K;from tensorflow.keras.layers import Dense, Dropout, Conv2D;from pprint.xyz.abc import pprint as ppr;from tensorflow.keras.backend import clear_session, set_session;from tf import trans as tf1;import torch.nn.functional as F;import tensorflow.compat.v1 as tf;import torch.nn as nn;from tensorflow.keras.layers import Dense, Dropout as drop;import numpy as np;from tensorflow.keras.layers import Dense as dense_layer, Dropout as drop, Conv2D as conv;import tensorflow.keras.layers as L;import torch.optim as optim;import torchvision.transforms as transforms;from pt import trans as tf2;import torchvision', function_to_run='tf.Variable(*args, **kwargs)', method_object=None, function_args=[eval('tf.random_uniform([layer_1_neurons, 1], minval = -1, maxval = 1, dtype = tf.float32)'), eval('test1'), eval('test2')], function_kwargs={'test3': eval('"test3"')}, max_wait_secs=30)
b_o = custom_method(
tf.Variable(tf.zeros([1, 1], dtype=tf.float32)), imports='from tensorflow.keras.layers import *;from os.path import exists;import math, random;from tensorflow.keras import layers;import tensorflow as tf;import torch;import sys;import matplotlib.pyplot as plt;from tensorflow import keras;import tensorflow.keras.backend as K;from tensorflow.keras.layers import Dense, Dropout, Conv2D;from pprint.xyz.abc import pprint as ppr;from tensorflow.keras.backend import clear_session, set_session;from tf import trans as tf1;import torch.nn.functional as F;import tensorflow.compat.v1 as tf;import torch.nn as nn;from tensorflow.keras.layers import Dense, Dropout as drop;import numpy as np;from tensorflow.keras.layers import Dense as dense_layer, Dropout as drop, Conv2D as conv;import tensorflow.keras.layers as L;import torch.optim as optim;import torchvision.transforms as transforms;from pt import trans as tf2;import torchvision', function_to_run='tf.Variable(*args)', method_object=None, function_args=[eval('tf.zeros([1, 1], dtype = tf.float32)')], function_kwargs={}, max_wait_secs=30)
cd = ab.test(a, b)
c_o = xf.test()
d_o = xy.test()
model = custom_method(
tf.keras.models.Sequential([tf.keras.layers.Flatten(input_shape=(28, 28)), tf.keras.layers.Dense(128, activation='relu'), tf.keras.layers.Dropout(0.2), tf.keras.layers.Dense(10)]), imports='from tensorflow.keras.layers import *;from os.path import exists;import math, random;from tensorflow.keras import layers;import tensorflow as tf;import torch;import sys;import matplotlib.pyplot as plt;from tensorflow import keras;import tensorflow.keras.backend as K;from tensorflow.keras.layers import Dense, Dropout, Conv2D;from pprint.xyz.abc import pprint as ppr;from tensorflow.keras.backend import clear_session, set_session;from tf import trans as tf1;import torch.nn.functional as F;import tensorflow.compat.v1 as tf;import torch.nn as nn;from tensorflow.keras.layers import Dense, Dropout as drop;import numpy as np;from tensorflow.keras.layers import Dense as dense_layer, Dropout as drop, Conv2D as conv;import tensorflow.keras.layers as L;import torch.optim as optim;import torchvision.transforms as transforms;from pt import trans as tf2;import torchvision', function_to_run='tf.keras.models.Sequential(*args)', method_object=None, function_args=[eval("[\n    tf.keras.layers.Flatten(input_shape=(28, 28)),\n    tf.keras.layers.Dense(128, activation='relu'),\n    tf.keras.layers.Dropout(0.2),\n    tf.keras.layers.Dense(10)\n    ]")], function_kwargs={}, max_wait_secs=30)
custom_method(
model.compile(), imports='from tensorflow.keras.layers import *;from os.path import exists;import math, random;from tensorflow.keras import layers;import tensorflow as tf;import torch;import sys;import matplotlib.pyplot as plt;from tensorflow import keras;import tensorflow.keras.backend as K;from tensorflow.keras.layers import Dense, Dropout, Conv2D;from pprint.xyz.abc import pprint as ppr;from tensorflow.keras.backend import clear_session, set_session;from tf import trans as tf1;import torch.nn.functional as F;import tensorflow.compat.v1 as tf;import torch.nn as nn;from tensorflow.keras.layers import Dense, Dropout as drop;import numpy as np;from tensorflow.keras.layers import Dense as dense_layer, Dropout as drop, Conv2D as conv;import tensorflow.keras.layers as L;import torch.optim as optim;import torchvision.transforms as transforms;from pt import trans as tf2;import torchvision', function_to_run='obj.compile()', method_object='model', function_args=[], function_kwargs={}, max_wait_secs=30)
f_o = custom_method(
b_o.test.compile(), imports='from tensorflow.keras.layers import *;from os.path import exists;import math, random;from tensorflow.keras import layers;import tensorflow as tf;import torch;import sys;import matplotlib.pyplot as plt;from tensorflow import keras;import tensorflow.keras.backend as K;from tensorflow.keras.layers import Dense, Dropout, Conv2D;from pprint.xyz.abc import pprint as ppr;from tensorflow.keras.backend import clear_session, set_session;from tf import trans as tf1;import torch.nn.functional as F;import tensorflow.compat.v1 as tf;import torch.nn as nn;from tensorflow.keras.layers import Dense, Dropout as drop;import numpy as np;from tensorflow.keras.layers import Dense as dense_layer, Dropout as drop, Conv2D as conv;import tensorflow.keras.layers as L;import torch.optim as optim;import torchvision.transforms as transforms;from pt import trans as tf2;import torchvision', function_to_run='obj.test.compile()', method_object='b_o', function_args=[], function_kwargs={}, max_wait_secs=30)
z_o = c_o.compile()
