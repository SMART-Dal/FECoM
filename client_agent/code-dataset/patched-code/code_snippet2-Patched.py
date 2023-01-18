requiredClass dict_keys(['Net', 'MyDenseLayer'])
requiredObjects ['X', 'Y', 'z', 'w_o', 'b_o', 'model', 'transform', 'trainset', 'trainloader', 'testset', 'testloader', 'x', 'x', 'x', 'net', 'layer', 'layer', 'criterion', 'optimizer', 'net']
requiredObjClassMapping {'net': 'Net', 'layer': 'MyDenseLayer'}
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
                            arg(
                                arg='imports',
                                annotation=Name(id='str', ctx=Load())),
                            arg(
                                arg='function_to_run',
                                annotation=Name(id='str', ctx=Load())),
                            arg(arg='method_object'),
                            arg(
                                arg='function_args',
                                annotation=Name(id='list', ctx=Load())),
                            arg(
                                arg='function_kwargs',
                                annotation=Name(id='dict', ctx=Load())),
                            arg(arg='max_wait_secs'),
                            arg(arg='custom_class')],
                        kwonlyargs=[],
                        kw_defaults=[],
                        defaults=[
                            Constant(value=None),
                            Constant(value=None),
                            Constant(value=None),
                            Constant(value=0),
                            Constant(value=None)]),
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
            module='torch',
            names=[
                alias(name='trans')],
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
                        value=Constant(value='import numpy as np;from tensorflow.keras.backend import clear_session, set_session;from pt import trans as tf2;import matplotlib.pyplot as plt;import torch.nn as nn;import tensorflow.keras.layers as L;from tensorflow import keras;from tensorflow.keras.layers import Dense, Dropout as drop;import tensorflow as tf;from tensorflow.keras.layers import Dense as dense_layer, Dropout as drop, Conv2D as conv;from tensorflow.keras.layers import *;import sys;from tensorflow.keras import layers;import torchvision.transforms as transforms;import torch.nn.functional as F;import math, random;from os.path import exists;from pprint.xyz.abc import pprint as ppr;import torch.optim as optim;import tensorflow.compat.v1 as tf;import tensorflow.keras.backend as K;import torchvision;from torch import trans;import torch;from tensorflow.keras.layers import Dense, Dropout, Conv2D')),
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
                        value=Constant(value='import numpy as np;from tensorflow.keras.backend import clear_session, set_session;from pt import trans as tf2;import matplotlib.pyplot as plt;import torch.nn as nn;import tensorflow.keras.layers as L;from tensorflow import keras;from tensorflow.keras.layers import Dense, Dropout as drop;import tensorflow as tf;from tensorflow.keras.layers import Dense as dense_layer, Dropout as drop, Conv2D as conv;from tensorflow.keras.layers import *;import sys;from tensorflow.keras import layers;import torchvision.transforms as transforms;import torch.nn.functional as F;import math, random;from os.path import exists;from pprint.xyz.abc import pprint as ppr;import torch.optim as optim;import tensorflow.compat.v1 as tf;import tensorflow.keras.backend as K;import torchvision;from torch import trans;import torch;from tensorflow.keras.layers import Dense, Dropout, Conv2D')),
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
                        value=Constant(value='import numpy as np;from tensorflow.keras.backend import clear_session, set_session;from pt import trans as tf2;import matplotlib.pyplot as plt;import torch.nn as nn;import tensorflow.keras.layers as L;from tensorflow import keras;from tensorflow.keras.layers import Dense, Dropout as drop;import tensorflow as tf;from tensorflow.keras.layers import Dense as dense_layer, Dropout as drop, Conv2D as conv;from tensorflow.keras.layers import *;import sys;from tensorflow.keras import layers;import torchvision.transforms as transforms;import torch.nn.functional as F;import math, random;from os.path import exists;from pprint.xyz.abc import pprint as ppr;import torch.optim as optim;import tensorflow.compat.v1 as tf;import tensorflow.keras.backend as K;import torchvision;from torch import trans;import torch;from tensorflow.keras.layers import Dense, Dropout, Conv2D')),
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
                        value=Constant(value='import numpy as np;from tensorflow.keras.backend import clear_session, set_session;from pt import trans as tf2;import matplotlib.pyplot as plt;import torch.nn as nn;import tensorflow.keras.layers as L;from tensorflow import keras;from tensorflow.keras.layers import Dense, Dropout as drop;import tensorflow as tf;from tensorflow.keras.layers import Dense as dense_layer, Dropout as drop, Conv2D as conv;from tensorflow.keras.layers import *;import sys;from tensorflow.keras import layers;import torchvision.transforms as transforms;import torch.nn.functional as F;import math, random;from os.path import exists;from pprint.xyz.abc import pprint as ppr;import torch.optim as optim;import tensorflow.compat.v1 as tf;import tensorflow.keras.backend as K;import torchvision;from torch import trans;import torch;from tensorflow.keras.layers import Dense, Dropout, Conv2D')),
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
                        value=Constant(value='import numpy as np;from tensorflow.keras.backend import clear_session, set_session;from pt import trans as tf2;import matplotlib.pyplot as plt;import torch.nn as nn;import tensorflow.keras.layers as L;from tensorflow import keras;from tensorflow.keras.layers import Dense, Dropout as drop;import tensorflow as tf;from tensorflow.keras.layers import Dense as dense_layer, Dropout as drop, Conv2D as conv;from tensorflow.keras.layers import *;import sys;from tensorflow.keras import layers;import torchvision.transforms as transforms;import torch.nn.functional as F;import math, random;from os.path import exists;from pprint.xyz.abc import pprint as ppr;import torch.optim as optim;import tensorflow.compat.v1 as tf;import tensorflow.keras.backend as K;import torchvision;from torch import trans;import torch;from tensorflow.keras.layers import Dense, Dropout, Conv2D')),
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
                        value=Constant(value='import numpy as np;from tensorflow.keras.backend import clear_session, set_session;from pt import trans as tf2;import matplotlib.pyplot as plt;import torch.nn as nn;import tensorflow.keras.layers as L;from tensorflow import keras;from tensorflow.keras.layers import Dense, Dropout as drop;import tensorflow as tf;from tensorflow.keras.layers import Dense as dense_layer, Dropout as drop, Conv2D as conv;from tensorflow.keras.layers import *;import sys;from tensorflow.keras import layers;import torchvision.transforms as transforms;import torch.nn.functional as F;import math, random;from os.path import exists;from pprint.xyz.abc import pprint as ppr;import torch.optim as optim;import tensorflow.compat.v1 as tf;import tensorflow.keras.backend as K;import torchvision;from torch import trans;import torch;from tensorflow.keras.layers import Dense, Dropout, Conv2D')),
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
                func=Name(id='custom_method', ctx=Load()),
                args=[
                    Expr(
                        value=Call(
                            func=Attribute(
                                value=Name(id='xf', ctx=Load()),
                                attr='test',
                                ctx=Load()),
                            args=[],
                            keywords=[]))],
                keywords=[
                    keyword(
                        arg='imports',
                        value=Constant(value='import numpy as np;from tensorflow.keras.backend import clear_session, set_session;from pt import trans as tf2;import matplotlib.pyplot as plt;import torch.nn as nn;import tensorflow.keras.layers as L;from tensorflow import keras;from tensorflow.keras.layers import Dense, Dropout as drop;import tensorflow as tf;from tensorflow.keras.layers import Dense as dense_layer, Dropout as drop, Conv2D as conv;from tensorflow.keras.layers import *;import sys;from tensorflow.keras import layers;import torchvision.transforms as transforms;import torch.nn.functional as F;import math, random;from os.path import exists;from pprint.xyz.abc import pprint as ppr;import torch.optim as optim;import tensorflow.compat.v1 as tf;import tensorflow.keras.backend as K;import torchvision;from torch import trans;import torch;from tensorflow.keras.layers import Dense, Dropout, Conv2D')),
                    keyword(
                        arg='function_to_run',
                        value=Constant(value='obj.test()')),
                    keyword(
                        arg='method_object',
                        value=Constant(value='xf')),
                    keyword(
                        arg='function_args',
                        value=List(elts=[], ctx=Load())),
                    keyword(
                        arg='function_kwargs',
                        value=Dict(keys=[], values=[])),
                    keyword(
                        arg='max_wait_secs',
                        value=Constant(value=30)),
                    keyword(
                        arg='custom_class',
                        value=Constant(value=None))])),
        Assign(
            targets=[
                Name(id='d_o', ctx=Store())],
            value=Call(
                func=Name(id='custom_method', ctx=Load()),
                args=[
                    Expr(
                        value=Call(
                            func=Attribute(
                                value=Name(id='xy', ctx=Load()),
                                attr='test',
                                ctx=Load()),
                            args=[],
                            keywords=[]))],
                keywords=[
                    keyword(
                        arg='imports',
                        value=Constant(value='import numpy as np;from tensorflow.keras.backend import clear_session, set_session;from pt import trans as tf2;import matplotlib.pyplot as plt;import torch.nn as nn;import tensorflow.keras.layers as L;from tensorflow import keras;from tensorflow.keras.layers import Dense, Dropout as drop;import tensorflow as tf;from tensorflow.keras.layers import Dense as dense_layer, Dropout as drop, Conv2D as conv;from tensorflow.keras.layers import *;import sys;from tensorflow.keras import layers;import torchvision.transforms as transforms;import torch.nn.functional as F;import math, random;from os.path import exists;from pprint.xyz.abc import pprint as ppr;import torch.optim as optim;import tensorflow.compat.v1 as tf;import tensorflow.keras.backend as K;import torchvision;from torch import trans;import torch;from tensorflow.keras.layers import Dense, Dropout, Conv2D')),
                    keyword(
                        arg='function_to_run',
                        value=Constant(value='obj.test()')),
                    keyword(
                        arg='method_object',
                        value=Constant(value='xy')),
                    keyword(
                        arg='function_args',
                        value=List(elts=[], ctx=Load())),
                    keyword(
                        arg='function_kwargs',
                        value=Dict(keys=[], values=[])),
                    keyword(
                        arg='max_wait_secs',
                        value=Constant(value=30)),
                    keyword(
                        arg='custom_class',
                        value=Constant(value=None))])),
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
                        value=Constant(value='import numpy as np;from tensorflow.keras.backend import clear_session, set_session;from pt import trans as tf2;import matplotlib.pyplot as plt;import torch.nn as nn;import tensorflow.keras.layers as L;from tensorflow import keras;from tensorflow.keras.layers import Dense, Dropout as drop;import tensorflow as tf;from tensorflow.keras.layers import Dense as dense_layer, Dropout as drop, Conv2D as conv;from tensorflow.keras.layers import *;import sys;from tensorflow.keras import layers;import torchvision.transforms as transforms;import torch.nn.functional as F;import math, random;from os.path import exists;from pprint.xyz.abc import pprint as ppr;import torch.optim as optim;import tensorflow.compat.v1 as tf;import tensorflow.keras.backend as K;import torchvision;from torch import trans;import torch;from tensorflow.keras.layers import Dense, Dropout, Conv2D')),
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
                        value=Constant(value='import numpy as np;from tensorflow.keras.backend import clear_session, set_session;from pt import trans as tf2;import matplotlib.pyplot as plt;import torch.nn as nn;import tensorflow.keras.layers as L;from tensorflow import keras;from tensorflow.keras.layers import Dense, Dropout as drop;import tensorflow as tf;from tensorflow.keras.layers import Dense as dense_layer, Dropout as drop, Conv2D as conv;from tensorflow.keras.layers import *;import sys;from tensorflow.keras import layers;import torchvision.transforms as transforms;import torch.nn.functional as F;import math, random;from os.path import exists;from pprint.xyz.abc import pprint as ppr;import torch.optim as optim;import tensorflow.compat.v1 as tf;import tensorflow.keras.backend as K;import torchvision;from torch import trans;import torch;from tensorflow.keras.layers import Dense, Dropout, Conv2D')),
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
                        value=Constant(value=30)),
                    keyword(
                        arg='custom_class',
                        value=Constant(value=None))])),
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
                        value=Constant(value='import numpy as np;from tensorflow.keras.backend import clear_session, set_session;from pt import trans as tf2;import matplotlib.pyplot as plt;import torch.nn as nn;import tensorflow.keras.layers as L;from tensorflow import keras;from tensorflow.keras.layers import Dense, Dropout as drop;import tensorflow as tf;from tensorflow.keras.layers import Dense as dense_layer, Dropout as drop, Conv2D as conv;from tensorflow.keras.layers import *;import sys;from tensorflow.keras import layers;import torchvision.transforms as transforms;import torch.nn.functional as F;import math, random;from os.path import exists;from pprint.xyz.abc import pprint as ppr;import torch.optim as optim;import tensorflow.compat.v1 as tf;import tensorflow.keras.backend as K;import torchvision;from torch import trans;import torch;from tensorflow.keras.layers import Dense, Dropout, Conv2D')),
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
                        value=Constant(value=30)),
                    keyword(
                        arg='custom_class',
                        value=Constant(value=None))])),
        Assign(
            targets=[
                Name(id='z_o', ctx=Store())],
            value=Call(
                func=Attribute(
                    value=Name(id='c_o', ctx=Load()),
                    attr='compile',
                    ctx=Load()),
                args=[],
                keywords=[])),
        Import(
            names=[
                alias(name='torch')]),
        Import(
            names=[
                alias(name='torchvision')]),
        Import(
            names=[
                alias(name='torchvision.transforms', asname='transforms')]),
        Import(
            names=[
                alias(name='torch.nn', asname='nn')]),
        Import(
            names=[
                alias(name='torch.nn.functional', asname='F')]),
        Import(
            names=[
                alias(name='torch.optim', asname='optim')]),
        Assign(
            targets=[
                Name(id='transform', ctx=Store())],
            value=Call(
                func=Name(id='custom_method', ctx=Load()),
                args=[
                    Expr(
                        value=Call(
                            func=Attribute(
                                value=Name(id='transforms', ctx=Load()),
                                attr='Compose',
                                ctx=Load()),
                            args=[
                                List(
                                    elts=[
                                        Call(
                                            func=Attribute(
                                                value=Name(id='transforms', ctx=Load()),
                                                attr='ToTensor',
                                                ctx=Load()),
                                            args=[],
                                            keywords=[]),
                                        Call(
                                            func=Attribute(
                                                value=Name(id='transforms', ctx=Load()),
                                                attr='Normalize',
                                                ctx=Load()),
                                            args=[
                                                Tuple(
                                                    elts=[
                                                        Constant(value=0.5),
                                                        Constant(value=0.5),
                                                        Constant(value=0.5)],
                                                    ctx=Load()),
                                                Tuple(
                                                    elts=[
                                                        Constant(value=0.5),
                                                        Constant(value=0.5),
                                                        Constant(value=0.5)],
                                                    ctx=Load())],
                                            keywords=[])],
                                    ctx=Load())],
                            keywords=[]))],
                keywords=[
                    keyword(
                        arg='imports',
                        value=Constant(value='import numpy as np;from tensorflow.keras.backend import clear_session, set_session;from pt import trans as tf2;import matplotlib.pyplot as plt;import torch.nn as nn;import tensorflow.keras.layers as L;from tensorflow import keras;from tensorflow.keras.layers import Dense, Dropout as drop;import tensorflow as tf;from tensorflow.keras.layers import Dense as dense_layer, Dropout as drop, Conv2D as conv;from tensorflow.keras.layers import *;import sys;from tensorflow.keras import layers;import torchvision.transforms as transforms;import torch.nn.functional as F;import math, random;from os.path import exists;from pprint.xyz.abc import pprint as ppr;import torch.optim as optim;import tensorflow.compat.v1 as tf;import tensorflow.keras.backend as K;import torchvision;from torch import trans;import torch;from tensorflow.keras.layers import Dense, Dropout, Conv2D')),
                    keyword(
                        arg='function_to_run',
                        value=Constant(value='transforms.Compose(*args)')),
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
                                        Constant(value='[transforms.ToTensor(),\n     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]')],
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
                Name(id='batch_size', ctx=Store())],
            value=Constant(value=4)),
        Assign(
            targets=[
                Name(id='trainset', ctx=Store())],
            value=Call(
                func=Name(id='custom_method', ctx=Load()),
                args=[
                    Expr(
                        value=Call(
                            func=Attribute(
                                value=Attribute(
                                    value=Name(id='torchvision', ctx=Load()),
                                    attr='datasets',
                                    ctx=Load()),
                                attr='CIFAR10',
                                ctx=Load()),
                            args=[],
                            keywords=[
                                keyword(
                                    arg='root',
                                    value=Constant(value='./data')),
                                keyword(
                                    arg='train',
                                    value=Constant(value=True)),
                                keyword(
                                    arg='download',
                                    value=Constant(value=True)),
                                keyword(
                                    arg='transform',
                                    value=Name(id='transform', ctx=Load()))]))],
                keywords=[
                    keyword(
                        arg='imports',
                        value=Constant(value='import numpy as np;from tensorflow.keras.backend import clear_session, set_session;from pt import trans as tf2;import matplotlib.pyplot as plt;import torch.nn as nn;import tensorflow.keras.layers as L;from tensorflow import keras;from tensorflow.keras.layers import Dense, Dropout as drop;import tensorflow as tf;from tensorflow.keras.layers import Dense as dense_layer, Dropout as drop, Conv2D as conv;from tensorflow.keras.layers import *;import sys;from tensorflow.keras import layers;import torchvision.transforms as transforms;import torch.nn.functional as F;import math, random;from os.path import exists;from pprint.xyz.abc import pprint as ppr;import torch.optim as optim;import tensorflow.compat.v1 as tf;import tensorflow.keras.backend as K;import torchvision;from torch import trans;import torch;from tensorflow.keras.layers import Dense, Dropout, Conv2D')),
                    keyword(
                        arg='function_to_run',
                        value=Constant(value='torchvision.datasets.CIFAR10(**kwargs)')),
                    keyword(
                        arg='method_object',
                        value=Constant(value=None)),
                    keyword(
                        arg='function_args',
                        value=List(elts=[], ctx=Load())),
                    keyword(
                        arg='function_kwargs',
                        value=Dict(
                            keys=[
                                Constant(value='root'),
                                Constant(value='train'),
                                Constant(value='download'),
                                Constant(value='transform')],
                            values=[
                                Call(
                                    func=Name(id='eval', ctx=Load()),
                                    args=[
                                        Constant(value="'./data'")],
                                    keywords=[]),
                                Call(
                                    func=Name(id='eval', ctx=Load()),
                                    args=[
                                        Constant(value='True')],
                                    keywords=[]),
                                Call(
                                    func=Name(id='eval', ctx=Load()),
                                    args=[
                                        Constant(value='True')],
                                    keywords=[]),
                                Call(
                                    func=Name(id='eval', ctx=Load()),
                                    args=[
                                        Constant(value='transform')],
                                    keywords=[])])),
                    keyword(
                        arg='max_wait_secs',
                        value=Constant(value=30))])),
        Assign(
            targets=[
                Name(id='trainloader', ctx=Store())],
            value=Call(
                func=Name(id='custom_method', ctx=Load()),
                args=[
                    Expr(
                        value=Call(
                            func=Attribute(
                                value=Attribute(
                                    value=Attribute(
                                        value=Name(id='torch', ctx=Load()),
                                        attr='utils',
                                        ctx=Load()),
                                    attr='data',
                                    ctx=Load()),
                                attr='DataLoader',
                                ctx=Load()),
                            args=[
                                Name(id='trainset', ctx=Load())],
                            keywords=[
                                keyword(
                                    arg='batch_size',
                                    value=Name(id='batch_size', ctx=Load())),
                                keyword(
                                    arg='shuffle',
                                    value=Constant(value=True)),
                                keyword(
                                    arg='num_workers',
                                    value=Constant(value=0))]))],
                keywords=[
                    keyword(
                        arg='imports',
                        value=Constant(value='import numpy as np;from tensorflow.keras.backend import clear_session, set_session;from pt import trans as tf2;import matplotlib.pyplot as plt;import torch.nn as nn;import tensorflow.keras.layers as L;from tensorflow import keras;from tensorflow.keras.layers import Dense, Dropout as drop;import tensorflow as tf;from tensorflow.keras.layers import Dense as dense_layer, Dropout as drop, Conv2D as conv;from tensorflow.keras.layers import *;import sys;from tensorflow.keras import layers;import torchvision.transforms as transforms;import torch.nn.functional as F;import math, random;from os.path import exists;from pprint.xyz.abc import pprint as ppr;import torch.optim as optim;import tensorflow.compat.v1 as tf;import tensorflow.keras.backend as K;import torchvision;from torch import trans;import torch;from tensorflow.keras.layers import Dense, Dropout, Conv2D')),
                    keyword(
                        arg='function_to_run',
                        value=Constant(value='torch.utils.data.DataLoader(*args, **kwargs)')),
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
                                        Constant(value='trainset')],
                                    keywords=[])],
                            ctx=Load())),
                    keyword(
                        arg='function_kwargs',
                        value=Dict(
                            keys=[
                                Constant(value='batch_size'),
                                Constant(value='shuffle'),
                                Constant(value='num_workers')],
                            values=[
                                Call(
                                    func=Name(id='eval', ctx=Load()),
                                    args=[
                                        Constant(value='batch_size')],
                                    keywords=[]),
                                Call(
                                    func=Name(id='eval', ctx=Load()),
                                    args=[
                                        Constant(value='True')],
                                    keywords=[]),
                                Call(
                                    func=Name(id='eval', ctx=Load()),
                                    args=[
                                        Constant(value='0')],
                                    keywords=[])])),
                    keyword(
                        arg='max_wait_secs',
                        value=Constant(value=30))])),
        Assign(
            targets=[
                Name(id='testset', ctx=Store())],
            value=Call(
                func=Name(id='custom_method', ctx=Load()),
                args=[
                    Expr(
                        value=Call(
                            func=Attribute(
                                value=Attribute(
                                    value=Name(id='torchvision', ctx=Load()),
                                    attr='datasets',
                                    ctx=Load()),
                                attr='CIFAR10',
                                ctx=Load()),
                            args=[],
                            keywords=[
                                keyword(
                                    arg='root',
                                    value=Constant(value='./data')),
                                keyword(
                                    arg='train',
                                    value=Constant(value=False)),
                                keyword(
                                    arg='download',
                                    value=Constant(value=True)),
                                keyword(
                                    arg='transform',
                                    value=Name(id='transform', ctx=Load()))]))],
                keywords=[
                    keyword(
                        arg='imports',
                        value=Constant(value='import numpy as np;from tensorflow.keras.backend import clear_session, set_session;from pt import trans as tf2;import matplotlib.pyplot as plt;import torch.nn as nn;import tensorflow.keras.layers as L;from tensorflow import keras;from tensorflow.keras.layers import Dense, Dropout as drop;import tensorflow as tf;from tensorflow.keras.layers import Dense as dense_layer, Dropout as drop, Conv2D as conv;from tensorflow.keras.layers import *;import sys;from tensorflow.keras import layers;import torchvision.transforms as transforms;import torch.nn.functional as F;import math, random;from os.path import exists;from pprint.xyz.abc import pprint as ppr;import torch.optim as optim;import tensorflow.compat.v1 as tf;import tensorflow.keras.backend as K;import torchvision;from torch import trans;import torch;from tensorflow.keras.layers import Dense, Dropout, Conv2D')),
                    keyword(
                        arg='function_to_run',
                        value=Constant(value='torchvision.datasets.CIFAR10(**kwargs)')),
                    keyword(
                        arg='method_object',
                        value=Constant(value=None)),
                    keyword(
                        arg='function_args',
                        value=List(elts=[], ctx=Load())),
                    keyword(
                        arg='function_kwargs',
                        value=Dict(
                            keys=[
                                Constant(value='root'),
                                Constant(value='train'),
                                Constant(value='download'),
                                Constant(value='transform')],
                            values=[
                                Call(
                                    func=Name(id='eval', ctx=Load()),
                                    args=[
                                        Constant(value="'./data'")],
                                    keywords=[]),
                                Call(
                                    func=Name(id='eval', ctx=Load()),
                                    args=[
                                        Constant(value='False')],
                                    keywords=[]),
                                Call(
                                    func=Name(id='eval', ctx=Load()),
                                    args=[
                                        Constant(value='True')],
                                    keywords=[]),
                                Call(
                                    func=Name(id='eval', ctx=Load()),
                                    args=[
                                        Constant(value='transform')],
                                    keywords=[])])),
                    keyword(
                        arg='max_wait_secs',
                        value=Constant(value=30))])),
        Assign(
            targets=[
                Name(id='testloader', ctx=Store())],
            value=Call(
                func=Name(id='custom_method', ctx=Load()),
                args=[
                    Expr(
                        value=Call(
                            func=Attribute(
                                value=Attribute(
                                    value=Attribute(
                                        value=Name(id='torch', ctx=Load()),
                                        attr='utils',
                                        ctx=Load()),
                                    attr='data',
                                    ctx=Load()),
                                attr='DataLoader',
                                ctx=Load()),
                            args=[
                                Name(id='testset', ctx=Load())],
                            keywords=[
                                keyword(
                                    arg='batch_size',
                                    value=Name(id='batch_size', ctx=Load())),
                                keyword(
                                    arg='shuffle',
                                    value=Constant(value=False)),
                                keyword(
                                    arg='num_workers',
                                    value=Constant(value=0))]))],
                keywords=[
                    keyword(
                        arg='imports',
                        value=Constant(value='import numpy as np;from tensorflow.keras.backend import clear_session, set_session;from pt import trans as tf2;import matplotlib.pyplot as plt;import torch.nn as nn;import tensorflow.keras.layers as L;from tensorflow import keras;from tensorflow.keras.layers import Dense, Dropout as drop;import tensorflow as tf;from tensorflow.keras.layers import Dense as dense_layer, Dropout as drop, Conv2D as conv;from tensorflow.keras.layers import *;import sys;from tensorflow.keras import layers;import torchvision.transforms as transforms;import torch.nn.functional as F;import math, random;from os.path import exists;from pprint.xyz.abc import pprint as ppr;import torch.optim as optim;import tensorflow.compat.v1 as tf;import tensorflow.keras.backend as K;import torchvision;from torch import trans;import torch;from tensorflow.keras.layers import Dense, Dropout, Conv2D')),
                    keyword(
                        arg='function_to_run',
                        value=Constant(value='torch.utils.data.DataLoader(*args, **kwargs)')),
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
                                        Constant(value='testset')],
                                    keywords=[])],
                            ctx=Load())),
                    keyword(
                        arg='function_kwargs',
                        value=Dict(
                            keys=[
                                Constant(value='batch_size'),
                                Constant(value='shuffle'),
                                Constant(value='num_workers')],
                            values=[
                                Call(
                                    func=Name(id='eval', ctx=Load()),
                                    args=[
                                        Constant(value='batch_size')],
                                    keywords=[]),
                                Call(
                                    func=Name(id='eval', ctx=Load()),
                                    args=[
                                        Constant(value='False')],
                                    keywords=[]),
                                Call(
                                    func=Name(id='eval', ctx=Load()),
                                    args=[
                                        Constant(value='0')],
                                    keywords=[])])),
                    keyword(
                        arg='max_wait_secs',
                        value=Constant(value=30))])),
        Assign(
            targets=[
                Name(id='classes', ctx=Store())],
            value=Tuple(
                elts=[
                    Constant(value='plane'),
                    Constant(value='car'),
                    Constant(value='bird'),
                    Constant(value='cat'),
                    Constant(value='deer'),
                    Constant(value='dog'),
                    Constant(value='frog'),
                    Constant(value='horse'),
                    Constant(value='ship'),
                    Constant(value='truck')],
                ctx=Load())),
        ClassDef(
            name='Net',
            bases=[
                Attribute(
                    value=Name(id='nn', ctx=Load()),
                    attr='Module',
                    ctx=Load()),
                Attribute(
                    value=Name(id='abc', ctx=Load()),
                    attr='Module',
                    ctx=Load())],
            keywords=[],
            body=[
                FunctionDef(
                    name='__init__',
                    args=arguments(
                        posonlyargs=[],
                        args=[
                            arg(arg='self')],
                        kwonlyargs=[],
                        kw_defaults=[],
                        defaults=[]),
                    body=[
                        Expr(
                            value=Call(
                                func=Attribute(
                                    value=Call(
                                        func=Name(id='super', ctx=Load()),
                                        args=[],
                                        keywords=[]),
                                    attr='__init__',
                                    ctx=Load()),
                                args=[],
                                keywords=[])),
                        Assign(
                            targets=[
                                Attribute(
                                    value=Name(id='self', ctx=Load()),
                                    attr='conv1',
                                    ctx=Store())],
                            value=Call(
                                func=Name(id='custom_method', ctx=Load()),
                                args=[
                                    Expr(
                                        value=Call(
                                            func=Attribute(
                                                value=Name(id='nn', ctx=Load()),
                                                attr='Conv2d',
                                                ctx=Load()),
                                            args=[
                                                Constant(value=3),
                                                Constant(value=6),
                                                Constant(value=5)],
                                            keywords=[]))],
                                keywords=[
                                    keyword(
                                        arg='imports',
                                        value=Constant(value='import numpy as np;from tensorflow.keras.backend import clear_session, set_session;from pt import trans as tf2;import matplotlib.pyplot as plt;import torch.nn as nn;import tensorflow.keras.layers as L;from tensorflow import keras;from tensorflow.keras.layers import Dense, Dropout as drop;import tensorflow as tf;from tensorflow.keras.layers import Dense as dense_layer, Dropout as drop, Conv2D as conv;from tensorflow.keras.layers import *;import sys;from tensorflow.keras import layers;import torchvision.transforms as transforms;import torch.nn.functional as F;import math, random;from os.path import exists;from pprint.xyz.abc import pprint as ppr;import torch.optim as optim;import tensorflow.compat.v1 as tf;import tensorflow.keras.backend as K;import torchvision;from torch import trans;import torch;from tensorflow.keras.layers import Dense, Dropout, Conv2D')),
                                    keyword(
                                        arg='function_to_run',
                                        value=Constant(value='nn.Conv2d(*args)')),
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
                                                        Constant(value='3')],
                                                    keywords=[]),
                                                Call(
                                                    func=Name(id='eval', ctx=Load()),
                                                    args=[
                                                        Constant(value='6')],
                                                    keywords=[]),
                                                Call(
                                                    func=Name(id='eval', ctx=Load()),
                                                    args=[
                                                        Constant(value='5')],
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
                                Attribute(
                                    value=Name(id='self', ctx=Load()),
                                    attr='pool',
                                    ctx=Store())],
                            value=Call(
                                func=Name(id='custom_method', ctx=Load()),
                                args=[
                                    Expr(
                                        value=Call(
                                            func=Attribute(
                                                value=Name(id='nn', ctx=Load()),
                                                attr='MaxPool2d',
                                                ctx=Load()),
                                            args=[
                                                Constant(value=2),
                                                Constant(value=2)],
                                            keywords=[]))],
                                keywords=[
                                    keyword(
                                        arg='imports',
                                        value=Constant(value='import numpy as np;from tensorflow.keras.backend import clear_session, set_session;from pt import trans as tf2;import matplotlib.pyplot as plt;import torch.nn as nn;import tensorflow.keras.layers as L;from tensorflow import keras;from tensorflow.keras.layers import Dense, Dropout as drop;import tensorflow as tf;from tensorflow.keras.layers import Dense as dense_layer, Dropout as drop, Conv2D as conv;from tensorflow.keras.layers import *;import sys;from tensorflow.keras import layers;import torchvision.transforms as transforms;import torch.nn.functional as F;import math, random;from os.path import exists;from pprint.xyz.abc import pprint as ppr;import torch.optim as optim;import tensorflow.compat.v1 as tf;import tensorflow.keras.backend as K;import torchvision;from torch import trans;import torch;from tensorflow.keras.layers import Dense, Dropout, Conv2D')),
                                    keyword(
                                        arg='function_to_run',
                                        value=Constant(value='nn.MaxPool2d(*args)')),
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
                                                        Constant(value='2')],
                                                    keywords=[]),
                                                Call(
                                                    func=Name(id='eval', ctx=Load()),
                                                    args=[
                                                        Constant(value='2')],
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
                                Attribute(
                                    value=Name(id='self', ctx=Load()),
                                    attr='conv2',
                                    ctx=Store())],
                            value=Call(
                                func=Name(id='custom_method', ctx=Load()),
                                args=[
                                    Expr(
                                        value=Call(
                                            func=Attribute(
                                                value=Name(id='nn', ctx=Load()),
                                                attr='Conv2d',
                                                ctx=Load()),
                                            args=[
                                                Constant(value=6),
                                                Constant(value=16),
                                                Constant(value=5)],
                                            keywords=[]))],
                                keywords=[
                                    keyword(
                                        arg='imports',
                                        value=Constant(value='import numpy as np;from tensorflow.keras.backend import clear_session, set_session;from pt import trans as tf2;import matplotlib.pyplot as plt;import torch.nn as nn;import tensorflow.keras.layers as L;from tensorflow import keras;from tensorflow.keras.layers import Dense, Dropout as drop;import tensorflow as tf;from tensorflow.keras.layers import Dense as dense_layer, Dropout as drop, Conv2D as conv;from tensorflow.keras.layers import *;import sys;from tensorflow.keras import layers;import torchvision.transforms as transforms;import torch.nn.functional as F;import math, random;from os.path import exists;from pprint.xyz.abc import pprint as ppr;import torch.optim as optim;import tensorflow.compat.v1 as tf;import tensorflow.keras.backend as K;import torchvision;from torch import trans;import torch;from tensorflow.keras.layers import Dense, Dropout, Conv2D')),
                                    keyword(
                                        arg='function_to_run',
                                        value=Constant(value='nn.Conv2d(*args)')),
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
                                                        Constant(value='6')],
                                                    keywords=[]),
                                                Call(
                                                    func=Name(id='eval', ctx=Load()),
                                                    args=[
                                                        Constant(value='16')],
                                                    keywords=[]),
                                                Call(
                                                    func=Name(id='eval', ctx=Load()),
                                                    args=[
                                                        Constant(value='5')],
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
                                Attribute(
                                    value=Name(id='self', ctx=Load()),
                                    attr='fc1',
                                    ctx=Store())],
                            value=Call(
                                func=Name(id='custom_method', ctx=Load()),
                                args=[
                                    Expr(
                                        value=Call(
                                            func=Attribute(
                                                value=Name(id='nn', ctx=Load()),
                                                attr='Linear',
                                                ctx=Load()),
                                            args=[
                                                BinOp(
                                                    left=BinOp(
                                                        left=Constant(value=16),
                                                        op=Mult(),
                                                        right=Constant(value=5)),
                                                    op=Mult(),
                                                    right=Constant(value=5)),
                                                Constant(value=120)],
                                            keywords=[]))],
                                keywords=[
                                    keyword(
                                        arg='imports',
                                        value=Constant(value='import numpy as np;from tensorflow.keras.backend import clear_session, set_session;from pt import trans as tf2;import matplotlib.pyplot as plt;import torch.nn as nn;import tensorflow.keras.layers as L;from tensorflow import keras;from tensorflow.keras.layers import Dense, Dropout as drop;import tensorflow as tf;from tensorflow.keras.layers import Dense as dense_layer, Dropout as drop, Conv2D as conv;from tensorflow.keras.layers import *;import sys;from tensorflow.keras import layers;import torchvision.transforms as transforms;import torch.nn.functional as F;import math, random;from os.path import exists;from pprint.xyz.abc import pprint as ppr;import torch.optim as optim;import tensorflow.compat.v1 as tf;import tensorflow.keras.backend as K;import torchvision;from torch import trans;import torch;from tensorflow.keras.layers import Dense, Dropout, Conv2D')),
                                    keyword(
                                        arg='function_to_run',
                                        value=Constant(value='nn.Linear(*args)')),
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
                                                        Constant(value='16 * 5 * 5')],
                                                    keywords=[]),
                                                Call(
                                                    func=Name(id='eval', ctx=Load()),
                                                    args=[
                                                        Constant(value='120')],
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
                                Attribute(
                                    value=Name(id='self', ctx=Load()),
                                    attr='fc2',
                                    ctx=Store())],
                            value=Call(
                                func=Name(id='custom_method', ctx=Load()),
                                args=[
                                    Expr(
                                        value=Call(
                                            func=Attribute(
                                                value=Name(id='nn', ctx=Load()),
                                                attr='Linear',
                                                ctx=Load()),
                                            args=[
                                                Constant(value=120),
                                                Constant(value=84)],
                                            keywords=[]))],
                                keywords=[
                                    keyword(
                                        arg='imports',
                                        value=Constant(value='import numpy as np;from tensorflow.keras.backend import clear_session, set_session;from pt import trans as tf2;import matplotlib.pyplot as plt;import torch.nn as nn;import tensorflow.keras.layers as L;from tensorflow import keras;from tensorflow.keras.layers import Dense, Dropout as drop;import tensorflow as tf;from tensorflow.keras.layers import Dense as dense_layer, Dropout as drop, Conv2D as conv;from tensorflow.keras.layers import *;import sys;from tensorflow.keras import layers;import torchvision.transforms as transforms;import torch.nn.functional as F;import math, random;from os.path import exists;from pprint.xyz.abc import pprint as ppr;import torch.optim as optim;import tensorflow.compat.v1 as tf;import tensorflow.keras.backend as K;import torchvision;from torch import trans;import torch;from tensorflow.keras.layers import Dense, Dropout, Conv2D')),
                                    keyword(
                                        arg='function_to_run',
                                        value=Constant(value='nn.Linear(*args)')),
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
                                                        Constant(value='120')],
                                                    keywords=[]),
                                                Call(
                                                    func=Name(id='eval', ctx=Load()),
                                                    args=[
                                                        Constant(value='84')],
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
                                Attribute(
                                    value=Name(id='self', ctx=Load()),
                                    attr='fc3',
                                    ctx=Store())],
                            value=Call(
                                func=Name(id='custom_method', ctx=Load()),
                                args=[
                                    Expr(
                                        value=Call(
                                            func=Attribute(
                                                value=Name(id='nn', ctx=Load()),
                                                attr='Linear',
                                                ctx=Load()),
                                            args=[
                                                Constant(value=84),
                                                Constant(value=10)],
                                            keywords=[]))],
                                keywords=[
                                    keyword(
                                        arg='imports',
                                        value=Constant(value='import numpy as np;from tensorflow.keras.backend import clear_session, set_session;from pt import trans as tf2;import matplotlib.pyplot as plt;import torch.nn as nn;import tensorflow.keras.layers as L;from tensorflow import keras;from tensorflow.keras.layers import Dense, Dropout as drop;import tensorflow as tf;from tensorflow.keras.layers import Dense as dense_layer, Dropout as drop, Conv2D as conv;from tensorflow.keras.layers import *;import sys;from tensorflow.keras import layers;import torchvision.transforms as transforms;import torch.nn.functional as F;import math, random;from os.path import exists;from pprint.xyz.abc import pprint as ppr;import torch.optim as optim;import tensorflow.compat.v1 as tf;import tensorflow.keras.backend as K;import torchvision;from torch import trans;import torch;from tensorflow.keras.layers import Dense, Dropout, Conv2D')),
                                    keyword(
                                        arg='function_to_run',
                                        value=Constant(value='nn.Linear(*args)')),
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
                                                        Constant(value='84')],
                                                    keywords=[]),
                                                Call(
                                                    func=Name(id='eval', ctx=Load()),
                                                    args=[
                                                        Constant(value='10')],
                                                    keywords=[])],
                                            ctx=Load())),
                                    keyword(
                                        arg='function_kwargs',
                                        value=Dict(keys=[], values=[])),
                                    keyword(
                                        arg='max_wait_secs',
                                        value=Constant(value=30))]))],
                    decorator_list=[]),
                FunctionDef(
                    name='forward',
                    args=arguments(
                        posonlyargs=[],
                        args=[
                            arg(arg='self'),
                            arg(arg='x')],
                        kwonlyargs=[],
                        kw_defaults=[],
                        defaults=[]),
                    body=[
                        Assign(
                            targets=[
                                Name(id='x', ctx=Store())],
                            value=Call(
                                func=Attribute(
                                    value=Name(id='self', ctx=Load()),
                                    attr='pool',
                                    ctx=Load()),
                                args=[
                                    Call(
                                        func=Attribute(
                                            value=Name(id='F', ctx=Load()),
                                            attr='relu',
                                            ctx=Load()),
                                        args=[
                                            Call(
                                                func=Attribute(
                                                    value=Name(id='self', ctx=Load()),
                                                    attr='conv1',
                                                    ctx=Load()),
                                                args=[
                                                    Name(id='x', ctx=Load())],
                                                keywords=[])],
                                        keywords=[])],
                                keywords=[])),
                        Assign(
                            targets=[
                                Name(id='x', ctx=Store())],
                            value=Call(
                                func=Attribute(
                                    value=Name(id='self', ctx=Load()),
                                    attr='pool',
                                    ctx=Load()),
                                args=[
                                    Call(
                                        func=Attribute(
                                            value=Name(id='F', ctx=Load()),
                                            attr='relu',
                                            ctx=Load()),
                                        args=[
                                            Call(
                                                func=Attribute(
                                                    value=Name(id='self', ctx=Load()),
                                                    attr='conv2',
                                                    ctx=Load()),
                                                args=[
                                                    Name(id='x', ctx=Load())],
                                                keywords=[])],
                                        keywords=[])],
                                keywords=[])),
                        Assign(
                            targets=[
                                Name(id='x', ctx=Store())],
                            value=Call(
                                func=Name(id='custom_method', ctx=Load()),
                                args=[
                                    Expr(
                                        value=Call(
                                            func=Attribute(
                                                value=Name(id='torch', ctx=Load()),
                                                attr='flatten',
                                                ctx=Load()),
                                            args=[
                                                Name(id='x', ctx=Load()),
                                                Constant(value=1)],
                                            keywords=[]))],
                                keywords=[
                                    keyword(
                                        arg='imports',
                                        value=Constant(value='import numpy as np;from tensorflow.keras.backend import clear_session, set_session;from pt import trans as tf2;import matplotlib.pyplot as plt;import torch.nn as nn;import tensorflow.keras.layers as L;from tensorflow import keras;from tensorflow.keras.layers import Dense, Dropout as drop;import tensorflow as tf;from tensorflow.keras.layers import Dense as dense_layer, Dropout as drop, Conv2D as conv;from tensorflow.keras.layers import *;import sys;from tensorflow.keras import layers;import torchvision.transforms as transforms;import torch.nn.functional as F;import math, random;from os.path import exists;from pprint.xyz.abc import pprint as ppr;import torch.optim as optim;import tensorflow.compat.v1 as tf;import tensorflow.keras.backend as K;import torchvision;from torch import trans;import torch;from tensorflow.keras.layers import Dense, Dropout, Conv2D')),
                                    keyword(
                                        arg='function_to_run',
                                        value=Constant(value='torch.flatten(*args)')),
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
                                                        Constant(value='x')],
                                                    keywords=[]),
                                                Call(
                                                    func=Name(id='eval', ctx=Load()),
                                                    args=[
                                                        Constant(value='1')],
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
                                Name(id='x', ctx=Store())],
                            value=Call(
                                func=Name(id='custom_method', ctx=Load()),
                                args=[
                                    Expr(
                                        value=Call(
                                            func=Attribute(
                                                value=Name(id='F', ctx=Load()),
                                                attr='relu',
                                                ctx=Load()),
                                            args=[
                                                Call(
                                                    func=Attribute(
                                                        value=Name(id='self', ctx=Load()),
                                                        attr='fc1',
                                                        ctx=Load()),
                                                    args=[
                                                        Name(id='x', ctx=Load())],
                                                    keywords=[])],
                                            keywords=[]))],
                                keywords=[
                                    keyword(
                                        arg='imports',
                                        value=Constant(value='import numpy as np;from tensorflow.keras.backend import clear_session, set_session;from pt import trans as tf2;import matplotlib.pyplot as plt;import torch.nn as nn;import tensorflow.keras.layers as L;from tensorflow import keras;from tensorflow.keras.layers import Dense, Dropout as drop;import tensorflow as tf;from tensorflow.keras.layers import Dense as dense_layer, Dropout as drop, Conv2D as conv;from tensorflow.keras.layers import *;import sys;from tensorflow.keras import layers;import torchvision.transforms as transforms;import torch.nn.functional as F;import math, random;from os.path import exists;from pprint.xyz.abc import pprint as ppr;import torch.optim as optim;import tensorflow.compat.v1 as tf;import tensorflow.keras.backend as K;import torchvision;from torch import trans;import torch;from tensorflow.keras.layers import Dense, Dropout, Conv2D')),
                                    keyword(
                                        arg='function_to_run',
                                        value=Constant(value='F.relu(*args)')),
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
                                                        Constant(value='self.fc1(x)')],
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
                                Name(id='x', ctx=Store())],
                            value=Call(
                                func=Name(id='custom_method', ctx=Load()),
                                args=[
                                    Expr(
                                        value=Call(
                                            func=Attribute(
                                                value=Name(id='F', ctx=Load()),
                                                attr='relu',
                                                ctx=Load()),
                                            args=[
                                                Call(
                                                    func=Attribute(
                                                        value=Name(id='self', ctx=Load()),
                                                        attr='fc2',
                                                        ctx=Load()),
                                                    args=[
                                                        Name(id='x', ctx=Load())],
                                                    keywords=[])],
                                            keywords=[]))],
                                keywords=[
                                    keyword(
                                        arg='imports',
                                        value=Constant(value='import numpy as np;from tensorflow.keras.backend import clear_session, set_session;from pt import trans as tf2;import matplotlib.pyplot as plt;import torch.nn as nn;import tensorflow.keras.layers as L;from tensorflow import keras;from tensorflow.keras.layers import Dense, Dropout as drop;import tensorflow as tf;from tensorflow.keras.layers import Dense as dense_layer, Dropout as drop, Conv2D as conv;from tensorflow.keras.layers import *;import sys;from tensorflow.keras import layers;import torchvision.transforms as transforms;import torch.nn.functional as F;import math, random;from os.path import exists;from pprint.xyz.abc import pprint as ppr;import torch.optim as optim;import tensorflow.compat.v1 as tf;import tensorflow.keras.backend as K;import torchvision;from torch import trans;import torch;from tensorflow.keras.layers import Dense, Dropout, Conv2D')),
                                    keyword(
                                        arg='function_to_run',
                                        value=Constant(value='F.relu(*args)')),
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
                                                        Constant(value='self.fc2(x)')],
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
                                Name(id='x', ctx=Store())],
                            value=Call(
                                func=Attribute(
                                    value=Name(id='self', ctx=Load()),
                                    attr='fc3',
                                    ctx=Load()),
                                args=[
                                    Name(id='x', ctx=Load())],
                                keywords=[])),
                        Return(
                            value=Name(id='x', ctx=Load()))],
                    decorator_list=[])],
            decorator_list=[]),
        ClassDef(
            name='dummyClass',
            bases=[
                Attribute(
                    value=Attribute(
                        value=Attribute(
                            value=Attribute(
                                value=Name(id='abc', ctx=Load()),
                                attr='tf',
                                ctx=Load()),
                            attr='keras',
                            ctx=Load()),
                        attr='layers',
                        ctx=Load()),
                    attr='Layer',
                    ctx=Load()),
                Attribute(
                    value=Attribute(
                        value=Name(id='abc', ctx=Load()),
                        attr='nn',
                        ctx=Load()),
                    attr='Module',
                    ctx=Load())],
            keywords=[],
            body=[
                FunctionDef(
                    name='__init__',
                    args=arguments(
                        posonlyargs=[],
                        args=[
                            arg(arg='self')],
                        kwonlyargs=[],
                        kw_defaults=[],
                        defaults=[]),
                    body=[
                        Assign(
                            targets=[
                                Attribute(
                                    value=Name(id='self', ctx=Load()),
                                    attr='dummy',
                                    ctx=Store())],
                            value=Constant(value=0))],
                    decorator_list=[]),
                FunctionDef(
                    name='dummyFunc',
                    args=arguments(
                        posonlyargs=[],
                        args=[
                            arg(arg='self')],
                        kwonlyargs=[],
                        kw_defaults=[],
                        defaults=[]),
                    body=[
                        AugAssign(
                            target=Attribute(
                                value=Name(id='self', ctx=Load()),
                                attr='dummy',
                                ctx=Store()),
                            op=Add(),
                            value=Constant(value=1)),
                        Return(
                            value=Attribute(
                                value=Name(id='self', ctx=Load()),
                                attr='dummy',
                                ctx=Load()))],
                    decorator_list=[])],
            decorator_list=[]),
        ClassDef(
            name='dummyClass2',
            bases=[],
            keywords=[],
            body=[
                FunctionDef(
                    name='__init__',
                    args=arguments(
                        posonlyargs=[],
                        args=[
                            arg(arg='self')],
                        kwonlyargs=[],
                        kw_defaults=[],
                        defaults=[]),
                    body=[
                        Assign(
                            targets=[
                                Attribute(
                                    value=Name(id='self', ctx=Load()),
                                    attr='dummy',
                                    ctx=Store())],
                            value=Constant(value=0))],
                    decorator_list=[]),
                FunctionDef(
                    name='dummyFunc',
                    args=arguments(
                        posonlyargs=[],
                        args=[
                            arg(arg='self')],
                        kwonlyargs=[],
                        kw_defaults=[],
                        defaults=[]),
                    body=[
                        AugAssign(
                            target=Attribute(
                                value=Name(id='self', ctx=Load()),
                                attr='dummy',
                                ctx=Store()),
                            op=Add(),
                            value=Constant(value=1)),
                        Return(
                            value=Attribute(
                                value=Name(id='self', ctx=Load()),
                                attr='dummy',
                                ctx=Load()))],
                    decorator_list=[])],
            decorator_list=[]),
        Assign(
            targets=[
                Name(id='net', ctx=Store())],
            value=Call(
                func=Name(id='Net', ctx=Load()),
                args=[],
                keywords=[])),
        ClassDef(
            name='MyDenseLayer',
            bases=[
                Attribute(
                    value=Attribute(
                        value=Attribute(
                            value=Name(id='tf', ctx=Load()),
                            attr='keras',
                            ctx=Load()),
                        attr='layers',
                        ctx=Load()),
                    attr='Layer',
                    ctx=Load()),
                Attribute(
                    value=Name(id='nn', ctx=Load()),
                    attr='Module',
                    ctx=Load())],
            keywords=[],
            body=[
                FunctionDef(
                    name='__init__',
                    args=arguments(
                        posonlyargs=[],
                        args=[
                            arg(arg='self'),
                            arg(arg='num_outputs')],
                        kwonlyargs=[],
                        kw_defaults=[],
                        defaults=[]),
                    body=[
                        Expr(
                            value=Call(
                                func=Name(id='custom_method', ctx=Load()),
                                args=[
                                    Expr(
                                        value=Call(
                                            func=Attribute(
                                                value=Call(
                                                    func=Name(id='super', ctx=Load()),
                                                    args=[
                                                        Name(id='MyDenseLayer', ctx=Load()),
                                                        Name(id='self', ctx=Load())],
                                                    keywords=[]),
                                                attr='__init__',
                                                ctx=Load()),
                                            args=[],
                                            keywords=[]))],
                                keywords=[
                                    keyword(
                                        arg='imports',
                                        value=Constant(value='import numpy as np;from tensorflow.keras.backend import clear_session, set_session;from pt import trans as tf2;import matplotlib.pyplot as plt;import torch.nn as nn;import tensorflow.keras.layers as L;from tensorflow import keras;from tensorflow.keras.layers import Dense, Dropout as drop;import tensorflow as tf;from tensorflow.keras.layers import Dense as dense_layer, Dropout as drop, Conv2D as conv;from tensorflow.keras.layers import *;import sys;from tensorflow.keras import layers;import torchvision.transforms as transforms;import torch.nn.functional as F;import math, random;from os.path import exists;from pprint.xyz.abc import pprint as ppr;import torch.optim as optim;import tensorflow.compat.v1 as tf;import tensorflow.keras.backend as K;import torchvision;from torch import trans;import torch;from tensorflow.keras.layers import Dense, Dropout, Conv2D')),
                                    keyword(
                                        arg='function_to_run',
                                        value=Constant(value='super(MyDenseLayer, self).__init__()')),
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
                                Attribute(
                                    value=Name(id='self', ctx=Load()),
                                    attr='num_outputs',
                                    ctx=Store())],
                            value=Name(id='num_outputs', ctx=Load()))],
                    decorator_list=[]),
                FunctionDef(
                    name='build',
                    args=arguments(
                        posonlyargs=[],
                        args=[
                            arg(arg='self'),
                            arg(arg='input_shape')],
                        kwonlyargs=[],
                        kw_defaults=[],
                        defaults=[]),
                    body=[
                        Assign(
                            targets=[
                                Attribute(
                                    value=Name(id='self', ctx=Load()),
                                    attr='kernel',
                                    ctx=Store())],
                            value=Call(
                                func=Attribute(
                                    value=Name(id='self', ctx=Load()),
                                    attr='add_weight',
                                    ctx=Load()),
                                args=[
                                    Constant(value='kernel')],
                                keywords=[
                                    keyword(
                                        arg='shape',
                                        value=List(
                                            elts=[
                                                Call(
                                                    func=Name(id='int', ctx=Load()),
                                                    args=[
                                                        Subscript(
                                                            value=Name(id='input_shape', ctx=Load()),
                                                            slice=UnaryOp(
                                                                op=USub(),
                                                                operand=Constant(value=1)),
                                                            ctx=Load())],
                                                    keywords=[]),
                                                Attribute(
                                                    value=Name(id='self', ctx=Load()),
                                                    attr='num_outputs',
                                                    ctx=Load())],
                                            ctx=Load()))]))],
                    decorator_list=[]),
                FunctionDef(
                    name='call',
                    args=arguments(
                        posonlyargs=[],
                        args=[
                            arg(arg='self'),
                            arg(arg='inputs')],
                        kwonlyargs=[],
                        kw_defaults=[],
                        defaults=[]),
                    body=[
                        Return(
                            value=Call(
                                func=Name(id='custom_method', ctx=Load()),
                                args=[
                                    Expr(
                                        value=Call(
                                            func=Attribute(
                                                value=Name(id='tf', ctx=Load()),
                                                attr='matmul',
                                                ctx=Load()),
                                            args=[
                                                Name(id='inputs', ctx=Load()),
                                                Attribute(
                                                    value=Name(id='self', ctx=Load()),
                                                    attr='kernel',
                                                    ctx=Load())],
                                            keywords=[]))],
                                keywords=[
                                    keyword(
                                        arg='imports',
                                        value=Constant(value='import numpy as np;from tensorflow.keras.backend import clear_session, set_session;from pt import trans as tf2;import matplotlib.pyplot as plt;import torch.nn as nn;import tensorflow.keras.layers as L;from tensorflow import keras;from tensorflow.keras.layers import Dense, Dropout as drop;import tensorflow as tf;from tensorflow.keras.layers import Dense as dense_layer, Dropout as drop, Conv2D as conv;from tensorflow.keras.layers import *;import sys;from tensorflow.keras import layers;import torchvision.transforms as transforms;import torch.nn.functional as F;import math, random;from os.path import exists;from pprint.xyz.abc import pprint as ppr;import torch.optim as optim;import tensorflow.compat.v1 as tf;import tensorflow.keras.backend as K;import torchvision;from torch import trans;import torch;from tensorflow.keras.layers import Dense, Dropout, Conv2D')),
                                    keyword(
                                        arg='function_to_run',
                                        value=Constant(value='tf.matmul(*args)')),
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
                                                        Constant(value='inputs')],
                                                    keywords=[]),
                                                Call(
                                                    func=Name(id='eval', ctx=Load()),
                                                    args=[
                                                        Constant(value='self.kernel')],
                                                    keywords=[])],
                                            ctx=Load())),
                                    keyword(
                                        arg='function_kwargs',
                                        value=Dict(keys=[], values=[])),
                                    keyword(
                                        arg='max_wait_secs',
                                        value=Constant(value=30))]))],
                    decorator_list=[])],
            decorator_list=[]),
        Assign(
            targets=[
                Name(id='layer', ctx=Store())],
            value=Call(
                func=Name(id='custom_method', ctx=Load()),
                args=[
                    Expr(
                        value=Call(
                            func=Name(id='MyDenseLayer', ctx=Load()),
                            args=[
                                Constant(value=10)],
                            keywords=[]))],
                keywords=[
                    keyword(
                        arg='imports',
                        value=Constant(value='import numpy as np;from tensorflow.keras.backend import clear_session, set_session;from pt import trans as tf2;import matplotlib.pyplot as plt;import torch.nn as nn;import tensorflow.keras.layers as L;from tensorflow import keras;from tensorflow.keras.layers import Dense, Dropout as drop;import tensorflow as tf;from tensorflow.keras.layers import Dense as dense_layer, Dropout as drop, Conv2D as conv;from tensorflow.keras.layers import *;import sys;from tensorflow.keras import layers;import torchvision.transforms as transforms;import torch.nn.functional as F;import math, random;from os.path import exists;from pprint.xyz.abc import pprint as ppr;import torch.optim as optim;import tensorflow.compat.v1 as tf;import tensorflow.keras.backend as K;import torchvision;from torch import trans;import torch;from tensorflow.keras.layers import Dense, Dropout, Conv2D')),
                    keyword(
                        arg='function_to_run',
                        value=Constant(value='MyDenseLayer(*args)')),
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
                                        Constant(value='10')],
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
                Name(id='criterion', ctx=Store())],
            value=Call(
                func=Name(id='custom_method', ctx=Load()),
                args=[
                    Expr(
                        value=Call(
                            func=Attribute(
                                value=Name(id='nn', ctx=Load()),
                                attr='CrossEntropyLoss',
                                ctx=Load()),
                            args=[],
                            keywords=[]))],
                keywords=[
                    keyword(
                        arg='imports',
                        value=Constant(value='import numpy as np;from tensorflow.keras.backend import clear_session, set_session;from pt import trans as tf2;import matplotlib.pyplot as plt;import torch.nn as nn;import tensorflow.keras.layers as L;from tensorflow import keras;from tensorflow.keras.layers import Dense, Dropout as drop;import tensorflow as tf;from tensorflow.keras.layers import Dense as dense_layer, Dropout as drop, Conv2D as conv;from tensorflow.keras.layers import *;import sys;from tensorflow.keras import layers;import torchvision.transforms as transforms;import torch.nn.functional as F;import math, random;from os.path import exists;from pprint.xyz.abc import pprint as ppr;import torch.optim as optim;import tensorflow.compat.v1 as tf;import tensorflow.keras.backend as K;import torchvision;from torch import trans;import torch;from tensorflow.keras.layers import Dense, Dropout, Conv2D')),
                    keyword(
                        arg='function_to_run',
                        value=Constant(value='nn.CrossEntropyLoss()')),
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
                Name(id='optimizer', ctx=Store())],
            value=Call(
                func=Name(id='custom_method', ctx=Load()),
                args=[
                    Expr(
                        value=Call(
                            func=Attribute(
                                value=Name(id='optim', ctx=Load()),
                                attr='SGD',
                                ctx=Load()),
                            args=[
                                Call(
                                    func=Attribute(
                                        value=Name(id='net', ctx=Load()),
                                        attr='parameters',
                                        ctx=Load()),
                                    args=[],
                                    keywords=[])],
                            keywords=[
                                keyword(
                                    arg='lr',
                                    value=Constant(value=0.001)),
                                keyword(
                                    arg='momentum',
                                    value=Constant(value=0.9))]))],
                keywords=[
                    keyword(
                        arg='imports',
                        value=Constant(value='import numpy as np;from tensorflow.keras.backend import clear_session, set_session;from pt import trans as tf2;import matplotlib.pyplot as plt;import torch.nn as nn;import tensorflow.keras.layers as L;from tensorflow import keras;from tensorflow.keras.layers import Dense, Dropout as drop;import tensorflow as tf;from tensorflow.keras.layers import Dense as dense_layer, Dropout as drop, Conv2D as conv;from tensorflow.keras.layers import *;import sys;from tensorflow.keras import layers;import torchvision.transforms as transforms;import torch.nn.functional as F;import math, random;from os.path import exists;from pprint.xyz.abc import pprint as ppr;import torch.optim as optim;import tensorflow.compat.v1 as tf;import tensorflow.keras.backend as K;import torchvision;from torch import trans;import torch;from tensorflow.keras.layers import Dense, Dropout, Conv2D')),
                    keyword(
                        arg='function_to_run',
                        value=Constant(value='optim.SGD(*args, **kwargs)')),
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
                                        Constant(value='net.parameters()')],
                                    keywords=[])],
                            ctx=Load())),
                    keyword(
                        arg='function_kwargs',
                        value=Dict(
                            keys=[
                                Constant(value='lr'),
                                Constant(value='momentum')],
                            values=[
                                Call(
                                    func=Name(id='eval', ctx=Load()),
                                    args=[
                                        Constant(value='0.001')],
                                    keywords=[]),
                                Call(
                                    func=Name(id='eval', ctx=Load()),
                                    args=[
                                        Constant(value='0.9')],
                                    keywords=[])])),
                    keyword(
                        arg='max_wait_secs',
                        value=Constant(value=30))])),
        For(
            target=Name(id='epoch', ctx=Store()),
            iter=Call(
                func=Name(id='range', ctx=Load()),
                args=[
                    Constant(value=2)],
                keywords=[]),
            body=[
                Assign(
                    targets=[
                        Name(id='running_loss', ctx=Store())],
                    value=Constant(value=0.0)),
                For(
                    target=Tuple(
                        elts=[
                            Name(id='i', ctx=Store()),
                            Name(id='data', ctx=Store())],
                        ctx=Store()),
                    iter=Call(
                        func=Name(id='enumerate', ctx=Load()),
                        args=[
                            Name(id='trainloader', ctx=Load()),
                            Constant(value=0)],
                        keywords=[]),
                    body=[
                        Assign(
                            targets=[
                                Tuple(
                                    elts=[
                                        Name(id='inputs', ctx=Store()),
                                        Name(id='labels', ctx=Store())],
                                    ctx=Store())],
                            value=Name(id='data', ctx=Load())),
                        Expr(
                            value=Call(
                                func=Name(id='custom_method', ctx=Load()),
                                args=[
                                    Expr(
                                        value=Call(
                                            func=Attribute(
                                                value=Name(id='optimizer', ctx=Load()),
                                                attr='zero_grad',
                                                ctx=Load()),
                                            args=[],
                                            keywords=[]))],
                                keywords=[
                                    keyword(
                                        arg='imports',
                                        value=Constant(value='import numpy as np;from tensorflow.keras.backend import clear_session, set_session;from pt import trans as tf2;import matplotlib.pyplot as plt;import torch.nn as nn;import tensorflow.keras.layers as L;from tensorflow import keras;from tensorflow.keras.layers import Dense, Dropout as drop;import tensorflow as tf;from tensorflow.keras.layers import Dense as dense_layer, Dropout as drop, Conv2D as conv;from tensorflow.keras.layers import *;import sys;from tensorflow.keras import layers;import torchvision.transforms as transforms;import torch.nn.functional as F;import math, random;from os.path import exists;from pprint.xyz.abc import pprint as ppr;import torch.optim as optim;import tensorflow.compat.v1 as tf;import tensorflow.keras.backend as K;import torchvision;from torch import trans;import torch;from tensorflow.keras.layers import Dense, Dropout, Conv2D')),
                                    keyword(
                                        arg='function_to_run',
                                        value=Constant(value='optimizer.zero_grad()')),
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
                                Name(id='outputs', ctx=Store())],
                            value=Call(
                                func=Name(id='custom_method', ctx=Load()),
                                args=[
                                    Expr(
                                        value=Call(
                                            func=Name(id='net', ctx=Load()),
                                            args=[
                                                Name(id='inputs', ctx=Load())],
                                            keywords=[]))],
                                keywords=[
                                    keyword(
                                        arg='imports',
                                        value=Constant(value='import numpy as np;from tensorflow.keras.backend import clear_session, set_session;from pt import trans as tf2;import matplotlib.pyplot as plt;import torch.nn as nn;import tensorflow.keras.layers as L;from tensorflow import keras;from tensorflow.keras.layers import Dense, Dropout as drop;import tensorflow as tf;from tensorflow.keras.layers import Dense as dense_layer, Dropout as drop, Conv2D as conv;from tensorflow.keras.layers import *;import sys;from tensorflow.keras import layers;import torchvision.transforms as transforms;import torch.nn.functional as F;import math, random;from os.path import exists;from pprint.xyz.abc import pprint as ppr;import torch.optim as optim;import tensorflow.compat.v1 as tf;import tensorflow.keras.backend as K;import torchvision;from torch import trans;import torch;from tensorflow.keras.layers import Dense, Dropout, Conv2D')),
                                    keyword(
                                        arg='function_to_run',
                                        value=Constant(value='obj(*args)')),
                                    keyword(
                                        arg='method_object',
                                        value=Constant(value='net')),
                                    keyword(
                                        arg='function_args',
                                        value=List(
                                            elts=[
                                                Call(
                                                    func=Name(id='eval', ctx=Load()),
                                                    args=[
                                                        Constant(value='inputs')],
                                                    keywords=[])],
                                            ctx=Load())),
                                    keyword(
                                        arg='function_kwargs',
                                        value=Dict(keys=[], values=[])),
                                    keyword(
                                        arg='max_wait_secs',
                                        value=Constant(value=30)),
                                    keyword(
                                        arg='custom_class',
                                        value=Constant(value='class Net(nn.Module, abc.Module):\n    def __init__(self):\n        super().__init__()\n        self.conv1 = nn.Conv2d(3, 6, 5)\n        self.pool = nn.MaxPool2d(2, 2)\n        self.conv2 = nn.Conv2d(6, 16, 5)\n        self.fc1 = nn.Linear(16 * 5 * 5, 120)\n        self.fc2 = nn.Linear(120, 84)\n        self.fc3 = nn.Linear(84, 10)\n\n    def forward(self, x):\n        x = self.pool(F.relu(self.conv1(x)))\n        x = self.pool(F.relu(self.conv2(x)))\n        x = torch.flatten(x, 1) # flatten all dimensions except batch\n        x = F.relu(self.fc1(x))\n        x = F.relu(self.fc2(x))\n        x = self.fc3(x)\n        return x'))])),
                        Assign(
                            targets=[
                                Name(id='loss', ctx=Store())],
                            value=Call(
                                func=Name(id='custom_method', ctx=Load()),
                                args=[
                                    Expr(
                                        value=Call(
                                            func=Name(id='criterion', ctx=Load()),
                                            args=[
                                                Name(id='outputs', ctx=Load()),
                                                Name(id='labels', ctx=Load())],
                                            keywords=[]))],
                                keywords=[
                                    keyword(
                                        arg='imports',
                                        value=Constant(value='import numpy as np;from tensorflow.keras.backend import clear_session, set_session;from pt import trans as tf2;import matplotlib.pyplot as plt;import torch.nn as nn;import tensorflow.keras.layers as L;from tensorflow import keras;from tensorflow.keras.layers import Dense, Dropout as drop;import tensorflow as tf;from tensorflow.keras.layers import Dense as dense_layer, Dropout as drop, Conv2D as conv;from tensorflow.keras.layers import *;import sys;from tensorflow.keras import layers;import torchvision.transforms as transforms;import torch.nn.functional as F;import math, random;from os.path import exists;from pprint.xyz.abc import pprint as ppr;import torch.optim as optim;import tensorflow.compat.v1 as tf;import tensorflow.keras.backend as K;import torchvision;from torch import trans;import torch;from tensorflow.keras.layers import Dense, Dropout, Conv2D')),
                                    keyword(
                                        arg='function_to_run',
                                        value=Constant(value='obj(*args)')),
                                    keyword(
                                        arg='method_object',
                                        value=Constant(value='criterion')),
                                    keyword(
                                        arg='function_args',
                                        value=List(
                                            elts=[
                                                Call(
                                                    func=Name(id='eval', ctx=Load()),
                                                    args=[
                                                        Constant(value='outputs')],
                                                    keywords=[]),
                                                Call(
                                                    func=Name(id='eval', ctx=Load()),
                                                    args=[
                                                        Constant(value='labels')],
                                                    keywords=[])],
                                            ctx=Load())),
                                    keyword(
                                        arg='function_kwargs',
                                        value=Dict(keys=[], values=[])),
                                    keyword(
                                        arg='max_wait_secs',
                                        value=Constant(value=30)),
                                    keyword(
                                        arg='custom_class',
                                        value=Constant(value=None))])),
                        Expr(
                            value=Call(
                                func=Attribute(
                                    value=Name(id='loss', ctx=Load()),
                                    attr='backward',
                                    ctx=Load()),
                                args=[],
                                keywords=[])),
                        Expr(
                            value=Call(
                                func=Name(id='custom_method', ctx=Load()),
                                args=[
                                    Expr(
                                        value=Call(
                                            func=Attribute(
                                                value=Name(id='optimizer', ctx=Load()),
                                                attr='step',
                                                ctx=Load()),
                                            args=[],
                                            keywords=[]))],
                                keywords=[
                                    keyword(
                                        arg='imports',
                                        value=Constant(value='import numpy as np;from tensorflow.keras.backend import clear_session, set_session;from pt import trans as tf2;import matplotlib.pyplot as plt;import torch.nn as nn;import tensorflow.keras.layers as L;from tensorflow import keras;from tensorflow.keras.layers import Dense, Dropout as drop;import tensorflow as tf;from tensorflow.keras.layers import Dense as dense_layer, Dropout as drop, Conv2D as conv;from tensorflow.keras.layers import *;import sys;from tensorflow.keras import layers;import torchvision.transforms as transforms;import torch.nn.functional as F;import math, random;from os.path import exists;from pprint.xyz.abc import pprint as ppr;import torch.optim as optim;import tensorflow.compat.v1 as tf;import tensorflow.keras.backend as K;import torchvision;from torch import trans;import torch;from tensorflow.keras.layers import Dense, Dropout, Conv2D')),
                                    keyword(
                                        arg='function_to_run',
                                        value=Constant(value='optimizer.step()')),
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
                        AugAssign(
                            target=Name(id='running_loss', ctx=Store()),
                            op=Add(),
                            value=Call(
                                func=Attribute(
                                    value=Name(id='loss', ctx=Load()),
                                    attr='item',
                                    ctx=Load()),
                                args=[],
                                keywords=[])),
                        If(
                            test=Compare(
                                left=BinOp(
                                    left=Name(id='i', ctx=Load()),
                                    op=Mod(),
                                    right=Constant(value=2000)),
                                ops=[
                                    Eq()],
                                comparators=[
                                    Constant(value=1999)]),
                            body=[
                                Expr(
                                    value=Call(
                                        func=Name(id='print', ctx=Load()),
                                        args=[
                                            JoinedStr(
                                                values=[
                                                    Constant(value='['),
                                                    FormattedValue(
                                                        value=BinOp(
                                                            left=Name(id='epoch', ctx=Load()),
                                                            op=Add(),
                                                            right=Constant(value=1)),
                                                        conversion=-1),
                                                    Constant(value=', '),
                                                    FormattedValue(
                                                        value=BinOp(
                                                            left=Name(id='i', ctx=Load()),
                                                            op=Add(),
                                                            right=Constant(value=1)),
                                                        conversion=-1,
                                                        format_spec=JoinedStr(
                                                            values=[
                                                                Constant(value='5d')])),
                                                    Constant(value='] loss: '),
                                                    FormattedValue(
                                                        value=BinOp(
                                                            left=Name(id='running_loss', ctx=Load()),
                                                            op=Div(),
                                                            right=Constant(value=2000)),
                                                        conversion=-1,
                                                        format_spec=JoinedStr(
                                                            values=[
                                                                Constant(value='.3f')]))])],
                                        keywords=[])),
                                Assign(
                                    targets=[
                                        Name(id='running_loss', ctx=Store())],
                                    value=Constant(value=0.0))],
                            orelse=[])],
                    orelse=[])],
            orelse=[]),
        Expr(
            value=Call(
                func=Name(id='print', ctx=Load()),
                args=[
                    Constant(value='Finished Training')],
                keywords=[])),
        Assign(
            targets=[
                Name(id='PATH', ctx=Store())],
            value=Constant(value='./cifar_net.pth')),
        Expr(
            value=Call(
                func=Name(id='custom_method', ctx=Load()),
                args=[
                    Expr(
                        value=Call(
                            func=Attribute(
                                value=Name(id='torch', ctx=Load()),
                                attr='save',
                                ctx=Load()),
                            args=[
                                Call(
                                    func=Attribute(
                                        value=Name(id='net', ctx=Load()),
                                        attr='state_dict',
                                        ctx=Load()),
                                    args=[],
                                    keywords=[]),
                                Name(id='PATH', ctx=Load())],
                            keywords=[]))],
                keywords=[
                    keyword(
                        arg='imports',
                        value=Constant(value='import numpy as np;from tensorflow.keras.backend import clear_session, set_session;from pt import trans as tf2;import matplotlib.pyplot as plt;import torch.nn as nn;import tensorflow.keras.layers as L;from tensorflow import keras;from tensorflow.keras.layers import Dense, Dropout as drop;import tensorflow as tf;from tensorflow.keras.layers import Dense as dense_layer, Dropout as drop, Conv2D as conv;from tensorflow.keras.layers import *;import sys;from tensorflow.keras import layers;import torchvision.transforms as transforms;import torch.nn.functional as F;import math, random;from os.path import exists;from pprint.xyz.abc import pprint as ppr;import torch.optim as optim;import tensorflow.compat.v1 as tf;import tensorflow.keras.backend as K;import torchvision;from torch import trans;import torch;from tensorflow.keras.layers import Dense, Dropout, Conv2D')),
                    keyword(
                        arg='function_to_run',
                        value=Constant(value='torch.save(*args)')),
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
                                        Constant(value='net.state_dict()')],
                                    keywords=[]),
                                Call(
                                    func=Name(id='eval', ctx=Load()),
                                    args=[
                                        Constant(value='PATH')],
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
                Name(id='dataiter', ctx=Store())],
            value=Call(
                func=Name(id='iter', ctx=Load()),
                args=[
                    Name(id='testloader', ctx=Load())],
                keywords=[])),
        Assign(
            targets=[
                Tuple(
                    elts=[
                        Name(id='images', ctx=Store()),
                        Name(id='labels', ctx=Store())],
                    ctx=Store())],
            value=Call(
                func=Name(id='custom_method', ctx=Load()),
                args=[
                    Expr(
                        value=Call(
                            func=Name(id='next', ctx=Load()),
                            args=[
                                Name(id='dataiter', ctx=Load())],
                            keywords=[]))],
                keywords=[
                    keyword(
                        arg='imports',
                        value=Constant(value='import numpy as np;from tensorflow.keras.backend import clear_session, set_session;from pt import trans as tf2;import matplotlib.pyplot as plt;import torch.nn as nn;import tensorflow.keras.layers as L;from tensorflow import keras;from tensorflow.keras.layers import Dense, Dropout as drop;import tensorflow as tf;from tensorflow.keras.layers import Dense as dense_layer, Dropout as drop, Conv2D as conv;from tensorflow.keras.layers import *;import sys;from tensorflow.keras import layers;import torchvision.transforms as transforms;import torch.nn.functional as F;import math, random;from os.path import exists;from pprint.xyz.abc import pprint as ppr;import torch.optim as optim;import tensorflow.compat.v1 as tf;import tensorflow.keras.backend as K;import torchvision;from torch import trans;import torch;from tensorflow.keras.layers import Dense, Dropout, Conv2D')),
                    keyword(
                        arg='function_to_run',
                        value=Constant(value='obj(*args)')),
                    keyword(
                        arg='method_object',
                        value=Constant(value='next')),
                    keyword(
                        arg='function_args',
                        value=List(
                            elts=[
                                Call(
                                    func=Name(id='eval', ctx=Load()),
                                    args=[
                                        Constant(value='dataiter')],
                                    keywords=[])],
                            ctx=Load())),
                    keyword(
                        arg='function_kwargs',
                        value=Dict(keys=[], values=[])),
                    keyword(
                        arg='max_wait_secs',
                        value=Constant(value=30)),
                    keyword(
                        arg='custom_class',
                        value=Constant(value=None))])),
        Assign(
            targets=[
                Name(id='net', ctx=Store())],
            value=Call(
                func=Name(id='Net', ctx=Load()),
                args=[],
                keywords=[])),
        Expr(
            value=Call(
                func=Name(id='custom_method', ctx=Load()),
                args=[
                    Expr(
                        value=Call(
                            func=Attribute(
                                value=Name(id='net', ctx=Load()),
                                attr='load_state_dict',
                                ctx=Load()),
                            args=[
                                Call(
                                    func=Attribute(
                                        value=Name(id='torch', ctx=Load()),
                                        attr='load',
                                        ctx=Load()),
                                    args=[
                                        Name(id='PATH', ctx=Load())],
                                    keywords=[])],
                            keywords=[]))],
                keywords=[
                    keyword(
                        arg='imports',
                        value=Constant(value='import numpy as np;from tensorflow.keras.backend import clear_session, set_session;from pt import trans as tf2;import matplotlib.pyplot as plt;import torch.nn as nn;import tensorflow.keras.layers as L;from tensorflow import keras;from tensorflow.keras.layers import Dense, Dropout as drop;import tensorflow as tf;from tensorflow.keras.layers import Dense as dense_layer, Dropout as drop, Conv2D as conv;from tensorflow.keras.layers import *;import sys;from tensorflow.keras import layers;import torchvision.transforms as transforms;import torch.nn.functional as F;import math, random;from os.path import exists;from pprint.xyz.abc import pprint as ppr;import torch.optim as optim;import tensorflow.compat.v1 as tf;import tensorflow.keras.backend as K;import torchvision;from torch import trans;import torch;from tensorflow.keras.layers import Dense, Dropout, Conv2D')),
                    keyword(
                        arg='function_to_run',
                        value=Constant(value='obj.load_state_dict(*args)')),
                    keyword(
                        arg='method_object',
                        value=Constant(value='net')),
                    keyword(
                        arg='function_args',
                        value=List(
                            elts=[
                                Call(
                                    func=Name(id='eval', ctx=Load()),
                                    args=[
                                        Constant(value='torch.load(PATH)')],
                                    keywords=[])],
                            ctx=Load())),
                    keyword(
                        arg='function_kwargs',
                        value=Dict(keys=[], values=[])),
                    keyword(
                        arg='max_wait_secs',
                        value=Constant(value=30)),
                    keyword(
                        arg='custom_class',
                        value=Constant(value='class Net(nn.Module, abc.Module):\n    def __init__(self):\n        super().__init__()\n        self.conv1 = nn.Conv2d(3, 6, 5)\n        self.pool = nn.MaxPool2d(2, 2)\n        self.conv2 = nn.Conv2d(6, 16, 5)\n        self.fc1 = nn.Linear(16 * 5 * 5, 120)\n        self.fc2 = nn.Linear(120, 84)\n        self.fc3 = nn.Linear(84, 10)\n\n    def forward(self, x):\n        x = self.pool(F.relu(self.conv1(x)))\n        x = self.pool(F.relu(self.conv2(x)))\n        x = torch.flatten(x, 1) # flatten all dimensions except batch\n        x = F.relu(self.fc1(x))\n        x = F.relu(self.fc2(x))\n        x = self.fc3(x)\n        return x'))])),
        Assign(
            targets=[
                Name(id='outputs', ctx=Store())],
            value=Call(
                func=Name(id='custom_method', ctx=Load()),
                args=[
                    Expr(
                        value=Call(
                            func=Name(id='net', ctx=Load()),
                            args=[
                                Name(id='images', ctx=Load())],
                            keywords=[]))],
                keywords=[
                    keyword(
                        arg='imports',
                        value=Constant(value='import numpy as np;from tensorflow.keras.backend import clear_session, set_session;from pt import trans as tf2;import matplotlib.pyplot as plt;import torch.nn as nn;import tensorflow.keras.layers as L;from tensorflow import keras;from tensorflow.keras.layers import Dense, Dropout as drop;import tensorflow as tf;from tensorflow.keras.layers import Dense as dense_layer, Dropout as drop, Conv2D as conv;from tensorflow.keras.layers import *;import sys;from tensorflow.keras import layers;import torchvision.transforms as transforms;import torch.nn.functional as F;import math, random;from os.path import exists;from pprint.xyz.abc import pprint as ppr;import torch.optim as optim;import tensorflow.compat.v1 as tf;import tensorflow.keras.backend as K;import torchvision;from torch import trans;import torch;from tensorflow.keras.layers import Dense, Dropout, Conv2D')),
                    keyword(
                        arg='function_to_run',
                        value=Constant(value='obj(*args)')),
                    keyword(
                        arg='method_object',
                        value=Constant(value='net')),
                    keyword(
                        arg='function_args',
                        value=List(
                            elts=[
                                Call(
                                    func=Name(id='eval', ctx=Load()),
                                    args=[
                                        Constant(value='images')],
                                    keywords=[])],
                            ctx=Load())),
                    keyword(
                        arg='function_kwargs',
                        value=Dict(keys=[], values=[])),
                    keyword(
                        arg='max_wait_secs',
                        value=Constant(value=30)),
                    keyword(
                        arg='custom_class',
                        value=Constant(value='class Net(nn.Module, abc.Module):\n    def __init__(self):\n        super().__init__()\n        self.conv1 = nn.Conv2d(3, 6, 5)\n        self.pool = nn.MaxPool2d(2, 2)\n        self.conv2 = nn.Conv2d(6, 16, 5)\n        self.fc1 = nn.Linear(16 * 5 * 5, 120)\n        self.fc2 = nn.Linear(120, 84)\n        self.fc3 = nn.Linear(84, 10)\n\n    def forward(self, x):\n        x = self.pool(F.relu(self.conv1(x)))\n        x = self.pool(F.relu(self.conv2(x)))\n        x = torch.flatten(x, 1) # flatten all dimensions except batch\n        x = F.relu(self.fc1(x))\n        x = F.relu(self.fc2(x))\n        x = self.fc3(x)\n        return x'))])),
        Assign(
            targets=[
                Tuple(
                    elts=[
                        Name(id='_', ctx=Store()),
                        Name(id='predicted', ctx=Store())],
                    ctx=Store())],
            value=Call(
                func=Name(id='custom_method', ctx=Load()),
                args=[
                    Expr(
                        value=Call(
                            func=Attribute(
                                value=Name(id='torch', ctx=Load()),
                                attr='max',
                                ctx=Load()),
                            args=[
                                Name(id='outputs', ctx=Load()),
                                Constant(value=1)],
                            keywords=[]))],
                keywords=[
                    keyword(
                        arg='imports',
                        value=Constant(value='import numpy as np;from tensorflow.keras.backend import clear_session, set_session;from pt import trans as tf2;import matplotlib.pyplot as plt;import torch.nn as nn;import tensorflow.keras.layers as L;from tensorflow import keras;from tensorflow.keras.layers import Dense, Dropout as drop;import tensorflow as tf;from tensorflow.keras.layers import Dense as dense_layer, Dropout as drop, Conv2D as conv;from tensorflow.keras.layers import *;import sys;from tensorflow.keras import layers;import torchvision.transforms as transforms;import torch.nn.functional as F;import math, random;from os.path import exists;from pprint.xyz.abc import pprint as ppr;import torch.optim as optim;import tensorflow.compat.v1 as tf;import tensorflow.keras.backend as K;import torchvision;from torch import trans;import torch;from tensorflow.keras.layers import Dense, Dropout, Conv2D')),
                    keyword(
                        arg='function_to_run',
                        value=Constant(value='torch.max(*args)')),
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
                                        Constant(value='outputs')],
                                    keywords=[]),
                                Call(
                                    func=Name(id='eval', ctx=Load()),
                                    args=[
                                        Constant(value='1')],
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
                func=Name(id='print', ctx=Load()),
                args=[
                    Constant(value='Predicted: '),
                    Call(
                        func=Attribute(
                            value=Constant(value=' '),
                            attr='join',
                            ctx=Load()),
                        args=[
                            GeneratorExp(
                                elt=JoinedStr(
                                    values=[
                                        FormattedValue(
                                            value=Subscript(
                                                value=Name(id='classes', ctx=Load()),
                                                slice=Subscript(
                                                    value=Name(id='predicted', ctx=Load()),
                                                    slice=Name(id='j', ctx=Load()),
                                                    ctx=Load()),
                                                ctx=Load()),
                                            conversion=-1,
                                            format_spec=JoinedStr(
                                                values=[
                                                    Constant(value='5s')]))]),
                                generators=[
                                    comprehension(
                                        target=Name(id='j', ctx=Store()),
                                        iter=Call(
                                            func=Name(id='range', ctx=Load()),
                                            args=[
                                                Constant(value=4)],
                                            keywords=[]),
                                        ifs=[],
                                        is_async=0)])],
                        keywords=[])],
                keywords=[])),
        Assign(
            targets=[
                Name(id='correct', ctx=Store())],
            value=Constant(value=0)),
        Assign(
            targets=[
                Name(id='total', ctx=Store())],
            value=Constant(value=0)),
        With(
            items=[
                withitem(
                    context_expr=Call(
                        func=Name(id='custom_method', ctx=Load()),
                        args=[
                            Expr(
                                value=Call(
                                    func=Attribute(
                                        value=Name(id='torch', ctx=Load()),
                                        attr='no_grad',
                                        ctx=Load()),
                                    args=[],
                                    keywords=[]))],
                        keywords=[
                            keyword(
                                arg='imports',
                                value=Constant(value='import numpy as np;from tensorflow.keras.backend import clear_session, set_session;from pt import trans as tf2;import matplotlib.pyplot as plt;import torch.nn as nn;import tensorflow.keras.layers as L;from tensorflow import keras;from tensorflow.keras.layers import Dense, Dropout as drop;import tensorflow as tf;from tensorflow.keras.layers import Dense as dense_layer, Dropout as drop, Conv2D as conv;from tensorflow.keras.layers import *;import sys;from tensorflow.keras import layers;import torchvision.transforms as transforms;import torch.nn.functional as F;import math, random;from os.path import exists;from pprint.xyz.abc import pprint as ppr;import torch.optim as optim;import tensorflow.compat.v1 as tf;import tensorflow.keras.backend as K;import torchvision;from torch import trans;import torch;from tensorflow.keras.layers import Dense, Dropout, Conv2D')),
                            keyword(
                                arg='function_to_run',
                                value=Constant(value='torch.no_grad()')),
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
                                value=Constant(value=30))]))],
            body=[
                For(
                    target=Name(id='data', ctx=Store()),
                    iter=Name(id='testloader', ctx=Load()),
                    body=[
                        Assign(
                            targets=[
                                Tuple(
                                    elts=[
                                        Name(id='images', ctx=Store()),
                                        Name(id='labels', ctx=Store())],
                                    ctx=Store())],
                            value=Name(id='data', ctx=Load())),
                        Assign(
                            targets=[
                                Name(id='outputs', ctx=Store())],
                            value=Call(
                                func=Name(id='custom_method', ctx=Load()),
                                args=[
                                    Expr(
                                        value=Call(
                                            func=Name(id='net', ctx=Load()),
                                            args=[
                                                Name(id='images', ctx=Load())],
                                            keywords=[]))],
                                keywords=[
                                    keyword(
                                        arg='imports',
                                        value=Constant(value='import numpy as np;from tensorflow.keras.backend import clear_session, set_session;from pt import trans as tf2;import matplotlib.pyplot as plt;import torch.nn as nn;import tensorflow.keras.layers as L;from tensorflow import keras;from tensorflow.keras.layers import Dense, Dropout as drop;import tensorflow as tf;from tensorflow.keras.layers import Dense as dense_layer, Dropout as drop, Conv2D as conv;from tensorflow.keras.layers import *;import sys;from tensorflow.keras import layers;import torchvision.transforms as transforms;import torch.nn.functional as F;import math, random;from os.path import exists;from pprint.xyz.abc import pprint as ppr;import torch.optim as optim;import tensorflow.compat.v1 as tf;import tensorflow.keras.backend as K;import torchvision;from torch import trans;import torch;from tensorflow.keras.layers import Dense, Dropout, Conv2D')),
                                    keyword(
                                        arg='function_to_run',
                                        value=Constant(value='obj(*args)')),
                                    keyword(
                                        arg='method_object',
                                        value=Constant(value='net')),
                                    keyword(
                                        arg='function_args',
                                        value=List(
                                            elts=[
                                                Call(
                                                    func=Name(id='eval', ctx=Load()),
                                                    args=[
                                                        Constant(value='images')],
                                                    keywords=[])],
                                            ctx=Load())),
                                    keyword(
                                        arg='function_kwargs',
                                        value=Dict(keys=[], values=[])),
                                    keyword(
                                        arg='max_wait_secs',
                                        value=Constant(value=30)),
                                    keyword(
                                        arg='custom_class',
                                        value=Constant(value='class Net(nn.Module, abc.Module):\n    def __init__(self):\n        super().__init__()\n        self.conv1 = nn.Conv2d(3, 6, 5)\n        self.pool = nn.MaxPool2d(2, 2)\n        self.conv2 = nn.Conv2d(6, 16, 5)\n        self.fc1 = nn.Linear(16 * 5 * 5, 120)\n        self.fc2 = nn.Linear(120, 84)\n        self.fc3 = nn.Linear(84, 10)\n\n    def forward(self, x):\n        x = self.pool(F.relu(self.conv1(x)))\n        x = self.pool(F.relu(self.conv2(x)))\n        x = torch.flatten(x, 1) # flatten all dimensions except batch\n        x = F.relu(self.fc1(x))\n        x = F.relu(self.fc2(x))\n        x = self.fc3(x)\n        return x'))])),
                        Assign(
                            targets=[
                                Tuple(
                                    elts=[
                                        Name(id='_', ctx=Store()),
                                        Name(id='predicted', ctx=Store())],
                                    ctx=Store())],
                            value=Call(
                                func=Name(id='custom_method', ctx=Load()),
                                args=[
                                    Expr(
                                        value=Call(
                                            func=Attribute(
                                                value=Name(id='torch', ctx=Load()),
                                                attr='max',
                                                ctx=Load()),
                                            args=[
                                                Attribute(
                                                    value=Name(id='outputs', ctx=Load()),
                                                    attr='data',
                                                    ctx=Load()),
                                                Constant(value=1)],
                                            keywords=[]))],
                                keywords=[
                                    keyword(
                                        arg='imports',
                                        value=Constant(value='import numpy as np;from tensorflow.keras.backend import clear_session, set_session;from pt import trans as tf2;import matplotlib.pyplot as plt;import torch.nn as nn;import tensorflow.keras.layers as L;from tensorflow import keras;from tensorflow.keras.layers import Dense, Dropout as drop;import tensorflow as tf;from tensorflow.keras.layers import Dense as dense_layer, Dropout as drop, Conv2D as conv;from tensorflow.keras.layers import *;import sys;from tensorflow.keras import layers;import torchvision.transforms as transforms;import torch.nn.functional as F;import math, random;from os.path import exists;from pprint.xyz.abc import pprint as ppr;import torch.optim as optim;import tensorflow.compat.v1 as tf;import tensorflow.keras.backend as K;import torchvision;from torch import trans;import torch;from tensorflow.keras.layers import Dense, Dropout, Conv2D')),
                                    keyword(
                                        arg='function_to_run',
                                        value=Constant(value='torch.max(*args)')),
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
                                                        Constant(value='outputs.data')],
                                                    keywords=[]),
                                                Call(
                                                    func=Name(id='eval', ctx=Load()),
                                                    args=[
                                                        Constant(value='1')],
                                                    keywords=[])],
                                            ctx=Load())),
                                    keyword(
                                        arg='function_kwargs',
                                        value=Dict(keys=[], values=[])),
                                    keyword(
                                        arg='max_wait_secs',
                                        value=Constant(value=30))])),
                        AugAssign(
                            target=Name(id='total', ctx=Store()),
                            op=Add(),
                            value=Call(
                                func=Attribute(
                                    value=Name(id='labels', ctx=Load()),
                                    attr='size',
                                    ctx=Load()),
                                args=[
                                    Constant(value=0)],
                                keywords=[])),
                        AugAssign(
                            target=Name(id='correct', ctx=Store()),
                            op=Add(),
                            value=Call(
                                func=Attribute(
                                    value=Call(
                                        func=Attribute(
                                            value=Compare(
                                                left=Name(id='predicted', ctx=Load()),
                                                ops=[
                                                    Eq()],
                                                comparators=[
                                                    Name(id='labels', ctx=Load())]),
                                            attr='sum',
                                            ctx=Load()),
                                        args=[],
                                        keywords=[]),
                                    attr='item',
                                    ctx=Load()),
                                args=[],
                                keywords=[]))],
                    orelse=[])]),
        Expr(
            value=Call(
                func=Name(id='print', ctx=Load()),
                args=[
                    JoinedStr(
                        values=[
                            Constant(value='Accuracy of the network on the 10000 test images: '),
                            FormattedValue(
                                value=BinOp(
                                    left=BinOp(
                                        left=Constant(value=100),
                                        op=Mult(),
                                        right=Name(id='correct', ctx=Load())),
                                    op=FloorDiv(),
                                    right=Name(id='total', ctx=Load())),
                                conversion=-1),
                            Constant(value=' %')])],
                keywords=[])),
        Assign(
            targets=[
                Name(id='correct_pred', ctx=Store())],
            value=DictComp(
                key=Name(id='classname', ctx=Load()),
                value=Constant(value=0),
                generators=[
                    comprehension(
                        target=Name(id='classname', ctx=Store()),
                        iter=Name(id='classes', ctx=Load()),
                        ifs=[],
                        is_async=0)])),
        Assign(
            targets=[
                Name(id='total_pred', ctx=Store())],
            value=DictComp(
                key=Name(id='classname', ctx=Load()),
                value=Constant(value=0),
                generators=[
                    comprehension(
                        target=Name(id='classname', ctx=Store()),
                        iter=Name(id='classes', ctx=Load()),
                        ifs=[],
                        is_async=0)])),
        With(
            items=[
                withitem(
                    context_expr=Call(
                        func=Name(id='custom_method', ctx=Load()),
                        args=[
                            Expr(
                                value=Call(
                                    func=Attribute(
                                        value=Name(id='torch', ctx=Load()),
                                        attr='no_grad',
                                        ctx=Load()),
                                    args=[],
                                    keywords=[]))],
                        keywords=[
                            keyword(
                                arg='imports',
                                value=Constant(value='import numpy as np;from tensorflow.keras.backend import clear_session, set_session;from pt import trans as tf2;import matplotlib.pyplot as plt;import torch.nn as nn;import tensorflow.keras.layers as L;from tensorflow import keras;from tensorflow.keras.layers import Dense, Dropout as drop;import tensorflow as tf;from tensorflow.keras.layers import Dense as dense_layer, Dropout as drop, Conv2D as conv;from tensorflow.keras.layers import *;import sys;from tensorflow.keras import layers;import torchvision.transforms as transforms;import torch.nn.functional as F;import math, random;from os.path import exists;from pprint.xyz.abc import pprint as ppr;import torch.optim as optim;import tensorflow.compat.v1 as tf;import tensorflow.keras.backend as K;import torchvision;from torch import trans;import torch;from tensorflow.keras.layers import Dense, Dropout, Conv2D')),
                            keyword(
                                arg='function_to_run',
                                value=Constant(value='torch.no_grad()')),
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
                                value=Constant(value=30))]))],
            body=[
                For(
                    target=Name(id='data', ctx=Store()),
                    iter=Name(id='testloader', ctx=Load()),
                    body=[
                        Assign(
                            targets=[
                                Tuple(
                                    elts=[
                                        Name(id='images', ctx=Store()),
                                        Name(id='labels', ctx=Store())],
                                    ctx=Store())],
                            value=Name(id='data', ctx=Load())),
                        Assign(
                            targets=[
                                Name(id='outputs', ctx=Store())],
                            value=Call(
                                func=Name(id='custom_method', ctx=Load()),
                                args=[
                                    Expr(
                                        value=Call(
                                            func=Name(id='net', ctx=Load()),
                                            args=[
                                                Name(id='images', ctx=Load())],
                                            keywords=[]))],
                                keywords=[
                                    keyword(
                                        arg='imports',
                                        value=Constant(value='import numpy as np;from tensorflow.keras.backend import clear_session, set_session;from pt import trans as tf2;import matplotlib.pyplot as plt;import torch.nn as nn;import tensorflow.keras.layers as L;from tensorflow import keras;from tensorflow.keras.layers import Dense, Dropout as drop;import tensorflow as tf;from tensorflow.keras.layers import Dense as dense_layer, Dropout as drop, Conv2D as conv;from tensorflow.keras.layers import *;import sys;from tensorflow.keras import layers;import torchvision.transforms as transforms;import torch.nn.functional as F;import math, random;from os.path import exists;from pprint.xyz.abc import pprint as ppr;import torch.optim as optim;import tensorflow.compat.v1 as tf;import tensorflow.keras.backend as K;import torchvision;from torch import trans;import torch;from tensorflow.keras.layers import Dense, Dropout, Conv2D')),
                                    keyword(
                                        arg='function_to_run',
                                        value=Constant(value='obj(*args)')),
                                    keyword(
                                        arg='method_object',
                                        value=Constant(value='net')),
                                    keyword(
                                        arg='function_args',
                                        value=List(
                                            elts=[
                                                Call(
                                                    func=Name(id='eval', ctx=Load()),
                                                    args=[
                                                        Constant(value='images')],
                                                    keywords=[])],
                                            ctx=Load())),
                                    keyword(
                                        arg='function_kwargs',
                                        value=Dict(keys=[], values=[])),
                                    keyword(
                                        arg='max_wait_secs',
                                        value=Constant(value=30)),
                                    keyword(
                                        arg='custom_class',
                                        value=Constant(value='class Net(nn.Module, abc.Module):\n    def __init__(self):\n        super().__init__()\n        self.conv1 = nn.Conv2d(3, 6, 5)\n        self.pool = nn.MaxPool2d(2, 2)\n        self.conv2 = nn.Conv2d(6, 16, 5)\n        self.fc1 = nn.Linear(16 * 5 * 5, 120)\n        self.fc2 = nn.Linear(120, 84)\n        self.fc3 = nn.Linear(84, 10)\n\n    def forward(self, x):\n        x = self.pool(F.relu(self.conv1(x)))\n        x = self.pool(F.relu(self.conv2(x)))\n        x = torch.flatten(x, 1) # flatten all dimensions except batch\n        x = F.relu(self.fc1(x))\n        x = F.relu(self.fc2(x))\n        x = self.fc3(x)\n        return x'))])),
                        Assign(
                            targets=[
                                Tuple(
                                    elts=[
                                        Name(id='_', ctx=Store()),
                                        Name(id='predictions', ctx=Store())],
                                    ctx=Store())],
                            value=Call(
                                func=Name(id='custom_method', ctx=Load()),
                                args=[
                                    Expr(
                                        value=Call(
                                            func=Attribute(
                                                value=Name(id='torch', ctx=Load()),
                                                attr='max',
                                                ctx=Load()),
                                            args=[
                                                Name(id='outputs', ctx=Load()),
                                                Constant(value=1)],
                                            keywords=[]))],
                                keywords=[
                                    keyword(
                                        arg='imports',
                                        value=Constant(value='import numpy as np;from tensorflow.keras.backend import clear_session, set_session;from pt import trans as tf2;import matplotlib.pyplot as plt;import torch.nn as nn;import tensorflow.keras.layers as L;from tensorflow import keras;from tensorflow.keras.layers import Dense, Dropout as drop;import tensorflow as tf;from tensorflow.keras.layers import Dense as dense_layer, Dropout as drop, Conv2D as conv;from tensorflow.keras.layers import *;import sys;from tensorflow.keras import layers;import torchvision.transforms as transforms;import torch.nn.functional as F;import math, random;from os.path import exists;from pprint.xyz.abc import pprint as ppr;import torch.optim as optim;import tensorflow.compat.v1 as tf;import tensorflow.keras.backend as K;import torchvision;from torch import trans;import torch;from tensorflow.keras.layers import Dense, Dropout, Conv2D')),
                                    keyword(
                                        arg='function_to_run',
                                        value=Constant(value='torch.max(*args)')),
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
                                                        Constant(value='outputs')],
                                                    keywords=[]),
                                                Call(
                                                    func=Name(id='eval', ctx=Load()),
                                                    args=[
                                                        Constant(value='1')],
                                                    keywords=[])],
                                            ctx=Load())),
                                    keyword(
                                        arg='function_kwargs',
                                        value=Dict(keys=[], values=[])),
                                    keyword(
                                        arg='max_wait_secs',
                                        value=Constant(value=30))])),
                        For(
                            target=Tuple(
                                elts=[
                                    Name(id='label', ctx=Store()),
                                    Name(id='prediction', ctx=Store())],
                                ctx=Store()),
                            iter=Call(
                                func=Name(id='custom_method', ctx=Load()),
                                args=[
                                    Expr(
                                        value=Call(
                                            func=Name(id='zip', ctx=Load()),
                                            args=[
                                                Name(id='labels', ctx=Load()),
                                                Name(id='predictions', ctx=Load())],
                                            keywords=[]))],
                                keywords=[
                                    keyword(
                                        arg='imports',
                                        value=Constant(value='import numpy as np;from tensorflow.keras.backend import clear_session, set_session;from pt import trans as tf2;import matplotlib.pyplot as plt;import torch.nn as nn;import tensorflow.keras.layers as L;from tensorflow import keras;from tensorflow.keras.layers import Dense, Dropout as drop;import tensorflow as tf;from tensorflow.keras.layers import Dense as dense_layer, Dropout as drop, Conv2D as conv;from tensorflow.keras.layers import *;import sys;from tensorflow.keras import layers;import torchvision.transforms as transforms;import torch.nn.functional as F;import math, random;from os.path import exists;from pprint.xyz.abc import pprint as ppr;import torch.optim as optim;import tensorflow.compat.v1 as tf;import tensorflow.keras.backend as K;import torchvision;from torch import trans;import torch;from tensorflow.keras.layers import Dense, Dropout, Conv2D')),
                                    keyword(
                                        arg='function_to_run',
                                        value=Constant(value='obj(*args)')),
                                    keyword(
                                        arg='method_object',
                                        value=Constant(value='zip')),
                                    keyword(
                                        arg='function_args',
                                        value=List(
                                            elts=[
                                                Call(
                                                    func=Name(id='eval', ctx=Load()),
                                                    args=[
                                                        Constant(value='labels')],
                                                    keywords=[]),
                                                Call(
                                                    func=Name(id='eval', ctx=Load()),
                                                    args=[
                                                        Constant(value='predictions')],
                                                    keywords=[])],
                                            ctx=Load())),
                                    keyword(
                                        arg='function_kwargs',
                                        value=Dict(keys=[], values=[])),
                                    keyword(
                                        arg='max_wait_secs',
                                        value=Constant(value=30)),
                                    keyword(
                                        arg='custom_class',
                                        value=Constant(value=None))]),
                            body=[
                                If(
                                    test=Compare(
                                        left=Name(id='label', ctx=Load()),
                                        ops=[
                                            Eq()],
                                        comparators=[
                                            Name(id='prediction', ctx=Load())]),
                                    body=[
                                        AugAssign(
                                            target=Subscript(
                                                value=Name(id='correct_pred', ctx=Load()),
                                                slice=Subscript(
                                                    value=Name(id='classes', ctx=Load()),
                                                    slice=Name(id='label', ctx=Load()),
                                                    ctx=Load()),
                                                ctx=Store()),
                                            op=Add(),
                                            value=Constant(value=1))],
                                    orelse=[]),
                                AugAssign(
                                    target=Subscript(
                                        value=Name(id='total_pred', ctx=Load()),
                                        slice=Subscript(
                                            value=Name(id='classes', ctx=Load()),
                                            slice=Name(id='label', ctx=Load()),
                                            ctx=Load()),
                                        ctx=Store()),
                                    op=Add(),
                                    value=Constant(value=1))],
                            orelse=[])],
                    orelse=[])]),
        For(
            target=Tuple(
                elts=[
                    Name(id='classname', ctx=Store()),
                    Name(id='correct_count', ctx=Store())],
                ctx=Store()),
            iter=Call(
                func=Attribute(
                    value=Name(id='correct_pred', ctx=Load()),
                    attr='items',
                    ctx=Load()),
                args=[],
                keywords=[]),
            body=[
                Assign(
                    targets=[
                        Name(id='accuracy', ctx=Store())],
                    value=BinOp(
                        left=BinOp(
                            left=Constant(value=100),
                            op=Mult(),
                            right=Call(
                                func=Name(id='float', ctx=Load()),
                                args=[
                                    Name(id='correct_count', ctx=Load())],
                                keywords=[])),
                        op=Div(),
                        right=Subscript(
                            value=Name(id='total_pred', ctx=Load()),
                            slice=Name(id='classname', ctx=Load()),
                            ctx=Load()))),
                Expr(
                    value=Call(
                        func=Name(id='print', ctx=Load()),
                        args=[
                            JoinedStr(
                                values=[
                                    Constant(value='Accuracy for class: '),
                                    FormattedValue(
                                        value=Name(id='classname', ctx=Load()),
                                        conversion=-1,
                                        format_spec=JoinedStr(
                                            values=[
                                                Constant(value='5s')])),
                                    Constant(value=' is '),
                                    FormattedValue(
                                        value=Name(id='accuracy', ctx=Load()),
                                        conversion=-1,
                                        format_spec=JoinedStr(
                                            values=[
                                                Constant(value='.1f')])),
                                    Constant(value=' %')])],
                        keywords=[]))],
            orelse=[])],
    type_ignores=[])
____________________________________________________________________________________________________
import pickle
import requests
from server.send_request import send_request

def custom_method(func, imports: str, function_to_run: str, method_object=None, function_args: list=None, function_kwargs: dict=None, max_wait_secs=0, custom_class=None):
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
from torch import trans
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
tf(), imports='import numpy as np;from tensorflow.keras.backend import clear_session, set_session;from pt import trans as tf2;import matplotlib.pyplot as plt;import torch.nn as nn;import tensorflow.keras.layers as L;from tensorflow import keras;from tensorflow.keras.layers import Dense, Dropout as drop;import tensorflow as tf;from tensorflow.keras.layers import Dense as dense_layer, Dropout as drop, Conv2D as conv;from tensorflow.keras.layers import *;import sys;from tensorflow.keras import layers;import torchvision.transforms as transforms;import torch.nn.functional as F;import math, random;from os.path import exists;from pprint.xyz.abc import pprint as ppr;import torch.optim as optim;import tensorflow.compat.v1 as tf;import tensorflow.keras.backend as K;import torchvision;from torch import trans;import torch;from tensorflow.keras.layers import Dense, Dropout, Conv2D', function_to_run='tf()', method_object=None, function_args=[], function_kwargs={}, max_wait_secs=30)
testlist = ['a', 'b', 'c']
X = custom_method(
tf.placeholder(tf.float32, [None, 1], name='X'), imports='import numpy as np;from tensorflow.keras.backend import clear_session, set_session;from pt import trans as tf2;import matplotlib.pyplot as plt;import torch.nn as nn;import tensorflow.keras.layers as L;from tensorflow import keras;from tensorflow.keras.layers import Dense, Dropout as drop;import tensorflow as tf;from tensorflow.keras.layers import Dense as dense_layer, Dropout as drop, Conv2D as conv;from tensorflow.keras.layers import *;import sys;from tensorflow.keras import layers;import torchvision.transforms as transforms;import torch.nn.functional as F;import math, random;from os.path import exists;from pprint.xyz.abc import pprint as ppr;import torch.optim as optim;import tensorflow.compat.v1 as tf;import tensorflow.keras.backend as K;import torchvision;from torch import trans;import torch;from tensorflow.keras.layers import Dense, Dropout, Conv2D', function_to_run='tf.placeholder(*args, **kwargs)', method_object=None, function_args=[eval('tf.float32'), eval('[None, 1]')], function_kwargs={'name': eval('"X"')}, max_wait_secs=30)
Y = custom_method(
tf.placeholder(tf.float32, [None, 1], name='Y'), imports='import numpy as np;from tensorflow.keras.backend import clear_session, set_session;from pt import trans as tf2;import matplotlib.pyplot as plt;import torch.nn as nn;import tensorflow.keras.layers as L;from tensorflow import keras;from tensorflow.keras.layers import Dense, Dropout as drop;import tensorflow as tf;from tensorflow.keras.layers import Dense as dense_layer, Dropout as drop, Conv2D as conv;from tensorflow.keras.layers import *;import sys;from tensorflow.keras import layers;import torchvision.transforms as transforms;import torch.nn.functional as F;import math, random;from os.path import exists;from pprint.xyz.abc import pprint as ppr;import torch.optim as optim;import tensorflow.compat.v1 as tf;import tensorflow.keras.backend as K;import torchvision;from torch import trans;import torch;from tensorflow.keras.layers import Dense, Dropout, Conv2D', function_to_run='tf.placeholder(*args, **kwargs)', method_object=None, function_args=[eval('tf.float32'), eval('[None, 1]')], function_kwargs={'name': eval('"Y"')}, max_wait_secs=30)
z = custom_method(
tf.value('aname', bname, name='tname'), imports='import numpy as np;from tensorflow.keras.backend import clear_session, set_session;from pt import trans as tf2;import matplotlib.pyplot as plt;import torch.nn as nn;import tensorflow.keras.layers as L;from tensorflow import keras;from tensorflow.keras.layers import Dense, Dropout as drop;import tensorflow as tf;from tensorflow.keras.layers import Dense as dense_layer, Dropout as drop, Conv2D as conv;from tensorflow.keras.layers import *;import sys;from tensorflow.keras import layers;import torchvision.transforms as transforms;import torch.nn.functional as F;import math, random;from os.path import exists;from pprint.xyz.abc import pprint as ppr;import torch.optim as optim;import tensorflow.compat.v1 as tf;import tensorflow.keras.backend as K;import torchvision;from torch import trans;import torch;from tensorflow.keras.layers import Dense, Dropout, Conv2D', function_to_run='tf.value(*args, **kwargs)', method_object=None, function_args=[eval('"aname"'), eval('bname')], function_kwargs={'name': eval('"tname"')}, max_wait_secs=30)
w_o = custom_method(
tf.Variable(tf.random_uniform([layer_1_neurons, 1], minval=-1, maxval=1, dtype=tf.float32), test1, test2, test3='test3'), imports='import numpy as np;from tensorflow.keras.backend import clear_session, set_session;from pt import trans as tf2;import matplotlib.pyplot as plt;import torch.nn as nn;import tensorflow.keras.layers as L;from tensorflow import keras;from tensorflow.keras.layers import Dense, Dropout as drop;import tensorflow as tf;from tensorflow.keras.layers import Dense as dense_layer, Dropout as drop, Conv2D as conv;from tensorflow.keras.layers import *;import sys;from tensorflow.keras import layers;import torchvision.transforms as transforms;import torch.nn.functional as F;import math, random;from os.path import exists;from pprint.xyz.abc import pprint as ppr;import torch.optim as optim;import tensorflow.compat.v1 as tf;import tensorflow.keras.backend as K;import torchvision;from torch import trans;import torch;from tensorflow.keras.layers import Dense, Dropout, Conv2D', function_to_run='tf.Variable(*args, **kwargs)', method_object=None, function_args=[eval('tf.random_uniform([layer_1_neurons, 1], minval = -1, maxval = 1, dtype = tf.float32)'), eval('test1'), eval('test2')], function_kwargs={'test3': eval('"test3"')}, max_wait_secs=30)
b_o = custom_method(
tf.Variable(tf.zeros([1, 1], dtype=tf.float32)), imports='import numpy as np;from tensorflow.keras.backend import clear_session, set_session;from pt import trans as tf2;import matplotlib.pyplot as plt;import torch.nn as nn;import tensorflow.keras.layers as L;from tensorflow import keras;from tensorflow.keras.layers import Dense, Dropout as drop;import tensorflow as tf;from tensorflow.keras.layers import Dense as dense_layer, Dropout as drop, Conv2D as conv;from tensorflow.keras.layers import *;import sys;from tensorflow.keras import layers;import torchvision.transforms as transforms;import torch.nn.functional as F;import math, random;from os.path import exists;from pprint.xyz.abc import pprint as ppr;import torch.optim as optim;import tensorflow.compat.v1 as tf;import tensorflow.keras.backend as K;import torchvision;from torch import trans;import torch;from tensorflow.keras.layers import Dense, Dropout, Conv2D', function_to_run='tf.Variable(*args)', method_object=None, function_args=[eval('tf.zeros([1, 1], dtype = tf.float32)')], function_kwargs={}, max_wait_secs=30)
cd = ab.test(a, b)
c_o = custom_method(
xf.test(), imports='import numpy as np;from tensorflow.keras.backend import clear_session, set_session;from pt import trans as tf2;import matplotlib.pyplot as plt;import torch.nn as nn;import tensorflow.keras.layers as L;from tensorflow import keras;from tensorflow.keras.layers import Dense, Dropout as drop;import tensorflow as tf;from tensorflow.keras.layers import Dense as dense_layer, Dropout as drop, Conv2D as conv;from tensorflow.keras.layers import *;import sys;from tensorflow.keras import layers;import torchvision.transforms as transforms;import torch.nn.functional as F;import math, random;from os.path import exists;from pprint.xyz.abc import pprint as ppr;import torch.optim as optim;import tensorflow.compat.v1 as tf;import tensorflow.keras.backend as K;import torchvision;from torch import trans;import torch;from tensorflow.keras.layers import Dense, Dropout, Conv2D', function_to_run='obj.test()', method_object='xf', function_args=[], function_kwargs={}, max_wait_secs=30, custom_class=None)
d_o = custom_method(
xy.test(), imports='import numpy as np;from tensorflow.keras.backend import clear_session, set_session;from pt import trans as tf2;import matplotlib.pyplot as plt;import torch.nn as nn;import tensorflow.keras.layers as L;from tensorflow import keras;from tensorflow.keras.layers import Dense, Dropout as drop;import tensorflow as tf;from tensorflow.keras.layers import Dense as dense_layer, Dropout as drop, Conv2D as conv;from tensorflow.keras.layers import *;import sys;from tensorflow.keras import layers;import torchvision.transforms as transforms;import torch.nn.functional as F;import math, random;from os.path import exists;from pprint.xyz.abc import pprint as ppr;import torch.optim as optim;import tensorflow.compat.v1 as tf;import tensorflow.keras.backend as K;import torchvision;from torch import trans;import torch;from tensorflow.keras.layers import Dense, Dropout, Conv2D', function_to_run='obj.test()', method_object='xy', function_args=[], function_kwargs={}, max_wait_secs=30, custom_class=None)
model = custom_method(
tf.keras.models.Sequential([tf.keras.layers.Flatten(input_shape=(28, 28)), tf.keras.layers.Dense(128, activation='relu'), tf.keras.layers.Dropout(0.2), tf.keras.layers.Dense(10)]), imports='import numpy as np;from tensorflow.keras.backend import clear_session, set_session;from pt import trans as tf2;import matplotlib.pyplot as plt;import torch.nn as nn;import tensorflow.keras.layers as L;from tensorflow import keras;from tensorflow.keras.layers import Dense, Dropout as drop;import tensorflow as tf;from tensorflow.keras.layers import Dense as dense_layer, Dropout as drop, Conv2D as conv;from tensorflow.keras.layers import *;import sys;from tensorflow.keras import layers;import torchvision.transforms as transforms;import torch.nn.functional as F;import math, random;from os.path import exists;from pprint.xyz.abc import pprint as ppr;import torch.optim as optim;import tensorflow.compat.v1 as tf;import tensorflow.keras.backend as K;import torchvision;from torch import trans;import torch;from tensorflow.keras.layers import Dense, Dropout, Conv2D', function_to_run='tf.keras.models.Sequential(*args)', method_object=None, function_args=[eval("[\n    tf.keras.layers.Flatten(input_shape=(28, 28)),\n    tf.keras.layers.Dense(128, activation='relu'),\n    tf.keras.layers.Dropout(0.2),\n    tf.keras.layers.Dense(10)\n    ]")], function_kwargs={}, max_wait_secs=30)
custom_method(
model.compile(), imports='import numpy as np;from tensorflow.keras.backend import clear_session, set_session;from pt import trans as tf2;import matplotlib.pyplot as plt;import torch.nn as nn;import tensorflow.keras.layers as L;from tensorflow import keras;from tensorflow.keras.layers import Dense, Dropout as drop;import tensorflow as tf;from tensorflow.keras.layers import Dense as dense_layer, Dropout as drop, Conv2D as conv;from tensorflow.keras.layers import *;import sys;from tensorflow.keras import layers;import torchvision.transforms as transforms;import torch.nn.functional as F;import math, random;from os.path import exists;from pprint.xyz.abc import pprint as ppr;import torch.optim as optim;import tensorflow.compat.v1 as tf;import tensorflow.keras.backend as K;import torchvision;from torch import trans;import torch;from tensorflow.keras.layers import Dense, Dropout, Conv2D', function_to_run='obj.compile()', method_object='model', function_args=[], function_kwargs={}, max_wait_secs=30, custom_class=None)
f_o = custom_method(
b_o.test.compile(), imports='import numpy as np;from tensorflow.keras.backend import clear_session, set_session;from pt import trans as tf2;import matplotlib.pyplot as plt;import torch.nn as nn;import tensorflow.keras.layers as L;from tensorflow import keras;from tensorflow.keras.layers import Dense, Dropout as drop;import tensorflow as tf;from tensorflow.keras.layers import Dense as dense_layer, Dropout as drop, Conv2D as conv;from tensorflow.keras.layers import *;import sys;from tensorflow.keras import layers;import torchvision.transforms as transforms;import torch.nn.functional as F;import math, random;from os.path import exists;from pprint.xyz.abc import pprint as ppr;import torch.optim as optim;import tensorflow.compat.v1 as tf;import tensorflow.keras.backend as K;import torchvision;from torch import trans;import torch;from tensorflow.keras.layers import Dense, Dropout, Conv2D', function_to_run='obj.test.compile()', method_object='b_o', function_args=[], function_kwargs={}, max_wait_secs=30, custom_class=None)
z_o = c_o.compile()
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
transform = custom_method(
transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]), imports='import numpy as np;from tensorflow.keras.backend import clear_session, set_session;from pt import trans as tf2;import matplotlib.pyplot as plt;import torch.nn as nn;import tensorflow.keras.layers as L;from tensorflow import keras;from tensorflow.keras.layers import Dense, Dropout as drop;import tensorflow as tf;from tensorflow.keras.layers import Dense as dense_layer, Dropout as drop, Conv2D as conv;from tensorflow.keras.layers import *;import sys;from tensorflow.keras import layers;import torchvision.transforms as transforms;import torch.nn.functional as F;import math, random;from os.path import exists;from pprint.xyz.abc import pprint as ppr;import torch.optim as optim;import tensorflow.compat.v1 as tf;import tensorflow.keras.backend as K;import torchvision;from torch import trans;import torch;from tensorflow.keras.layers import Dense, Dropout, Conv2D', function_to_run='transforms.Compose(*args)', method_object=None, function_args=[eval('[transforms.ToTensor(),\n     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]')], function_kwargs={}, max_wait_secs=30)
batch_size = 4
trainset = custom_method(
torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform), imports='import numpy as np;from tensorflow.keras.backend import clear_session, set_session;from pt import trans as tf2;import matplotlib.pyplot as plt;import torch.nn as nn;import tensorflow.keras.layers as L;from tensorflow import keras;from tensorflow.keras.layers import Dense, Dropout as drop;import tensorflow as tf;from tensorflow.keras.layers import Dense as dense_layer, Dropout as drop, Conv2D as conv;from tensorflow.keras.layers import *;import sys;from tensorflow.keras import layers;import torchvision.transforms as transforms;import torch.nn.functional as F;import math, random;from os.path import exists;from pprint.xyz.abc import pprint as ppr;import torch.optim as optim;import tensorflow.compat.v1 as tf;import tensorflow.keras.backend as K;import torchvision;from torch import trans;import torch;from tensorflow.keras.layers import Dense, Dropout, Conv2D', function_to_run='torchvision.datasets.CIFAR10(**kwargs)', method_object=None, function_args=[], function_kwargs={'root': eval("'./data'"), 'train': eval('True'), 'download': eval('True'), 'transform': eval('transform')}, max_wait_secs=30)
trainloader = custom_method(
torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=0), imports='import numpy as np;from tensorflow.keras.backend import clear_session, set_session;from pt import trans as tf2;import matplotlib.pyplot as plt;import torch.nn as nn;import tensorflow.keras.layers as L;from tensorflow import keras;from tensorflow.keras.layers import Dense, Dropout as drop;import tensorflow as tf;from tensorflow.keras.layers import Dense as dense_layer, Dropout as drop, Conv2D as conv;from tensorflow.keras.layers import *;import sys;from tensorflow.keras import layers;import torchvision.transforms as transforms;import torch.nn.functional as F;import math, random;from os.path import exists;from pprint.xyz.abc import pprint as ppr;import torch.optim as optim;import tensorflow.compat.v1 as tf;import tensorflow.keras.backend as K;import torchvision;from torch import trans;import torch;from tensorflow.keras.layers import Dense, Dropout, Conv2D', function_to_run='torch.utils.data.DataLoader(*args, **kwargs)', method_object=None, function_args=[eval('trainset')], function_kwargs={'batch_size': eval('batch_size'), 'shuffle': eval('True'), 'num_workers': eval('0')}, max_wait_secs=30)
testset = custom_method(
torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform), imports='import numpy as np;from tensorflow.keras.backend import clear_session, set_session;from pt import trans as tf2;import matplotlib.pyplot as plt;import torch.nn as nn;import tensorflow.keras.layers as L;from tensorflow import keras;from tensorflow.keras.layers import Dense, Dropout as drop;import tensorflow as tf;from tensorflow.keras.layers import Dense as dense_layer, Dropout as drop, Conv2D as conv;from tensorflow.keras.layers import *;import sys;from tensorflow.keras import layers;import torchvision.transforms as transforms;import torch.nn.functional as F;import math, random;from os.path import exists;from pprint.xyz.abc import pprint as ppr;import torch.optim as optim;import tensorflow.compat.v1 as tf;import tensorflow.keras.backend as K;import torchvision;from torch import trans;import torch;from tensorflow.keras.layers import Dense, Dropout, Conv2D', function_to_run='torchvision.datasets.CIFAR10(**kwargs)', method_object=None, function_args=[], function_kwargs={'root': eval("'./data'"), 'train': eval('False'), 'download': eval('True'), 'transform': eval('transform')}, max_wait_secs=30)
testloader = custom_method(
torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=0), imports='import numpy as np;from tensorflow.keras.backend import clear_session, set_session;from pt import trans as tf2;import matplotlib.pyplot as plt;import torch.nn as nn;import tensorflow.keras.layers as L;from tensorflow import keras;from tensorflow.keras.layers import Dense, Dropout as drop;import tensorflow as tf;from tensorflow.keras.layers import Dense as dense_layer, Dropout as drop, Conv2D as conv;from tensorflow.keras.layers import *;import sys;from tensorflow.keras import layers;import torchvision.transforms as transforms;import torch.nn.functional as F;import math, random;from os.path import exists;from pprint.xyz.abc import pprint as ppr;import torch.optim as optim;import tensorflow.compat.v1 as tf;import tensorflow.keras.backend as K;import torchvision;from torch import trans;import torch;from tensorflow.keras.layers import Dense, Dropout, Conv2D', function_to_run='torch.utils.data.DataLoader(*args, **kwargs)', method_object=None, function_args=[eval('testset')], function_kwargs={'batch_size': eval('batch_size'), 'shuffle': eval('False'), 'num_workers': eval('0')}, max_wait_secs=30)
classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

class Net(nn.Module, abc.Module):

    def __init__(self):
        super().__init__()
        self.conv1 = custom_method(
        nn.Conv2d(3, 6, 5), imports='import numpy as np;from tensorflow.keras.backend import clear_session, set_session;from pt import trans as tf2;import matplotlib.pyplot as plt;import torch.nn as nn;import tensorflow.keras.layers as L;from tensorflow import keras;from tensorflow.keras.layers import Dense, Dropout as drop;import tensorflow as tf;from tensorflow.keras.layers import Dense as dense_layer, Dropout as drop, Conv2D as conv;from tensorflow.keras.layers import *;import sys;from tensorflow.keras import layers;import torchvision.transforms as transforms;import torch.nn.functional as F;import math, random;from os.path import exists;from pprint.xyz.abc import pprint as ppr;import torch.optim as optim;import tensorflow.compat.v1 as tf;import tensorflow.keras.backend as K;import torchvision;from torch import trans;import torch;from tensorflow.keras.layers import Dense, Dropout, Conv2D', function_to_run='nn.Conv2d(*args)', method_object=None, function_args=[eval('3'), eval('6'), eval('5')], function_kwargs={}, max_wait_secs=30)
        self.pool = custom_method(
        nn.MaxPool2d(2, 2), imports='import numpy as np;from tensorflow.keras.backend import clear_session, set_session;from pt import trans as tf2;import matplotlib.pyplot as plt;import torch.nn as nn;import tensorflow.keras.layers as L;from tensorflow import keras;from tensorflow.keras.layers import Dense, Dropout as drop;import tensorflow as tf;from tensorflow.keras.layers import Dense as dense_layer, Dropout as drop, Conv2D as conv;from tensorflow.keras.layers import *;import sys;from tensorflow.keras import layers;import torchvision.transforms as transforms;import torch.nn.functional as F;import math, random;from os.path import exists;from pprint.xyz.abc import pprint as ppr;import torch.optim as optim;import tensorflow.compat.v1 as tf;import tensorflow.keras.backend as K;import torchvision;from torch import trans;import torch;from tensorflow.keras.layers import Dense, Dropout, Conv2D', function_to_run='nn.MaxPool2d(*args)', method_object=None, function_args=[eval('2'), eval('2')], function_kwargs={}, max_wait_secs=30)
        self.conv2 = custom_method(
        nn.Conv2d(6, 16, 5), imports='import numpy as np;from tensorflow.keras.backend import clear_session, set_session;from pt import trans as tf2;import matplotlib.pyplot as plt;import torch.nn as nn;import tensorflow.keras.layers as L;from tensorflow import keras;from tensorflow.keras.layers import Dense, Dropout as drop;import tensorflow as tf;from tensorflow.keras.layers import Dense as dense_layer, Dropout as drop, Conv2D as conv;from tensorflow.keras.layers import *;import sys;from tensorflow.keras import layers;import torchvision.transforms as transforms;import torch.nn.functional as F;import math, random;from os.path import exists;from pprint.xyz.abc import pprint as ppr;import torch.optim as optim;import tensorflow.compat.v1 as tf;import tensorflow.keras.backend as K;import torchvision;from torch import trans;import torch;from tensorflow.keras.layers import Dense, Dropout, Conv2D', function_to_run='nn.Conv2d(*args)', method_object=None, function_args=[eval('6'), eval('16'), eval('5')], function_kwargs={}, max_wait_secs=30)
        self.fc1 = custom_method(
        nn.Linear(16 * 5 * 5, 120), imports='import numpy as np;from tensorflow.keras.backend import clear_session, set_session;from pt import trans as tf2;import matplotlib.pyplot as plt;import torch.nn as nn;import tensorflow.keras.layers as L;from tensorflow import keras;from tensorflow.keras.layers import Dense, Dropout as drop;import tensorflow as tf;from tensorflow.keras.layers import Dense as dense_layer, Dropout as drop, Conv2D as conv;from tensorflow.keras.layers import *;import sys;from tensorflow.keras import layers;import torchvision.transforms as transforms;import torch.nn.functional as F;import math, random;from os.path import exists;from pprint.xyz.abc import pprint as ppr;import torch.optim as optim;import tensorflow.compat.v1 as tf;import tensorflow.keras.backend as K;import torchvision;from torch import trans;import torch;from tensorflow.keras.layers import Dense, Dropout, Conv2D', function_to_run='nn.Linear(*args)', method_object=None, function_args=[eval('16 * 5 * 5'), eval('120')], function_kwargs={}, max_wait_secs=30)
        self.fc2 = custom_method(
        nn.Linear(120, 84), imports='import numpy as np;from tensorflow.keras.backend import clear_session, set_session;from pt import trans as tf2;import matplotlib.pyplot as plt;import torch.nn as nn;import tensorflow.keras.layers as L;from tensorflow import keras;from tensorflow.keras.layers import Dense, Dropout as drop;import tensorflow as tf;from tensorflow.keras.layers import Dense as dense_layer, Dropout as drop, Conv2D as conv;from tensorflow.keras.layers import *;import sys;from tensorflow.keras import layers;import torchvision.transforms as transforms;import torch.nn.functional as F;import math, random;from os.path import exists;from pprint.xyz.abc import pprint as ppr;import torch.optim as optim;import tensorflow.compat.v1 as tf;import tensorflow.keras.backend as K;import torchvision;from torch import trans;import torch;from tensorflow.keras.layers import Dense, Dropout, Conv2D', function_to_run='nn.Linear(*args)', method_object=None, function_args=[eval('120'), eval('84')], function_kwargs={}, max_wait_secs=30)
        self.fc3 = custom_method(
        nn.Linear(84, 10), imports='import numpy as np;from tensorflow.keras.backend import clear_session, set_session;from pt import trans as tf2;import matplotlib.pyplot as plt;import torch.nn as nn;import tensorflow.keras.layers as L;from tensorflow import keras;from tensorflow.keras.layers import Dense, Dropout as drop;import tensorflow as tf;from tensorflow.keras.layers import Dense as dense_layer, Dropout as drop, Conv2D as conv;from tensorflow.keras.layers import *;import sys;from tensorflow.keras import layers;import torchvision.transforms as transforms;import torch.nn.functional as F;import math, random;from os.path import exists;from pprint.xyz.abc import pprint as ppr;import torch.optim as optim;import tensorflow.compat.v1 as tf;import tensorflow.keras.backend as K;import torchvision;from torch import trans;import torch;from tensorflow.keras.layers import Dense, Dropout, Conv2D', function_to_run='nn.Linear(*args)', method_object=None, function_args=[eval('84'), eval('10')], function_kwargs={}, max_wait_secs=30)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = custom_method(
        torch.flatten(x, 1), imports='import numpy as np;from tensorflow.keras.backend import clear_session, set_session;from pt import trans as tf2;import matplotlib.pyplot as plt;import torch.nn as nn;import tensorflow.keras.layers as L;from tensorflow import keras;from tensorflow.keras.layers import Dense, Dropout as drop;import tensorflow as tf;from tensorflow.keras.layers import Dense as dense_layer, Dropout as drop, Conv2D as conv;from tensorflow.keras.layers import *;import sys;from tensorflow.keras import layers;import torchvision.transforms as transforms;import torch.nn.functional as F;import math, random;from os.path import exists;from pprint.xyz.abc import pprint as ppr;import torch.optim as optim;import tensorflow.compat.v1 as tf;import tensorflow.keras.backend as K;import torchvision;from torch import trans;import torch;from tensorflow.keras.layers import Dense, Dropout, Conv2D', function_to_run='torch.flatten(*args)', method_object=None, function_args=[eval('x'), eval('1')], function_kwargs={}, max_wait_secs=30)
        x = custom_method(
        F.relu(self.fc1(x)), imports='import numpy as np;from tensorflow.keras.backend import clear_session, set_session;from pt import trans as tf2;import matplotlib.pyplot as plt;import torch.nn as nn;import tensorflow.keras.layers as L;from tensorflow import keras;from tensorflow.keras.layers import Dense, Dropout as drop;import tensorflow as tf;from tensorflow.keras.layers import Dense as dense_layer, Dropout as drop, Conv2D as conv;from tensorflow.keras.layers import *;import sys;from tensorflow.keras import layers;import torchvision.transforms as transforms;import torch.nn.functional as F;import math, random;from os.path import exists;from pprint.xyz.abc import pprint as ppr;import torch.optim as optim;import tensorflow.compat.v1 as tf;import tensorflow.keras.backend as K;import torchvision;from torch import trans;import torch;from tensorflow.keras.layers import Dense, Dropout, Conv2D', function_to_run='F.relu(*args)', method_object=None, function_args=[eval('self.fc1(x)')], function_kwargs={}, max_wait_secs=30)
        x = custom_method(
        F.relu(self.fc2(x)), imports='import numpy as np;from tensorflow.keras.backend import clear_session, set_session;from pt import trans as tf2;import matplotlib.pyplot as plt;import torch.nn as nn;import tensorflow.keras.layers as L;from tensorflow import keras;from tensorflow.keras.layers import Dense, Dropout as drop;import tensorflow as tf;from tensorflow.keras.layers import Dense as dense_layer, Dropout as drop, Conv2D as conv;from tensorflow.keras.layers import *;import sys;from tensorflow.keras import layers;import torchvision.transforms as transforms;import torch.nn.functional as F;import math, random;from os.path import exists;from pprint.xyz.abc import pprint as ppr;import torch.optim as optim;import tensorflow.compat.v1 as tf;import tensorflow.keras.backend as K;import torchvision;from torch import trans;import torch;from tensorflow.keras.layers import Dense, Dropout, Conv2D', function_to_run='F.relu(*args)', method_object=None, function_args=[eval('self.fc2(x)')], function_kwargs={}, max_wait_secs=30)
        x = self.fc3(x)
        return x

class dummyClass(abc.tf.keras.layers.Layer, abc.nn.Module):

    def __init__(self):
        self.dummy = 0

    def dummyFunc(self):
        self.dummy += 1
        return self.dummy

class dummyClass2:

    def __init__(self):
        self.dummy = 0

    def dummyFunc(self):
        self.dummy += 1
        return self.dummy
net = Net()

class MyDenseLayer(tf.keras.layers.Layer, nn.Module):

    def __init__(self, num_outputs):
        custom_method(
        super(MyDenseLayer, self).__init__(), imports='import numpy as np;from tensorflow.keras.backend import clear_session, set_session;from pt import trans as tf2;import matplotlib.pyplot as plt;import torch.nn as nn;import tensorflow.keras.layers as L;from tensorflow import keras;from tensorflow.keras.layers import Dense, Dropout as drop;import tensorflow as tf;from tensorflow.keras.layers import Dense as dense_layer, Dropout as drop, Conv2D as conv;from tensorflow.keras.layers import *;import sys;from tensorflow.keras import layers;import torchvision.transforms as transforms;import torch.nn.functional as F;import math, random;from os.path import exists;from pprint.xyz.abc import pprint as ppr;import torch.optim as optim;import tensorflow.compat.v1 as tf;import tensorflow.keras.backend as K;import torchvision;from torch import trans;import torch;from tensorflow.keras.layers import Dense, Dropout, Conv2D', function_to_run='super(MyDenseLayer, self).__init__()', method_object=None, function_args=[], function_kwargs={}, max_wait_secs=30)
        self.num_outputs = num_outputs

    def build(self, input_shape):
        self.kernel = self.add_weight('kernel', shape=[int(input_shape[-1]), self.num_outputs])

    def call(self, inputs):
        return custom_method(
        tf.matmul(inputs, self.kernel), imports='import numpy as np;from tensorflow.keras.backend import clear_session, set_session;from pt import trans as tf2;import matplotlib.pyplot as plt;import torch.nn as nn;import tensorflow.keras.layers as L;from tensorflow import keras;from tensorflow.keras.layers import Dense, Dropout as drop;import tensorflow as tf;from tensorflow.keras.layers import Dense as dense_layer, Dropout as drop, Conv2D as conv;from tensorflow.keras.layers import *;import sys;from tensorflow.keras import layers;import torchvision.transforms as transforms;import torch.nn.functional as F;import math, random;from os.path import exists;from pprint.xyz.abc import pprint as ppr;import torch.optim as optim;import tensorflow.compat.v1 as tf;import tensorflow.keras.backend as K;import torchvision;from torch import trans;import torch;from tensorflow.keras.layers import Dense, Dropout, Conv2D', function_to_run='tf.matmul(*args)', method_object=None, function_args=[eval('inputs'), eval('self.kernel')], function_kwargs={}, max_wait_secs=30)
layer = custom_method(
MyDenseLayer(10), imports='import numpy as np;from tensorflow.keras.backend import clear_session, set_session;from pt import trans as tf2;import matplotlib.pyplot as plt;import torch.nn as nn;import tensorflow.keras.layers as L;from tensorflow import keras;from tensorflow.keras.layers import Dense, Dropout as drop;import tensorflow as tf;from tensorflow.keras.layers import Dense as dense_layer, Dropout as drop, Conv2D as conv;from tensorflow.keras.layers import *;import sys;from tensorflow.keras import layers;import torchvision.transforms as transforms;import torch.nn.functional as F;import math, random;from os.path import exists;from pprint.xyz.abc import pprint as ppr;import torch.optim as optim;import tensorflow.compat.v1 as tf;import tensorflow.keras.backend as K;import torchvision;from torch import trans;import torch;from tensorflow.keras.layers import Dense, Dropout, Conv2D', function_to_run='MyDenseLayer(*args)', method_object=None, function_args=[eval('10')], function_kwargs={}, max_wait_secs=30)
criterion = custom_method(
nn.CrossEntropyLoss(), imports='import numpy as np;from tensorflow.keras.backend import clear_session, set_session;from pt import trans as tf2;import matplotlib.pyplot as plt;import torch.nn as nn;import tensorflow.keras.layers as L;from tensorflow import keras;from tensorflow.keras.layers import Dense, Dropout as drop;import tensorflow as tf;from tensorflow.keras.layers import Dense as dense_layer, Dropout as drop, Conv2D as conv;from tensorflow.keras.layers import *;import sys;from tensorflow.keras import layers;import torchvision.transforms as transforms;import torch.nn.functional as F;import math, random;from os.path import exists;from pprint.xyz.abc import pprint as ppr;import torch.optim as optim;import tensorflow.compat.v1 as tf;import tensorflow.keras.backend as K;import torchvision;from torch import trans;import torch;from tensorflow.keras.layers import Dense, Dropout, Conv2D', function_to_run='nn.CrossEntropyLoss()', method_object=None, function_args=[], function_kwargs={}, max_wait_secs=30)
optimizer = custom_method(
optim.SGD(net.parameters(), lr=0.001, momentum=0.9), imports='import numpy as np;from tensorflow.keras.backend import clear_session, set_session;from pt import trans as tf2;import matplotlib.pyplot as plt;import torch.nn as nn;import tensorflow.keras.layers as L;from tensorflow import keras;from tensorflow.keras.layers import Dense, Dropout as drop;import tensorflow as tf;from tensorflow.keras.layers import Dense as dense_layer, Dropout as drop, Conv2D as conv;from tensorflow.keras.layers import *;import sys;from tensorflow.keras import layers;import torchvision.transforms as transforms;import torch.nn.functional as F;import math, random;from os.path import exists;from pprint.xyz.abc import pprint as ppr;import torch.optim as optim;import tensorflow.compat.v1 as tf;import tensorflow.keras.backend as K;import torchvision;from torch import trans;import torch;from tensorflow.keras.layers import Dense, Dropout, Conv2D', function_to_run='optim.SGD(*args, **kwargs)', method_object=None, function_args=[eval('net.parameters()')], function_kwargs={'lr': eval('0.001'), 'momentum': eval('0.9')}, max_wait_secs=30)
for epoch in range(2):
    running_loss = 0.0
    for (i, data) in enumerate(trainloader, 0):
        (inputs, labels) = data
        custom_method(
        optimizer.zero_grad(), imports='import numpy as np;from tensorflow.keras.backend import clear_session, set_session;from pt import trans as tf2;import matplotlib.pyplot as plt;import torch.nn as nn;import tensorflow.keras.layers as L;from tensorflow import keras;from tensorflow.keras.layers import Dense, Dropout as drop;import tensorflow as tf;from tensorflow.keras.layers import Dense as dense_layer, Dropout as drop, Conv2D as conv;from tensorflow.keras.layers import *;import sys;from tensorflow.keras import layers;import torchvision.transforms as transforms;import torch.nn.functional as F;import math, random;from os.path import exists;from pprint.xyz.abc import pprint as ppr;import torch.optim as optim;import tensorflow.compat.v1 as tf;import tensorflow.keras.backend as K;import torchvision;from torch import trans;import torch;from tensorflow.keras.layers import Dense, Dropout, Conv2D', function_to_run='optimizer.zero_grad()', method_object=None, function_args=[], function_kwargs={}, max_wait_secs=30)
        outputs = custom_method(
        net(inputs), imports='import numpy as np;from tensorflow.keras.backend import clear_session, set_session;from pt import trans as tf2;import matplotlib.pyplot as plt;import torch.nn as nn;import tensorflow.keras.layers as L;from tensorflow import keras;from tensorflow.keras.layers import Dense, Dropout as drop;import tensorflow as tf;from tensorflow.keras.layers import Dense as dense_layer, Dropout as drop, Conv2D as conv;from tensorflow.keras.layers import *;import sys;from tensorflow.keras import layers;import torchvision.transforms as transforms;import torch.nn.functional as F;import math, random;from os.path import exists;from pprint.xyz.abc import pprint as ppr;import torch.optim as optim;import tensorflow.compat.v1 as tf;import tensorflow.keras.backend as K;import torchvision;from torch import trans;import torch;from tensorflow.keras.layers import Dense, Dropout, Conv2D', function_to_run='obj(*args)', method_object='net', function_args=[eval('inputs')], function_kwargs={}, max_wait_secs=30, custom_class='class Net(nn.Module, abc.Module):\n    def __init__(self):\n        super().__init__()\n        self.conv1 = nn.Conv2d(3, 6, 5)\n        self.pool = nn.MaxPool2d(2, 2)\n        self.conv2 = nn.Conv2d(6, 16, 5)\n        self.fc1 = nn.Linear(16 * 5 * 5, 120)\n        self.fc2 = nn.Linear(120, 84)\n        self.fc3 = nn.Linear(84, 10)\n\n    def forward(self, x):\n        x = self.pool(F.relu(self.conv1(x)))\n        x = self.pool(F.relu(self.conv2(x)))\n        x = torch.flatten(x, 1) # flatten all dimensions except batch\n        x = F.relu(self.fc1(x))\n        x = F.relu(self.fc2(x))\n        x = self.fc3(x)\n        return x')
        loss = custom_method(
        criterion(outputs, labels), imports='import numpy as np;from tensorflow.keras.backend import clear_session, set_session;from pt import trans as tf2;import matplotlib.pyplot as plt;import torch.nn as nn;import tensorflow.keras.layers as L;from tensorflow import keras;from tensorflow.keras.layers import Dense, Dropout as drop;import tensorflow as tf;from tensorflow.keras.layers import Dense as dense_layer, Dropout as drop, Conv2D as conv;from tensorflow.keras.layers import *;import sys;from tensorflow.keras import layers;import torchvision.transforms as transforms;import torch.nn.functional as F;import math, random;from os.path import exists;from pprint.xyz.abc import pprint as ppr;import torch.optim as optim;import tensorflow.compat.v1 as tf;import tensorflow.keras.backend as K;import torchvision;from torch import trans;import torch;from tensorflow.keras.layers import Dense, Dropout, Conv2D', function_to_run='obj(*args)', method_object='criterion', function_args=[eval('outputs'), eval('labels')], function_kwargs={}, max_wait_secs=30, custom_class=None)
        loss.backward()
        custom_method(
        optimizer.step(), imports='import numpy as np;from tensorflow.keras.backend import clear_session, set_session;from pt import trans as tf2;import matplotlib.pyplot as plt;import torch.nn as nn;import tensorflow.keras.layers as L;from tensorflow import keras;from tensorflow.keras.layers import Dense, Dropout as drop;import tensorflow as tf;from tensorflow.keras.layers import Dense as dense_layer, Dropout as drop, Conv2D as conv;from tensorflow.keras.layers import *;import sys;from tensorflow.keras import layers;import torchvision.transforms as transforms;import torch.nn.functional as F;import math, random;from os.path import exists;from pprint.xyz.abc import pprint as ppr;import torch.optim as optim;import tensorflow.compat.v1 as tf;import tensorflow.keras.backend as K;import torchvision;from torch import trans;import torch;from tensorflow.keras.layers import Dense, Dropout, Conv2D', function_to_run='optimizer.step()', method_object=None, function_args=[], function_kwargs={}, max_wait_secs=30)
        running_loss += loss.item()
        if i % 2000 == 1999:
            print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
            running_loss = 0.0
print('Finished Training')
PATH = './cifar_net.pth'
custom_method(
torch.save(net.state_dict(), PATH), imports='import numpy as np;from tensorflow.keras.backend import clear_session, set_session;from pt import trans as tf2;import matplotlib.pyplot as plt;import torch.nn as nn;import tensorflow.keras.layers as L;from tensorflow import keras;from tensorflow.keras.layers import Dense, Dropout as drop;import tensorflow as tf;from tensorflow.keras.layers import Dense as dense_layer, Dropout as drop, Conv2D as conv;from tensorflow.keras.layers import *;import sys;from tensorflow.keras import layers;import torchvision.transforms as transforms;import torch.nn.functional as F;import math, random;from os.path import exists;from pprint.xyz.abc import pprint as ppr;import torch.optim as optim;import tensorflow.compat.v1 as tf;import tensorflow.keras.backend as K;import torchvision;from torch import trans;import torch;from tensorflow.keras.layers import Dense, Dropout, Conv2D', function_to_run='torch.save(*args)', method_object=None, function_args=[eval('net.state_dict()'), eval('PATH')], function_kwargs={}, max_wait_secs=30)
dataiter = iter(testloader)
(images, labels) = custom_method(
next(dataiter), imports='import numpy as np;from tensorflow.keras.backend import clear_session, set_session;from pt import trans as tf2;import matplotlib.pyplot as plt;import torch.nn as nn;import tensorflow.keras.layers as L;from tensorflow import keras;from tensorflow.keras.layers import Dense, Dropout as drop;import tensorflow as tf;from tensorflow.keras.layers import Dense as dense_layer, Dropout as drop, Conv2D as conv;from tensorflow.keras.layers import *;import sys;from tensorflow.keras import layers;import torchvision.transforms as transforms;import torch.nn.functional as F;import math, random;from os.path import exists;from pprint.xyz.abc import pprint as ppr;import torch.optim as optim;import tensorflow.compat.v1 as tf;import tensorflow.keras.backend as K;import torchvision;from torch import trans;import torch;from tensorflow.keras.layers import Dense, Dropout, Conv2D', function_to_run='obj(*args)', method_object='next', function_args=[eval('dataiter')], function_kwargs={}, max_wait_secs=30, custom_class=None)
net = Net()
custom_method(
net.load_state_dict(torch.load(PATH)), imports='import numpy as np;from tensorflow.keras.backend import clear_session, set_session;from pt import trans as tf2;import matplotlib.pyplot as plt;import torch.nn as nn;import tensorflow.keras.layers as L;from tensorflow import keras;from tensorflow.keras.layers import Dense, Dropout as drop;import tensorflow as tf;from tensorflow.keras.layers import Dense as dense_layer, Dropout as drop, Conv2D as conv;from tensorflow.keras.layers import *;import sys;from tensorflow.keras import layers;import torchvision.transforms as transforms;import torch.nn.functional as F;import math, random;from os.path import exists;from pprint.xyz.abc import pprint as ppr;import torch.optim as optim;import tensorflow.compat.v1 as tf;import tensorflow.keras.backend as K;import torchvision;from torch import trans;import torch;from tensorflow.keras.layers import Dense, Dropout, Conv2D', function_to_run='obj.load_state_dict(*args)', method_object='net', function_args=[eval('torch.load(PATH)')], function_kwargs={}, max_wait_secs=30, custom_class='class Net(nn.Module, abc.Module):\n    def __init__(self):\n        super().__init__()\n        self.conv1 = nn.Conv2d(3, 6, 5)\n        self.pool = nn.MaxPool2d(2, 2)\n        self.conv2 = nn.Conv2d(6, 16, 5)\n        self.fc1 = nn.Linear(16 * 5 * 5, 120)\n        self.fc2 = nn.Linear(120, 84)\n        self.fc3 = nn.Linear(84, 10)\n\n    def forward(self, x):\n        x = self.pool(F.relu(self.conv1(x)))\n        x = self.pool(F.relu(self.conv2(x)))\n        x = torch.flatten(x, 1) # flatten all dimensions except batch\n        x = F.relu(self.fc1(x))\n        x = F.relu(self.fc2(x))\n        x = self.fc3(x)\n        return x')
outputs = custom_method(
net(images), imports='import numpy as np;from tensorflow.keras.backend import clear_session, set_session;from pt import trans as tf2;import matplotlib.pyplot as plt;import torch.nn as nn;import tensorflow.keras.layers as L;from tensorflow import keras;from tensorflow.keras.layers import Dense, Dropout as drop;import tensorflow as tf;from tensorflow.keras.layers import Dense as dense_layer, Dropout as drop, Conv2D as conv;from tensorflow.keras.layers import *;import sys;from tensorflow.keras import layers;import torchvision.transforms as transforms;import torch.nn.functional as F;import math, random;from os.path import exists;from pprint.xyz.abc import pprint as ppr;import torch.optim as optim;import tensorflow.compat.v1 as tf;import tensorflow.keras.backend as K;import torchvision;from torch import trans;import torch;from tensorflow.keras.layers import Dense, Dropout, Conv2D', function_to_run='obj(*args)', method_object='net', function_args=[eval('images')], function_kwargs={}, max_wait_secs=30, custom_class='class Net(nn.Module, abc.Module):\n    def __init__(self):\n        super().__init__()\n        self.conv1 = nn.Conv2d(3, 6, 5)\n        self.pool = nn.MaxPool2d(2, 2)\n        self.conv2 = nn.Conv2d(6, 16, 5)\n        self.fc1 = nn.Linear(16 * 5 * 5, 120)\n        self.fc2 = nn.Linear(120, 84)\n        self.fc3 = nn.Linear(84, 10)\n\n    def forward(self, x):\n        x = self.pool(F.relu(self.conv1(x)))\n        x = self.pool(F.relu(self.conv2(x)))\n        x = torch.flatten(x, 1) # flatten all dimensions except batch\n        x = F.relu(self.fc1(x))\n        x = F.relu(self.fc2(x))\n        x = self.fc3(x)\n        return x')
(_, predicted) = custom_method(
torch.max(outputs, 1), imports='import numpy as np;from tensorflow.keras.backend import clear_session, set_session;from pt import trans as tf2;import matplotlib.pyplot as plt;import torch.nn as nn;import tensorflow.keras.layers as L;from tensorflow import keras;from tensorflow.keras.layers import Dense, Dropout as drop;import tensorflow as tf;from tensorflow.keras.layers import Dense as dense_layer, Dropout as drop, Conv2D as conv;from tensorflow.keras.layers import *;import sys;from tensorflow.keras import layers;import torchvision.transforms as transforms;import torch.nn.functional as F;import math, random;from os.path import exists;from pprint.xyz.abc import pprint as ppr;import torch.optim as optim;import tensorflow.compat.v1 as tf;import tensorflow.keras.backend as K;import torchvision;from torch import trans;import torch;from tensorflow.keras.layers import Dense, Dropout, Conv2D', function_to_run='torch.max(*args)', method_object=None, function_args=[eval('outputs'), eval('1')], function_kwargs={}, max_wait_secs=30)
print('Predicted: ', ' '.join((f'{classes[predicted[j]]:5s}' for j in range(4))))
correct = 0
total = 0
with custom_method(
torch.no_grad(), imports='import numpy as np;from tensorflow.keras.backend import clear_session, set_session;from pt import trans as tf2;import matplotlib.pyplot as plt;import torch.nn as nn;import tensorflow.keras.layers as L;from tensorflow import keras;from tensorflow.keras.layers import Dense, Dropout as drop;import tensorflow as tf;from tensorflow.keras.layers import Dense as dense_layer, Dropout as drop, Conv2D as conv;from tensorflow.keras.layers import *;import sys;from tensorflow.keras import layers;import torchvision.transforms as transforms;import torch.nn.functional as F;import math, random;from os.path import exists;from pprint.xyz.abc import pprint as ppr;import torch.optim as optim;import tensorflow.compat.v1 as tf;import tensorflow.keras.backend as K;import torchvision;from torch import trans;import torch;from tensorflow.keras.layers import Dense, Dropout, Conv2D', function_to_run='torch.no_grad()', method_object=None, function_args=[], function_kwargs={}, max_wait_secs=30):
    for data in testloader:
        (images, labels) = data
        outputs = custom_method(
        net(images), imports='import numpy as np;from tensorflow.keras.backend import clear_session, set_session;from pt import trans as tf2;import matplotlib.pyplot as plt;import torch.nn as nn;import tensorflow.keras.layers as L;from tensorflow import keras;from tensorflow.keras.layers import Dense, Dropout as drop;import tensorflow as tf;from tensorflow.keras.layers import Dense as dense_layer, Dropout as drop, Conv2D as conv;from tensorflow.keras.layers import *;import sys;from tensorflow.keras import layers;import torchvision.transforms as transforms;import torch.nn.functional as F;import math, random;from os.path import exists;from pprint.xyz.abc import pprint as ppr;import torch.optim as optim;import tensorflow.compat.v1 as tf;import tensorflow.keras.backend as K;import torchvision;from torch import trans;import torch;from tensorflow.keras.layers import Dense, Dropout, Conv2D', function_to_run='obj(*args)', method_object='net', function_args=[eval('images')], function_kwargs={}, max_wait_secs=30, custom_class='class Net(nn.Module, abc.Module):\n    def __init__(self):\n        super().__init__()\n        self.conv1 = nn.Conv2d(3, 6, 5)\n        self.pool = nn.MaxPool2d(2, 2)\n        self.conv2 = nn.Conv2d(6, 16, 5)\n        self.fc1 = nn.Linear(16 * 5 * 5, 120)\n        self.fc2 = nn.Linear(120, 84)\n        self.fc3 = nn.Linear(84, 10)\n\n    def forward(self, x):\n        x = self.pool(F.relu(self.conv1(x)))\n        x = self.pool(F.relu(self.conv2(x)))\n        x = torch.flatten(x, 1) # flatten all dimensions except batch\n        x = F.relu(self.fc1(x))\n        x = F.relu(self.fc2(x))\n        x = self.fc3(x)\n        return x')
        (_, predicted) = custom_method(
        torch.max(outputs.data, 1), imports='import numpy as np;from tensorflow.keras.backend import clear_session, set_session;from pt import trans as tf2;import matplotlib.pyplot as plt;import torch.nn as nn;import tensorflow.keras.layers as L;from tensorflow import keras;from tensorflow.keras.layers import Dense, Dropout as drop;import tensorflow as tf;from tensorflow.keras.layers import Dense as dense_layer, Dropout as drop, Conv2D as conv;from tensorflow.keras.layers import *;import sys;from tensorflow.keras import layers;import torchvision.transforms as transforms;import torch.nn.functional as F;import math, random;from os.path import exists;from pprint.xyz.abc import pprint as ppr;import torch.optim as optim;import tensorflow.compat.v1 as tf;import tensorflow.keras.backend as K;import torchvision;from torch import trans;import torch;from tensorflow.keras.layers import Dense, Dropout, Conv2D', function_to_run='torch.max(*args)', method_object=None, function_args=[eval('outputs.data'), eval('1')], function_kwargs={}, max_wait_secs=30)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
print(f'Accuracy of the network on the 10000 test images: {100 * correct // total} %')
correct_pred = {classname: 0 for classname in classes}
total_pred = {classname: 0 for classname in classes}
with custom_method(
torch.no_grad(), imports='import numpy as np;from tensorflow.keras.backend import clear_session, set_session;from pt import trans as tf2;import matplotlib.pyplot as plt;import torch.nn as nn;import tensorflow.keras.layers as L;from tensorflow import keras;from tensorflow.keras.layers import Dense, Dropout as drop;import tensorflow as tf;from tensorflow.keras.layers import Dense as dense_layer, Dropout as drop, Conv2D as conv;from tensorflow.keras.layers import *;import sys;from tensorflow.keras import layers;import torchvision.transforms as transforms;import torch.nn.functional as F;import math, random;from os.path import exists;from pprint.xyz.abc import pprint as ppr;import torch.optim as optim;import tensorflow.compat.v1 as tf;import tensorflow.keras.backend as K;import torchvision;from torch import trans;import torch;from tensorflow.keras.layers import Dense, Dropout, Conv2D', function_to_run='torch.no_grad()', method_object=None, function_args=[], function_kwargs={}, max_wait_secs=30):
    for data in testloader:
        (images, labels) = data
        outputs = custom_method(
        net(images), imports='import numpy as np;from tensorflow.keras.backend import clear_session, set_session;from pt import trans as tf2;import matplotlib.pyplot as plt;import torch.nn as nn;import tensorflow.keras.layers as L;from tensorflow import keras;from tensorflow.keras.layers import Dense, Dropout as drop;import tensorflow as tf;from tensorflow.keras.layers import Dense as dense_layer, Dropout as drop, Conv2D as conv;from tensorflow.keras.layers import *;import sys;from tensorflow.keras import layers;import torchvision.transforms as transforms;import torch.nn.functional as F;import math, random;from os.path import exists;from pprint.xyz.abc import pprint as ppr;import torch.optim as optim;import tensorflow.compat.v1 as tf;import tensorflow.keras.backend as K;import torchvision;from torch import trans;import torch;from tensorflow.keras.layers import Dense, Dropout, Conv2D', function_to_run='obj(*args)', method_object='net', function_args=[eval('images')], function_kwargs={}, max_wait_secs=30, custom_class='class Net(nn.Module, abc.Module):\n    def __init__(self):\n        super().__init__()\n        self.conv1 = nn.Conv2d(3, 6, 5)\n        self.pool = nn.MaxPool2d(2, 2)\n        self.conv2 = nn.Conv2d(6, 16, 5)\n        self.fc1 = nn.Linear(16 * 5 * 5, 120)\n        self.fc2 = nn.Linear(120, 84)\n        self.fc3 = nn.Linear(84, 10)\n\n    def forward(self, x):\n        x = self.pool(F.relu(self.conv1(x)))\n        x = self.pool(F.relu(self.conv2(x)))\n        x = torch.flatten(x, 1) # flatten all dimensions except batch\n        x = F.relu(self.fc1(x))\n        x = F.relu(self.fc2(x))\n        x = self.fc3(x)\n        return x')
        (_, predictions) = custom_method(
        torch.max(outputs, 1), imports='import numpy as np;from tensorflow.keras.backend import clear_session, set_session;from pt import trans as tf2;import matplotlib.pyplot as plt;import torch.nn as nn;import tensorflow.keras.layers as L;from tensorflow import keras;from tensorflow.keras.layers import Dense, Dropout as drop;import tensorflow as tf;from tensorflow.keras.layers import Dense as dense_layer, Dropout as drop, Conv2D as conv;from tensorflow.keras.layers import *;import sys;from tensorflow.keras import layers;import torchvision.transforms as transforms;import torch.nn.functional as F;import math, random;from os.path import exists;from pprint.xyz.abc import pprint as ppr;import torch.optim as optim;import tensorflow.compat.v1 as tf;import tensorflow.keras.backend as K;import torchvision;from torch import trans;import torch;from tensorflow.keras.layers import Dense, Dropout, Conv2D', function_to_run='torch.max(*args)', method_object=None, function_args=[eval('outputs'), eval('1')], function_kwargs={}, max_wait_secs=30)
        for (label, prediction) in custom_method(
        zip(labels, predictions), imports='import numpy as np;from tensorflow.keras.backend import clear_session, set_session;from pt import trans as tf2;import matplotlib.pyplot as plt;import torch.nn as nn;import tensorflow.keras.layers as L;from tensorflow import keras;from tensorflow.keras.layers import Dense, Dropout as drop;import tensorflow as tf;from tensorflow.keras.layers import Dense as dense_layer, Dropout as drop, Conv2D as conv;from tensorflow.keras.layers import *;import sys;from tensorflow.keras import layers;import torchvision.transforms as transforms;import torch.nn.functional as F;import math, random;from os.path import exists;from pprint.xyz.abc import pprint as ppr;import torch.optim as optim;import tensorflow.compat.v1 as tf;import tensorflow.keras.backend as K;import torchvision;from torch import trans;import torch;from tensorflow.keras.layers import Dense, Dropout, Conv2D', function_to_run='obj(*args)', method_object='zip', function_args=[eval('labels'), eval('predictions')], function_kwargs={}, max_wait_secs=30, custom_class=None):
            if label == prediction:
                correct_pred[classes[label]] += 1
            total_pred[classes[label]] += 1
for (classname, correct_count) in correct_pred.items():
    accuracy = 100 * float(correct_count) / total_pred[classname]
    print(f'Accuracy for class: {classname:5s} is {accuracy:.1f} %')
