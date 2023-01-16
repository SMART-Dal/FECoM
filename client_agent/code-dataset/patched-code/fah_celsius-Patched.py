requiredClass dict_keys([])
requiredObjects ['logger', 'l0', 'model', 'l0', 'l1', 'l2', 'model']
requiredObjClassMapping {}
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
                alias(name='logging')]),
        Assign(
            targets=[
                Name(id='logger', ctx=Store())],
            value=Call(
                func=Name(id='custom_method', ctx=Load()),
                args=[
                    Expr(
                        value=Call(
                            func=Attribute(
                                value=Name(id='tf', ctx=Load()),
                                attr='get_logger',
                                ctx=Load()),
                            args=[],
                            keywords=[]))],
                keywords=[
                    keyword(
                        arg='imports',
                        value=Constant(value='import logging;import matplotlib.pyplot as plt;import numpy as np;import tensorflow as tf')),
                    keyword(
                        arg='function_to_run',
                        value=Constant(value='tf.get_logger()')),
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
        Expr(
            value=Call(
                func=Name(id='custom_method', ctx=Load()),
                args=[
                    Expr(
                        value=Call(
                            func=Attribute(
                                value=Name(id='logger', ctx=Load()),
                                attr='setLevel',
                                ctx=Load()),
                            args=[
                                Attribute(
                                    value=Name(id='logging', ctx=Load()),
                                    attr='ERROR',
                                    ctx=Load())],
                            keywords=[]))],
                keywords=[
                    keyword(
                        arg='imports',
                        value=Constant(value='import logging;import matplotlib.pyplot as plt;import numpy as np;import tensorflow as tf')),
                    keyword(
                        arg='function_to_run',
                        value=Constant(value='obj.setLevel(*args)')),
                    keyword(
                        arg='method_object',
                        value=Constant(value='logger')),
                    keyword(
                        arg='function_args',
                        value=List(
                            elts=[
                                Call(
                                    func=Name(id='eval', ctx=Load()),
                                    args=[
                                        Constant(value='logging.ERROR')],
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
                Name(id='celsius_q', ctx=Store())],
            value=Call(
                func=Attribute(
                    value=Name(id='np', ctx=Load()),
                    attr='array',
                    ctx=Load()),
                args=[
                    List(
                        elts=[
                            UnaryOp(
                                op=USub(),
                                operand=Constant(value=40)),
                            UnaryOp(
                                op=USub(),
                                operand=Constant(value=10)),
                            Constant(value=0),
                            Constant(value=8),
                            Constant(value=15),
                            Constant(value=22),
                            Constant(value=38)],
                        ctx=Load())],
                keywords=[
                    keyword(
                        arg='dtype',
                        value=Name(id='float', ctx=Load()))])),
        Assign(
            targets=[
                Name(id='fahrenheit_a', ctx=Store())],
            value=Call(
                func=Attribute(
                    value=Name(id='np', ctx=Load()),
                    attr='array',
                    ctx=Load()),
                args=[
                    List(
                        elts=[
                            UnaryOp(
                                op=USub(),
                                operand=Constant(value=40)),
                            Constant(value=14),
                            Constant(value=32),
                            Constant(value=46),
                            Constant(value=59),
                            Constant(value=72),
                            Constant(value=100)],
                        ctx=Load())],
                keywords=[
                    keyword(
                        arg='dtype',
                        value=Name(id='float', ctx=Load()))])),
        For(
            target=Tuple(
                elts=[
                    Name(id='i', ctx=Store()),
                    Name(id='c', ctx=Store())],
                ctx=Store()),
            iter=Call(
                func=Name(id='enumerate', ctx=Load()),
                args=[
                    Name(id='celsius_q', ctx=Load())],
                keywords=[]),
            body=[
                Expr(
                    value=Call(
                        func=Name(id='print', ctx=Load()),
                        args=[
                            Call(
                                func=Attribute(
                                    value=Constant(value='{} degrees Celsius = {} degrees Fahrenheit'),
                                    attr='format',
                                    ctx=Load()),
                                args=[
                                    Name(id='c', ctx=Load()),
                                    Subscript(
                                        value=Name(id='fahrenheit_a', ctx=Load()),
                                        slice=Name(id='i', ctx=Load()),
                                        ctx=Load())],
                                keywords=[])],
                        keywords=[]))],
            orelse=[]),
        Assign(
            targets=[
                Name(id='l0', ctx=Store())],
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
                                    attr='layers',
                                    ctx=Load()),
                                attr='Dense',
                                ctx=Load()),
                            args=[],
                            keywords=[
                                keyword(
                                    arg='units',
                                    value=Constant(value=1)),
                                keyword(
                                    arg='input_shape',
                                    value=List(
                                        elts=[
                                            Constant(value=1)],
                                        ctx=Load()))]))],
                keywords=[
                    keyword(
                        arg='imports',
                        value=Constant(value='import logging;import matplotlib.pyplot as plt;import numpy as np;import tensorflow as tf')),
                    keyword(
                        arg='function_to_run',
                        value=Constant(value='tf.keras.layers.Dense(**kwargs)')),
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
                                Constant(value='units'),
                                Constant(value='input_shape')],
                            values=[
                                Call(
                                    func=Name(id='eval', ctx=Load()),
                                    args=[
                                        Constant(value='1')],
                                    keywords=[]),
                                Call(
                                    func=Name(id='eval', ctx=Load()),
                                    args=[
                                        Constant(value='[1]')],
                                    keywords=[])])),
                    keyword(
                        arg='max_wait_secs',
                        value=Constant(value=30))])),
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
                                    value=Name(id='tf', ctx=Load()),
                                    attr='keras',
                                    ctx=Load()),
                                attr='Sequential',
                                ctx=Load()),
                            args=[
                                List(
                                    elts=[
                                        Name(id='l0', ctx=Load())],
                                    ctx=Load())],
                            keywords=[]))],
                keywords=[
                    keyword(
                        arg='imports',
                        value=Constant(value='import logging;import matplotlib.pyplot as plt;import numpy as np;import tensorflow as tf')),
                    keyword(
                        arg='function_to_run',
                        value=Constant(value='tf.keras.Sequential(*args)')),
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
                                        Constant(value='[l0]')],
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
                            keywords=[
                                keyword(
                                    arg='loss',
                                    value=Constant(value='mean_squared_error')),
                                keyword(
                                    arg='optimizer',
                                    value=Call(
                                        func=Attribute(
                                            value=Attribute(
                                                value=Attribute(
                                                    value=Name(id='tf', ctx=Load()),
                                                    attr='keras',
                                                    ctx=Load()),
                                                attr='optimizers',
                                                ctx=Load()),
                                            attr='Adam',
                                            ctx=Load()),
                                        args=[
                                            Constant(value=0.1)],
                                        keywords=[]))]))],
                keywords=[
                    keyword(
                        arg='imports',
                        value=Constant(value='import logging;import matplotlib.pyplot as plt;import numpy as np;import tensorflow as tf')),
                    keyword(
                        arg='function_to_run',
                        value=Constant(value='obj.compile(**kwargs)')),
                    keyword(
                        arg='method_object',
                        value=Constant(value='model')),
                    keyword(
                        arg='function_args',
                        value=List(elts=[], ctx=Load())),
                    keyword(
                        arg='function_kwargs',
                        value=Dict(
                            keys=[
                                Constant(value='loss'),
                                Constant(value='optimizer')],
                            values=[
                                Call(
                                    func=Name(id='eval', ctx=Load()),
                                    args=[
                                        Constant(value="'mean_squared_error'")],
                                    keywords=[]),
                                Call(
                                    func=Name(id='eval', ctx=Load()),
                                    args=[
                                        Constant(value='tf.keras.optimizers.Adam(0.1)')],
                                    keywords=[])])),
                    keyword(
                        arg='max_wait_secs',
                        value=Constant(value=30)),
                    keyword(
                        arg='custom_class',
                        value=Constant(value=None))])),
        Assign(
            targets=[
                Name(id='history', ctx=Store())],
            value=Call(
                func=Name(id='custom_method', ctx=Load()),
                args=[
                    Expr(
                        value=Call(
                            func=Attribute(
                                value=Name(id='model', ctx=Load()),
                                attr='fit',
                                ctx=Load()),
                            args=[
                                Name(id='celsius_q', ctx=Load()),
                                Name(id='fahrenheit_a', ctx=Load())],
                            keywords=[
                                keyword(
                                    arg='epochs',
                                    value=Constant(value=500)),
                                keyword(
                                    arg='verbose',
                                    value=Constant(value=False))]))],
                keywords=[
                    keyword(
                        arg='imports',
                        value=Constant(value='import logging;import matplotlib.pyplot as plt;import numpy as np;import tensorflow as tf')),
                    keyword(
                        arg='function_to_run',
                        value=Constant(value='obj.fit(*args, **kwargs)')),
                    keyword(
                        arg='method_object',
                        value=Constant(value='model')),
                    keyword(
                        arg='function_args',
                        value=List(
                            elts=[
                                Call(
                                    func=Name(id='eval', ctx=Load()),
                                    args=[
                                        Constant(value='celsius_q')],
                                    keywords=[]),
                                Call(
                                    func=Name(id='eval', ctx=Load()),
                                    args=[
                                        Constant(value='fahrenheit_a')],
                                    keywords=[])],
                            ctx=Load())),
                    keyword(
                        arg='function_kwargs',
                        value=Dict(
                            keys=[
                                Constant(value='epochs'),
                                Constant(value='verbose')],
                            values=[
                                Call(
                                    func=Name(id='eval', ctx=Load()),
                                    args=[
                                        Constant(value='500')],
                                    keywords=[]),
                                Call(
                                    func=Name(id='eval', ctx=Load()),
                                    args=[
                                        Constant(value='False')],
                                    keywords=[])])),
                    keyword(
                        arg='max_wait_secs',
                        value=Constant(value=30)),
                    keyword(
                        arg='custom_class',
                        value=Constant(value=None))])),
        Expr(
            value=Call(
                func=Name(id='print', ctx=Load()),
                args=[
                    Constant(value='Finished training the model')],
                keywords=[])),
        Import(
            names=[
                alias(name='matplotlib.pyplot', asname='plt')]),
        Expr(
            value=Call(
                func=Attribute(
                    value=Name(id='plt', ctx=Load()),
                    attr='xlabel',
                    ctx=Load()),
                args=[
                    Constant(value='Epoch Number')],
                keywords=[])),
        Expr(
            value=Call(
                func=Attribute(
                    value=Name(id='plt', ctx=Load()),
                    attr='ylabel',
                    ctx=Load()),
                args=[
                    Constant(value='Loss Magnitude')],
                keywords=[])),
        Expr(
            value=Call(
                func=Attribute(
                    value=Name(id='plt', ctx=Load()),
                    attr='plot',
                    ctx=Load()),
                args=[
                    Subscript(
                        value=Attribute(
                            value=Name(id='history', ctx=Load()),
                            attr='history',
                            ctx=Load()),
                        slice=Constant(value='loss'),
                        ctx=Load())],
                keywords=[])),
        Expr(
            value=Call(
                func=Name(id='print', ctx=Load()),
                args=[
                    Call(
                        func=Attribute(
                            value=Name(id='model', ctx=Load()),
                            attr='predict',
                            ctx=Load()),
                        args=[
                            List(
                                elts=[
                                    Constant(value=100.0)],
                                ctx=Load())],
                        keywords=[])],
                keywords=[])),
        Expr(
            value=Call(
                func=Name(id='print', ctx=Load()),
                args=[
                    Call(
                        func=Attribute(
                            value=Constant(value='These are the layer variables: {}'),
                            attr='format',
                            ctx=Load()),
                        args=[
                            Call(
                                func=Attribute(
                                    value=Name(id='l0', ctx=Load()),
                                    attr='get_weights',
                                    ctx=Load()),
                                args=[],
                                keywords=[])],
                        keywords=[])],
                keywords=[])),
        Assign(
            targets=[
                Name(id='l0', ctx=Store())],
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
                                    attr='layers',
                                    ctx=Load()),
                                attr='Dense',
                                ctx=Load()),
                            args=[],
                            keywords=[
                                keyword(
                                    arg='units',
                                    value=Constant(value=4)),
                                keyword(
                                    arg='input_shape',
                                    value=List(
                                        elts=[
                                            Constant(value=1)],
                                        ctx=Load()))]))],
                keywords=[
                    keyword(
                        arg='imports',
                        value=Constant(value='import logging;import matplotlib.pyplot as plt;import numpy as np;import tensorflow as tf')),
                    keyword(
                        arg='function_to_run',
                        value=Constant(value='tf.keras.layers.Dense(**kwargs)')),
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
                                Constant(value='units'),
                                Constant(value='input_shape')],
                            values=[
                                Call(
                                    func=Name(id='eval', ctx=Load()),
                                    args=[
                                        Constant(value='4')],
                                    keywords=[]),
                                Call(
                                    func=Name(id='eval', ctx=Load()),
                                    args=[
                                        Constant(value='[1]')],
                                    keywords=[])])),
                    keyword(
                        arg='max_wait_secs',
                        value=Constant(value=30))])),
        Assign(
            targets=[
                Name(id='l1', ctx=Store())],
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
                                    attr='layers',
                                    ctx=Load()),
                                attr='Dense',
                                ctx=Load()),
                            args=[],
                            keywords=[
                                keyword(
                                    arg='units',
                                    value=Constant(value=4))]))],
                keywords=[
                    keyword(
                        arg='imports',
                        value=Constant(value='import logging;import matplotlib.pyplot as plt;import numpy as np;import tensorflow as tf')),
                    keyword(
                        arg='function_to_run',
                        value=Constant(value='tf.keras.layers.Dense(**kwargs)')),
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
                                Constant(value='units')],
                            values=[
                                Call(
                                    func=Name(id='eval', ctx=Load()),
                                    args=[
                                        Constant(value='4')],
                                    keywords=[])])),
                    keyword(
                        arg='max_wait_secs',
                        value=Constant(value=30))])),
        Assign(
            targets=[
                Name(id='l2', ctx=Store())],
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
                                    attr='layers',
                                    ctx=Load()),
                                attr='Dense',
                                ctx=Load()),
                            args=[],
                            keywords=[
                                keyword(
                                    arg='units',
                                    value=Constant(value=1))]))],
                keywords=[
                    keyword(
                        arg='imports',
                        value=Constant(value='import logging;import matplotlib.pyplot as plt;import numpy as np;import tensorflow as tf')),
                    keyword(
                        arg='function_to_run',
                        value=Constant(value='tf.keras.layers.Dense(**kwargs)')),
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
                                Constant(value='units')],
                            values=[
                                Call(
                                    func=Name(id='eval', ctx=Load()),
                                    args=[
                                        Constant(value='1')],
                                    keywords=[])])),
                    keyword(
                        arg='max_wait_secs',
                        value=Constant(value=30))])),
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
                                    value=Name(id='tf', ctx=Load()),
                                    attr='keras',
                                    ctx=Load()),
                                attr='Sequential',
                                ctx=Load()),
                            args=[
                                List(
                                    elts=[
                                        Name(id='l0', ctx=Load()),
                                        Name(id='l1', ctx=Load()),
                                        Name(id='l2', ctx=Load())],
                                    ctx=Load())],
                            keywords=[]))],
                keywords=[
                    keyword(
                        arg='imports',
                        value=Constant(value='import logging;import matplotlib.pyplot as plt;import numpy as np;import tensorflow as tf')),
                    keyword(
                        arg='function_to_run',
                        value=Constant(value='tf.keras.Sequential(*args)')),
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
                                        Constant(value='[l0, l1, l2]')],
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
                            keywords=[
                                keyword(
                                    arg='loss',
                                    value=Constant(value='mean_squared_error')),
                                keyword(
                                    arg='optimizer',
                                    value=Call(
                                        func=Attribute(
                                            value=Attribute(
                                                value=Attribute(
                                                    value=Name(id='tf', ctx=Load()),
                                                    attr='keras',
                                                    ctx=Load()),
                                                attr='optimizers',
                                                ctx=Load()),
                                            attr='Adam',
                                            ctx=Load()),
                                        args=[
                                            Constant(value=0.1)],
                                        keywords=[]))]))],
                keywords=[
                    keyword(
                        arg='imports',
                        value=Constant(value='import logging;import matplotlib.pyplot as plt;import numpy as np;import tensorflow as tf')),
                    keyword(
                        arg='function_to_run',
                        value=Constant(value='obj.compile(**kwargs)')),
                    keyword(
                        arg='method_object',
                        value=Constant(value='model')),
                    keyword(
                        arg='function_args',
                        value=List(elts=[], ctx=Load())),
                    keyword(
                        arg='function_kwargs',
                        value=Dict(
                            keys=[
                                Constant(value='loss'),
                                Constant(value='optimizer')],
                            values=[
                                Call(
                                    func=Name(id='eval', ctx=Load()),
                                    args=[
                                        Constant(value="'mean_squared_error'")],
                                    keywords=[]),
                                Call(
                                    func=Name(id='eval', ctx=Load()),
                                    args=[
                                        Constant(value='tf.keras.optimizers.Adam(0.1)')],
                                    keywords=[])])),
                    keyword(
                        arg='max_wait_secs',
                        value=Constant(value=30)),
                    keyword(
                        arg='custom_class',
                        value=Constant(value=None))])),
        Expr(
            value=Call(
                func=Name(id='custom_method', ctx=Load()),
                args=[
                    Expr(
                        value=Call(
                            func=Attribute(
                                value=Name(id='model', ctx=Load()),
                                attr='fit',
                                ctx=Load()),
                            args=[
                                Name(id='celsius_q', ctx=Load()),
                                Name(id='fahrenheit_a', ctx=Load())],
                            keywords=[
                                keyword(
                                    arg='epochs',
                                    value=Constant(value=500)),
                                keyword(
                                    arg='verbose',
                                    value=Constant(value=False))]))],
                keywords=[
                    keyword(
                        arg='imports',
                        value=Constant(value='import logging;import matplotlib.pyplot as plt;import numpy as np;import tensorflow as tf')),
                    keyword(
                        arg='function_to_run',
                        value=Constant(value='obj.fit(*args, **kwargs)')),
                    keyword(
                        arg='method_object',
                        value=Constant(value='model')),
                    keyword(
                        arg='function_args',
                        value=List(
                            elts=[
                                Call(
                                    func=Name(id='eval', ctx=Load()),
                                    args=[
                                        Constant(value='celsius_q')],
                                    keywords=[]),
                                Call(
                                    func=Name(id='eval', ctx=Load()),
                                    args=[
                                        Constant(value='fahrenheit_a')],
                                    keywords=[])],
                            ctx=Load())),
                    keyword(
                        arg='function_kwargs',
                        value=Dict(
                            keys=[
                                Constant(value='epochs'),
                                Constant(value='verbose')],
                            values=[
                                Call(
                                    func=Name(id='eval', ctx=Load()),
                                    args=[
                                        Constant(value='500')],
                                    keywords=[]),
                                Call(
                                    func=Name(id='eval', ctx=Load()),
                                    args=[
                                        Constant(value='False')],
                                    keywords=[])])),
                    keyword(
                        arg='max_wait_secs',
                        value=Constant(value=30)),
                    keyword(
                        arg='custom_class',
                        value=Constant(value=None))])),
        Expr(
            value=Call(
                func=Name(id='print', ctx=Load()),
                args=[
                    Constant(value='Finished training the model')],
                keywords=[])),
        Expr(
            value=Call(
                func=Name(id='print', ctx=Load()),
                args=[
                    Call(
                        func=Attribute(
                            value=Name(id='model', ctx=Load()),
                            attr='predict',
                            ctx=Load()),
                        args=[
                            List(
                                elts=[
                                    Constant(value=100.0)],
                                ctx=Load())],
                        keywords=[])],
                keywords=[])),
        Expr(
            value=Call(
                func=Name(id='print', ctx=Load()),
                args=[
                    Call(
                        func=Attribute(
                            value=Constant(value='Model predicts that 100 degrees Celsius is: {} degrees Fahrenheit'),
                            attr='format',
                            ctx=Load()),
                        args=[
                            Call(
                                func=Attribute(
                                    value=Name(id='model', ctx=Load()),
                                    attr='predict',
                                    ctx=Load()),
                                args=[
                                    List(
                                        elts=[
                                            Constant(value=100.0)],
                                        ctx=Load())],
                                keywords=[])],
                        keywords=[])],
                keywords=[])),
        Expr(
            value=Call(
                func=Name(id='print', ctx=Load()),
                args=[
                    Call(
                        func=Attribute(
                            value=Constant(value='These are the l0 variables: {}'),
                            attr='format',
                            ctx=Load()),
                        args=[
                            Call(
                                func=Attribute(
                                    value=Name(id='l0', ctx=Load()),
                                    attr='get_weights',
                                    ctx=Load()),
                                args=[],
                                keywords=[])],
                        keywords=[])],
                keywords=[])),
        Expr(
            value=Call(
                func=Name(id='print', ctx=Load()),
                args=[
                    Call(
                        func=Attribute(
                            value=Constant(value='These are the l1 variables: {}'),
                            attr='format',
                            ctx=Load()),
                        args=[
                            Call(
                                func=Attribute(
                                    value=Name(id='l1', ctx=Load()),
                                    attr='get_weights',
                                    ctx=Load()),
                                args=[],
                                keywords=[])],
                        keywords=[])],
                keywords=[])),
        Expr(
            value=Call(
                func=Name(id='print', ctx=Load()),
                args=[
                    Call(
                        func=Attribute(
                            value=Constant(value='These are the l2 variables: {}'),
                            attr='format',
                            ctx=Load()),
                        args=[
                            Call(
                                func=Attribute(
                                    value=Name(id='l2', ctx=Load()),
                                    attr='get_weights',
                                    ctx=Load()),
                                args=[],
                                keywords=[])],
                        keywords=[])],
                keywords=[]))],
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
import logging
logger = custom_method(
tf.get_logger(), imports='import logging;import matplotlib.pyplot as plt;import numpy as np;import tensorflow as tf', function_to_run='tf.get_logger()', method_object=None, function_args=[], function_kwargs={}, max_wait_secs=30)
custom_method(
logger.setLevel(logging.ERROR), imports='import logging;import matplotlib.pyplot as plt;import numpy as np;import tensorflow as tf', function_to_run='obj.setLevel(*args)', method_object='logger', function_args=[eval('logging.ERROR')], function_kwargs={}, max_wait_secs=30, custom_class=None)
celsius_q = np.array([-40, -10, 0, 8, 15, 22, 38], dtype=float)
fahrenheit_a = np.array([-40, 14, 32, 46, 59, 72, 100], dtype=float)
for (i, c) in enumerate(celsius_q):
    print('{} degrees Celsius = {} degrees Fahrenheit'.format(c, fahrenheit_a[i]))
l0 = custom_method(
tf.keras.layers.Dense(units=1, input_shape=[1]), imports='import logging;import matplotlib.pyplot as plt;import numpy as np;import tensorflow as tf', function_to_run='tf.keras.layers.Dense(**kwargs)', method_object=None, function_args=[], function_kwargs={'units': eval('1'), 'input_shape': eval('[1]')}, max_wait_secs=30)
model = custom_method(
tf.keras.Sequential([l0]), imports='import logging;import matplotlib.pyplot as plt;import numpy as np;import tensorflow as tf', function_to_run='tf.keras.Sequential(*args)', method_object=None, function_args=[eval('[l0]')], function_kwargs={}, max_wait_secs=30)
custom_method(
model.compile(loss='mean_squared_error', optimizer=tf.keras.optimizers.Adam(0.1)), imports='import logging;import matplotlib.pyplot as plt;import numpy as np;import tensorflow as tf', function_to_run='obj.compile(**kwargs)', method_object='model', function_args=[], function_kwargs={'loss': eval("'mean_squared_error'"), 'optimizer': eval('tf.keras.optimizers.Adam(0.1)')}, max_wait_secs=30, custom_class=None)
history = custom_method(
model.fit(celsius_q, fahrenheit_a, epochs=500, verbose=False), imports='import logging;import matplotlib.pyplot as plt;import numpy as np;import tensorflow as tf', function_to_run='obj.fit(*args, **kwargs)', method_object='model', function_args=[eval('celsius_q'), eval('fahrenheit_a')], function_kwargs={'epochs': eval('500'), 'verbose': eval('False')}, max_wait_secs=30, custom_class=None)
print('Finished training the model')
import matplotlib.pyplot as plt
plt.xlabel('Epoch Number')
plt.ylabel('Loss Magnitude')
plt.plot(history.history['loss'])
print(model.predict([100.0]))
print('These are the layer variables: {}'.format(l0.get_weights()))
l0 = custom_method(
tf.keras.layers.Dense(units=4, input_shape=[1]), imports='import logging;import matplotlib.pyplot as plt;import numpy as np;import tensorflow as tf', function_to_run='tf.keras.layers.Dense(**kwargs)', method_object=None, function_args=[], function_kwargs={'units': eval('4'), 'input_shape': eval('[1]')}, max_wait_secs=30)
l1 = custom_method(
tf.keras.layers.Dense(units=4), imports='import logging;import matplotlib.pyplot as plt;import numpy as np;import tensorflow as tf', function_to_run='tf.keras.layers.Dense(**kwargs)', method_object=None, function_args=[], function_kwargs={'units': eval('4')}, max_wait_secs=30)
l2 = custom_method(
tf.keras.layers.Dense(units=1), imports='import logging;import matplotlib.pyplot as plt;import numpy as np;import tensorflow as tf', function_to_run='tf.keras.layers.Dense(**kwargs)', method_object=None, function_args=[], function_kwargs={'units': eval('1')}, max_wait_secs=30)
model = custom_method(
tf.keras.Sequential([l0, l1, l2]), imports='import logging;import matplotlib.pyplot as plt;import numpy as np;import tensorflow as tf', function_to_run='tf.keras.Sequential(*args)', method_object=None, function_args=[eval('[l0, l1, l2]')], function_kwargs={}, max_wait_secs=30)
custom_method(
model.compile(loss='mean_squared_error', optimizer=tf.keras.optimizers.Adam(0.1)), imports='import logging;import matplotlib.pyplot as plt;import numpy as np;import tensorflow as tf', function_to_run='obj.compile(**kwargs)', method_object='model', function_args=[], function_kwargs={'loss': eval("'mean_squared_error'"), 'optimizer': eval('tf.keras.optimizers.Adam(0.1)')}, max_wait_secs=30, custom_class=None)
custom_method(
model.fit(celsius_q, fahrenheit_a, epochs=500, verbose=False), imports='import logging;import matplotlib.pyplot as plt;import numpy as np;import tensorflow as tf', function_to_run='obj.fit(*args, **kwargs)', method_object='model', function_args=[eval('celsius_q'), eval('fahrenheit_a')], function_kwargs={'epochs': eval('500'), 'verbose': eval('False')}, max_wait_secs=30, custom_class=None)
print('Finished training the model')
print(model.predict([100.0]))
print('Model predicts that 100 degrees Celsius is: {} degrees Fahrenheit'.format(model.predict([100.0])))
print('These are the l0 variables: {}'.format(l0.get_weights()))
print('These are the l1 variables: {}'.format(l1.get_weights()))
print('These are the l2 variables: {}'.format(l2.get_weights()))
