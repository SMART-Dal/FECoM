{'col_offset': 0,
 'end_col_offset': 53,
 'end_lineno': 10,
 'lineno': 10,
 'targets': [<ast.Name object at 0x7f8e9d573ac0>],
 'type_comment': None,
 'value': <ast.Call object at 0x7f8e9d573a90>}
{'col_offset': 0,
 'end_col_offset': 53,
 'end_lineno': 11,
 'lineno': 11,
 'targets': [<ast.Name object at 0x7f8e9d573880>],
 'type_comment': None,
 'value': <ast.Call object at 0x7f8e9d573850>}
{'col_offset': 0,
 'end_col_offset': 40,
 'end_lineno': 12,
 'lineno': 12,
 'targets': [<ast.Name object at 0x7f8e9d573640>],
 'type_comment': None,
 'value': <ast.Call object at 0x7f8e9d573610>}
{'col_offset': 0,
 'end_col_offset': 114,
 'end_lineno': 17,
 'lineno': 16,
 'targets': [<ast.Name object at 0x7f8e9d573490>],
 'type_comment': None,
 'value': <ast.Call object at 0x7f8e9d573460>}
{'col_offset': 0,
 'end_col_offset': 55,
 'end_lineno': 18,
 'lineno': 18,
 'targets': [<ast.Name object at 0x7f8e9d573040>],
 'type_comment': None,
 'value': <ast.Call object at 0x7f8e9d573010>}
{'col_offset': 0,
 'end_col_offset': 17,
 'end_lineno': 20,
 'lineno': 20,
 'targets': [<ast.Name object at 0x7f8e9d572da0>],
 'type_comment': None,
 'value': <ast.Call object at 0x7f8e9d572d70>}
{'col_offset': 0,
 'end_col_offset': 15,
 'end_lineno': 21,
 'lineno': 21,
 'targets': [<ast.Name object at 0x7f8e9d572c50>],
 'type_comment': None,
 'value': <ast.Call object at 0x7f8e9d572c20>}
{'col_offset': 0,
 'end_col_offset': 15,
 'end_lineno': 22,
 'lineno': 22,
 'targets': [<ast.Name object at 0x7f8e9d572b60>],
 'type_comment': None,
 'value': <ast.Call object at 0x7f8e9d572b30>}
{'col_offset': 0,
 'end_col_offset': 6,
 'end_lineno': 29,
 'lineno': 24,
 'targets': [<ast.Name object at 0x7f8e9d572a70>],
 'type_comment': None,
 'value': <ast.Call object at 0x7f8e9d572a40>}
{'col_offset': 0,
 'end_col_offset': 24,
 'end_lineno': 32,
 'lineno': 32,
 'targets': [<ast.Name object at 0x7f8e9d571900>],
 'type_comment': None,
 'value': <ast.Call object at 0x7f8e9d5718d0>}
{'col_offset': 0,
 'end_col_offset': 19,
 'end_lineno': 33,
 'lineno': 33,
 'targets': [<ast.Name object at 0x7f8e9d5717e0>],
 'type_comment': None,
 'value': <ast.Call object at 0x7f8e9d5717b0>}
requiredObjects ['X', 'Y', 'z', 'w_o', 'b_o', 'model']
blah1 <ast.Name object at 0x7f8e9d573c70>  Value: tf  Split:  tf
tf()
blah1 <ast.Attribute object at 0x7f8e9d573a60>  Value: tf.placeholder  Split:  tf
tf.placeholder(tf.float32, [None, 1], name = "X")
blah1 <ast.Attribute object at 0x7f8e9d573820>  Value: tf.placeholder  Split:  tf
tf.placeholder(tf.float32, [None, 1], name = "Y")
blah1 <ast.Attribute object at 0x7f8e9d5735e0>  Value: tf.value  Split:  tf
tf.value("aname",bname,name="tname")
blah1 <ast.Attribute object at 0x7f8e9d573430>  Value: tf.Variable  Split:  tf
tf.Variable(
   tf.random_uniform([layer_1_neurons, 1], minval = -1, maxval = 1, dtype = tf.float32),test1,test2,test3="test3")
blah1 <ast.Attribute object at 0x7f8e9d572fe0>  Value: tf.Variable  Split:  tf
tf.Variable(tf.zeros([1, 1], dtype = tf.float32))
blah1 <ast.Attribute object at 0x7f8e9d572d40>  Value: ab.test  Split:  ab
blah1 <ast.Attribute object at 0x7f8e9d572bf0>  Value: xf.test  Split:  xf
blah1 <ast.Attribute object at 0x7f8e9d572b00>  Value: xy.test  Split:  xy
blah1 <ast.Attribute object at 0x7f8e9d572a10>  Value: tf.keras.models.Sequential  Split:  tf
tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(10)
    ])
blah1 <ast.Attribute object at 0x7f8e9d5719c0>  Value: model.compile  Split:  model
blah1 <ast.Attribute object at 0x7f8e9d5718a0>  Value: b_o.test.compile  Split:  b_o
blah1 <ast.Attribute object at 0x7f8e9d571780>  Value: c_o.compile  Split:  c_o
++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
Module(
    body=[
        Module(
            body=[
                FunctionDef(
                    name='custom_method',
                    args=arguments(
                        posonlyargs=[],
                        args=[
                            arg(arg='func'),
                            arg(arg='imports'),
                            arg(arg='functionDef'),
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
                                Name(id='method_details', ctx=Store())],
                            value=Dict(
                                keys=[
                                    Constant(value='imports'),
                                    Constant(value='function'),
                                    Constant(value='method_object'),
                                    Constant(value='args'),
                                    Constant(value='kwargs'),
                                    Constant(value='max_wait_secs')],
                                values=[
                                    Name(id='imports', ctx=Load()),
                                    Name(id='function_to_run', ctx=Load()),
                                    Name(id='method_object', ctx=Load()),
                                    Name(id='function_args', ctx=Load()),
                                    Name(id='function_kwargs', ctx=Load()),
                                    Name(id='max_wait_secs', ctx=Load())])),
                        Assign(
                            targets=[
                                Tuple(
                                    elts=[
                                        Name(id='sig', ctx=Store()),
                                        Name(id='func_locals', ctx=Store())],
                                    ctx=Store())],
                            value=Tuple(
                                elts=[
                                    Call(
                                        func=Attribute(
                                            value=Name(id='inspect', ctx=Load()),
                                            attr='signature',
                                            ctx=Load()),
                                        args=[
                                            Name(id='functionDef', ctx=Load())],
                                        keywords=[]),
                                    Call(
                                        func=Name(id='locals', ctx=Load()),
                                        args=[],
                                        keywords=[])],
                                ctx=Load())),
                        Assign(
                            targets=[
                                Name(id='arglist', ctx=Store())],
                            value=ListComp(
                                elt=Subscript(
                                    value=Name(id='func_locals', ctx=Load()),
                                    slice=Attribute(
                                        value=Name(id='param', ctx=Load()),
                                        attr='name',
                                        ctx=Load()),
                                    ctx=Load()),
                                generators=[
                                    comprehension(
                                        target=Name(id='param', ctx=Store()),
                                        iter=Call(
                                            func=Attribute(
                                                value=Attribute(
                                                    value=Name(id='sig', ctx=Load()),
                                                    attr='parameters',
                                                    ctx=Load()),
                                                attr='values',
                                                ctx=Load()),
                                            args=[],
                                            keywords=[]),
                                        ifs=[
                                            Compare(
                                                left=Attribute(
                                                    value=Name(id='param', ctx=Load()),
                                                    attr='kind',
                                                    ctx=Load()),
                                                ops=[
                                                    Eq()],
                                                comparators=[
                                                    Attribute(
                                                        value=Name(id='param', ctx=Load()),
                                                        attr='POSITIONAL_OR_KEYWORD',
                                                        ctx=Load())])],
                                        is_async=0)])),
                        Assign(
                            targets=[
                                Name(id='keywordDict', ctx=Store())],
                            value=ListComp(
                                elt=Subscript(
                                    value=Name(id='func_locals', ctx=Load()),
                                    slice=Attribute(
                                        value=Name(id='param', ctx=Load()),
                                        attr='name',
                                        ctx=Load()),
                                    ctx=Load()),
                                generators=[
                                    comprehension(
                                        target=Name(id='param', ctx=Store()),
                                        iter=Call(
                                            func=Attribute(
                                                value=Attribute(
                                                    value=Name(id='sig', ctx=Load()),
                                                    attr='parameters',
                                                    ctx=Load()),
                                                attr='values',
                                                ctx=Load()),
                                            args=[],
                                            keywords=[]),
                                        ifs=[
                                            Compare(
                                                left=Attribute(
                                                    value=Name(id='param', ctx=Load()),
                                                    attr='kind',
                                                    ctx=Load()),
                                                ops=[
                                                    Eq()],
                                                comparators=[
                                                    Attribute(
                                                        value=Name(id='param', ctx=Load()),
                                                        attr='KEYWORD_ONLY',
                                                        ctx=Load())])],
                                        is_async=0)])),
                        Assign(
                            targets=[
                                Name(id='data', ctx=Store())],
                            value=Call(
                                func=Attribute(
                                    value=Name(id='pickle', ctx=Load()),
                                    attr='dumps',
                                    ctx=Load()),
                                args=[
                                    Name(id='method_details', ctx=Load())],
                                keywords=[])),
                        Assign(
                            targets=[
                                Name(id='resp', ctx=Store())],
                            value=Call(
                                func=Attribute(
                                    value=Name(id='requests', ctx=Load()),
                                    attr='post',
                                    ctx=Load()),
                                args=[
                                    Name(id='url', ctx=Load())],
                                keywords=[
                                    keyword(
                                        arg='data',
                                        value=Name(id='data', ctx=Load())),
                                    keyword(
                                        arg='headers',
                                        value=Dict(
                                            keys=[
                                                Constant(value='Content-Type')],
                                            values=[
                                                Constant(value='application/octet-stream')]))])),
                        Expr(
                            value=Call(
                                func=Name(id='custom_method', ctx=Load()),
                                args=[
                                    Call(
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
                                                value=Constant(value='X'))])],
                                keywords=[
                                    keyword(
                                        arg='imports',
                                        value=Constant(value='import tensorflow as tf;import math, random;from pprint import pprint;import matplotlib.pyplot as plt;import numpy as np')),
                                    keyword(
                                        arg='functionDef',
                                        value=Constant(value='tf.placeholder')),
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
                                                Constant(value='name'),
                                                Constant(value='keyname')],
                                            values=[
                                                Call(
                                                    func=Name(id='eval', ctx=Load()),
                                                    args=[
                                                        Constant(value='"X"')],
                                                    keywords=[]),
                                                Call(
                                                    func=Name(id='eval', ctx=Load()),
                                                    args=[
                                                        Constant(value='"Y"')],
                                                    keywords=[])])),
                                    keyword(
                                        arg='max_wait_secs',
                                        value=Constant(value=30))])),
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
            module='pprint',
            names=[
                alias(name='pprint')],
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
                        value=Constant(value='import matplotlib.pyplot as plt;import tensorflow as tf;from pprint import pprint;import numpy as np;import math, random')),
                    keyword(
                        arg='functionDef',
                        value=Constant(value='tf')),
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
                        value=Constant(value='import matplotlib.pyplot as plt;import tensorflow as tf;from pprint import pprint;import numpy as np;import math, random')),
                    keyword(
                        arg='functionDef',
                        value=Constant(value='tf.placeholder')),
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
                        value=Constant(value='import matplotlib.pyplot as plt;import tensorflow as tf;from pprint import pprint;import numpy as np;import math, random')),
                    keyword(
                        arg='functionDef',
                        value=Constant(value='tf.placeholder')),
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
                        value=Constant(value='import matplotlib.pyplot as plt;import tensorflow as tf;from pprint import pprint;import numpy as np;import math, random')),
                    keyword(
                        arg='functionDef',
                        value=Constant(value='tf.value')),
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
                        value=Constant(value='import matplotlib.pyplot as plt;import tensorflow as tf;from pprint import pprint;import numpy as np;import math, random')),
                    keyword(
                        arg='functionDef',
                        value=Constant(value='tf.Variable')),
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
                        value=Constant(value='import matplotlib.pyplot as plt;import tensorflow as tf;from pprint import pprint;import numpy as np;import math, random')),
                    keyword(
                        arg='functionDef',
                        value=Constant(value='tf.Variable')),
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
                        value=Constant(value='import matplotlib.pyplot as plt;import tensorflow as tf;from pprint import pprint;import numpy as np;import math, random')),
                    keyword(
                        arg='functionDef',
                        value=Constant(value='tf.keras.models.Sequential')),
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
                        value=Constant(value='import matplotlib.pyplot as plt;import tensorflow as tf;from pprint import pprint;import numpy as np;import math, random')),
                    keyword(
                        arg='functionDef',
                        value=Constant(value='model.compile')),
                    keyword(
                        arg='function_to_run',
                        value=Constant(value='obj.compile()')),
                    keyword(
                        arg='method_object',
                        value=Constant(value='model')),
                    keyword(
                        arg='function_args',
                        value=Constant(value=[])),
                    keyword(
                        arg='function_kwargs',
                        value=Constant(value={})),
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
                        value=Constant(value='import matplotlib.pyplot as plt;import tensorflow as tf;from pprint import pprint;import numpy as np;import math, random')),
                    keyword(
                        arg='functionDef',
                        value=Constant(value='b_o.test.compile')),
                    keyword(
                        arg='function_to_run',
                        value=Constant(value='obj.test.compile()')),
                    keyword(
                        arg='method_object',
                        value=Constant(value='b_o')),
                    keyword(
                        arg='function_args',
                        value=Constant(value=[])),
                    keyword(
                        arg='function_kwargs',
                        value=Constant(value={})),
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
def custom_method(func, imports, functionDef, function_to_run, method_object, function_args, function_kwargs, max_wait_secs):
    method_details = {'imports': imports, 'function': function_to_run, 'method_object': method_object, 'args': function_args, 'kwargs': function_kwargs, 'max_wait_secs': max_wait_secs}
    (sig, func_locals) = (inspect.signature(functionDef), locals())
    arglist = [func_locals[param.name] for param in sig.parameters.values() if param.kind == param.POSITIONAL_OR_KEYWORD]
    keywordDict = [func_locals[param.name] for param in sig.parameters.values() if param.kind == param.KEYWORD_ONLY]
    data = pickle.dumps(method_details)
    resp = requests.post(url, data=data, headers={'Content-Type': 'application/octet-stream'})
    custom_method(tf.placeholder(tf.float32, [None, 1], name='X'), imports='import tensorflow as tf;import math, random;from pprint import pprint;import matplotlib.pyplot as plt;import numpy as np', functionDef='tf.placeholder', function_to_run='tf.placeholder(*args, **kwargs)', method_object=None, function_args=[eval('tf.float32'), eval('[None, 1]')], function_kwargs={'name': eval('"X"'), 'keyname': eval('"Y"')}, max_wait_secs=30)
    return func
import tensorflow as tf
import numpy as np
import math, random
import matplotlib.pyplot as plt
from pprint import pprint
custom_method(
tf(), imports='import matplotlib.pyplot as plt;import tensorflow as tf;from pprint import pprint;import numpy as np;import math, random', functionDef='tf', function_to_run='tf()', method_object=None, function_args=[], function_kwargs={}, max_wait_secs=30)
testlist = ['a', 'b', 'c']
X = custom_method(
tf.placeholder(tf.float32, [None, 1], name='X'), imports='import matplotlib.pyplot as plt;import tensorflow as tf;from pprint import pprint;import numpy as np;import math, random', functionDef='tf.placeholder', function_to_run='tf.placeholder(*args, **kwargs)', method_object=None, function_args=[eval('tf.float32'), eval('[None, 1]')], function_kwargs={'name': eval('"X"')}, max_wait_secs=30)
Y = custom_method(
tf.placeholder(tf.float32, [None, 1], name='Y'), imports='import matplotlib.pyplot as plt;import tensorflow as tf;from pprint import pprint;import numpy as np;import math, random', functionDef='tf.placeholder', function_to_run='tf.placeholder(*args, **kwargs)', method_object=None, function_args=[eval('tf.float32'), eval('[None, 1]')], function_kwargs={'name': eval('"Y"')}, max_wait_secs=30)
z = custom_method(
tf.value('aname', bname, name='tname'), imports='import matplotlib.pyplot as plt;import tensorflow as tf;from pprint import pprint;import numpy as np;import math, random', functionDef='tf.value', function_to_run='tf.value(*args, **kwargs)', method_object=None, function_args=[eval('"aname"'), eval('bname')], function_kwargs={'name': eval('"tname"')}, max_wait_secs=30)
w_o = custom_method(
tf.Variable(tf.random_uniform([layer_1_neurons, 1], minval=-1, maxval=1, dtype=tf.float32), test1, test2, test3='test3'), imports='import matplotlib.pyplot as plt;import tensorflow as tf;from pprint import pprint;import numpy as np;import math, random', functionDef='tf.Variable', function_to_run='tf.Variable(*args, **kwargs)', method_object=None, function_args=[eval('tf.random_uniform([layer_1_neurons, 1], minval = -1, maxval = 1, dtype = tf.float32)'), eval('test1'), eval('test2')], function_kwargs={'test3': eval('"test3"')}, max_wait_secs=30)
b_o = custom_method(
tf.Variable(tf.zeros([1, 1], dtype=tf.float32)), imports='import matplotlib.pyplot as plt;import tensorflow as tf;from pprint import pprint;import numpy as np;import math, random', functionDef='tf.Variable', function_to_run='tf.Variable(*args)', method_object=None, function_args=[eval('tf.zeros([1, 1], dtype = tf.float32)')], function_kwargs={}, max_wait_secs=30)
cd = ab.test(a, b)
c_o = xf.test()
d_o = xy.test()
model = custom_method(
tf.keras.models.Sequential([tf.keras.layers.Flatten(input_shape=(28, 28)), tf.keras.layers.Dense(128, activation='relu'), tf.keras.layers.Dropout(0.2), tf.keras.layers.Dense(10)]), imports='import matplotlib.pyplot as plt;import tensorflow as tf;from pprint import pprint;import numpy as np;import math, random', functionDef='tf.keras.models.Sequential', function_to_run='tf.keras.models.Sequential(*args)', method_object=None, function_args=[eval("[\n    tf.keras.layers.Flatten(input_shape=(28, 28)),\n    tf.keras.layers.Dense(128, activation='relu'),\n    tf.keras.layers.Dropout(0.2),\n    tf.keras.layers.Dense(10)\n    ]")], function_kwargs={}, max_wait_secs=30)
custom_method(
model.compile(), imports='import matplotlib.pyplot as plt;import tensorflow as tf;from pprint import pprint;import numpy as np;import math, random', functionDef='model.compile', function_to_run='obj.compile()', method_object='model', function_args=[], function_kwargs={}, max_wait_secs=30)
f_o = custom_method(
b_o.test.compile(), imports='import matplotlib.pyplot as plt;import tensorflow as tf;from pprint import pprint;import numpy as np;import math, random', functionDef='b_o.test.compile', function_to_run='obj.test.compile()', method_object='b_o', function_args=[], function_kwargs={}, max_wait_secs=30)
z_o = c_o.compile()
++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
