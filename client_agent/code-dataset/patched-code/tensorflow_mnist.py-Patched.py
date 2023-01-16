requiredClass dict_keys([])
requiredObjects ['model', 'loss_fn', 'probability_model']
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
        Assign(
            targets=[
                Name(id='mnist', ctx=Store())],
            value=Attribute(
                value=Attribute(
                    value=Attribute(
                        value=Name(id='tf', ctx=Load()),
                        attr='keras',
                        ctx=Load()),
                    attr='datasets',
                    ctx=Load()),
                attr='mnist',
                ctx=Load())),
        Assign(
            targets=[
                Tuple(
                    elts=[
                        Tuple(
                            elts=[
                                Name(id='x_train', ctx=Store()),
                                Name(id='y_train', ctx=Store())],
                            ctx=Store()),
                        Tuple(
                            elts=[
                                Name(id='x_test', ctx=Store()),
                                Name(id='y_test', ctx=Store())],
                            ctx=Store())],
                    ctx=Store())],
            value=Call(
                func=Attribute(
                    value=Name(id='mnist', ctx=Load()),
                    attr='load_data',
                    ctx=Load()),
                args=[],
                keywords=[])),
        Assign(
            targets=[
                Tuple(
                    elts=[
                        Name(id='x_train', ctx=Store()),
                        Name(id='x_test', ctx=Store())],
                    ctx=Store())],
            value=Tuple(
                elts=[
                    BinOp(
                        left=Name(id='x_train', ctx=Load()),
                        op=Div(),
                        right=Constant(value=255.0)),
                    BinOp(
                        left=Name(id='x_test', ctx=Load()),
                        op=Div(),
                        right=Constant(value=255.0))],
                ctx=Load())),
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
                        value=Constant(value='import tensorflow as tf')),
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
                                        Constant(value="[\n  tf.keras.layers.Flatten(input_shape=(28, 28)),\n  tf.keras.layers.Dense(128, activation='relu'),\n  tf.keras.layers.Dropout(0.2),\n  tf.keras.layers.Dense(10)\n]")],
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
                Name(id='predictions', ctx=Store())],
            value=Call(
                func=Attribute(
                    value=Call(
                        func=Name(id='model', ctx=Load()),
                        args=[
                            Subscript(
                                value=Name(id='x_train', ctx=Load()),
                                slice=Slice(
                                    upper=Constant(value=1)),
                                ctx=Load())],
                        keywords=[]),
                    attr='numpy',
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
                                value=Call(
                                    func=Attribute(
                                        value=Attribute(
                                            value=Name(id='tf', ctx=Load()),
                                            attr='nn',
                                            ctx=Load()),
                                        attr='softmax',
                                        ctx=Load()),
                                    args=[
                                        Name(id='predictions', ctx=Load())],
                                    keywords=[]),
                                attr='numpy',
                                ctx=Load()),
                            args=[],
                            keywords=[]))],
                keywords=[
                    keyword(
                        arg='imports',
                        value=Constant(value='import tensorflow as tf')),
                    keyword(
                        arg='function_to_run',
                        value=Constant(value='tf.nn.softmax(predictions).numpy()')),
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
                Name(id='loss_fn', ctx=Store())],
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
                                    attr='losses',
                                    ctx=Load()),
                                attr='SparseCategoricalCrossentropy',
                                ctx=Load()),
                            args=[],
                            keywords=[
                                keyword(
                                    arg='from_logits',
                                    value=Constant(value=True))]))],
                keywords=[
                    keyword(
                        arg='imports',
                        value=Constant(value='import tensorflow as tf')),
                    keyword(
                        arg='function_to_run',
                        value=Constant(value='tf.keras.losses.SparseCategoricalCrossentropy(**kwargs)')),
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
                                Constant(value='from_logits')],
                            values=[
                                Call(
                                    func=Name(id='eval', ctx=Load()),
                                    args=[
                                        Constant(value='True')],
                                    keywords=[])])),
                    keyword(
                        arg='max_wait_secs',
                        value=Constant(value=30))])),
        Expr(
            value=Call(
                func=Attribute(
                    value=Call(
                        func=Name(id='loss_fn', ctx=Load()),
                        args=[
                            Subscript(
                                value=Name(id='y_train', ctx=Load()),
                                slice=Slice(
                                    upper=Constant(value=1)),
                                ctx=Load()),
                            Name(id='predictions', ctx=Load())],
                        keywords=[]),
                    attr='numpy',
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
                                value=Name(id='model', ctx=Load()),
                                attr='compile',
                                ctx=Load()),
                            args=[],
                            keywords=[
                                keyword(
                                    arg='optimizer',
                                    value=Constant(value='adam')),
                                keyword(
                                    arg='loss',
                                    value=Name(id='loss_fn', ctx=Load())),
                                keyword(
                                    arg='metrics',
                                    value=List(
                                        elts=[
                                            Constant(value='accuracy')],
                                        ctx=Load()))]))],
                keywords=[
                    keyword(
                        arg='imports',
                        value=Constant(value='import tensorflow as tf')),
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
                                Constant(value='optimizer'),
                                Constant(value='loss'),
                                Constant(value='metrics')],
                            values=[
                                Call(
                                    func=Name(id='eval', ctx=Load()),
                                    args=[
                                        Constant(value="'adam'")],
                                    keywords=[]),
                                Call(
                                    func=Name(id='eval', ctx=Load()),
                                    args=[
                                        Constant(value='loss_fn')],
                                    keywords=[]),
                                Call(
                                    func=Name(id='eval', ctx=Load()),
                                    args=[
                                        Constant(value="['accuracy']")],
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
                                Name(id='x_train', ctx=Load()),
                                Name(id='y_train', ctx=Load())],
                            keywords=[
                                keyword(
                                    arg='epochs',
                                    value=Constant(value=5))]))],
                keywords=[
                    keyword(
                        arg='imports',
                        value=Constant(value='import tensorflow as tf')),
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
                                        Constant(value='x_train')],
                                    keywords=[]),
                                Call(
                                    func=Name(id='eval', ctx=Load()),
                                    args=[
                                        Constant(value='y_train')],
                                    keywords=[])],
                            ctx=Load())),
                    keyword(
                        arg='function_kwargs',
                        value=Dict(
                            keys=[
                                Constant(value='epochs')],
                            values=[
                                Call(
                                    func=Name(id='eval', ctx=Load()),
                                    args=[
                                        Constant(value='5')],
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
                                attr='evaluate',
                                ctx=Load()),
                            args=[
                                Name(id='x_test', ctx=Load()),
                                Name(id='y_test', ctx=Load())],
                            keywords=[
                                keyword(
                                    arg='verbose',
                                    value=Constant(value=2))]))],
                keywords=[
                    keyword(
                        arg='imports',
                        value=Constant(value='import tensorflow as tf')),
                    keyword(
                        arg='function_to_run',
                        value=Constant(value='obj.evaluate(*args, **kwargs)')),
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
                                        Constant(value='x_test')],
                                    keywords=[]),
                                Call(
                                    func=Name(id='eval', ctx=Load()),
                                    args=[
                                        Constant(value='y_test')],
                                    keywords=[])],
                            ctx=Load())),
                    keyword(
                        arg='function_kwargs',
                        value=Dict(
                            keys=[
                                Constant(value='verbose')],
                            values=[
                                Call(
                                    func=Name(id='eval', ctx=Load()),
                                    args=[
                                        Constant(value='2')],
                                    keywords=[])])),
                    keyword(
                        arg='max_wait_secs',
                        value=Constant(value=30)),
                    keyword(
                        arg='custom_class',
                        value=Constant(value=None))])),
        Assign(
            targets=[
                Name(id='probability_model', ctx=Store())],
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
                                        Name(id='model', ctx=Load()),
                                        Call(
                                            func=Attribute(
                                                value=Attribute(
                                                    value=Attribute(
                                                        value=Name(id='tf', ctx=Load()),
                                                        attr='keras',
                                                        ctx=Load()),
                                                    attr='layers',
                                                    ctx=Load()),
                                                attr='Softmax',
                                                ctx=Load()),
                                            args=[],
                                            keywords=[])],
                                    ctx=Load())],
                            keywords=[]))],
                keywords=[
                    keyword(
                        arg='imports',
                        value=Constant(value='import tensorflow as tf')),
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
                                        Constant(value='[\n  model,\n  tf.keras.layers.Softmax()\n]')],
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
                            func=Name(id='probability_model', ctx=Load()),
                            args=[
                                Subscript(
                                    value=Name(id='x_test', ctx=Load()),
                                    slice=Slice(
                                        upper=Constant(value=5)),
                                    ctx=Load())],
                            keywords=[]))],
                keywords=[
                    keyword(
                        arg='imports',
                        value=Constant(value='import tensorflow as tf')),
                    keyword(
                        arg='function_to_run',
                        value=Constant(value='obj(*args)')),
                    keyword(
                        arg='method_object',
                        value=Constant(value='probability_model')),
                    keyword(
                        arg='function_args',
                        value=List(
                            elts=[
                                Call(
                                    func=Name(id='eval', ctx=Load()),
                                    args=[
                                        Constant(value='x_test[:5]')],
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
                        value=Constant(value=None))]))],
    type_ignores=[])
____________________________________________________________________________________________________
import pickle
import requests
from server.send_request import send_request

def custom_method(func, imports: str, function_to_run: str, method_object=None, function_args: list=None, function_kwargs: dict=None, max_wait_secs=0, custom_class=None):
    result = send_request(imports, function_to_run, function_args, function_kwargs, max_wait_secs, method_object)
    return func
import tensorflow as tf
mnist = tf.keras.datasets.mnist
((x_train, y_train), (x_test, y_test)) = mnist.load_data()
(x_train, x_test) = (x_train / 255.0, x_test / 255.0)
model = custom_method(
tf.keras.models.Sequential([tf.keras.layers.Flatten(input_shape=(28, 28)), tf.keras.layers.Dense(128, activation='relu'), tf.keras.layers.Dropout(0.2), tf.keras.layers.Dense(10)]), imports='import tensorflow as tf', function_to_run='tf.keras.models.Sequential(*args)', method_object=None, function_args=[eval("[\n  tf.keras.layers.Flatten(input_shape=(28, 28)),\n  tf.keras.layers.Dense(128, activation='relu'),\n  tf.keras.layers.Dropout(0.2),\n  tf.keras.layers.Dense(10)\n]")], function_kwargs={}, max_wait_secs=30)
predictions = model(x_train[:1]).numpy()
custom_method(
tf.nn.softmax(predictions).numpy(), imports='import tensorflow as tf', function_to_run='tf.nn.softmax(predictions).numpy()', method_object=None, function_args=[], function_kwargs={}, max_wait_secs=30)
loss_fn = custom_method(
tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), imports='import tensorflow as tf', function_to_run='tf.keras.losses.SparseCategoricalCrossentropy(**kwargs)', method_object=None, function_args=[], function_kwargs={'from_logits': eval('True')}, max_wait_secs=30)
loss_fn(y_train[:1], predictions).numpy()
custom_method(
model.compile(optimizer='adam', loss=loss_fn, metrics=['accuracy']), imports='import tensorflow as tf', function_to_run='obj.compile(**kwargs)', method_object='model', function_args=[], function_kwargs={'optimizer': eval("'adam'"), 'loss': eval('loss_fn'), 'metrics': eval("['accuracy']")}, max_wait_secs=30, custom_class=None)
custom_method(
model.fit(x_train, y_train, epochs=5), imports='import tensorflow as tf', function_to_run='obj.fit(*args, **kwargs)', method_object='model', function_args=[eval('x_train'), eval('y_train')], function_kwargs={'epochs': eval('5')}, max_wait_secs=30, custom_class=None)
custom_method(
model.evaluate(x_test, y_test, verbose=2), imports='import tensorflow as tf', function_to_run='obj.evaluate(*args, **kwargs)', method_object='model', function_args=[eval('x_test'), eval('y_test')], function_kwargs={'verbose': eval('2')}, max_wait_secs=30, custom_class=None)
probability_model = custom_method(
tf.keras.Sequential([model, tf.keras.layers.Softmax()]), imports='import tensorflow as tf', function_to_run='tf.keras.Sequential(*args)', method_object=None, function_args=[eval('[\n  model,\n  tf.keras.layers.Softmax()\n]')], function_kwargs={}, max_wait_secs=30)
custom_method(
probability_model(x_test[:5]), imports='import tensorflow as tf', function_to_run='obj(*args)', method_object='probability_model', function_args=[eval('x_test[:5]')], function_kwargs={}, max_wait_secs=30, custom_class=None)
