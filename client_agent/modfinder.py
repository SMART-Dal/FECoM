import ast
import inspect

def get_class_def(obj):
    if obj.__class__.__module__ == "__main__":
        source = inspect.getsource(obj.__class__)
        module_ast = ast.parse(source)
        for node in module_ast.body:
            if isinstance(node, ast.ClassDef) and node.name == obj.__class__.__name__:
                return node
    return None

class A:
    pass

a = A()
class_def = get_class_def(a)
print(class_def)
