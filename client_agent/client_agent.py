import ast
from pprint import pprint
from collections import deque
import json

requiredLibraries = ["tensorflow"]
requiredAlias = []

def main():
    with open("code_snippet.py", "r") as source:
        tree = ast.parse(source.read())
    analyzer = Analyzer()
    analyzer.visit(tree)
    # analyzer.report()
    global requiredAlias
    requiredAlias = analyzer.stats['required']
    print(get_func_calls(tree))
    # print(ast.dump(tree, indent=4))

class FuncCallVisitor(ast.NodeVisitor):
    def __init__(self):
        self._name = deque()

    @property
    def name(self):
        return '.'.join(self._name)

    @name.deleter
    def name(self):
        self._name.clear()

    def visit_Name(self, node):
        # if((node.id in requiredAlias) or (node.id in self._name)):
        self._name.appendleft(node.id)

    def visit_Attribute(self, node):
        try:
            # if((node.attr in requiredAlias) or (node.value.id in requiredAlias)):
            self._name.appendleft(node.attr)
            self._name.appendleft(node.value.id)
        except AttributeError:
            self.generic_visit(node)

class FuncArgVisitor(ast.NodeVisitor):
    def __init__(self):
        self._name = deque()

    @property
    def name(self):
        return '.'.join(self._name)

    @name.deleter
    def name(self):
        self._name.clear()

    def visit_Name(self, node):
        self._name.appendleft(node.id)

    def visit_Attribute(self, node):
        try:
            # if isinstance(node, ast.Attribute):
            #     arVisitor = FuncArgVisitor()
            #     arVisitor.visit(node)
                self._name.appendleft(node.attr)
                self._name.appendleft(node.value.id)
        except AttributeError:
            self.generic_visit(node)


def get_func_calls(tree):
    global requiredAlias
    func_calls = []
    for node in ast.walk(tree):
        func_arguments = []
        func_keywords = []
        if isinstance(node, ast.Call):
            callvisitor = FuncCallVisitor()
            callvisitor.visit(node.func)
            # func_calls.append(callvisitor.name)
            for arg in node.args:
                argVisitor = FuncArgVisitor()
                argVisitor.visit(arg)
                func_arguments.append(argVisitor.name)
            # print("Func args :",list(filter(None, func_arguments)))
            
            for kw in node.keywords:
                kwVisitor = FuncArgVisitor()
                kwVisitor.visit(kw)
                func_keywords.append(kwVisitor.name)
            # print("Func keywords :",list(filter(None, func_keywords)))
            if(any(lib in callvisitor.name for lib in requiredAlias)):
                data = {}
                data['funcCall']=callvisitor.name
                data['funcArgs']=func_arguments
                data['funcKeywords']=func_keywords
                func_calls.append(json.dumps(data))
    # pprint("Required Function Calls and their arguments :")
    # pprint(func_calls)
    return func_calls

class Analyzer(ast.NodeVisitor):
    def __init__(self):
        self.stats = {"import": [], "from": [], "required": []}

    def visit_Import(self, node):
        for alias in node.names:
            self.stats["import"].append(alias.name)
            if(any(lib in alias.name for lib in requiredLibraries)):
                self.stats["required"].append(alias.asname)
        self.generic_visit(node)

    def visit_ImportFrom(self, node):
        for alias in node.names:
            self.stats["from"].append(alias.name)
        self.generic_visit(node)

    def report(self):
        pprint(self.stats)


if __name__ == "__main__":
    main()