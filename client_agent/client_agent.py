import ast
from pprint import pprint
from collections import deque
import json
import inspect
import copy

requiredLibraries = ["tensorflow"]
requiredAlias = []
importScriptList = ''
sourceCode = ''

def main():
    with open("code_snippet2.py", "r") as source:
        global sourceCode
        sourceCode = source.read()
        tree = ast.parse(sourceCode)
        # ast.get_source_segment(source.read(), node)
        # code = """if 1 == 1 and 2 == 2 and 3 == 3:
        #     test = 1
        # """
        # node1 = ast.parse(code)
        # print(ast.get_source_segment(code, node1.body[0]))
    analyzer = Analyzer()
    analyzer.visit(tree)
    # pprint(sourceCode)
    # print("Test analyzer :",analyzer.lineno)
    global requiredAlias
    global importScriptList
    requiredAlias = analyzer.stats['required']
    importScriptList = ';'.join(list(set(analyzer.stats['importScript'])))
    transf = TransformCall()
    transf.visit(tree)
    # tree.body.insert(0, new_node)
    print('+'*100)
    print(ast.dump(tree, indent=4))
    print('_'*100)
    # for fcall in tree.body:
    #     # if(isinstance(fcall[7], ast.Call)):
    #     print("FCALL:",fcall)
    print(ast.unparse(tree))


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


class TransformCall(ast.NodeTransformer):
    def __init__(self):
        global requiredAlias

    def visit_Call(self, node):
        callvisitor = FuncCallVisitor()
        callvisitor.visit(node.func)

        if(any(lib in callvisitor.name for lib in requiredAlias)):
            print("SSR1:",ast.get_source_segment(sourceCode, node))
            dummyNode=copy.deepcopy(node)
            print("type is:",type(dummyNode.args))
            dummyNode.args.clear()
            dummyNode.keywords.clear()
            if(node.args):
                dummyNode.args.append(ast.Name(id='*args', ctx=ast.Load()))
            if(node.keywords):
                dummyNode.keywords.append(ast.Name(id='**kwargs', ctx=ast.Load()))
            print("SSR_dummynode:",ast.dump(dummyNode))
            print("SSR_node:",ast.dump(node))
            new_node = ast.Call(func=ast.Name(id='custom_method', ctx=ast.Load()),
                                args=[ast.Expr(node)],
                                keywords=[
                                    ast.keyword(
                                        arg='imports',
                                        value=ast.Constant(importScriptList)),
                                    ast.keyword(
                                        arg='function_to_run',
                                        value=ast.Constant(ast.unparse(dummyNode))),
                                    ast.keyword(
                                        arg='method_object',
                                        value=ast.Constant('')),
                                    ast.keyword(
                                        arg='function_args',
                                        value=ast.Constant('')),
                                    ast.keyword(
                                        arg='function_kwargs',
                                        value=ast.Constant('')),
                                    ast.keyword(
                                        arg='max_wait_secs',
                                        value=ast.Constant('')),
                                        ],
                                        starargs=None, kwargs=None
                                )
            ast.copy_location(new_node, node)
            ast.fix_missing_locations(new_node)
            return new_node
        

def get_func_calls(tree):
    global requiredAlias
    func_calls = []
    for node in ast.walk(tree):
        # func_arguments = []
        # func_keywords = []
        if isinstance(node, ast.Call):
            callvisitor = FuncCallVisitor()
            callTransformer = TransformCall()
            callvisitor.visit(node.func)
            if(any(lib in callvisitor.name for lib in requiredAlias)):
                tree = callTransformer.visit(node)
                print("---",ast.unparse(node))
                print(ast.unparse(tree))

class Analyzer(ast.NodeVisitor):
    def __init__(self):
        self.stats = {"import": [], "from": [], "required": [],"importScript": []}
        global sourceCode

    def visit_Import(self, node):
        for alias in node.names:
            self.stats["importScript"].append(ast.get_source_segment(sourceCode, node))
            # pprint(vars(node))
            self.stats["import"].append(alias.name)
            if(any(lib in alias.name for lib in requiredLibraries)):
                self.stats["required"].append(alias.asname)
        self.generic_visit(node)

    def visit_ImportFrom(self, node):
        for alias in node.names:
            self.stats["importScript"].append(ast.get_source_segment(sourceCode, node))
            self.stats["from"].append(alias.name)
        self.generic_visit(node)

    def report(self):
        pprint(self.stats)


if __name__ == "__main__":
    main()