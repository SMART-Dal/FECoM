import ast
from ast import literal_eval
from pprint import pprint
from collections import deque
import json
import inspect
import copy
import json

requiredLibraries = ["tensorflow"]
requiredAlias = []
requiredObjects = []
importScriptList = ''
sourceCode = ''

def main():

    #Step1: Create an AST from the client python code
    with open("code_snippet2.py", "r") as source:
        global sourceCode
        sourceCode = source.read()
        tree = ast.parse(sourceCode)

    #Step2: Extract list of libraries and aliases for energy calculation
    analyzer = Analyzer()
    analyzer.visit(tree)
    global requiredAlias
    global importScriptList
    requiredAlias = analyzer.stats['required']
    importScriptList = ';'.join(list(set(analyzer.stats['importScript'])))

    #Step3: Get list of objects created from the required libraries
    global requiredObjects
    objAnalyzer = ObjectAnalyzer()
    objAnalyzer.visit(tree)
    requiredObjects = objAnalyzer.stats['objects']
    print("requiredObjects",requiredObjects)

    #Step4: Tranform the client script by adding custom method calls
    transf = TransformCall()
    transf.visit(tree)
    with open("custom_method.py", "r") as source:
        cm = source.read()
        cm_node = ast.parse(cm)
        tree.body.insert(0, cm_node)

    print('+'*100)
    print(ast.dump(tree, indent=4))
    print('_'*100)

    #Step5: Unparse and convert AST to final code
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
        global sourceCode

    def visit_Call(self, node):
        callvisitor = FuncCallVisitor()
        callvisitor.visit(node.func)
        print("blah1",node.func," Value:",callvisitor.name)

        if(any(lib in callvisitor.name for lib in requiredAlias)):
            dummyNode=copy.deepcopy(node)
            dummyNode.args.clear()
            dummyNode.keywords.clear()
            # print("SSR args:",ast.get_source_segment(sourceCode, node))
            print(ast.get_source_segment(sourceCode, node))
            argList = [ast.get_source_segment(sourceCode, a) for a in node.args]
            # argList = ast.literal_eval(ast.get_source_segment(sourceCode, node.args))
            # print("SSR arguments:",argList)
            # print("SSR args:",node.args)
            keywordsDict = {a.arg:ast.get_source_segment(sourceCode, a.value) for a in node.keywords}
            # print("SSR keywords:",keywordsDict)
            if(node.args):
                dummyNode.args.append(ast.Name(id='*args', ctx=ast.Load()))
            if(node.keywords):
                dummyNode.keywords.append(ast.Name(id='**kwargs', ctx=ast.Load()))
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
                                        value=ast.Constant(argList)),
                                    ast.keyword(
                                        arg='function_kwargs',
                                        value=ast.Constant(keywordsDict)),
                                    ast.keyword(
                                        arg='max_wait_secs',
                                        value=ast.Constant(30)),
                                        ],
                                        starargs=None, kwargs=None
                                )
            ast.copy_location(new_node, node)
            ast.fix_missing_locations(new_node)
            return new_node
        
        return node

    # def visit_Expr(self, node):
    #     # --------------------
    #     return node
    #     # --------------------

    #     callvisitor = FuncCallVisitor()
    #     callvisitor.visit(node.func)

    #     if(any(lib in callvisitor.name for lib in requiredAlias)):
    #         dummyNode=copy.deepcopy(node)
    #         dummyNode.args.clear()
    #         dummyNode.keywords.clear()
    #         # print("SSR args:",ast.get_source_segment(sourceCode, node))
    #         print(ast.get_source_segment(sourceCode, node))
    #         argList = [ast.get_source_segment(sourceCode, a) for a in node.args]
    #         # argList = ast.literal_eval(ast.get_source_segment(sourceCode, node.args))
    #         # print("SSR arguments:",argList)
    #         # print("SSR args:",node.args)
    #         keywordsDict = {a.arg:ast.get_source_segment(sourceCode, a.value) for a in node.keywords}
    #         # print("SSR keywords:",keywordsDict)
    #         if(node.args):
    #             dummyNode.args.append(ast.Name(id='*args', ctx=ast.Load()))
    #         if(node.keywords):
    #             dummyNode.keywords.append(ast.Name(id='**kwargs', ctx=ast.Load()))
    #         new_node = ast.Call(func=ast.Name(id='custom_method', ctx=ast.Load()),
    #                             args=[ast.Expr(node)],
    #                             keywords=[
    #                                 ast.keyword(
    #                                     arg='imports',
    #                                     value=ast.Constant(importScriptList)),
    #                                 ast.keyword(
    #                                     arg='function_to_run',
    #                                     value=ast.Constant(ast.unparse(dummyNode))),
    #                                 ast.keyword(
    #                                     arg='method_object',
    #                                     value=ast.Constant('')),
    #                                 ast.keyword(
    #                                     arg='function_args',
    #                                     value=ast.Constant(argList)),
    #                                 ast.keyword(
    #                                     arg='function_kwargs',
    #                                     value=ast.Constant(keywordsDict)),
    #                                 ast.keyword(
    #                                     arg='max_wait_secs',
    #                                     value=ast.Constant(30)),
    #                                     ],
    #                                     starargs=None, kwargs=None
    #                             )
    #         ast.copy_location(new_node, node)
    #         ast.fix_missing_locations(new_node)
    #         return new_node  

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
            print("blah2",node.func," Value:",callvisitor.name)
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

class ObjectAnalyzer(ast.NodeVisitor):
    def __init__(self):
        self.stats = {"objects": []}
        global sourceCode
        
    def visit_Assign(self, node):
        if isinstance(node.value, ast.Call):
            print("Inside OA:",node.targets[0].id)
            pprint(vars(node))
            callvisitor2 = FuncCallVisitor()
            callvisitor2.visit(node.value.func)
            print("blah3",node.value," Value:",callvisitor2.name)
            pprint(vars(node.value.func))
            if((any(lib in callvisitor2.name for lib in requiredAlias)) and (isinstance(node.targets[0], ast.Name))):
                print("Lib is :",requiredAlias," got name :",callvisitor2.name)
                self.stats["objects"].append(node.targets[0].id)
        self.generic_visit(node)


if __name__ == "__main__":
    main()