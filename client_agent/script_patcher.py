import ast
from ast import literal_eval
from pprint import pprint
from collections import deque
import json
import inspect
import copy
import json
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('input_files', type=argparse.FileType('r'))
args = parser.parse_args()

requiredLibraries = ["tensorflow"]
requiredAlias = []
requiredObjects = []
requiredClassDefs = {}
requiredObjClassMapping = {}
importScriptList = ''
sourceCode = ''

def main():
    #Step1: Create an AST from the client python code
    global sourceCode
    global args
    sourceCode = args.input_files.read()
    tree = ast.parse(sourceCode)

    #Step2: Extract list of libraries and aliases for energy calculation
    analyzer = Analyzer()
    analyzer.visit(tree)
    global requiredAlias
    global importScriptList
    requiredAlias = analyzer.stats['required']
    # print("required Alias:",requiredAlias)
    importScriptList = ';'.join(list(set(analyzer.stats['importScript'])))

    # Step3: Get list of Classdefs having bases from the required libraries
    global requiredClassDefs
    classDefAnalyzer = ClassDefAnalyzer()
    classDefAnalyzer.visit(tree)
    requiredClassDefs = classDefAnalyzer.classDef

    # Step4: Get list of objects created from the required libraries
    global requiredObjects
    global requiredObjClassMapping
    objAnalyzer = ObjectAnalyzer()
    objAnalyzer.visit(tree)
    requiredObjects = objAnalyzer.stats['objects']
    requiredObjClassMapping = objAnalyzer.userDefObjects

    #Step5: Tranform the client script by adding custom method calls
    transf = TransformCall()
    transf.visit(tree)
    with open("custom_method.py", "r") as source:
        cm = source.read()
        cm_node = ast.parse(cm)
        tree.body.insert(0, cm_node)

    # print('+'*100)
    # print(ast.dump(tree, indent=4))
    # print('_'*100)

    #Step6: Unparse and convert AST to final code
    print(ast.unparse(tree))
    # print('+'*100)


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
        global requiredObjects
        global requiredObjClassMapping

    def visit_Call(self, node):
        callvisitor = FuncCallVisitor()
        callvisitor.visit(node.func)

        if(any(lib in callvisitor.name for lib in requiredAlias)):
            dummyNode=copy.deepcopy(node)
            dummyNode.args.clear()
            dummyNode.keywords.clear()
            argList = [ast.get_source_segment(sourceCode, a) for a in node.args]
            keywordsDict = {a.arg:ast.get_source_segment(sourceCode, a.value) for a in node.keywords}
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
                                        value=ast.Constant(None)),
                                    ast.keyword(
                                        arg='function_args',
                                        value=ast.List(
                                            elts=[
                                                ast.Call(
                                                    func=ast.Name(id='eval', ctx=ast.Load()),
                                                    args=[
                                                        ast.Constant(value=argItem)],
                                                    keywords=[]) for argItem in argList],
                                            ctx=ast.Load())),
                                    ast.keyword(
                                        arg='function_kwargs',
                                        value=ast.Dict(
                                            keys=[
                                                ast.Constant(value=KWItem) for KWItem in keywordsDict],
                                            values=[
                                                ast.Call(
                                                    func=ast.Name(id='eval', ctx=ast.Load()),
                                                    args=[
                                                        ast.Constant(value=keywordsDict[KWItem])],
                                                    keywords=[]) for KWItem in keywordsDict])),
                                    ast.keyword(
                                        arg='max_wait_secs',
                                        value=ast.Constant(0)),
                                        ],
                                        starargs=None, kwargs=None
                                )
            ast.copy_location(new_node, node)
            ast.fix_missing_locations(new_node)
            return new_node

        if(any(obj in callvisitor.name.split('.')[0] for obj in requiredObjects)):
            dummyNode=copy.deepcopy(node)
            dummyNode.args.clear()
            dummyNode.keywords.clear()
            argList = [ast.get_source_segment(sourceCode, a) for a in node.args]
            keywordsDict = {a.arg:ast.get_source_segment(sourceCode, a.value) for a in node.keywords}
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
                                        value=ast.Constant(ast.unparse(dummyNode).replace(callvisitor.name.split('.')[0], 'obj', 1))),
                                    ast.keyword(
                                        arg='method_object',
                                        value=ast.Constant(callvisitor.name.split('.')[0])),
                                    ast.keyword(
                                        arg='function_args',
                                        value=ast.List(
                                            elts=[
                                                ast.Call(
                                                    func=ast.Name(id='eval', ctx=ast.Load()),
                                                    args=[
                                                        ast.Constant(value=argItem)],
                                                    keywords=[]) for argItem in argList],
                                            ctx=ast.Load())),
                                    ast.keyword(
                                        arg='function_kwargs',
                                        value=ast.Dict(
                                            keys=[
                                                ast.Constant(value=KWItem) for KWItem in keywordsDict],
                                            values=[
                                                ast.Call(
                                                    func=ast.Name(id='eval', ctx=ast.Load()),
                                                    args=[
                                                        ast.Constant(value=keywordsDict[KWItem])],
                                                    keywords=[]) for KWItem in keywordsDict])),
                                    ast.keyword(
                                        arg='max_wait_secs',
                                        value=ast.Constant(0)),
                                    ast.keyword(
                                        arg='custom_class',
                                        value=ast.Constant(requiredClassDefs.get(requiredObjClassMapping.get(callvisitor.name.split('.')[0]))))
                                        ],
                                        starargs=None, kwargs=None
                                )
            ast.copy_location(new_node, node)
            ast.fix_missing_locations(new_node)
            return new_node
        
        return node

class Analyzer(ast.NodeVisitor):
    def __init__(self):
        self.stats = {"import": [], "from": [], "required": [],"importScript": []}
        global sourceCode

    def visit_Import(self, node):
        for alias in node.names:
            self.stats["importScript"].append(ast.get_source_segment(sourceCode, node))
            self.stats["import"].append(alias.name)
            lib_path = alias.name.split('.')
            if(any(lib in lib_path for lib in requiredLibraries)):
                if(alias.asname):
                    self.stats["required"].append(alias.asname)
                else:
                    self.stats["required"].append(lib_path[-1])
        self.generic_visit(node)

    def visit_ImportFrom(self, node):
        lib_path = node.module.split('.')
        for alias in node.names:
            self.stats["importScript"].append(ast.get_source_segment(sourceCode, node))
            self.stats["from"].append(alias.name)
            if(any(lib in lib_path for lib in requiredLibraries)):
                if(alias.asname):
                    self.stats["required"].append(alias.asname)
                elif(alias.name == '*'):
                    pass
                    # TODO: Need to find a way to get all the methods from the module without installing the libraries if possible
                    # print("Star import not supported and is not required for this script")
                else:
                    self.stats["required"].append(alias.name)
        self.generic_visit(node)

    def report(self):
        pprint(self.stats)

class ClassDefAnalyzer(ast.NodeVisitor):
    def __init__(self):
        self.classDef = {}
        global sourceCode

    def visit_ClassDef(self, node):
        if((node.bases) and any(ast.get_source_segment(sourceCode,lib).split('.')[0] in requiredAlias for lib in node.bases)):
            self.classDef[node.name] = ast.get_source_segment(sourceCode, node)

class ObjectAnalyzer(ast.NodeVisitor):
    def __init__(self):
        self.stats = {"objects": []}
        self.userDefObjects = {}
        global sourceCode
        global requiredClassDefs
        
    def visit_Assign(self, node):
        if isinstance(node.value, ast.Call):
            callvisitor2 = FuncCallVisitor()
            callvisitor2.visit(node.value.func)
            if((any(lib in callvisitor2.name for lib in requiredAlias)) and (isinstance(node.targets[0], ast.Name))):
                self.stats["objects"].append(node.targets[0].id)
            
            if((any(lib in callvisitor2.name.split('.')[0] for lib in list(requiredClassDefs.keys()))) and (isinstance(node.targets[0], ast.Name))):
                self.stats["objects"].append(node.targets[0].id)
                self.userDefObjects[node.targets[0].id] = callvisitor2.name.split('.')[0]
        self.generic_visit(node)


if __name__ == "__main__":
    main()