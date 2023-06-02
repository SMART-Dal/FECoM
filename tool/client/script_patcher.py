import ast
from pprint import pprint
import copy
import json
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('input_files', type=argparse.FileType('r'))
args = parser.parse_args()

requiredLibraries = ["tensorflow"]
requiredAlias = []
requiredObjects = []
requiredObjectsSignature = {}
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
    # print('+'*100)
    # print(ast.dump(tree, indent=4))
    # print('_'*100)

    #Step2: Extract list of libraries and aliases for energy calculation
    analyzer = Analyzer()
    analyzer.visit(tree)
    global requiredAlias
    global importScriptList
    requiredAlias = analyzer.stats['required']
    importScriptList = list(set(analyzer.stats['importScript']))

    # Get list of libraries and aliases with __future__ imports as they need to be moved to the beginning
    future_imports = [imp for imp in importScriptList if imp.startswith("from __future__")]

    # Get list of libraries and aliases without __future__ imports
    importScriptList = [imp for imp in importScriptList if not imp.startswith("from __future__")]

    # Add __future__ imports to the beginning of the list
    importScriptList = future_imports + importScriptList

    importScriptList = ';'.join(importScriptList)

    # Step3: Get list of Classdefs having bases from the required libraries
    global requiredClassDefs
    classDefAnalyzer = ClassDefAnalyzer()
    classDefAnalyzer.visit(tree)
    requiredClassDefs = classDefAnalyzer.classDef

    # Step4: Get list of objects created from the required libraries
    global requiredObjects
    global requiredObjectsSignature
    global requiredObjClassMapping
    objAnalyzer = ObjectAnalyzer()
    objAnalyzer.visit(tree)
    requiredObjects = objAnalyzer.stats['objects']
    # print('requiredObjects', requiredObjects)

    # create nodes to add before and after the method call
    before_execution_call="start_times_INSERTED_INTO_SCRIPT = before_execution_INSERTED_INTO_SCRIPT()"
    global before_execution_call_node
    before_execution_call_node = ast.parse(before_execution_call)

    #Step5: Tranform the client script by adding custom method calls
    transf = TransformCall()
    transf.visit(tree)
    with open("custom_method.py", "r") as source:
        cm = source.read()
        cm_node = ast.parse(cm)

        first_import = 0
        while first_import < len(tree.body) and not isinstance(tree.body[first_import], (ast.Import, ast.ImportFrom)):
            first_import += 1

        first_non_import = first_import
        while first_non_import < len(tree.body) and isinstance(tree.body[first_non_import], (ast.Import, ast.ImportFrom)):
            first_non_import += 1
        # Insert the new import statement before the first non-import statement
        tree.body.insert(first_non_import, cm_node)

    # print('+'*100)
    # import traceback
    # try:
    #     print(ast.dump(tree, indent=4))
    # except:
    #     traceback.print_exc()
    # print('_'*100)

    #Step6: Unparse and convert AST to final code
    print(ast.unparse(tree))


class FuncCallVisitor(ast.NodeVisitor):
    def __init__(self):
        self.name_list = []

    def visit_Name(self, node):
        self.name_list.append(node.id)
        return self.name_list

    def visit_Attribute(self, node):
        self.visit(node.value)
        self.name_list.append(node.attr)
        return self.name_list

    def visit_Call(self, node):
        callvisitor = FuncCallVisitor()
        callvisitor.visit(node.func)
        call_list = callvisitor.get_name_list()
        self.name_list.extend(call_list)
        return call_list

    def get_name_list(self):
        return self.name_list



class TransformCall(ast.NodeTransformer):
    def __init__(self):
        global requiredAlias
        global sourceCode
        global requiredObjects
        global requiredObjClassMapping
        global requiredObjectsSignature
        global requiredClassDefs
        global before_execution_call_node
        self.objectname = None

    def get_target_id(self, target):
        if isinstance(target, ast.Name):
            return target.id
        elif isinstance(target, ast.Tuple):
            return ", ".join(self.get_target_id(elt) for elt in target.elts)
        elif isinstance(target, ast.Attribute):
            return target.attr
        elif isinstance(target, ast.Subscript):
            target_id = self.get_target_id(target.value)
            if isinstance(target.slice, ast.Index):
                index_id = self.get_target_id(target.slice.value)
                return f"{target_id}[{index_id}]"
            elif isinstance(target.slice, ast.Slice):
                start_id = self.get_target_id(target.slice.lower) if target.slice.lower else ""
                stop_id = self.get_target_id(target.slice.upper) if target.slice.upper else ""
                step_id = self.get_target_id(target.slice.step) if target.slice.step else ""
                if start_id or stop_id or step_id:
                    return f"{target_id}[{start_id}:{stop_id}:{step_id}]"
                else:
                    return f"{target_id}[:]"
            elif isinstance(target.slice, ast.ExtSlice):
                dim_ids = [self.get_target_id(d) for d in target.slice.dims]
                return f"{target_id}[{', '.join(dim_ids)}]"
            elif isinstance(target.slice, ast.Name):
                return f"{target_id}[{target.slice.id}]"
            elif isinstance(target.slice, ast.Constant):
                return f"{target_id}[{target.slice.value}]"
            elif isinstance(target.slice, ast.Tuple):
                return f"{target_id}[{', '.join(self.get_target_id(elt) for elt in target.slice.elts)}]"
            else:
                return ""
                # raise ValueError("Unsupported target type")
        elif isinstance(target, ast.Starred):
            return f"*{self.get_target_id(target.value)}"
        else:
            return "" #covered multiple types of object, add in future if some complex type are missing
            # raise ValueError("Unsupported target type")


    def get_func_name(self, value):
        if isinstance(value, ast.Call):
            if isinstance(value.func, ast.Name):
                return value.func.id
            elif isinstance(value.func, ast.Attribute):
                return value.func.attr
            else:
                return None                       
                # raise ValueError("Unsupported function type")
        elif isinstance(value, ast.BinOp):
            return self.get_func_name(value.left)
        else:
            # print("Unsupported value type")
            return None

    # def visit_Assign(self, node):
    #     target = node.targets[0]
    #     self.objectname = self.get_target_id(target)
    #     classname = self.get_func_name(node.value)
    #     if classname and classname in list(requiredClassDefs.keys()):
    #         requiredObjClassMapping[self.objectname] = classname
    #         requiredObjectsSignature[self.objectname] = ast.get_source_segment(sourceCode, node.value)
        
    #     if isinstance(node.value, ast.Call):
    #          node.value = self.visit_Call(node.value)
            
    #     return node

    def visit_Assign(self, node):
        target = node.targets[0]
        self.objectname = self.get_target_id(target)
        classname = self.get_func_name(node.value)
        # print("node.value: ", ast.dump(node.value))
        modified_node = None  # Initialize modified_node as None

        if classname and classname in list(requiredClassDefs.keys()):
            requiredObjClassMapping[self.objectname] = classname
            requiredObjectsSignature[self.objectname] = ast.get_source_segment(sourceCode, node.value)

        if isinstance(node.value, ast.Call):
            requiredObjectsSignature[self.objectname] = ast.get_source_segment(sourceCode, node.value.func)
            # print("self.objectname : ", self.objectname, " --- value: ", ast.get_source_segment(sourceCode, node.value))
            modified_node = self.custom_Call(node.value)
        
        if modified_node and modified_node != node.value:
            return [before_execution_call_node, node, ast.Expr(value=modified_node[0])]

        return node


    def visit_Expr(self, node):
        # print("visit_Expr", ast.dump(node))
        if isinstance(node.value, ast.Call):
            modified_node = None  # Initialize modified_node as None
            modified_node = self.custom_Call(node.value)
            if modified_node and modified_node != node.value:
                return [before_execution_call_node, node, ast.Expr(value=modified_node[0])]
        return node
    
    def custom_Call(self, node):
        callvisitor = FuncCallVisitor()
        callvisitor.visit(node.func)
        callvisitor_list = callvisitor.get_name_list()
        # print("callvisitor_list", ast.dump(node))

        if(any(lib in callvisitor.get_name_list() for lib in requiredAlias)):
            dummyNode=copy.deepcopy(node)
            dummyNode.args.clear()
            dummyNode.keywords.clear()
            # argList = [ast.get_source_segment(sourceCode, a) for a in node.args]
            argList = [a for a in node.args]
            # print("argList", [ast.dump(a) for a in node.args])
            # keywordsDict = {a.arg:ast.get_source_segment(sourceCode, a.value) for a in node.keywords}
            keywordsDict = {a.arg:a.value for a in node.keywords}
            # print("keywordsDict", [kwitem for kwitem in keywordsDict])
            # if(node.args):
            #     dummyNode.args.append(ast.Name(id='*args', ctx=ast.Load()))
            # if(node.keywords):
            #     dummyNode.keywords.append(ast.Name(id='**kwargs', ctx=ast.Load()))

            new_node = ast.Call(func=ast.Name(id='after_execution_INSERTED_INTO_SCRIPT', ctx=ast.Load()),
                                args=[],
                                keywords=[
                                    ast.keyword(
                                        arg='start_times',
                                        value=ast.Name(id='start_times_INSERTED_INTO_SCRIPT')),
                                    ast.keyword(
                                        arg='experiment_file_path',
                                        value=ast.Name(id='EXPERIMENT_FILE_PATH')),
                                    ast.keyword(
                                        arg='function_to_run',
                                        value=ast.Constant(ast.unparse(dummyNode))),
                                    ast.keyword(
                                        arg='method_object',
                                        value=ast.Constant(None)),
                                    ast.keyword(
                                        arg='function_args',
                                        value=ast.List(
                                            elts=[argItem for argItem in argList],
                                            ctx=ast.Load()) if argList else ast.Constant(None)
                                            ),
                                    ast.keyword(
                                        arg='function_kwargs',
                                        value=ast.Dict(
                                            keys=[ast.Constant(KWItem) for KWItem in keywordsDict],
                                            values=[keywordsDict[KWItem] for KWItem in keywordsDict]) if keywordsDict else ast.Constant(None)
                                            )
                                        ],
                                        starargs=None, kwargs=None
                                )
            
            ast.copy_location(new_node, node)
            ast.fix_missing_locations(new_node)
            # return [ast.Expr(value=new_node), ast.Expr(value=node)]
            return [new_node, node]
        elif(callvisitor_list and (callvisitor_list[0] in requiredObjects) and (requiredObjClassMapping.get(callvisitor_list[0]) in list(requiredClassDefs.keys()))):
            
            dummyNode=copy.deepcopy(node)
            dummyNode.args.clear()
            dummyNode.keywords.clear()
            # argList = [ast.get_source_segment(sourceCode, a) for a in node.args]
            argList = [a for a in node.args]
            # keywordsDict = {a.arg:ast.get_source_segment(sourceCode, a.value) for a in node.keywords}
            keywordsDict = {a.arg:a.value for a in node.keywords}
            # print("keywordsDict", [kwitem for kwitem in keywordsDict])
            # if(node.args):
            #     dummyNode.args.append(ast.Name(id='*args', ctx=ast.Load()))
            # if(node.keywords):
            #     dummyNode.keywords.append(ast.Name(id='**kwargs', ctx=ast.Load()))
            new_node = ast.Call(func=ast.Name(id='after_execution_INSERTED_INTO_SCRIPT', ctx=ast.Load()),
                                args=[],
                                keywords=[
                                    ast.keyword(
                                        arg='start_times',
                                        value=ast.Name(id='start_times_INSERTED_INTO_SCRIPT')),
                                    ast.keyword(
                                        arg='experiment_file_path',
                                        value=ast.Name(id='EXPERIMENT_FILE_PATH')),
                                    ast.keyword(
                                        arg='function_to_run',
                                        value=ast.Constant(ast.unparse(dummyNode).replace(callvisitor_list[0], requiredObjectsSignature.get(callvisitor_list[0]), 1))),
                                    ast.keyword(
                                        arg='method_object',
                                        value= ast.Name(callvisitor_list[0])
                                        ),
                                    ast.keyword(
                                        arg='function_args',
                                        value=ast.List(
                                            elts=[argItem for argItem in argList],
                                            ctx=ast.Load()) if argList else ast.Constant(None)
                                            ),
                                    ast.keyword(
                                        arg='function_kwargs',
                                        value=ast.Dict(
                                            keys=[ast.Constant(KWItem) for KWItem in keywordsDict],
                                            values=[keywordsDict[KWItem] for KWItem in keywordsDict]) if keywordsDict else ast.Constant(None)
                                            )
                                        ],
                                        starargs=None, kwargs=None
                                )
            ast.copy_location(new_node, node)
            ast.fix_missing_locations(new_node)
            # return [ast.Expr(value=new_node), ast.Expr(value=node)]
            return [new_node, node]
        elif(callvisitor_list and (callvisitor_list[0] in requiredObjects)):
            
            dummyNode=copy.deepcopy(node)
            dummyNode.args.clear()
            dummyNode.keywords.clear()
            # argList = [ast.get_source_segment(sourceCode, a) for a in node.args]
            argList = [a for a in node.args]
            # keywordsDict = {a.arg:ast.get_source_segment(sourceCode, a.value) for a in node.keywords}
            keywordsDict = {a.arg:a.value for a in node.keywords}
            # print("keywordsDict", [kwitem for kwitem in keywordsDict])
            # if(node.args):
            #     dummyNode.args.append(ast.Name(id='*args', ctx=ast.Load()))
            # if(node.keywords):
            #     dummyNode.keywords.append(ast.Name(id='**kwargs', ctx=ast.Load()))
            new_node = ast.Call(func=ast.Name(id='after_execution_INSERTED_INTO_SCRIPT', ctx=ast.Load()),
                                args=[],
                                keywords=[
                                    ast.keyword(
                                        arg='start_times',
                                        value=ast.Name(id='start_times_INSERTED_INTO_SCRIPT')),
                                    ast.keyword(
                                        arg='experiment_file_path',
                                        value=ast.Name(id='EXPERIMENT_FILE_PATH')),
                                    ast.keyword(
                                        arg='function_to_run',
                                        value=ast.Constant(ast.unparse(dummyNode).replace(callvisitor_list[0], requiredObjectsSignature.get(callvisitor_list[0]), 1))),
                                    ast.keyword(
                                        arg='method_object',
                                        value= ast.Name(callvisitor_list[0])
                                        ),
                                    ast.keyword(
                                        arg='function_args',
                                        value=ast.List(
                                            elts=[argItem for argItem in argList],
                                            ctx=ast.Load()) if argList else ast.Constant(None)
                                            ),
                                    ast.keyword(
                                        arg='function_kwargs',
                                        value=ast.Dict(
                                            keys=[ast.Constant(KWItem) for KWItem in keywordsDict],
                                            values=[keywordsDict[KWItem] for KWItem in keywordsDict]) if keywordsDict else ast.Constant(None)
                                            )
                                        ],
                                        starargs=None, kwargs=None
                                )
            ast.copy_location(new_node, node)
            ast.fix_missing_locations(new_node)
            # print("new_node",ast.dump(new_node))
            # print("new_node_type",type(new_node))
            # return [ast.Expr(value=new_node), ast.Expr(value=node)]
            return [new_node, node]
            

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
        global sourceCode
        global requiredClassDefs
        
    def visit_Assign(self, node):
        if isinstance(node.value, ast.Call):
            callvisitor2 = FuncCallVisitor()
            callvisitor2.visit(node.value.func)
            name_list = callvisitor2.get_name_list()
            if(name_list and (any(lib in name_list for lib in requiredAlias)) and (isinstance(node.targets[0], ast.Name))):
                self.stats["objects"].append(node.targets[0].id)
            
            if(name_list and (any(lib in name_list[0] for lib in list(requiredClassDefs.keys()))) and (isinstance(node.targets[0], ast.Name))):
                self.stats["objects"].append(node.targets[0].id)
        self.generic_visit(node)


if __name__ == "__main__":
    main()