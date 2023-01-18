requiredClass dict_keys([])
requiredObjects []
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
                alias(name='math'),
                alias(name='random')]),
        Import(
            names=[
                alias(name='matplotlib.pyplot', asname='plt')]),
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
        ImportFrom(
            module='os.path',
            names=[
                alias(name='exists')],
            level=0),
        Import(
            names=[
                alias(name='sys')])],
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
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from os.path import exists
import sys
