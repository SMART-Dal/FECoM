import pickle
import requests
import sys
sys.path.append('../../../server')
from send_request import send_request

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
