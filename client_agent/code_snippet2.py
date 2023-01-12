#Importing the necessary modules 
import tensorflow as tf 
import numpy as np 
import math, random 
import matplotlib.pyplot as plt 
from pprint.xyz.abc import pprint as ppr
import torch
import torchvision
import torchvision.transforms as transforms
from tf import trans as tf1
from pt import trans as tf2
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from os.path import exists
# add parent directory to path
import sys

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers import Dense, Dropout as drop
from tensorflow.keras.layers import *
import tensorflow.keras.backend as K
import tensorflow.keras.layers as L
import tensorflow.compat.v1 as tf
from tensorflow.keras.layers import Dense, Dropout, Conv2D
from tensorflow.keras.backend import clear_session, set_session
from tensorflow.keras.layers import Dense as dense_layer, Dropout as drop, Conv2D as conv


tf()
testlist = ['a','b','c']
X = tf.placeholder(tf.float32, [None, 1], name = "X")
Y = tf.placeholder(tf.float32, [None, 1], name = "Y")
z = tf.value("aname",bname,name="tname")
# testlist.func()
#output layer 
#Number of neurons = 10 
w_o = tf.Variable(
   tf.random_uniform([layer_1_neurons, 1], minval = -1, maxval = 1, dtype = tf.float32),test1,test2,test3="test3") 
b_o = tf.Variable(tf.zeros([1, 1], dtype = tf.float32)) 

cd = ab.test(a,b)
c_o = xf.test()
d_o = xy.test()

model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(10)
    ])

model.compile()
f_o = b_o.test.compile()
z_o = c_o.compile()