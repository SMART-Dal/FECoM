#Importing the necessary modules 
import tensorflow as tf 
import numpy as np 
import math, random 
import matplotlib.pyplot as plt 
from pprint import pprint

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