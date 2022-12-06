#Importing the necessary modules 
import tensorflow as tf 
import numpy as np 
import math, random 
import matplotlib.pyplot as plt 
from pprint import pprint

tf()

X = tf.placeholder(tf.float32, [None, 1], name = "X")
Y = tf.placeholder(tf.float32, [None, 1], name = "Y")

#output layer 
#Number of neurons = 10 
w_o = tf.Variable(
   tf.random_uniform([layer_1_neurons, 1], minval = -1, maxval = 1, dtype = tf.float32)) 
b_o = tf.Variable(tf.zeros([1, 1], dtype = tf.float32)) 