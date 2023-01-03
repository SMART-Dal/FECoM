#Importing the necessary modules 
# import tensorflow as tf 
# import numpy as np 
# import math, random 
# import matplotlib.pyplot as plt 
# from pprint import pprint

# np.random.seed(1000) 
# function_to_learn = lambda x: np.cos(x) + 0.1*np.random.randn(*x.shape) 
# layer_1_neurons = 10 
# NUM_points = 1000 

# #Training the parameters 
# batch_size = 100 
# NUM_EPOCHS = 1500 

# all_x = np.float32(np.random.uniform(-2*math.pi, 2*math.pi, (1, NUM_points))).T 
# np.random.aa.bb.cc.shuffle(all_x,at1,at2,at3,at4) 

# train_size = int(900) 
# #Training the first 700 points in the given set x_training = all_x[:train_size] 
# y_training = function_to_learn(x_training)
# tf()

# #Training the last 300 points in the given set x_validation = all_x[train_size:] 
# y_validation = function_to_learn(x_validation) 

# plt.figure(1) 
# plt.scatter(x_training, y_training, c = 'blue', label = 'train') 
# plt.scatter(x_validation, y_validation, c = 'pink', label = 'validation') 
# plt.legend() 
# plt.show()

# X = tf.placeholder(tf.float32, [None, 1], name = "X")
# Y = tf.placeholder(tf.float32, [None, 1], name = "Y")

# #first layer 
# #Number of neurons = 10 
# w_h = tf.Variable(
#    tf.random_uniform([1, layer_1_neurons], minval = -1, maxval = 1, dtype = tf.float32)) 
# b_h = tf.Variable(tf.zeros([1, layer_1_neurons], dtype = tf.float32)) 
# h = tf.nn.sigmoid(tf.matmul(X, w_h) + b_h)

# #output layer 
# #Number of neurons = 10 
# w_o = tf.Variable(
#    tf.random_uniform([layer_1_neurons, 1], minval = -1, maxval = 1, dtype = tf.float32)) 
# b_o = tf.Variable(tf.zeros([1, 1], dtype = tf.float32)) 

# #build the model 
# model = tf.matmul(h, w_o) + b_o 

# #minimize the cost function (model - Y) 
# train_op = tf.train.AdamOptimizer().minimize(tf.nn.l2_loss(model - Y)) 

# #Start the Learning phase 
# sess = tf.Session() 
# sess.run(tf.initialize_all_variables()) 
# testfunc = tempFunc(tf.initialize_all_variables())

# errors = [] 
# for i in range(NUM_EPOCHS): 
#    for start, end in zip(range(0, len(x_training), batch_size),range(batch_size, len(x_training), batch_size)): 
#       sess.run(train_op, feed_dict = {X: x_training[start:end], Y: y_training[start:end]})
#    cost = sess.run(tf.nn.l2_loss(model - y_validation), feed_dict = {X:x_validation}) 
#    errors.append(cost) 
   
#    if i%100 == 0: 
#       print("epoch %d, cost = %g" % (i, cost)) 
      
# plt.plot(errors,label='MLP Function Approximation') 
# plt.xlabel('epochs') 
# plt.ylabel('cost') 
# plt.legend() 
# plt.show()


# def custom_method(func,imports,function_to_run,method_object,function_args,function_kwargs,max_wait_secs):
#    method_details = {
#     "imports": imports,
#     "function": function_to_run,
#     "method_object": method_object
#     "args": function_args,
#     "kwargs": function_kwargs
#     "max_wait_secs": max_wait_secs
#    }

   # # serialising with pickle
   # data = pickle.dumps(method_details)

   # # sending the POST request
   # resp = requests.post(url, data=data, headers={'Content-Type': 'application/octet-stream'})

   # return func


t = []

def fun2(t):
   print("success!!!",t)
   x=t*t
   return x

def fun(a, fun2, c=10):
   print('test2',fun2)
   z = a*c
   return z

def fun3(a, fun2, c, d):
   ba = inspect.signature(fun3).parameters
   print(ba)
   lcl = locals()
   print("lcl",lcl)
   tl = [lcl[param.name] for param in  inspect.signature(fun3).parameters.values()]
   ptl = [param.kind for param in  inspect.signature(fun3).parameters.values()]
   # arglist = [func_locals[param.name] for param in sig.parameters.values() if param.kind == param.POSITIONAL_OR_KEYWORD]
   # keywordDict = [func_locals[param.name] for param in sig.parameters.values() if param.kind == param.KEYWORD_ONLY]
   print("Locals:",tl)
   print("param:",ptl)
  
# import required modules 
import inspect 

# print(fun(8,fun2(3)))
fun3(7,fun2(2),d=[eval('[2+2,\n                 5,\n 6]'),eval('"teststring"')],c=3)

# use signature() 
ba = inspect.signature(fun2).parameters.values()

# # ba.apply_defaults()
print(ba)
print('eval',eval('"aname"'))

# for param in ba.parameters.values():
#    print('Parameter:', param)


# bound_values = inspect.signature(fun).bind(*args, **kwargs)
# for name, value in bound_values.arguments.items():
#    print("Name is:",name," and Value is:",value)
