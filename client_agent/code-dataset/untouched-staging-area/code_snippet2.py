# #Importing the necessary modules 
# import tensorflow as tf 
# import numpy as np 
# import math, random 
# import matplotlib.pyplot as plt 
# # from pprint.xyz.abc import pprint as ppr
# import torch
# import torchvision
# import torchvision.transforms as transforms
# from torch import trans
# from pt import trans as tf2
# import torch.nn as nn
# import torch.nn.functional as F
# import torch.optim as optim
# from os.path import exists
# # add parent directory to path
# import sys

# import tensorflow as tf
# from tensorflow import keras
# from tensorflow.keras import layers
# from tensorflow.keras.layers import Dense, Dropout as drop
# from tensorflow.keras.layers import *
# import tensorflow.keras.backend as K
# import tensorflow.keras.layers as L
# import tensorflow.compat.v1 as tf
# from tensorflow.keras.layers import Dense, Dropout, Conv2D
# from tensorflow.keras.backend import clear_session, set_session
# from tensorflow.keras.layers import Dense as dense_layer, Dropout as drop, Conv2D as conv


# tf()
# testlist = ['a','b','c']
# X = tf.placeholder(tf.float32, [None, 1], name = "X")
# Y = tf.placeholder(tf.float32, [None, 1], name = "Y")
# z = tf.value("aname",bname,name="tname")
# # testlist.func()
# #output layer 
# #Number of neurons = 10 
# w_o = tf.Variable(
#    tf.random_uniform([layer_1_neurons, 1], minval = -1, maxval = 1, dtype = tf.float32),test1,test2,test3="test3") 
# b_o = tf.Variable(tf.zeros([1, 1], dtype = tf.float32)) 

# cd = ab.test(a,b)
# c_o = xf.test()
# d_o = xy.test()

# model = tf.keras.models.Sequential([
#     tf.keras.layers.Flatten(input_shape=(28, 28)),
#     tf.keras.layers.Dense(128, activation='relu'),
#     tf.keras.layers.Dropout(0.2),
#     tf.keras.layers.Dense(10)
#     ])

# model.compile()
# f_o = b_o.test.compile()
# z_o = c_o.compile()


# # source: https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html
# # also see cifar10_tutorial.ipynb

# import torch
# import torchvision
# import torchvision.transforms as transforms
# import torch.nn as nn
# import torch.nn.functional as F
# import torch.optim as optim

# # (1) load dataset & normalise tensors
# transform = transforms.Compose(
#     [transforms.ToTensor(),
#      transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

# batch_size = 4

# trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
# trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=0)

# testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
# testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=0)

# classes = ('plane', 'car', 'bird', 'cat',
#            'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


# # (2) define a convolutional neural network
# class Net(nn.Module, abc.Module):
#     def __init__(self):
#         super().__init__()
#         self.conv1 = nn.Conv2d(3, 6, 5)
#         self.pool = nn.MaxPool2d(2, 2)
#         self.conv2 = nn.Conv2d(6, 16, 5)
#         self.fc1 = nn.Linear(16 * 5 * 5, 120)
#         self.fc2 = nn.Linear(120, 84)
#         self.fc3 = nn.Linear(84, 10)

#     def forward(self, x):
#         x = self.pool(F.relu(self.conv1(x)))
#         x = self.pool(F.relu(self.conv2(x)))
#         x = torch.flatten(x, 1) # flatten all dimensions except batch
#         x = F.relu(self.fc1(x))
#         x = F.relu(self.fc2(x))
#         x = self.fc3(x)
#         return x

# class dummyClass(abc.tf.keras.layers.Layer, abc.nn.Module):
#     def __init__(self):
#         self.dummy = 0

#     def dummyFunc(self):
#         self.dummy += 1
#         return self.dummy

# class dummyClass2():
#     def __init__(self):
#         self.dummy = 0

#     def dummyFunc(self):
#         self.dummy += 1
#         return self.dummy

# net = Net()

# class MyDenseLayer(tf.keras.layers.Layer, nn.Module):
#   def __init__(self, num_outputs):
#     super(MyDenseLayer, self).__init__()
#     self.num_outputs = num_outputs

#   def build(self, input_shape):
#     self.kernel = self.add_weight("kernel",
#                                   shape=[int(input_shape[-1]),
#                                          self.num_outputs])

#   def call(self, inputs):
#     return tf.matmul(inputs, self.kernel)

# layer = MyDenseLayer(10)

# # (3) define loss function & optimiser
# criterion = nn.CrossEntropyLoss()
# optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)


# # (4) train the network
# for epoch in range(2):  # loop over the dataset multiple times

#     running_loss = 0.0
#     for i, data in enumerate(trainloader, 0):
#         # get the inputs; data is a list of [inputs, labels]
#         inputs, labels = data

#         # zero the parameter gradients
#         optimizer.zero_grad()

#         # forward + backward + optimize
#         outputs = net(inputs)
#         loss = criterion(outputs, labels)
#         loss.backward()
#         optimizer.step()

#         # print statistics
#         running_loss += loss.item()
#         if i % 2000 == 1999:    # print every 2000 mini-batches
#             print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
#             running_loss = 0.0

# print('Finished Training')

# PATH = './cifar_net.pth'
# torch.save(net.state_dict(), PATH)


# # (5) test the network
# dataiter = iter(testloader)
# images, labels = next(dataiter)

# net = Net()
# net.load_state_dict(torch.load(PATH))

# outputs = net(images)

# _, predicted = torch.max(outputs, 1)

# print('Predicted: ', ' '.join(f'{classes[predicted[j]]:5s}' for j in range(4)))


# # (6) test on the whole dataset
# correct = 0
# total = 0
# # since we're not training, we don't need to calculate the gradients for our outputs
# with torch.no_grad():
#     for data in testloader:
#         images, labels = data
#         # calculate outputs by running images through the network
#         outputs = net(images)
#         # the class with the highest energy is what we choose as prediction
#         _, predicted = torch.max(outputs.data, 1)
#         total += labels.size(0)
#         correct += (predicted == labels).sum().item()

# print(f'Accuracy of the network on the 10000 test images: {100 * correct // total} %')


# # (7) check which classes performed well, and which did not
# # prepare to count predictions for each class
# correct_pred = {classname: 0 for classname in classes}
# total_pred = {classname: 0 for classname in classes}

# # again no gradients needed
# with torch.no_grad():
#     for data in testloader:
#         images, labels = data
#         outputs = net(images)
#         _, predictions = torch.max(outputs, 1)
#         # collect the correct predictions for each class
#         for label, prediction in zip(labels, predictions):
#             if label == prediction:
#                 correct_pred[classes[label]] += 1
#             total_pred[classes[label]] += 1


# # print accuracy for each class
# for classname, correct_count in correct_pred.items():
#     accuracy = 100 * float(correct_count) / total_pred[classname]
#     print(f'Accuracy for class: {classname:5s} is {accuracy:.1f} %')


# # (8) Train on the GPU (add this before and into the training above, section (4))
# # device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# # # Assuming that we are on a CUDA machine, this should print a CUDA device:

# # print(device)

# # net.to(device)

# # inputs, labels = data[0].to(device), data[1].to(device)