#!/usr/bin/env python
# coding: utf-8

# In[25]:


# Testing of my own (new) data
import json
import random
import sys
sys.path.append('../src/')

import matplotlib.pyplot as plt
import numpy as np


# In[51]:


import mnist_loader_new
training_data , validation_data , new_test_data = mnist_loader_new.load_data_wrapper()
import network
net = network.Network([784, 100, 10], cost=network.CrossEntropyCost)
net.large_weight_initializer()
test_cost, test_accuracy, training_cost, training_accuracy     = net.SGD(training_data, 100, 10, 0.1, lmbda = 5.0,  
            evaluation_data=new_test_data,
            monitor_evaluation_cost=True,
            monitor_evaluation_accuracy=True,
            monitor_training_cost=True,
            monitor_training_accuracy=True)

net.save('test_my_digits.json')

training_cost_xmin = 0
test_cost_xmin = 0
test_accuracy_xmin = 0
training_accuracy_xmin = 0
num_epochs = 100
training_set_size = 50000

import winsound
duration = 1000  # milliseconds
freq = 440  # Hz
winsound.Beep(freq, duration)


# In[67]:


def plot_test_accuracy(test_accuracy, num_epochs, test_accuracy_xmin):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(np.arange(test_accuracy_xmin, num_epochs), 
            [accuracy*100/6
             for accuracy in test_accuracy[test_accuracy_xmin:num_epochs]],
            color='#FF338D')
    ax.set_xlim([test_accuracy_xmin, num_epochs])
    ax.grid(True, linestyle='--')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Classificatioin accuracy, %')
    ax.set_title('Accuracy on test data')
    plt.show()
    fig.savefig('images/my_data_accuracy_alldata_100epochs_reg.jpeg')
    
plot_test_accuracy(test_accuracy, num_epochs, test_accuracy_xmin)

def plot_training_cost(training_cost, num_epochs, training_cost_xmin):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(np.arange(training_cost_xmin, num_epochs), 
            training_cost[training_cost_xmin:num_epochs],
            color='#4FDC98')
    ax.set_xlim([training_cost_xmin, num_epochs])
    ax.grid(True, linestyle='--')
    ax.set_xlabel('Epoch')
    ax.set_title('Cost on the training data')
    plt.show()
    fig.savefig('images/training_cost_alldata_100epochs_reg_mydata.jpeg')
    
plot_training_cost(training_cost, num_epochs, training_cost_xmin)


# In[ ]:




