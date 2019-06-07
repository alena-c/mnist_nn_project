#!/usr/bin/env python
# coding: utf-8

# In[8]:


# Shows that accuracy is less than random without a hidden layer
import mnist_loader
import network_0

training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
net = network_0.Network([784, 0, 10])
net.SGD(training_data, 30, 10, 3.0, test_data=test_data)

