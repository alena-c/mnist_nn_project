#!/usr/bin/env python
# coding: utf-8

# In[4]:


# Visualizes specified image from the MNIST database

import pickle
import gzip

import matplotlib.pyplot as plt
import numpy as np
get_ipython().run_line_magic('matplotlib', 'inline')


# In[17]:


def vectorized_result(i):
    label = np.zeros((10, 1))
    label[i] = 1.0
    return e
    
def load_data():
    f = gzip.open('mnist.pkl.gz', 'rb')
    training_data, validation_data, test_data = pickle.load(f, encoding="latin1")
    f.close()
    return (training_data, validation_data, test_data)

tr_d, va_d, te_d = load_data()
training_inputs = [np.reshape(x, (784, 1)) for x in tr_d[0]]
training_results = [vectorized_result(y) for y in tr_d[1]]
training_data = list(zip(training_inputs, training_results))

type(training_data[0][0])

type(training_inputs[0][0])
len(training_inputs[0])
arr = training_inputs[987]

def displayMNIST(imageAsArray):
    imageAsArray = imageAsArray.reshape(28, 28);
    plt.imshow(imageAsArray, cmap='gray')
#     plt.show()

displayMNIST(arr) 

