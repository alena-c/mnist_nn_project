#!/usr/bin/env python
# coding: utf-8

# ![](https://paper-attachments.dropbox.com/s_A6DC005F6FFD997D371B7F58E6BDAB0FE6D3CC4E36D1A209D8D89A728077A289_1559613357589_test.png)

# In[73]:


"""
Overfitting: 50,000 training images, 60 epochs, 
             100 hidden neurons
             10 mini-batches,
             0.1 learining rate
             5.0 reg parameter
             
             Validation data is used to compare with the 
             earlier graphs.
~~~~~~~~~~~
Plot graphs to illustrate the problem of overfitting.  9780
"""
import json
import random
import sys
import time

import matplotlib.pyplot as plt
import numpy as np

sys.path.append('../src/')
import mnist_loader
import network


# In[85]:


def main(filename):
    start = time.time()
    
    NUM_HIDDEN = 30
#     NUM_HIDDEN = 100
    run_network(filename)
    plot_overlay_old_new(filename)
    
    end = time.time()
    print(end - start)

    import winsound
    duration = 1000 
    freq = 440
    winsound.Beep(freq, duration)
    
def run_network(filename):
    random.seed(12345678)
    np.random.seed(12345678)

    training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
    net = network.Network([784, NUM_HIDDEN, 10], cost=network.CrossEntropyCost)
    # The new starting weights, whose standard deviation is 1 over the
    # square root of the number of input neurons.
    print("Train the network with the new starting weights.")
    
    new_vc, new_va, new_tc, new_ta         = net.SGD(training_data, 30, 10, 0.1, lmbda = 5.0, 
                  evaluation_data=validation_data,
                  monitor_evaluation_accuracy=True)
    
    # the old(previous and large) starting weights, whose standard deviation is 1
    print("\n\nTrain the network using the old starting weights.")
    net.large_weight_initializer()
    old_vc, old_va, old_tc, old_ta         = net.SGD(training_data, 30, 10, 0.1, lmbda = 5.0, 
                  evaluation_data=validation_data,
                  monitor_evaluation_accuracy=True)
              
    f = open(filename, 'w')
    json.dump({'new_starting_weights':
               [new_vc, new_va, new_tc, new_ta],
               'old_starting_weights':
               [old_vc, old_va, old_tc, old_ta]}, f)
    f.close()

    
def plot_overlay_old_new(filename):
    f = open(filename, 'r')
    results = json.load(f)
    f.close
    
    new_vc, new_va, new_tc, new_ta = results['new_starting_weights']
    old_vc, old_va, old_tc, old_ta = results['old_starting_weights']

    #convert to %
    new_va = [x/100 for x in new_va]
    old_va = [x/100 for x in old_va]
    
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(np.arange(0,30), new_va, color='#7e7f9e', linewidth=2,
            label="weight_initialization_new")
    ax.plot(np.arange(0,30), old_va, color='#7b3f84', linewidth=2,
            label="weight_initialization_old")
    ax.grid(True, linestyle='--')
    ax.set_xlim([0, 30])
    ax.set_xlabel('Epoch')
    ax.set_ylim([90, 98])
    ax.set_ylabel('Classification accuracy, %')
    plt.legend(loc='lower right')
    ax.set_title('Effect of Weights Initialization Approach on Learning \n (30 hiddin layers)')
#     ax.set_title('Effect of Weights Initialization Approach on Learning \n (100 hiddin layers)')
    plt.show()
    fig.savefig('images/w_b_overlayAccu_allData_30epochs_30hl_5Lmbda_valid.jpg')
    
main('w_b_30.json')
# main('w_b_100.json')

