#!/usr/bin/env python
# coding: utf-8

# In[15]:


# Multiple learning rate
# This program shows how different values for the learning rate affect
# training.  In particular, we'll plot out how the cost changes using
# three different values for α.

import json
import random
import sys
sys.path.append('../src/')

import matplotlib.pyplot as plt
import numpy as np
import mnist_loader
import network

LEARNING_RATES = [0.025, 0.25, 1.0, 2.5]
COLORS = ['#37b767', '#FF338D', '#135ca5', '#84737a']
NUM_EPOCHS = 30

def main():
    run_networks()
    make_plot()

def run_networks():
    # Trains the network and saves the training_costs in a json file
    random.seed(12345678)
    np.random.seed(12345678)
    training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
    results = []
    for rate in LEARNING_RATES:
        print("\nTrain a network using learning rate = " + str(rate))
        net = network.Network([784, 30, 10])
        results.append(
            net.SGD(training_data, NUM_EPOCHS, 10, rate, lmbda=5.0,
                    evaluation_data=validation_data, 
                    monitor_training_cost=True))
    f = open("multiple_lr.json", "w")
    json.dump(results, f)
    f.close()

def make_plot():
    f = open("multiple_lr_1000_400epochs_valid.json", "r")
    results = json.load(f)
    f.close()
    fig = plt.figure()
    ax = fig.add_subplot(111)
    for rate, result, color in zip(LEARNING_RATES, results, COLORS):
        _, _, training_cost, _ = result
        ax.plot(np.arange(NUM_EPOCHS), training_cost, "o-",
                label="α = " + str(rate),
                color=color)
    ax.set_xlim([0, 30])
    ax.set_title('Cost at Multiple Learning Rates \n (unregulazed, 1000 training images, 400 epochs)')
    ax.grid(True, linestyle='--')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Training_cost')
    plt.legend(loc='upper right')
    plt.show()
    fig.savefig('images/multiple_lr_1000_400epochs_valid.png')
    
make_plot()
if __name__ == "__main__":
    main()


# In[ ]:




