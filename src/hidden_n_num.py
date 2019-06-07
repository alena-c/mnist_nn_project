#!/usr/bin/env python
# coding: utf-8

# In[1]:


# The number of hidden neurons variation
# Plot shows the performance of MNIST with different hidden layer architectures

# Standard library
import time
import json
import random
import sys
sys.path.append('../src/')

import matplotlib.pyplot as plt
import numpy as np

import mnist_loader
import network


# In[2]:


HIDDEN_NS = [10, 20, 50, 100, 200, 500] 


# In[3]:


def main():
    start = time.time()
    run_networks()
    make_plots()
    
    end = time.time()
    print(end - start)

    import winsound
    duration = 1000  # milliseconds
    freq = 440  # Hz
    winsound.Beep(freq, duration)
                       
def run_networks():
    # Make results more easily reproducible
    random.seed(12345678)
    np.random.seed(12345678)
    training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
    accuracies = []
    for hidden_n in HIDDEN_NS:
        net = network.Network([784, hidden_n, 10], cost=network.CrossEntropyCost())
        print("\n\nTraining network with %s hidden neurons" % hidden_n)
        net.large_weight_initializer()
        num_epochs = 400 
        net.SGD(training_data[:1000], num_epochs, 10, 0.5)
        accuracy = net.accuracy(validation_data) / 100.0
        print("Accuracy was %s percent" % accuracy)
        accuracies.append(accuracy)
    f = open("more_data_hn.json", "w")
    json.dump(accuracies, f)
    f.close()

def make_plots():
    f = open("more_data_hn.json", "r") 
    accuracies = json.load(f)
    f.close()
    make_linear_plot(accuracies)
    make_log_plot(accuracies)

def make_linear_plot(accuracies):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(HIDDEN_NS, accuracies, color='#37b767',
            linewidth=2)
    ax.plot(HIDDEN_NS, accuracies, "o", color='#FF338D')
    ax.set_xlim(10, 500)
    ax.set_ylim(60, 100)
    ax.grid(True, linestyle='--')
    ax.set_xlabel('Number of hidden neurons')
    ax.set_ylabel('Classification accuracy (%)')
    ax.set_title('Accuracy on validation data')
    plt.show()
    fig.savefig('images/class_accuracy_hidNNum_overfit_1000data_400epo_0.5lr_unreg_valid.png')


def make_log_plot(accuracies):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(HIDDEN_NS, accuracies, color='#37b767',
            linewidth=2)
    ax.plot(HIDDEN_NS, accuracies, "o", color='#FF338D')
    ax.set_xlim(10, 500)
    ax.set_ylim(60, 100)
    ax.set_xscale('log')
    ax.grid(True, linestyle='--')
    ax.set_xlabel('Number of hidden neurons')
    ax.set_ylabel('Classification accuracy (%)')
    ax.set_title('Accuracy on validation data')
    plt.show()
    fig.savefig('images/class_accuracy_hidNNum_log_overfit_1000data_400epo_0.5lr_unreg_valid.png')

if __name__ == "__main__":
    main()

