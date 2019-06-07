#!/usr/bin/env python
# coding: utf-8

# In[59]:


import time
import json
import random
import sys
sys.path.append('../src/')

import matplotlib.pyplot as plt
import numpy as np
from sklearn import svm

import mnist_loader
import network


# In[68]:


# multiple learning rates
RATES = [0.1, 0.2, 0.5, 1.0, 2.0, 5.0,10.0] 

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


# In[69]:


def run_networks():
    # Make results more easily reproducible
    random.seed(12345678)
    np.random.seed(12345678)
    training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
    net = network.Network([784, 30, 10], cost=network.CrossEntropyCost())
    accuracies = []
    t_costs = []
    for rate in RATES: 
        print("\n\nTraining network with data set size %s" % rate)
        net.large_weight_initializer()
        num_epochs = 400 
        net.SGD(training_data[:1000], num_epochs, 10, rate, lmbda = 0.0)
        accuracy = net.accuracy(validation_data) / 100.0
        print("Accuracy was %s percent" % accuracy)
        cost = net.total_cost(training_data, 0.0)
        print("Training cost was %s " % cost)
        accuracies.append(accuracy)
        t_costs.append(cost)
        
    f = open("lr_var_accuracies_1000_unreg_.json", "w")
    json.dump(accuracies, f)
    f.close()
    f = open("lr_var_trCost_1000_unreg_.json", "w")
    json.dump(t_costs, f)
    f.close()


# In[70]:


def make_plots():
    f = open("lr_var_accuracies_1000_unreg_.json", "r") 
    accuracies = json.load(f)
    f.close()
    f = open("lr_var_trCost_1000_unreg_.json", "r")
    t_costs  = json.load(f)
    f.close()
    
    plots(accuracies, t_costs) 
    
def plots(accuracies, t_costs):
    fig, axes = plt.subplots(2, 2, figsize=(8,8))
    axes[0, 0].tick_params(color='w', labelcolor='w')
    axes[0, 1].tick_params(color='w', labelcolor='w')
    axes[1, 0].tick_params(color='w', labelcolor='w')
    axes[1, 1].tick_params(color='w', labelcolor='w')

    fig.suptitle('Effect of Learning Rates on Network \n (Unregularized, 1000 training images)', fontsize=12)
    
    ax = fig.add_subplot(221)
    ax.plot(RATES, t_costs, color='#e01a84', linewidth=1.5,
            label='training_cost')
    ax.plot(RATES, t_costs, "o", color='#5b0131')
    ax.set_ylim(np.amin(t_costs), np.amax(t_costs) + 0.1)
#     ax.set_ylim(0, 0.9)
    ax.grid(True, linestyle='--')
    ax.set_ylabel('Training cost')
    plt.legend(loc='best')
    
    ax = fig.add_subplot(222)    
    ax.plot(RATES, t_costs, color='#e01a84', linewidth=1.5,
            label='training_cost_log_scaled')
    ax.plot(RATES, t_costs, "o", color='#5b0131')
    ax.set_xscale('Log')
#     ax.set_ylim(0, 0.9)
    ax.set_ylim(np.amin(t_costs), np.amax(t_costs) + 0.1)
    ax.grid(True, linestyle='--')
    plt.legend(loc='best')
    
    ax = fig.add_subplot(223)
    ax.plot(RATES, accuracies, color='#37b767', linewidth=1.5,
            label='accuracy %')
    ax.plot(RATES, accuracies, "o", color='#FF338D')
    ax.set_ylim(60, 90)
    ax.grid(True, linestyle='--')
    ax.set_xlabel('Learning rate linear')
    ax.set_ylabel('Test accuracy %')
    plt.legend(loc='lower left')
    
    ax = fig.add_subplot(224)
    ax.plot(RATES, accuracies, color='#37b767', linewidth=1.5,
            label='accuracy %_scaled')
    ax.plot(RATES, accuracies, "o", color='#FF338D')
    ax.set_ylim(60, 90)
    ax.set_xscale('log')
    ax.grid(True, linestyle='--')
    ax.set_xlabel('Learning rate log scaled')
    plt.legend(loc='lower left')
    
    plt.show()
#     fig.savefig('images/lr_var_1000_unreg.png')
    
if __name__ == "__main__":
    main() 


# In[ ]:




