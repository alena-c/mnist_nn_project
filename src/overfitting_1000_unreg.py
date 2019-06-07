#!/usr/bin/env python
# coding: utf-8

# In[9]:


# The Plots illustrate the overfitting proble
# np.amax(test_accuracy) #8203,8293
import time
import json
import random
import sys
sys.path.append('../src/')

import matplotlib.pyplot as plt
import numpy as np

import mnist_loader
import network

def main():
    start = time.time()
    
    run_network()
    make_plots()
    
    end = time.time()
    print(end - start)

    import winsound
    duration = 1000  # milliseconds
    freq = 440  # Hz
    winsound.Beep(freq, duration)


# In[6]:


def run_network():
    training_data , validation_data , test_data = mnist_loader.load_data_wrapper()
    net = network.Network([784, 30, 10], cost=network.CrossEntropyCost)
    net.large_weight_initializer()
    test_cost, test_accuracy, training_cost, training_accuracy             = net.SGD(training_data[:1000], 400, 10, 0.5,
                      evaluation_data=test_data,
                      monitor_evaluation_cost=True,
                      monitor_evaluation_accuracy=True,
                      monitor_training_cost=True,
                      monitor_training_accuracy=True)
    
    f = open("overfitting_results_1000_unreg.json", "w") 
    result_data = {
        'test_cost' : test_cost,
        'test_accuracy' : test_accuracy,
        'training_cost': training_cost,
        'training_accuracy': training_accuracy
        }
    json.dump(result_data, f)
    f.close()
    
def make_plots():
    f = open("overfitting_results_1000_unreg.json", "r") 
    result_data = json.load(f)
    f.close()
    
    test_cost = result_data['test_cost']
    test_accuracy = result_data['test_accuracy']
    training_cost = result_data['training_cost']
    training_accuracy = result_data['training_accuracy']
    
    plots(test_cost, test_accuracy,  training_cost, training_accuracy) 
    
def plots(test_cost, test_accuracy,  training_cost, training_accuracy):
    ymax = np.amax(test_accuracy)
    xpos = test_accuracy.index(ymax)
    max_acc = (xpos, ymax/100)
    
    training_cost_xmin = 0
    test_cost_xmin = 0
    test_accuracy_xmin = 200
    training_accuracy_xmin = 0
    num_epochs = 400
    training_set_size = 1000
    
    fig, axes = plt.subplots(2, 2, figsize=(12,8))
    axes[0, 0].tick_params(color='w', labelcolor='w')
    axes[0, 1].tick_params(color='w', labelcolor='w')
    axes[1, 0].tick_params(color='w', labelcolor='w')
    axes[1, 1].tick_params(color='w', labelcolor='w')

    fig.suptitle('Overfitting_unregularized (1000 training images)', fontsize=12)
    
    ax = fig.add_subplot(221)
    ax.plot(np.arange(test_accuracy_xmin, num_epochs), 
            [accuracy/100.0
             for accuracy in test_accuracy[test_accuracy_xmin:num_epochs]],
            color='#5FDC98', label='test_accuracy')    #5FDC98
    ax.annotate(str(np.amax(test_accuracy)/100), xy=max_acc, color='gray', 
                xytext=(xpos-30, np.amax(test_accuracy)/100),
                ha='center', arrowprops=dict(facecolor='gray'))
            
    ax.set_xlim([test_accuracy_xmin, num_epochs])    
    ax.grid(True, linestyle='--')
    ax.set_title('Classificatioin Accuracy (%)')
    ax.set_ylabel('Test data')
    plt.legend(loc='lower right')
    
    ax = fig.add_subplot(222)
    ax.plot(np.arange(test_cost_xmin, num_epochs),
            test_cost[test_cost_xmin:num_epochs],               # test_cost_xmin:num_epochs],
            color='#5a378e', label='test_cost')    # #4FDC98
    ax.grid(True, linestyle='--')
    ax.set_title('Cost')
    plt.legend(loc='lower right')

    ax = fig.add_subplot(223)
    ax.plot(np.arange(training_accuracy_xmin, num_epochs), 
            [accuracy*100.0/training_set_size 
             for accuracy in training_accuracy[training_accuracy_xmin:num_epochs]],
            color='#37b767', label='training_accuracy')
    ax.grid(True, linestyle='--')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Training data')
    plt.legend(loc='best')
    
    ax = fig.add_subplot(224)
    ax.plot(np.arange(training_cost_xmin, 200), 
            training_cost[training_cost_xmin:200],  # training_cost_xmin:num_epochs],
            color='#FF338D', label='training_cost')
    ax.grid(True, linestyle='--')
    ax.set_xlabel('Epoch')
    plt.legend(loc='best')
    plt.show()
    fig.savefig('Overfitting_results_1000_unreg.png')

if __name__ == "__main__":
    main() 


# In[10]:


make_plots()


# In[ ]:




