#!/usr/bin/env python
# coding: utf-8

# In[605]:


# Draw a neural network architecture diagram using matplotilb.

import matplotlib.pyplot as plt
import sys
sys.path.append('../src/')

def draw_neural_net(ax, left, right, bottom, top, layer_sizes):
    '''
    parameters:
        - ax : matplotlib.axes.AxesSubplot
            The axes on which to plot the cartoon (get e.g. by plt.gca())
        - left : float
            The center of the leftmost node(s) will be placed here
        - right : float
            The center of the rightmost node(s) will be placed here
        - bottom : float
            The center of the bottommost node(s) will be placed here
        - top : float
            The center of the topmost node(s) will be placed here
        - layer_sizes : list of int
            List of layer sizes, including input and output dimensionality
    '''
    n_layers = len(layer_sizes)
    v_spacing = (top - bottom)/float(max(layer_sizes))
    h_spacing = (right - left)/float(len(layer_sizes) - 1)
    
    # Nodes
    for n, layer_size in enumerate(layer_sizes):
        layer_top = v_spacing*(layer_size - 1)/2. + (top + bottom)/2.
        layer_top2 = v_spacing*(layer_size - 1)/2. + (top*.95 + bottom)/2.
        layer_top3 = v_spacing*(layer_size - 1)/2. + (top + bottom*1.35)/2.
        
        if n == 0:
            for m in range(layer_size):
                if m == 3: 
                    dot = plt.Circle((n*h_spacing + left, layer_top2 - m*v_spacing), v_spacing/32,
                                color='w', ec='k', zorder=4)
                    ax.add_artist(dot)
                elif m == 4:
                    dot = plt.Circle((n*h_spacing + left, layer_top - m*v_spacing), v_spacing/32,
                                color='w', ec='k', zorder=4)
                    ax.add_artist(dot)
                elif m == 5:
                    dot = plt.Circle((n*h_spacing + left, layer_top3 - m*v_spacing), v_spacing/32,
                                color='w', ec='k', zorder=4)
                    ax.add_artist(dot)
                else: 
                    circle = plt.Circle((n*h_spacing + left, layer_top - m*v_spacing), v_spacing/4.,
                                color='#FF338D', ec='#FF338D', zorder=4)
                    ax.add_artist(circle)
        elif n == 1:
            for m in range(layer_size):
                if m == 6:
                    dot = plt.Circle((n*h_spacing + left, layer_top2 - m*v_spacing), v_spacing/32,
                                color='w', ec='k', zorder=4)
                    ax.add_artist(dot)
                elif m == 7:
                    dot = plt.Circle((n*h_spacing + left, layer_top - m*v_spacing), v_spacing/32,
                                color='w', ec='k', zorder=4)
                    ax.add_artist(dot)
                elif m == 8:
                    dot = plt.Circle((n*h_spacing + left, layer_top3 - m*v_spacing), v_spacing/32,
                                color='w', ec='k', zorder=4)
                    ax.add_artist(dot)
                else: 
                    circle = plt.Circle((n*h_spacing + left, layer_top - m*v_spacing), v_spacing/4.,
                                color='w', ec='k', zorder=4)
                    ax.add_artist(circle)
        else:
            for m in range(layer_size):
                circle = plt.Circle((n*h_spacing + left, layer_top - m*v_spacing), v_spacing/4.,
                                color='#4FDC98', ec='#4FDC98', zorder=4)
                ax.add_artist(circle)
                
    # Edges
    for n, (layer_size_a, layer_size_b) in enumerate(zip(layer_sizes[:-1], layer_sizes[1:])):
        layer_top_a = v_spacing*(layer_size_a - 1)/2. + (top + bottom)/2.
        layer_top_b = v_spacing*(layer_size_b - 1)/2. + (top + bottom)/2.
        if n == 0:
            for m in range(layer_size_a):
                if not m == 3 and not m == 4 and not m == 5:
                    for o in range(layer_size_b):
                        if not o == 6 and not o == 7 and not o == 8:
                            line = plt.Line2D([n*h_spacing + left, (n + 1)*h_spacing + left], 
                                [layer_top_a - m*v_spacing, layer_top_b - o*v_spacing], c='#b2b0b0')
                            ax.add_artist(line)
        else:
            for m in range(layer_size_a):
                if not m == 6 and not m == 7 and not m==8:
                    for o in range(layer_size_b):
                        line = plt.Line2D([n*h_spacing + left, (n + 1)*h_spacing + left],
                                  [layer_top_a - m*v_spacing, layer_top_b - o*v_spacing], c='#b2b0b0')
                        ax.add_artist(line)


# In[606]:


def axes(x, w, y, z, str1, str2, bbox_props, fs):
    ax.annotate(str1, xy=(x,w), xytext=(x, y), xycoords='axes fraction',
            fontsize=fs*1.5, ha='center', va='bottom', bbox=bbox_props,
            arrowprops=dict(arrowstyle='-[, widthB=3.5, lengthB=0.5'))
    ax.annotate(str2, xy=(x,z), xytext=(x, z), xycoords='axes fraction', 
            fontsize=fs*1.5, ha='center', va='bottom', bbox=bbox_props)


# In[613]:


fig= plt.figure(figsize=(12, 12))

ax = fig.add_subplot(111)
ax.tick_params(bottom=False, left=False, labelcolor='w')
# removing the spines
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.spines['bottom'].set_visible(False)
ax.spines['left'].set_visible(False)

fs = 8
bbox_props = dict(boxstyle='round,pad=0.,rounding_size=0.2', fc='w', ec='w', lw=2)
axes(0.1, 0.06, 0.05, 0.03, 'input layer', '784', bbox_props, fs)
axes(0.5, 0.06, 0.05, 0.03, 'hidden layer', 'n', bbox_props, fs)
axes(0.9, 0.06, 0.05, 0.03, 'output layer', '[0-9]', bbox_props, fs)
axes(0.5, 0.0, 0.0, 0.92, '', 'Tree-Layer Neural Network', bbox_props, fs=fs*2.25)

draw_neural_net(fig.gca(), .1, .9, .1, .9, [9, 15, 10])
fig.savefig('architecture.png',bbox_inches='tight')

