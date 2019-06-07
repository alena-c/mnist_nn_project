#!/usr/bin/env python
# coding: utf-8

# In[6]:


#!/usr/bin/env python
# coding: utf-8

# In[5]:

"""
Loads the MNIST data and my data.
"""

import tensorflow as tf
import input_data
import numpy as np
import cv2
import sys
import math
from scipy import ndimage

# get the center_of_mass mass
def getBestShift(img):
    cy,cx = ndimage.measurements.center_of_mass(img)
    rows,cols = img.shape
    shiftx = np.round(cols/2.0-cx).astype(int)
    shifty = np.round(rows/2.0-cy).astype(int)
    return shiftx,shifty

# shifts the image in the given directions
def shift(img,sx,sy):
    rows,cols = img.shape
    M = np.float32([[1,0,sx],[0,1,sy]])
    shifted = cv2.warpAffine(img,M,(cols,rows))
    return shifted  

def preprocess(input_images):
    # create an array where we can store our 4 pictures
    images = np.zeros((6,784))
    # and the correct values
    correct_vals = np.zeros((6,10))
    
    # testing my images
    i = 0
    for no in input_images:  #[2,6,7,9]:

        gray = cv2.imread("f/d_"+ str(no) +".jpg", 0)
        # rescale
        gray = cv2.resize(255-gray, (28, 28))

        # better black and white version
        (thresh, gray) = cv2.threshold(gray, 128, 255,
                                    cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        # fit the images into a 20x20 pixel box
        # remove every row and column at the sides of the image which are completely black
        while np.sum(gray[0]) == 0:
            gray = gray[1:]

        while np.sum(gray[:,0]) == 0:
            gray = np.delete(gray,0,1)

        while np.sum(gray[-1]) == 0:
            gray = gray[:-1]

        while np.sum(gray[:,-1]) == 0:
            gray = np.delete(gray,-1,1)

        rows,cols = gray.shape

        # resize our outer box to fit it into a 20x20 box. We need a resize factor for this
        if rows > cols:
            factor = 20.0/rows
            rows = 20
            cols = int(round(cols*factor))
            gray = cv2.resize(gray, (cols,rows))
        else:
            factor = 20.0/cols
            cols = 20
            rows = int(round(rows*factor))
            gray = cv2.resize(gray, (cols, rows))

        # But at the end we need a 28x28 pixel image so we add the missing black
        # rows and columns using the np.lib.pad function which adds 0s to the sides
        colsPadding = (int(math.ceil((28-cols)/2.0)),int(math.floor((28-cols)/2.0)))
        rowsPadding = (int(math.ceil((28-rows)/2.0)),int(math.floor((28-rows)/2.0)))
        gray = np.lib.pad(gray,(rowsPadding,colsPadding),'constant')

        shiftx,shifty = getBestShift(gray)
        shifted = shift(gray,shiftx,shifty)
        gray = shifted

        # save new images
        cv2.imwrite("f/image_"+ str(no) +"s.png", gray)

        """
        all images in the training set have an range from 0-1
        and not from 0-255 so we divide our flatten images
        (a one dimensional vector with our 784 pixels)
        to use the same 0-1 based range
        """
        flatten = gray.flatten() / 255.0

        """
        we need to store the flatten image and generate
        the correct_vals array
        correct_val for the first digit (9) would be
        [0,0,0,0,0,0,0,0,0,1]
        """
        images[i] = flatten
        correct_val = np.zeros((10))
        correct_val[no] = 1
        correct_vals[i] = correct_val
        i += 1
    
    return images
         

# %load mnist_loader.py

import pickle
import gzip
import numpy as np

def load_data():
    f = gzip.open('mnist.pkl.gz', 'rb')
    training_data, validation_data, test_data = pickle.load(f, encoding="latin1")
    f.close()
    return (training_data, validation_data, test_data)

def load_data_wrapper():
    tr_d, va_d, te_d = load_data()
    training_inputs = [np.reshape(x, (784, 1)) for x in tr_d[0]]
    training_results = [vectorized_result(y) for y in tr_d[1]]
    training_data = list(zip(training_inputs, training_results))
    validation_inputs = [np.reshape(x, (784, 1)) for x in va_d[0]]
    validation_data = list(zip(validation_inputs, va_d[1]))
    test_inputs = [np.reshape(x, (784, 1)) for x in te_d[0]]
    test_data = list(zip(test_inputs, te_d[1]))
    
    input_images = ([2, 7, 6, 9, 8, 5])
    images = preprocess(input_images)
    new_test_inputs = [np.reshape(x, (784, 1)) for x in images]
    new_test_data = zip(new_test_inputs, input_images)
    
    return (training_data, validation_data, new_test_data)

def vectorized_result(j):
    e = np.zeros((10, 1))
    e[j] = 1.0
    return e


