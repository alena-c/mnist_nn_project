"""
mnist_loader
~~~~~~~~~~~~
Loads the MNIST image data. Returns the transformed structure of the MNIST data.
Called by the neural network code.
"""

#### Libraries
# Standard library
import pickle
import gzip

# Third-party libraries
import numpy as np

# Returns MNIST data as a tuple of training, validation and testing data,
# each of which in turn is a tuple of 2 ndarrays (pixel values of the images 
# and their lables)
def load_data():
    f = gzip.open('mnist.pkl.gz', 'rb')
    training_data, validation_data, test_data = pickle.load(f, encoding="latin1")
    f.close()
    return (training_data, validation_data, test_data)

# Data transformation:
def load_data_wrapper():
    tr_d, va_d, te_d = load_data()
    training_inputs = [np.reshape(x, (784, 1)) for x in tr_d[0]]
    training_results = [vectorized_result(y) for y in tr_d[1]]
    training_data = list(zip(training_inputs, training_results))
    validation_inputs = [np.reshape(x, (784, 1)) for x in va_d[0]]
    validation_data = list(zip(validation_inputs, va_d[1]))
    test_inputs = [np.reshape(x, (784, 1)) for x in te_d[0]]
    test_data = list(zip(test_inputs, te_d[1]))
    return (training_data, validation_data, test_data)

# Converts a label [0-9] into a 10-dimensional unit vector with 
# 1 at the ith position and 0s elsewhere ((e.g., [0,0,1,0,0,0,0,0,0,0] as 2).
def vectorized_result(i):
    label = np.zeros((10, 1))
    label[i] = 1.0
    return label
