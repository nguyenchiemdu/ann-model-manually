import numpy as np


def cal_dot_matrix(a, b):
    a = np.array(a)
    b = np.array(b)
    # check if a is a vector
    if len(a.shape) != 1 or len(b.shape) != 1:
        return np.dot(a.T, b)

    # convert a to a column vector, b to a row vector and then do the dot product
    a = a.reshape(a.shape[0], 1)
    b = b.reshape(1, b.shape[0])
    return np.dot(a, b)
