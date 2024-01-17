import numpy as np
from abc import ABC, abstractclassmethod


class ActivationFunction(ABC):
    def __init__(self):
        pass

    def cal(self, x):
        pass

    @abstractclassmethod
    def derivative(self, x):
        pass


class Relu(ActivationFunction):
    def __init__(self):
        pass

    def cal(self, x):
        return np.maximum(0, x)

    def derivative(self, x):
        return np.where(x > 0, 1, 0)


class Linear(ActivationFunction):
    def __init__(self):
        pass

    def cal(self, x):
        return x

    def derivative(self, x):
        return np.ones_like(x)


class Sigmoid(ActivationFunction):
    def __init__(self):
        pass

    def cal(self, x):
        return 1 / (1 + np.exp(-x))

    def derivative(self, x):
        return x * (1 - x)


def getActivationFunction(name):
    """
    Returns an instance of the activation function based on the given name.

    Parameters:
    name (str): The name of the activation function.

    Returns:
    object: An instance of the activation function.

    Raises:
    Exception: If the given name is not recognized.
    """
    if name == "relu":
        return Relu()
    elif name == "linear":
        return Linear()
    elif name == "sigmoid":
        return Sigmoid()
    else:
        raise Exception("Unknown activation function: {}".format(name))
