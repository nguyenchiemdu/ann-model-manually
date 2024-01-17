import numpy as np
from lib.activation import getActivationFunction


class Layer:
    """
    Represents a layer in an artificial neural network.

    Attributes:
    - input_nodes (int): The number of input nodes in the layer.
    - output_nodes (int): The number of output nodes in the layer.
    - activation (str): The activation function to be used in the layer.
    - index (int): The index of the layer.
    - learning_rate (float): The learning rate for updating the layer's parameters.

    Methods:
    - __init__(self, input_nodes, output_nodes, activation, index=1, learning_rate=00.001): Initializes the Layer object.
    - calculate_n(self, a): Calculates the weighted sum of inputs and biases.
    - forward(self, a): Performs forward propagation for the layer.
    - predict_only(self, a): Performs forward propagation without storing intermediate values.
    - linear_backward(self, dn): Performs backward propagation for the linear part of the layer.
    - backward(self, da): Performs backward propagation for the layer.
    - update_params(self, dw, db): Updates the layer's parameters.
    - change_w(self, w): Changes the weight matrix of the layer.
    - change_b(self, b): Changes the bias vector of the layer.
    - print(self): Prints the weight matrix and bias vector of the layer.
    """

    def __init__(
        self, input_nodes, output_nodes, activation, index=1, learning_rate=00.001
    ):
        self.w = np.random.randn(input_nodes, output_nodes) * 0.01
        self.activation = getActivationFunction(activation)
        self.b = np.zeros(output_nodes)
        self.b.resize(1, output_nodes)
        self.index = index
        self.learning_rate = learning_rate

    def calculate_n(self, a):
        return np.dot(self.w, a.T) + self.b

    def forward(self, a):
        self.a_prev = a
        self.n = np.dot(a, self.w) + self.b
        self.a = self.activation.cal(self.n)
        return self.a

    def predict_only(self, a):
        n = np.dot(a, self.w) + self.b
        return self.activation.cal(n)

    def linear_backward(self, dn):
        dn = np.array(dn)
        if len(dn.shape) == 1:
            dn.resize(1, dn.shape[0])
        m = self.a_prev.shape[0]
        dw = np.dot(self.a_prev.T, dn) / m

        db = np.sum(dn, axis=0, keepdims=True) * 2 / m
        da = np.dot(dn, self.w.T)
        return da, dw, db

    def backward(self, da):
        dn = da * self.activation.derivative(self.n)
        da_prev, dw, db = self.linear_backward(dn)
        return da_prev, dw, db

    def update_params(self, dw, db):
        self.w = self.w + self.learning_rate * dw
        self.b = self.b + self.learning_rate * db

    def change_w(self, w):
        self.w = np.array(w)
        if self.w.shape.__len__() == 1:
            self.w.resize((self.w.shape[0], 1))

    def change_b(self, b):
        self.b = b
        self.b.resize((1, b.shape[0]))

    def print(self):
        print("w: \n", self.w)
        print(self.w.shape)
        print("b: \n", self.b)
        print(self.b.shape)
