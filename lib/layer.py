import numpy as np

from lib.activation import getActivationFunction
from lib.utils import cal_dot_matrix

# a => x-N
# w => N-M  với n là số node trên input layer
# b => 1-M với m là số node trên output layer


class Layer:
    def __init__(self, input_nodes, output_nodes, activation, learning_rate=0.1):
        self.w = np.random.randn(input_nodes, output_nodes)
        self.activation = getActivationFunction(activation)
        self.b = np.zeros(output_nodes)
        self.b.resize(1, output_nodes)
        self.learning_rate = learning_rate

    def calculate_n(self, a):
        return np.dot(self.w, a.T) + self.b

    def forward(self, a):
        self.a_prev = a
        self.n = np.dot(a, self.w) + self.b
        self.a = self.activation.cal(self.n)
        return self.a

    def linear_backward(self, dn):
        # print("dn: ", dn)
        dn = np.array(dn)
        if len(dn.shape) == 1:
            dn.resize(1, dn.shape[0])
        # print("aT", self.a_prev)
        dw = np.dot(self.a_prev, dn.T)
        # print("dw", dw)

        db = np.sum(dn, axis=1, keepdims=True)

        da = (np.dot(self.w.T, dn.T)).T
        return da, dw, db

    def update_params(self, dw, db):
        # print("w: ", self.w)
        # print("db: \n", db)
        # print("b:\n", self.b)
        self.w = self.w + self.learning_rate * dw
        self.b = self.b + self.learning_rate * db.T

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
