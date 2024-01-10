import numpy as np
from sklearn.metrics import mean_squared_error


class ANN:
    def __init__(self):
        self.layers = []

    def addLayer(self, layer):
        self.layers.append(layer)

    def forward(self):
        a = self.x
        for layer in self.layers:
            a = layer.forward(a)
        self.predicted = self.layers[-1].a
        # print("Predicted: ", self.predicted)
        print("MSE:", mean_squared_error(self.y, self.predicted))
        return self.predicted

    def backward(self):
        da = self.cal_dal()
        # print("da:", da)
        for layer in reversed(self.layers):
            da, dw, db = layer.linear_backward(da)
            layer.update_params(dw, db)

    def cal_dal(self):
        res = -1 * (np.array(self.predicted) - self.y)
        return res

    def train(self, X, y, epoch):
        self.x = X
        self.y = np.array(y)
        if self.y.shape.__len__() == 1:
            self.y.resize(self.y.shape[0], 1)
        for _ in range(epoch):
            self.forward()
            self.backward()
