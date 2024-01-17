import numpy as np
from sklearn.metrics import mean_squared_error


class ANN:
    """
    Artificial Neural Network (ANN) class.

    This class represents a feedforward neural network model.
    It consists of multiple layers and supports forward and backward propagation.

    Attributes:
        layers (list): List of layers in the neural network.

    Methods:
        addLayer(layer): Adds a layer to the neural network.
        forward(): Performs forward propagation through the neural network.
        backward(): Performs backward propagation through the neural network.
        cal_dal(): Calculates the derivative of the activation layer.
        train(X, y, epoch, cal_mse=False): Trains the neural network on the given data.
        calculate_mse(): Calculates the Mean Squared Error (MSE) of the predictions.
        predict(x): Makes predictions using the trained neural network.

    """

    def __init__(self):
        self.layers = []

    def addLayer(self, layer):
        self.layers.append(layer)

    def forward(self):
        """
        Performs forward propagation through the neural network.

        This method passes the input data through each layer in the network,
        updating the activations and predictions along the way.

        """
        a = self.x
        for layer in self.layers:
            a = layer.forward(a)
        self.predicted = self.layers[-1].a

    def backward(self):
        """
        Performs backward propagation through the neural network.

        This method calculates the gradients of the loss with respect to the
        parameters of each layer, and updates the parameters accordingly.

        """
        da = self.cal_dal()
        for layer in reversed(self.layers):
            da, dw, db = layer.backward(da)
            layer.update_params(dw, db)

    def cal_dal(self):
        """
        Calculates the derivative of the activation layer.

        Returns:
            res (numpy.ndarray): The derivative of the activation layer.

        """
        res = -1 * (np.array(self.predicted) - self.y)
        return res

    def train(self, X, y, epoch, cal_mse=False):
        """
        Trains the neural network on the given data.

        Args:
            X (numpy.ndarray): The input data.
            y (numpy.ndarray): The target labels.
            epoch (int): The number of training epochs.
            cal_mse (bool): Whether to calculate the Mean Squared Error (MSE) during training.

        """
        list_mse = []
        self.x = X
        self.y = np.array(y)
        if self.y.shape.__len__() == 1:
            self.y.resize(self.y.shape[0], 1)
        for _ in range(epoch):
            self.forward()
            if cal_mse:
                mse = self.calculate_mse()
                list_mse.append(mse)
            self.backward()
        return list_mse

    def calculate_mse(self):
        """
        Calculates the Mean Squared Error (MSE) of the predictions.

        """
        mse = mean_squared_error(self.y, self.predicted)
        print("MSE:", mse)
        return mse

    def predict(self, x):
        """
        Makes predictions using the trained neural network.

        Args:
            x (numpy.ndarray): The input data.

        Returns:
            numpy.ndarray: The predicted output.

        """
        a = x
        for layer in self.layers:
            a = layer.predict_only(a)
        return a
