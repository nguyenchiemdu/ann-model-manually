from lib.ann import ANN
from lib.layer import Layer
import numpy as np

layer1 = Layer(2, 2, "linear")


input = np.array([[0, 1], [0, 1]]).reshape(2, 2)
output = np.array([[1, 0], [1, 0]]).reshape(2, 2)
w1 = np.array([[-1, 0], [0, 1]])
b1 = np.array([1, 1])
layer1.change_w(w1)
layer1.change_b(b1)
# layer1.calculate_n(input)
# layer1.print()

# a1 = layer1.forward()
# print("a1: ", a1)

layer2 = Layer(2, 2, "linear")

w2 = np.array([[1, 0], [-1, 1]])
b2 = np.array([1, 1])

layer2.change_w(w2)
layer2.change_b(b2)
# layer2.calculate_n(a1)
layer2.print()
# a2 = layer2.forward()
# print("a2: ", a2)


ann = ANN()

ann.addLayer(layer1)
ann.addLayer(layer2)


epochs = 23
ann.train(input, output, epoch=epochs)
