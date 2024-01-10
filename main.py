import pandas as pd
from lib.ann import ANN
from lib.layer import Layer
import numpy as np

data = pd.read_csv("./AKH_WQI.csv")
y = data["WQI"]
X = data.drop(columns=["WQI"])
print(X.__len__())

model = ANN()

layer1 = Layer(X.columns.__len__(), 2, "linear")
layer2 = Layer(2, 1, "linear")

model.addLayer(layer1)
model.addLayer(layer2)
# layer1.print()
# layer2.print()

model.train(X, y, epoch=3)
