import matplotlib
import matplotlib.pyplot as plt
import pandas as pd

filename = 'Datasets/MNIST/mnist_train.csv'

def loadData(path):
    return pd.read_csv(path, header=None)

XTrain = loadData(filename)
# XTrain = XTrain.to_numpy
# print(XTrain.shape)
someDigit = XTrain.iloc[36000, :-1]
sdImg = someDigit.reshape(28, 28)

plt.imshow(sdImg)
plt.axis("off")
plt.show()