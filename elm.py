import numpy as np
import csv
import random
# import plotly.graph_objects as go
import matplotlib.pyplot as plt
import pandas as pd


filename = 'Datasets/Iris/iris.data'
trainRatio = 0.8


def loadData(filename):
    data = pd.read_csv(filename, header=None)
    X = data.to_numpy(dtype=None, copy=False)
    
    return X

def splitData(data, trainRatio):
    # Find which classes are which and separate them
    notSpam = np.array(data[np.where(data[:, -1] == 0)]) 
    spam = np.array(data[np.where(data[:, -1] == 1)])

    # Shuffle the classed data
    np.random.shuffle(notSpam)
    np.random.shuffle(spam)

    # Take the first trainRatio data
    trainIndex0 = round(trainRatio*len(notSpam))
    trainData = notSpam[0:trainIndex0, :]
    testData = notSpam[trainIndex0:len(notSpam), :]

    # Take the first trainRatio data
    trainIndex1 = round(trainRatio * len(spam))
    trainData = np.concatenate((trainData, spam[0:trainIndex1, :]))
    testData = np.concatenate((testData, spam[trainIndex1:len(spam), :]))

    np.random.shuffle(trainData)
    np.random.shuffle(testData)

    return trainData, testData

def takePercentTrain(trainData, percentage):
    # Get percentage in float
    percentage /= 100.

    # Find which classes are which and separate them
    notSpam = np.array(trainData[np.where(trainData[:, -1] == 0)])
    spam = np.array(trainData[np.where(trainData[:, -1] == 1)])

    # Shuffle the classed data
    np.random.shuffle(notSpam)
    np.random.shuffle(spam)

    # Take the first percentage data
    trainIndex0 = round(percentage * len(notSpam))
    trainSplitData = notSpam[0:trainIndex0, :]

    # Take the first percentage data
    trainIndex1 = round(percentage * len(spam))
    trainSplitData = np.concatenate((trainSplitData, spam[0:trainIndex1, :]))

    # Shuffle again
    np.random.shuffle(trainSplitData)

    return trainSplitData


def main():
    # Load the data into features and labels
    X = loadData(filename)
    # print(X[np.where(X[:, -1] == 'Iris-versicolor')][:,-1])
    # X[np.where(X[:, -1] == 'Iris-setosa')][:,-1] = 2
    # X[np.where(X[:, -1] == 'Iris-virginica')][:,-1] = 3
    np.put(X, [np.where(X[:, -1] == 'Iris-versicolor')], 1)
    print(X[np.where(X[:, -1] == 'Iris-versicolor')])
    # 'Iris-setosa'
    # 'Iris-virginica'
    
    

if __name__ == '__main__':
    main()