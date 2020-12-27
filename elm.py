import numpy as np
import csv
import matplotlib.pyplot as plt
import pandas as pd


filename = 'Datasets/Iris/iris.data'
trainRatio = 0.8

# Loads the data given a filename
def loadData(filename):
    # Create the dataframe then convert it to a
    # numpy array
    data = pd.read_csv(filename, header=None)
    data = data.to_numpy(dtype=None, copy=False)
    
    return data

# Splits the data according to the given trainRatio
def splitData(data, trainRatio):
    # Find which classes are which and separate them
    setosa = data[np.where(data[:, -1] == 'Iris-setosa')]
    setosa[:,-1] = 1
    versicolor = data[np.where(data[:, -1] == 'Iris-versicolor')]
    versicolor[:,-1] = 2
    virginica = data[np.where(data[:, -1] == 'Iris-virginica')]
    virginica[:,-1] = 3

    # Shuffle the classed data
    np.random.shuffle(setosa)
    np.random.shuffle(versicolor)
    np.random.shuffle(virginica)

    # Take the first trainNum of each class
    trainNum = int(trainRatio*setosa.shape[0])
    trainSet = setosa[:trainNum,:]
    trainSet = np.concatenate((trainSet, versicolor[:trainNum,:]), axis=0)
    trainSet = np.concatenate((trainSet, virginica[:trainNum,:]), axis=0)

    # Take the rest as test data
    testSet = setosa[trainNum:,:]
    testSet = np.concatenate((testSet, versicolor[trainNum:,:]), axis=0)
    testSet = np.concatenate((testSet, virginica[trainNum:,:]), axis=0)

    # Shuffle the train and test sets
    np.random.shuffle(trainSet)
    np.random.shuffle(testSet)

    trainSet = trainSet.astype(np.float32)
    testSet = testSet.astype(np.float32)

    return trainSet, testSet

# Activation function
def activate(z, activation):
    if activation == 'sigmoid':
        return 1/(1+np.exp(-z))
    elif activation == 'relu':
        return np.maximum(0.0,z)

# Compute the accuracy of predictions
def accuracy(YPred, YTrue):
    sumAcc = 0
    if(YPred.size == YTrue.size):
        for i in np.arange(YTrue.size):
            if(YPred[i] == YTrue[i]):
                sumAcc += 1
    return (sumAcc/YTrue.size) * 100

# MAIN FUNCTION
def main():
    # Load the data
    data = loadData(filename)

    # Shuffle and split the data
    XTrain, XTest = splitData(data, trainRatio)
    print(XTrain.shape)
    print(XTest.shape)

    # Separate the data into features and their labels
    YTrain = XTrain[:,-1]
    XTrain = XTrain[:,:-1]
    YTest = XTest[:,-1]
    XTest = XTest[:,:-1]

    # Now the data is prepped, we can train and 
    # test the single-layer ELM
    nHidden = 2
    activation = 'relu'
    np.random.seed(0)
    inputWeights = np.random.rand(XTrain.shape[1], nHidden)
    inputBias = np.random.rand(nHidden, 1)

    # Compute the forward calculation
    z = np.matmul(inputWeights.T, XTrain.T) + inputBias
    z = z.T

    # Activate the computation
    H = activate(z, activation)
    H.astype(np.double)
    print("H Shape:", H.shape)

    # Take the pseudoinverse of H and multiply
    # it by the labels
    Beta = np.matmul(np.linalg.pinv(H), YTrain)
    print("Beta Shape:", Beta.shape)

    # With this Beta, we should be able to carry out 
    # the classification task on the test data
    z = np.dot(inputWeights.T, XTest.T) + inputBias
    z = z.T
    H = activate(z, activation)
    print(H.shape)

    YPred = np.matmul(H, Beta)

    # For multi-class classification, it is important that 
    # we threshold the data in some manner to select
    # its predicted label
    for i in range(1,4):
        pos = np.argwhere(np.abs(YPred-i) < 0.5)
        for j in pos:
            YPred[j] = i
    pos = np.argwhere(YPred < 1)
    for j in pos:
            YPred[j] = 1
    pos = np.argwhere(YPred > 3)
    for j in pos:
            YPred[j] = 3
    print("Predictions Shape:", YPred.shape)
    acc = accuracy(YPred, YTest)
    print(acc)

if __name__ == '__main__':
    main()