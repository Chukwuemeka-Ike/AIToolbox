import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedShuffleSplit

HOUSING_PATH = "Datasets/Hands-On ML"

def loadData(path):
    csvPath = os.path.join(path, "housing.csv")
    return pd.read_csv(csvPath)

def splitData(data, testRatio, seed):
    np.random.seed(seed)
    shuffledIndices = np.random.permutation(len(data))
    testSetSize = int(len(data) * testRatio)
    testIndices = shuffledIndices[:testSetSize]
    trainIndices = shuffledIndices[testSetSize:]
    return data.iloc[trainIndices], data.iloc[testIndices]

def main():
    # Load the data located in HOUSING_PATH
    housingData = loadData(HOUSING_PATH)

    # # Understanding the data
    # print(housingData.info())
    # print(housingData["ocean_proximity"].value_counts())
    # housingData.hist(bins=50, figsize=(7,7))
    # plt.show()

    # Split into train and test set
    # trainSet, testSet = splitData(housingData, 0.2, 42)
    # trainSet, testSet = train_test_split(housingData, test_size=0.2, random_state=42)
    split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)

    housingData["income_cat"] = np.ceil(housingData["median_income"]/1.5)
    housingData["income_cat"].where(housingData["income_cat"] < 5, 5.0, inplace=True)
    for trainIndex, testIndex in split.split(housingData, housingData["income_cat"]):
        stratTrainSet = housingData.loc[trainIndex]
        stratTestSet = housingData.loc[testIndex]
    print(housingData["income_cat"].value_counts()/len(housingData))

    # Remove the "income_cat" attribute so the data is back to original state
    for set in (stratTrainSet, stratTestSet):
        set.drop(["income_cat"], axis=1, inplace=True)
    
    # Visualize the Geographical Data
    housingData = stratTrainSet.copy()
    housingData.plot(kind="scatter", x="longitude", y="latitude", alpha=0.4,
        s=housingData["population"]/100, label="population", c="median_house_value",
        cmap=plt.get_cmap("jet"), colorbar=True)
    plt.legend()
    plt.show()

    # Fill in the median total_bedrooms value wherever that feature is missing
    median = housingData["total_bedrooms"].median()
    housingData["total_bedrooms"].fillna(median)
    



if __name__ == "__main__":
    main()