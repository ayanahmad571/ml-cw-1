# classifier.py
# Lin Li/26-dec-2021

# Classifier extended and implemented by Ayan Ahmad
# K-Number: 19002255


# Import
import numpy as np
import math
from statistics import mode



### Start Global Functions

# Calculates mathematical distances between two lists of numbers of equal size
def listDiffCalc(listA, listB):
    """
    :param listA: List A, list of values of n features and k samples
    :param listB: List B, list of values of n features and k samples
    :return: distance between the two lists as a float 
    """
    
    listA = np.asarray(listA) 
    differenceLists = (listA - listB)
    squareLists = differenceLists ** 2
    sumElements = np.sum(squareLists) 
    sqrtValue = np.sqrt(sumElements)
    return sqrtValue

# Taken two arrays as input and shuffles both the arrays in unison
# https://stackoverflow.com/questions/4601373/better-way-to-shuffle-two-numpy-arrays-in-unison
def shuffleArray(listA, listB):
    """
    :param listA: List A, list of values of n features and k samples
    :param listB: List B, list of labels of k samples
    :return: Two randomly shuffled lists of the same number of samples, shuffled in unison.
    """
    listA = np.asarray(listA)
    listB = np.asarray(listB)
    assert len(listA) == len(listB)
    shuffledIndexes = np.random.permutation(len(listA)) # returns a list of shuffled indexes
    return listA[shuffledIndexes], listB[shuffledIndexes] # returns arrays with the aforementioned shuffled list as indexes

# Shuffles the Features and Labels in unison and returns a desired train and test split of the data
def trainTestSplit(X, y, splitSize):
    """
    :param X: list of values of n features and k samples
    :param y: list of labels of k samples
    :param splitSize: the percentage of samples to be used as training data
    :return: 2 arrays for the feature values and 2 arrays for labelled values of k samples.
    """
    shuffleX, shuffleY = shuffleArray(X,y) # shuffle the arrays each time to ensure fresh sample is split
    XTrain = list()
    XTest = list()
    yTrain = list()
    yTest = list()

    trainLen = math.floor(splitSize * len(shuffleX)) # Approximation of the number of samples, since it has to be an int

    # loop trainlen many times and add data to both train and test sets
    for i in range(trainLen):
        XTrain.append(shuffleX[i])
        yTrain.append(shuffleY[i])
        newI = i + trainLen
        
        # Key Assumption for the below condition, training set is always more than test set
        # the current iteration index is added to the total size of the dataset.
        # hence that accesses the test set

        if(newI < len(shuffleX)):
            XTest.append(shuffleX[newI])
            yTest.append(shuffleY[newI])

    return XTrain, XTest, yTrain, yTest

# Calculates the gini impurity of a given features provided all data is labelled
def giniImpurity(data):
    """
    :param data: list of values of n features and k samples
    :return: gini impurity of the given branch, of a particular feature
    """
    
    lables = {0,1,2,3}
    dataLen = len(data)

    # If there is no data, then safeguard against zero division.
    if dataLen == 0:
        dataLen = 1

    # list of number of elements of each label divided by total number of elements, squared
    giniValues = [(data.count(l) / dataLen) ** 2 for l in lables] 
    giniImp =1-sum(giniValues) 

    return giniImp

# Calculates the gini impurity of a given features provided all data is labelled
def giniFeat(X, y, featIndex):
    """
    :param X: list of values of n features and k samples
    :param y: list of labels of k samples
    :param featIndex: the percentage of samples to be used as training data
    :return: gini impurity value of the current feature
    """
    zeroVals = list() # For when feature value is 0
    oneVals = list() # For when feature value is 1
    
    for rowIndex in range(len(X)):
        atRowFeatureVal = X[rowIndex][featIndex]
        atRowLabel = y[rowIndex]
        if(atRowFeatureVal == 0):
            zeroVals.append(atRowLabel)
        else:
            oneVals.append(atRowLabel)

    # Calculate Individual impurities for each branch
    zeroGiniImp = giniImpurity(zeroVals)
    oneGiniImp = giniImpurity(oneVals)
    
    # Weighted combination of the individual gini values
    # We could just do len(feat) but if in case some data is missing
    featureLen = len(zeroVals) + len(oneVals)
    totalGini = (len(zeroVals) * zeroGiniImp + len(oneVals) * oneGiniImp)/featureLen # Gini Formula
    return totalGini

# Calculates the gini value of all features given a complete labelled dataset
def giniAll(X,y):
    """
    :param X: list of values of n features and k samples
    :param y: list of labels of k samples
    :return: list of gini values for all features of a dataset
    """
    numFeatures = len(X[0]) 
    
    featureGini = list()

    # for each feature calculate the gini and append to the array
    for fIndex in range(numFeatures):
        featureGini.append(giniFeat(X,y,fIndex))
    
    return featureGini

# Shuffles the dataset and returns two thirds of the data
def bootstrapData(X,y):
    """
    :param X: list of values of n features and k samples
    :param y: list of labels of k samples
    :return: Returns two arrays, both split and shuffled versions of the dataset
    """
    XBts, _, yBts, _ = trainTestSplit(X, y, 0.66) # We do not need the test sets here
    return XBts, yBts

# Calculates the new dataset values for both the Left(Zero) and Right(One) Branches
def split(X,y,feat):
    zeroValsX = list()
    zeroValsY = list()
    oneValsX = list()
    oneValsY = list()
    
    for rowIndex in range(len(X)):
       
        atIndexVal = X[rowIndex][feat]
        atIndexLabel = y[rowIndex]
        if(atIndexVal == 0):
            zeroValsX.append(X[rowIndex])
            zeroValsY.append(atIndexLabel)
        else:
            oneValsX.append(X[rowIndex])
            oneValsY.append(atIndexLabel)
    
    return zeroValsX, zeroValsY, oneValsX, oneValsY

### End Global Functions


class Node:
    def __init__(
        self, feature=None, left=None, right=None, *, value=None
    ):
        self.feature = feature
        self.left = left
        self.right = right
        self.value = value

    def isLeafNode(self):
        return self.value is not None

class KNNClassifier:
    def __init__(self):
        # print("Init KNN ", self)
        self.k = 5
    
    def predict(self, XTrain, yTrain, data):
        # Compute distances between x and all examples in the training set        
        eDiff = [listDiffCalc(data, xTrain) for xTrain in XTrain]

        # Sort by distance and return indices of the first k neighbors        
        minDiffIndexes = np.argsort(eDiff)
        
        # https://stackoverflow.com/questions/5234090/how-to-take-the-first-n-items-from-a-generator-or-list
        kIndexes = minDiffIndexes[: self.k]
        
        # Extract the labels of the k nearest neighbor training samples
        labelledIndexes = [yTrain[i] for i in kIndexes]
        yPred = mode(labelledIndexes)
        return yPred

class DTClassifier:
    def __init__(self, maxDepth = 100, numFeats = 25):
        # print("Init DT-CLASS", self)
        self.maxDepth = maxDepth
        self.numFeats = numFeats
        self.root = None
    
    
    def fit(self, XTrain, yTrain):
        self.root = self.buildTree(XTrain, yTrain)
        pass

    def buildTree(self, X, y, depth=0, pluralityParent = 0):
        numUniqueLabels = len(np.unique(y))
        
        #stop-condition
        if(depth >= self.maxDepth or numUniqueLabels < 2):
            if len(y)<1:
                return Node(value = pluralityParent) # TODO: Add some parent code   
            return Node(value = mode(y))
        
        yMode = mode(y)
        
        giniVals = giniAll(X, y)
        giniValsNp = np.asarray(giniVals)
        lowestGiniFeatureIndex = np.argmin(giniValsNp)
        zeroValsX, zeroValsY, oneValsX, oneValsY = split(X, y, lowestGiniFeatureIndex)
        left = self.buildTree(zeroValsX, zeroValsY, depth + 1, yMode)
        right = self.buildTree(oneValsX, oneValsY, depth + 1, yMode)

        return Node(lowestGiniFeatureIndex, left, right)

    def search(self, x, node):
        if node.isLeafNode():
            return node.value

        if x[node.feature] < 1:
            return self.search(x, node.left)
        return self.search(x, node.right)


    def predict(self, data):
        yPred = self.search(data, self.root)
        return yPred

class RFClassifier:
    def __init__(self, numTrees=20, maxDepth=100, numFeats=None):
        # print("Init RF-CLASS", self)
        self.numTrees = numTrees
        self.maxDepth = maxDepth
        self.numFeats = numFeats
        self.trees = []
    
    def fit(self, X, y):
        self.trees = []
        for t in range(self.numTrees):
            print("Processing Tree", t)
            dt = DTClassifier(maxDepth=self.maxDepth, numFeats=self.numFeats)
            XBts, yBts = bootstrapData(X, y)
            dt.fit(XBts, yBts)
            self.trees.append(dt)
    
    def predict(self, data):
        yPredsAll = [dt.predict(data) for dt in self.trees]
        print(yPredsAll)
        yPred = mode(yPredsAll)
        return yPred

class Classifier:
    def __init__(self):
        self.knn = KNNClassifier()
        self.dt = DTClassifier()
        self.rf = RFClassifier()

    def reset(self):
        print("RESET-ALL", self)
        self.knn = None
        self.dt = None
        self.rf = None
    
    def fit(self, data, target):
        print("FIT", data, target)
        self.X = data
        self.y = target
        self.XTrain, self.XTest, self.yTrain, self.yTest = trainTestSplit(data, target, 0.8)

        self.dt.fit(self.XTrain, self.yTrain)
        self.rf.fit(self.XTrain, self.yTrain)
    

    def predict(self, data, legal=None):
        print("_______________")
        ret = [
            self.knn.predict(self.XTrain, self.yTrain, data), 
            self.dt.predict(data), 
            self.rf.predict(data)
        ]
        print("Options:", ret)
        return mode(ret)
