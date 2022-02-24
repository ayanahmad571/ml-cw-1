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
    """
    :param X: list of values of n features and k samples
    :param y: list of labels of k samples
    :param feat: index of the feature to split on
    :return: Returns 4 arrays, 2 contain X and y datasets of the left branch and 2 contain that of the right branch
    """
    zeroValsX = list()
    zeroValsY = list()
    oneValsX = list()
    oneValsY = list()
    
    for rowIndex in range(len(X)):
       
        atRowFeatureVal = X[rowIndex][feat]
        atRowLabel = y[rowIndex]
        
        # A feature has two values, either 1 or 0
        if(atRowFeatureVal == 0):
            zeroValsX.append(X[rowIndex])
            zeroValsY.append(atRowLabel)
        else:
            oneValsX.append(X[rowIndex])
            oneValsY.append(atRowLabel)
    
    return zeroValsX, zeroValsY, oneValsX, oneValsY

### End Global Functions

# Class Definition for a Node, to be used in Decision Trees and Random Forest
class Node:
    def __init__(self, feature=None, left=None, right=None, value=None):
        self.feature = feature
        self.left = left
        self.right = right
        self.value = value

    # A node is a leaf node if it has a value
    def isLeafNode(self):
        """
        :param self: class information
        :return: boolean, true if it has a non null value
        """
        return self.value is not None

# KNN Classifier Class - Current Configuration = 5NN
class KNNClassifier:
    def __init__(self):
        self.k = 5 

    # Returns the predicted label for a given feature set
    def predict(self, XTrain, yTrain, data):
        """
        :param self: class information
        :param XTrain: list of values of n features and k samples
        :param yTrain: list of labels of k samples
        :param data: list of values of n features that have to be predicted (data to label)
        :return: predicted value of the given features values 
        """

        # Compute distances between x and all examples in the training set        
        featDiff = [listDiffCalc(data, xTrain) for xTrain in XTrain]

        # Sort the array by distance and return their indices  
        minDiffFeatures = np.argsort(featDiff)
        
        # Get the indices of the first 5 values, (since they are the lowest values)
        # https://stackoverflow.com/questions/5234090/how-to-take-the-first-n-items-from-a-generator-or-list
        kFeatures = minDiffFeatures[: self.k]
        
        # Get labels of the k(5) nearest neighbor training data values
        labelledFeatures = [yTrain[i] for i in kFeatures]

        yPred = mode(labelledFeatures)
        return yPred

# Decision Tree Classifier Class - Current Configuration = 100 max depth
class DTClassifier:
    def __init__(self, maxDepth = 100):
        self.maxDepth = maxDepth
        self.root = None
    
    # Builds the tree based on the provided training dataset
    def fit(self, XTrain, yTrain):
        """
        :param self: class information
        :param XTrain: list of values of n features and k samples
        :param yTrain: list of labels of k samples
        """
        
        self.root = self.buildTree(XTrain, yTrain)
        pass

    # Recursive function responsible for building the nodes and linking them
    def buildTree(self, X, y, depth=0, pluralityParent = 0):
        """
        :param self: class information
        :param X: list of values of n features and k samples, is sent down recursively
        :param y: list of labels of k samples, is sent down recursively
        :param depth: the current depth of the node
        :param pluralityParent: the value of the parent node
        :return: Nodes with values and parameters for the tree
        """
        # Number of Unique Labels that are currently present in the data
        numUniqueLabels = len(np.unique(y))
        
        #stop-condition, either depth has reached Max Depth, or there isn't any useful split
        if(depth >= self.maxDepth or numUniqueLabels < 2):
            # if there are no labels, return the value of the parent
            if len(y)<1:
                return Node(value = pluralityParent)
            return Node(value = mode(y))
        
        yMode = mode(y)
        giniVals = giniAll(X, y) # Extract Gini values for all features
        giniValsNp = np.asarray(giniVals)
        lowestGiniFeatureIndex = np.argmin(giniValsNp) # Get the index of the feature with the lowest gini value
        zeroValsX, zeroValsY, oneValsX, oneValsY = split(X, y, lowestGiniFeatureIndex)
        
        # Build the tree recursively
        left = self.buildTree(zeroValsX, zeroValsY, depth + 1, yMode)
        right = self.buildTree(oneValsX, oneValsY, depth + 1, yMode)

        # Nodes with a left and right branch will not hold a value
        return Node(lowestGiniFeatureIndex, left, right)

    # Searches through the tree and return the value of the node
    def search(self, x, node):
        """
        :param self: class information
        :param x: list of values of n features, data to be labelled
        :param node: the node that needs to be searched
        :return: predicted value of the given features values based on the current node 
        """

        # For leaf nodes, return the value
        if node.isLeafNode():
            return node.value

        if x[node.feature] < 1:
            return self.search(x, node.left)
        return self.search(x, node.right)

    # Returns the predicted label for a given feature set 
    def predict(self, data):
        """
        :param self: class information
        :param data: list of values of n features that have to be predicted (data to label)
        :return: predicted value of the given features values 
        """
        yPred = self.search(data, self.root)
        return yPred

# Random Forest Classifier, Current Configuration = 20 Trees
class RFClassifier:
    def __init__(self, numTrees=20, maxDepth=100):
        self.numTrees = numTrees
        self.maxDepth = maxDepth
        self.trees = []
    
    # Builds the desired number of trees and stores them
    def fit(self, X, y):
        """
        :param self: class information
        :param X: list of values of n features and k samples
        :param y: list of labels of k samples
        """
        self.trees = []
        for t in range(self.numTrees):
            print("Processing Tree", t)
            dt = DTClassifier(maxDepth=self.maxDepth)
            XBts, yBts = bootstrapData(X, y) # Get a random shuffle and split of data
            dt.fit(XBts, yBts)
            self.trees.append(dt)
    
    # Returns the predicted label for a given feature set
    def predict(self, data):
        """
        :param self: class information
        :param data: list of values of n features that have to be predicted (data to label)
        :return: predicted value of the given features values 
        """
        labelPredAll = [dt.predict(data) for dt in self.trees] # Predict values for each decision tree
        labelPred = mode(labelPredAll)
        return labelPred

# Classifier Class, serves as an ensemble class.
# Utilizes and extracts predictions from all features and weights them.
class Classifier:
    def __init__(self):
        self.knn = KNNClassifier()
        self.dt = DTClassifier()
        self.rf = RFClassifier()
    
    # Un-sets all classifier instances
    def reset(self):
        """
        :param self: class information
        """
        self.knn = None
        self.dt = None
        self.rf = None

    # Calls the Fit function on Decision Trees and Random Forest classifiers 
    def fit(self, data, target):
        """
        :param self: class information
        :param data: list of values of n features and k samples
        :param target: list of labels of k samples
        """
        self.XTrain, self.XTest, self.yTrain, self.yTest = trainTestSplit(data, target, 0.8)

        self.dt.fit(self.XTrain, self.yTrain)
        self.rf.fit(self.XTrain, self.yTrain)
    
    # Returns the predicted label for a given feature set, mode of predicted values by all classifiers
    def predict(self, data, legal=None):
        """
        :param self: class information
        :param data: list of values of n features that have to be predicted (data to label)
        :param legal: The legal moves, set to None by default
        :return: predicted value of the given features values 
        """

        ret = [
            self.knn.predict(self.XTrain, self.yTrain, data), 
            self.dt.predict(data), 
            self.rf.predict(data)
        ]
        # Currently the mode is being returned, but we could weight them by replicating the value of one classifier
        # and thereby giving it more importance
        return mode(ret)
