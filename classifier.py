from cProfile import label
from random import shuffle
import numpy as np
import math
import pandas as pd 
from statistics import mode
# from sklearn.neighbors import KNeighborsClassifier
from collections import Counter

# classifier.py
# Lin Li/26-dec-2021
#
# Use the skeleton below for the classifier and insert your code here.
#Euclidean Distance
def e_diff_calc(a, b):
    a = np.asarray(a)
    # print(x1,x2)
    # print("______________")
    # print(x1 - x2)
    # print((x1 - x2) ** 2)
    # print(np.sum((x1 - x2) ** 2))
    # print("SUM:", np.sqrt(np.sum((x1 - x2) ** 2)))

    return np.sqrt(np.sum((a - b) ** 2))

# https://stackoverflow.com/questions/4601373/better-way-to-shuffle-two-numpy-arrays-in-unison
def shuffle_array(a, b):
    a = np.asarray(a)
    b = np.asarray(b)
    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    return a[p], b[p]

def train_test_split(X, y, split):
    shuffle_X, shuffle_y = shuffle_array(X,y) 
    X_train = list()
    X_test = list()
    y_train = list()
    y_test = list()

    train_len = math.floor(split * len(shuffle_X))

    for i in range(train_len):
        X_train.append(shuffle_X[i])
        y_train.append(shuffle_y[i])
        new_i = i+train_len
        if(new_i < len(shuffle_X)):
            X_test.append(shuffle_X[new_i])
            y_test.append(shuffle_y[new_i])

    return X_train, X_test, y_train, y_test

def gini_impurity(data):
    # print("Gini Impurity Calculation")
    lables = {0,1,2,3}
    data_len = len(data)
    if data_len == 0:
        data_len = 1
    gini_values = [(data.count(l)/data_len)**2 for l in lables]
    gini_imp =1-sum(gini_values) 
    # print(gini_imp)
    return gini_imp

def gini_feat(X,y, feat):
    zeroVals = list()
    oneVals = list()
    
    for val_index in range(len(X)):
        
        
        at_index_val = X[val_index][feat]
        at_index_label = y[val_index]
        if(at_index_val == 0):
            zeroVals.append(at_index_label)
        else:
            oneVals.append(at_index_label)

    # print(zeroVals)
    # print(oneVals)
    
    # print("Zero Gini", len(zeroVals))
    zeroGiniImp = gini_impurity(zeroVals)
    
    # print("One Gini", len(oneVals))
    oneGiniImp = gini_impurity(oneVals)
    
    # We could just do len(feat) but if in case some data is missing
    feature_len = len(zeroVals) + len(oneVals)
    
    total_gini = (len(zeroVals)*zeroGiniImp + len(oneVals)*oneGiniImp)/feature_len
    # print("total gini")
    # print(total_gini)
    

    # print("_____")
    # gini_impurity()
    return total_gini
    
def gini_all(X,y):
    # print(len(X))
    num_feats = len(X[0]) 
    
    feature_gini = list()
    for f in range(num_feats):
        # print("Processing Feature ",f)
        feature_gini.append(gini_feat(X,y,f))
    
    return feature_gini



class KNNClassifier:
    def __init__(self):
        print("Init KNN ", self)
        self.k = 5
    
    def predict(self, X_train, y_train, data):
        # Compute distances between x and all examples in the training set        
        e_diff = [e_diff_calc(data, x_train) for x_train in X_train]

        # Sort by distance and return indices of the first k neighbors        
        min_diff_indexes = np.argsort(e_diff)
        
        # https://stackoverflow.com/questions/5234090/how-to-take-the-first-n-items-from-a-generator-or-list
        k_indexes = min_diff_indexes[: self.k]
        
        # Extract the labels of the k nearest neighbor training samples
        labelled_indexes = [y_train[i] for i in k_indexes]
        
        return mode(labelled_indexes)





class Node:
    def __init__(
        self, feature=None, threshold=None, left=None, right=None, *, value=None
    ):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value

    def is_leaf_node(self):
        return self.value is not None

class DTClassifier:
    def __init__(self):
        print("Init DT-CLASS", self)
        self.depth = 10
    
    
    def fit(self, X_train, y_train, X_test, y_test,):
        # self.X = X_train
        # self.Y = y_train
        num_cols = len(X_train[0])
        col_labels = [f'C{c}' for c in range(len(X_train[0]))]
        lables = [0,1,2,3]
        print(gini_all(X_train, y_train))

        print(col_labels)
        pass

    def predict(self, data):
        # y_pred = np.array([self._traverse_tree(x, self.root) for x in X_test])
        # print(y_pred)
        return 1

class RFClassifier:
    def __init__(self):
        print("Init RF-CLASS", self)
        self.depth = 100
    
    def predict(self, X_train, y_train, data):
        return 1

class Classifier:
    def __init__(self):
        self.knn = KNNClassifier()
        self.dt = DTClassifier()
        self.rf = RFClassifier()



    def reset(self):
        print("RESET-ALL", self)
        self.knn = ""
        self.dt = ""
        self.rf = ""
    
    def fit(self, data, target):
        print("FIT", data, target)
        self.X = data
        self.y = target
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(data, target, 0.8)

        # self.knn.fit(self.X_train, self.y_train)
        self.dt.fit(self.X_train, self.y_train, self.X_test, self.y_test)
        # self.rf.fit(self.X_train, self.y_train)
    

    def predict(self, data, legal=None):
        print("_______________")
        ret = [
            self.knn.predict(self.X_train, self.y_train, data), 
            self.dt.predict(data), 
            self.rf.predict(self.X_train, self.y_train ,data)]
        print("KNN", ret[0])
        print("DT", ret[1])
        print("RF", ret[2])
        return ret[0]
