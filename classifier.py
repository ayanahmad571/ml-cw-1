from cProfile import label
from random import random, shuffle
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

def bootstraping(X,y):
    num_rows = len(X)
    num_samples = int((2*num_rows)/3)
    random_indexes = list(np.random.randint(low = 0,high = num_rows, size=num_samples))
    X = np.asarray(X)
    y = np.asarray(y)
    return X[random_indexes], y[random_indexes]

def convertNumberToMove(number):
        if number == 0:
            return "NORTH"
        elif number == 1:
            return "EAST"
        elif number == 2:
            return "SOUTH"
        elif number == 3:
            return "WEST"


class Node:
    def __init__(
        self, feature=None, left=None, right=None, *, value=None
    ):
        self.feature = feature
        self.left = left
        self.right = right
        self.value = value

    def is_leaf_node(self):
        return self.value is not None

def split(X,y,feat):
    zeroVals_X = list()
    zeroVals_y = list()
    oneVals_X = list()
    oneVals_y = list()
    
    for row_index in range(len(X)):
       
        at_index_val = X[row_index][feat]
        at_index_label = y[row_index]
        if(at_index_val == 0):
            zeroVals_X.append(X[row_index])
            zeroVals_y.append(at_index_label)
        else:
            oneVals_X.append(X[row_index])
            oneVals_y.append(at_index_label)
    
    return zeroVals_X, zeroVals_y, oneVals_X, oneVals_y

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
        y_pred = mode(labelled_indexes)
        print("K-P:",self.convertNumberToMove(y_pred))
        return y_pred

class DTClassifier:
    def __init__(self, max_depth = 100, num_feats = 25):
        print("Init DT-CLASS", self)
        self.max_depth = max_depth
        self.num_feats = num_feats
        self.root = None
    
    
    def fit(self, X_train, y_train):
        self.root = self.build_tree(X_train, y_train)
        pass

    def build_tree(self, X, y, depth=0):
        num_unique_labels = len(np.unique(y))
        
        #stop-condition
        if(depth >= self.max_depth or num_unique_labels < 2):
            if len(y)<1:
                return Node(value = 0)    
            return Node(value = mode(y))
        
        gini_vals = gini_all(X, y)
        gini_vals_np = np.asarray(gini_vals)
        lowest_gini_feature_index = np.argmin(gini_vals_np)
        zeroVals_X, zeroVals_y, oneVals_X, oneVals_y = split(X, y, lowest_gini_feature_index)
        left = self.build_tree(zeroVals_X, zeroVals_y, depth + 1)
        right = self.build_tree(oneVals_X, oneVals_y, depth + 1)

        return Node(lowest_gini_feature_index, left, right)

    def search(self, x, node):
        if node.is_leaf_node():
            return node.value

        if x[node.feature] < 1:
            return self.search(x, node.left)
        return self.search(x, node.right)


    def predict(self, data):
        y_pred = self.search(data, self.root)
        print("D-P:",convertNumberToMove(y_pred))
        return y_pred

class RFClassifier:
    def __init__(self, num_trees=100, max_depth=100, num_feats=None):
        print("Init RF-CLASS", self)
        self.num_trees = num_trees
        self.max_depth = max_depth
        self.num_feats = num_feats
        self.trees = []
    
    def fit(self, X, y):
        self.trees = []
        for t in range(self.num_trees):
            print("Processing Tree", t)
            dt = DTClassifier(max_depth=self.max_depth, num_feats=self.num_feats)
            X_bts, y_bts = bootstraping(X, y)
            dt.fit(X_bts, y_bts)
            self.trees.append(dt)
    
    def predict(self, data):
        y_preds_all = [dt.predict(data) for dt in self.trees]
        print(y_preds_all)
        y_pred = mode(y_preds_all)
        print("R-P:",convertNumberToMove(y_pred))
        return y_pred

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

        self.dt.fit(self.X_train, self.y_train)
        self.rf.fit(self.X_train, self.y_train)
    

    def predict(self, data, legal=None):
        print("_______________")
        # ret = [
        #     self.knn.predict(self.X_train, self.y_train, data), 
        #     self.dt.predict(data), 
        #     self.rf.predict(self.X_train, self.y_train ,data)]
        # print("KNN", ret[0])
        # print("DT", ret[1])
        # print("RF", ret[2])
        # return ret[1]
        # return self.dt.predict(data)
        return self.rf.predict(data)

