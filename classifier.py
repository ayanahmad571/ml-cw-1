from cProfile import label
from random import shuffle
import numpy
import math
from statistics import mode
from sklearn.neighbors import KNeighborsClassifier
from collections import Counter

# classifier.py
# Lin Li/26-dec-2021
#
# Use the skeleton below for the classifier and insert your code here.
#Euclidean Distance
def e_diff_calc(a, b):
    a = numpy.asarray(a)
    # print(x1,x2)
    # print("______________")
    # print(x1 - x2)
    # print((x1 - x2) ** 2)
    # print(numpy.sum((x1 - x2) ** 2))
    # print("SUM:", numpy.sqrt(numpy.sum((x1 - x2) ** 2)))

    return numpy.sqrt(numpy.sum((a - b) ** 2))

# https://stackoverflow.com/questions/4601373/better-way-to-shuffle-two-numpy-arrays-in-unison
def shuffle_array(a, b):
    a = numpy.asarray(a)
    b = numpy.asarray(b)
    assert len(a) == len(b)
    p = numpy.random.permutation(len(a))
    return a[p], b[p]

class Classifier:
    def __init__(self):
        print("INIT-CLASS", self)
        self.k = 5
   
       
    def train_test_split(self, X, y, split):
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

    def reset(self):
        print("RESET", self)
        self.X_test = []
        self.y_test = []
    
    def fit(self, data, target):
        print("FIT", data, target)
        self.X = data
        self.y = target
        self.X_train, self.X_test, self.y_train, self.y_test = self.train_test_split(data, target, 0.8)
        
       
    def predict(self, data, legal=None):
        y_pred = list()
        # for x in data:
        # Compute distances between x and all examples in the training set
        
        e_diff = [e_diff_calc(data, x_train) for x_train in self.X_train]
        # print(e_diff)
        # Sort by distance and return indices of the first k neighbors
        
        min_diff_indexes = numpy.argsort(e_diff)
        # https://stackoverflow.com/questions/5234090/how-to-take-the-first-n-items-from-a-generator-or-list
        
        k_indexes = min_diff_indexes[: self.k]
        # print(k_indexes)
        
        # Extract the labels of the k nearest neighbor training samples
        labelled_indexes = [self.y_train[i] for i in k_indexes]
        # print(labelled_indexes)
        
        return mode(labelled_indexes)
