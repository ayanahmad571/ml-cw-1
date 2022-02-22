from random import shuffle
import numpy
import math
from scipy.stats import mode
from sklearn.neighbors import KNeighborsClassifier
from collections import Counter

# classifier.py
# Lin Li/26-dec-2021
#
# Use the skeleton below for the classifier and insert your code here.
#Euclidean Distance
def euclidean_distance(x1, x2):
    return numpy.sqrt(numpy.sum((x1 - x2) ** 2))

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
        self.k = 3
   
       
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
        for x in data:
            # Compute distances between x and all examples in the training set
            distances = [euclidean_distance(x, x_train) for x_train in self.X_train]
            # Sort by distance and return indices of the first k neighbors
            k_idx = numpy.argsort(distances)[: self.k]
            # Extract the labels of the k nearest neighbor training samples
            k_neighbor_labels = [self.y_train[i] for i in k_idx]
            # return the most common class label
            most_common = Counter(k_neighbor_labels).most_common(1)
            y_pred.append(most_common[0][0])
        print(numpy.array(y_pred))
        return y_pred[0]
