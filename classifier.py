from random import shuffle
import numpy
import math
# classifier.py
# Lin Li/26-dec-2021
#
# Use the skeleton below for the classifier and insert your code here.

class Classifier:
    def __init__(self):
        print("INIT-CLASS", self)
        self.k = 3
    
    # https://stackoverflow.com/questions/4601373/better-way-to-shuffle-two-numpy-arrays-in-unison
    def shuffle_array(self, a, b):
        a = numpy.asarray(a)
        b = numpy.asarray(b)
        assert len(a) == len(b)
        p = numpy.random.permutation(len(a))
        return a[p], b[p]
        
    def train_test_split(self, X, y, split):
        shuffle_X, shuffle_y = self.shuffle_array(X,y) 
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
        self.X_train, self.X_test, self.y_train, self.y_test = self.train_test_split(data, target, 0.8)
        print("x-train", self.X_train)
        print("x-test", self.X_test)
        print("y_train", self.y_train)
        print("y_test", self.y_test)
        
    def predict(self, data, legal=None):
        return 1
        