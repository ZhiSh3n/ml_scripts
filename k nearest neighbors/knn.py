# K-Nearest Neighbors: Supervised Machine Learning
# - use the data points nearest to make a prediction
# - training is simple; simply store all the known data points so you can use them later
# - prediction is where the magic happens
# KNN is known as a lazy classifier because training doesn't do anything
# Predict does all the work by loking through the stored X and Y

import numpy as np
from sortedcontainers import SortedList
from datetime import datetime
import pandas as pd

def get_data(limit=None):
    print("Reading in and transforming data...")
    df = pd.read_csv('../train.csv')
    data = df.as_matrix()
    np.random.shuffle(data)
    X = data[:, 1:] / 255.0 # data is from 0..255
    Y = data[:, 0]
    if limit is not None:
        X, Y = X[:limit], Y[:limit]
    return X, Y

class KNN(object):
    def __init__(self, k):
        # k is equal to how many nearest neighbors we want
        self.k = k 

    def fit(self, X, y):
        # X equals the inputs
        self.X = X
        # y equals the labels
        self.y = y 

    def predict(self, X):
        # set y equal to an array of length X, filled with zeros
        # this is because we need a prediction for every input
        y = np.zeros(len(X))

        # loop through every input
        for i,x in enumerate(X):
            # create a sorted list of size k
            # stores (distance, class) tuples
            sl = SortedList(load=self.k)

            # for each input test point, we 
            # loop through all the training points
            # to find the K nearest neighbors
            for j,xt in enumerate(self.X): # self.X is all the training points
                diff = x - xt # xt is the training point, j is the index, ie input vs training point
                d = diff.dot(diff)
                if len(sl) < self.k:
                    # if sorted list is less than size K, just add the point
                    # basically add up to K neighbors
                    sl.add( (d, self.y[j]) )
                else:
                    if d < sl[-1][0]: # check value at the end, because that is the biggest distance (last index)
                        # but why is this? is it not possible for the first index of sl to be the smallest
                        del sl[-1]
                        sl.add( (d, self.y[j]) )
            # print "input:", x
            # print "sl:", sl

            # create a dictionary called votes
            votes = {}

            # loop through the sorted list of the
            # k nearest neighbors
            for _, v in sl:
                # print "v:", v
                votes[v] = votes.get(v,0) + 1
            # print "votes:", votes, "true:", Ytest[i]
            max_votes = 0
            max_votes_class = -1
            for v,count in votes.items():
                if count > max_votes:
                    max_votes = count
                    max_votes_class = v
            y[i] = max_votes_class # set label of particular input to the max votes
        return y # return the label array!

    def score(self, X, Y):
        P = self.predict(X)
        return np.mean(P == Y)


if __name__ == '__main__':
    X, Y = get_data(2000)
    Ntrain = 1000
    Xtrain, Ytrain = X[:Ntrain], Y[:Ntrain]
    Xtest, Ytest = X[Ntrain:], Y[Ntrain:]
    for k in (1,2,3,4,5):
        knn = KNN(k)
        print("K:", k)
        t0 = datetime.now()
        knn.fit(Xtrain, Ytrain)
        print("Training time:", (datetime.now() - t0))

        t0 = datetime.now()
        print("Train accuracy:", knn.score(Xtrain, Ytrain))
        print("Time to compute train accuracy:", (datetime.now() - t0), "Train size:", len(Ytrain))

        t0 = datetime.now()
        print("Test accuracy:", knn.score(Xtest, Ytest))
        print("Time to compute test accuracy:", (datetime.now() - t0), "Test size:", len(Ytest))

"""
When can KNN fail?
=====================
well, since KNN depends on how close an input point is to K training points,
it probably won't work very well when we don't have training labels for those
training inputs, since we won't be able to fit in the first place.

also for non-linear data sets it might be hard. for example, when red dots
and blue dots are spiraled against each other, it might be harder. or when
data is very mixed together in a cloud formation, since then KNN might
result in very low accuracy

Answers
===================
grid of alternating dots
3nn
2/3 vote for wrong class, as it counts itself

Fixes
==================
using 1NN
weighing each point by distance


"""
