# naive bayes is another machine learning method
# it deals more with probability
# naive bayes is naive because it doesn't
# know that eg cash and money are correlated,
# so it ends up basically doubling a certain probability

import numpy as np
from datetime import datetime
from scipy.stats import norm
from scipy.stats import multivariate_normal as mvn
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

class Bayes(object):
    def fit(self, X, Y, smoothing=10e-3):
        N, D = X.shape
        self.gaussians = dict()
        self.priors = dict()
        labels = set(Y)
        for c in labels:
            current_x = X[Y == c]
            self.gaussians[c] = {
                'mean': current_x.mean(axis=0),
                # calculate covariance instead of variance
                'cov': np.cov(current_x.T) + np.eye(D) * smoothing,
            }
            # assert(self.gaussians[c]['mean'].shape[0] == D)
            self.priors[c] = float(len(Y[Y == c])) / len(Y)
        # print "gaussians:", self.gaussians
        # print "priors:", self.priors

    def score(self, X, Y):
        P = self.predict(X)
        return np.mean(P == Y)

    def predict(self, X):
        N, D = X.shape
        K = len(self.gaussians)
        P = np.zeros((N, K))
        for c, g in self.gaussians.items():
            # print "c:", c
            # also changed from nbayes to nnbayes
            mean, cov = g['mean'], g['cov']
            P[:,c] = mvn.logpdf(X, mean=mean, cov=cov) + np.log(self.priors[c])
        return np.argmax(P, axis=1)


if __name__ == '__main__':
    X, Y = get_data(10000)
    Ntrain = int(len(Y) / 2)
    Xtrain, Ytrain = X[:Ntrain], Y[:Ntrain]
    Xtest, Ytest = X[Ntrain:], Y[Ntrain:]

    model = Bayes()
    t0 = datetime.now()
    model.fit(Xtrain, Ytrain)
    print ("Training time:", (datetime.now() - t0))

    t0 = datetime.now()
    print ("Train accuracy:", model.score(Xtrain, Ytrain))
    print ("Time to compute train accuracy:", (datetime.now() - t0), "Train size:", len(Ytrain))

    t0 = datetime.now()
    print ("Test accuracy:", model.score(Xtest, Ytest))
    print ("Time to compute test accuracy:", (datetime.now() - t0), "Test size:", len(Ytest))
