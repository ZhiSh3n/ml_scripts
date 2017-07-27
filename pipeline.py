from scipy.spatial import distance

# a function that returns the distance between two points
def euc(a,b):
    return distance.euclidean(a,b)

# this is our classifier
# it is basically a nearest-neighbor classifier
class ScrappyKNN():
    # store values and labels 
    def fit(self, features_train, labels_train):
        self.features_train = features_train
        self.labels_train = labels_train
        
    def predict(self, features_test):
        # make a predictions array
        predictions = []
        # for every feature in the test set,
        # the label we predict is defined by the closest label
        # in the method def(test feature)
        for row in features_test:
            label = self.closest(row)
            predictions.append(label)
        return predictions

    # finds distance between a feature in the testing set and
    # features in the training set
    # return the label of the best distance
    def closest(self, row):
        best_dist = euc(row, self.features_train[0])
        best_index = 0
        for i in range(1, len(self.features_train)):
            dist = euc(row, self.features_train[i])
            if dist < best_dist:
                best_dist = dist
                best_index = i
        return self.labels_train[best_index]
    

# let's write a classifier!
from sklearn import datasets
iris = datasets.load_iris()

# think of a classifier as a function f(x) = y
features = iris.data
labels = iris.target

# cross_validation is depreceated
# we are basically partitioning our features and labels into 2 sets
from sklearn.model_selection import train_test_split
features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size = .5)

# create the classifier
"""
# this is one way of creating a classifier - it is a decision tree
from sklearn import tree
my_classifier = tree.DecisionTreeClassifier()

# another way of making a classifier
from sklearn.neighbors import KNeighborsClassifier
my_classifier = KNeighborsClassifier()
"""
# let's write our own classifier!
my_classifier = ScrappyKNN()

# the takeaway is that while there are many types of classifiers
# at a high level they have a similar interface

# train the classifier
my_classifier.fit(features_train, labels_train)

predictions = my_classifier.predict(features_test)
# print(predictions)
# we get what the classifier predicts each iris will be

# let's calculate how accurate we were
from sklearn.metrics import accuracy_score
print(accuracy_score(labels_test, predictions))
