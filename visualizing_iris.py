#following https://dzone.com/refcardz/data-mining-discovering-and


# get the data
import urllib3
import shutil
http = urllib3.PoolManager()
url = "http://aima.cs.berkeley.edu/data/iris.csv"
with http.request('GET', url, preload_content=False) as r, open("./%s" % ("iris.csv"), 'wb') as out_file:       
    shutil.copyfileobj(r, out_file)

# organize the data
from numpy import *
data = genfromtxt("iris.csv", delimiter=',',usecols=(0,1, 2, 3))
target = genfromtxt("iris.csv", delimiter=",", usecols=(4), dtype=str)
print("Shape of the inputs:",data.shape)
print("Shape of the labels:", target.shape)
# note that this is not ordered
print("Unique elements in labels:", set(target))

# visualize the data

# plot sepal lengh by sepal width
# no need to create figure
from matplotlib import pylab
from pylab import *
"""
plot(data[target=="setosa",0],data[target=="setosa",2],"bo")
plot(data[target=='versicolor',0],data[target=='versicolor',2],'ro')
plot(data[target=='virginica',0],data[target=='virginica',2],'go')
"""

# plot histograms
"""
xmin = min(data[:,0])
xmax = max(data[:,0])
figure() # create new window
subplot(411)
subplot(411) # distribution of the setosa class (1st, on the top)
hist(data[target=='setosa',0],color='b',alpha=.7)
xlim(xmin,xmax)
subplot(412) # distribution of the versicolor class (2nd)
hist(data[target=='versicolor',0],color='r',alpha=.7)
xlim(xmin,xmax)
subplot(413) # distribution of the virginica class (3rd)
hist(data[target=='virginica',0],color='g',alpha=.7)
xlim(xmin,xmax)
subplot(414) # global histogram (4th, on the bottom)
hist(data[:,0],color='y',alpha=.7)
xlim(xmin,xmax)
"""
#show()

# now we will use gaussian naive bayes to classify
# first we will convert the vector strings in target into integers
t = zeros(len(target))
t[target == "setosa"] = 1
t[target == "versicolor"] = 2
t[target == "virginica"] = 3

from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(data,t)

# reshape data because prediction needs array that looks like training data
# each index needs to be training example with same number of features as in training
# ie. reshaped == ([[150, 0]])
# non reshaped == ([150, 0])
# data == ([[ ... ]])
print(classifier.predict(data[0].reshape(1, -1)))
print(t[0])

# ok now we will split the set with train_test_split
from sklearn.model_selection import train_test_split
train, test, ttrain, ttest = train_test_split(data, t, test_size=0.4, random_state=0)
# the size of the test is 40% the size of the original

# train again
classifier.fit(train,ttrain)
print(classifier.score(test, ttest))


# we can use a confusion matrix to see examine the performance of this classifier
from sklearn.metrics import confusion_matrix
print(confusion_matrix(classifier.predict(test),ttest))


# or we can use classification_report which is more in depth
from sklearn.metrics import classification_report
print(classification_report(classifier.predict(test), ttest, target_names=["setosa", "versicolor", "virginica"]))



# we can use cross validation to evaluate a classifier, which repeats the above
# process multiple times

from sklearn.model_selection import cross_val_score
# cross validation with 6 iterations
scores = cross_val_score(classifier, data, t, cv=6)
print(scores)

# get the mean accuracy
from numpy import mean
print(mean(scores))

# clustering ???unsupervised data analysis
from sklearn.cluster import KMeans
# groups data in 3 clusters
kmeans = KMeans(3,init="random")
# assign each input to one cluster
kmeans.fit(data)
c = kmeans.predict(data)

# evaluate results of clustering???
# compare it with the labels we already have
from sklearn.metrics import completeness_score, homogeneity_score
# completeness approaches 1 when most data point are elements of the same cluster
print(completeness_score(t,c))
# homogeneity approaches 1 when all clusters contain only data points that are members of a single class
print(homogeneity_score(t,c))


# visualize this 
#figure() uncomment if you include the other figures above
# remember we are plotting two of the features of the flowers on x y axis
subplot(211) # top figure with the real classes
plot(data[t==2,0],data[t==2,2],'ro')
plot(data[t==1,0],data[t==1,2],'bo')
plot(data[t==3,0],data[t==3,2],'mo')

subplot(212) # bottom figure with classes assigned automatically
plot(data[c==1,0],data[c==1,2],'bo')
plot(data[c==2,0],data[c==2,2],'go')
plot(data[c==0,0],data[c==0,2],'mo')
show()


# regression
