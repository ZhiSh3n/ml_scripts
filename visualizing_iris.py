
# get the data
import urllib3
import shutil
http = urllib3.PoolManager()
url = "http://aima.cs.berkeley.edu/data/iris.csv"
with http.request('GET', url, preload_content=False) as r, open("./%s" % ("iris.csv"), 'wb') as out_file:       
    shutil.copyfileobj(r, out_file)

# organize the data
from numpy import genfromtxt, zeros
data = genfromtxt("iris.csv", delimiter=',',usecols=(0,1, 2, 3))
target = genfromtxt("iris.csv", delimiter=",", usecols=(4), dtype=str)
print("Shape of the inputs:",data.shape)
print("Shape of the labels:", target.shape)
print("Unique elements in labels:", set(target))

# visualize the data

# plot sepal lengh by sepal width
# no need to create figure
from matplotlib import pylab
from pylab import *
plot(data[target=="setosa",0],data[target=="setosa",2],"bo")
plot(data[target=='versicolor',0],data[target=='versicolor',2],'ro')
plot(data[target=='virginica',0],data[target=='virginica',2],'go')

# plot histograms
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
show()

# now we will use gaussian naive bayes to classify
# first we will convert the vector strings in target into integers
t = zeros(len(target))
t[target == "setosa"] = 1
t[target == "versicolor"] = 2
t[target == "virginicia"] = 3
