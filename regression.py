#regression
# inspired by https://dzone.com/refcardz/data-mining-discovering-and

# build a synthetic dataset
from numpy.random import rand
x = rand(40,1) #independent variable
y = x*x*x+rand(40,1)/5 # dependent variable

# now we draw a best fit line
from sklearn.linear_model import LinearRegression
linreg = LinearRegression()
linreg.fit(x,y)

# plot this line over actual data points
from numpy import linspace, matrix
from matplotlib.pyplot import *
xx = linspace(0,1,40)
plot(x,y,'o',xx,linreg.predict(matrix(xx).T),'--r')
show()

# how accurate is our best fit line?
# closer to 0 is more accurate
from sklearn.metrics import mean_squared_error
print(mean_squared_error(linreg.predict(x),y))
