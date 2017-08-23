import numpy as np

# set a seed for computer's pseudorandom number generator
# this is for reproductability purposes
np.random.seed(123)

# import sequential model type from keras
# basically a linear stack of neural network layers
from keras.models import Sequential

# import core layers
from keras.layers import Dense, Dropout, Activation, Flatten

# import CNN layers
# these are convolutional layers used in training
from keras.layers import Convolution2D, MaxPooling2D

# import some utilities
from keras.utils import np_utils

# import MNIST
from keras.datasets import mnist
from keras import backend as K
K.set_image_dim_ordering('th')

# load pre-shuffled mnist data into train and test sets
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# let's look at the shape of the dataset
print(X_train.shape)
# prints (60000, 28, 28)
# we have 60,000 samples, images are 28x28

# we can plot any sample in matplotlib
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
plt.imshow(X_train[0])
#plt.show()

# i wonder if this applies to tensorflow backend?
# we must declare the depth, which is 1
X_train = X_train.reshape(X_train.shape[0], 1, 28, 28)
X_test = X_test.reshape(X_test.shape[0], 1, 28, 28)
print(X_train.shape)

# final step is to convert datatype to float32
# and change range of values to 0,1
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255

print(y_train.shape)
# we need to split the y groups into 10 distinct class labels
Y_train = np_utils.to_categorical(y_train, 10)
Y_test = np_utils.to_categorical(y_test, 10)
print(Y_train.shape)

# start by declaring a squential model format
model = Sequential()

# declare input layer
model.add(Convolution2D(32, (3, 3), activation="relu", input_shape=(1,28,28)))
# the first 3 params represent eh number of convolutional filters to use,
# the number of rows in each convolutional kernel,
# and the number of columns in each kernal

# we add more layers to our model
model.add(Convolution2D(32, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))

# add a fully connected layer and then the output
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(10, activation='softmax'))

# now all we need to do is comile the model and train it
model.compile(loss='categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

# fit the model on training data
model.fit(X_train, Y_train, 
          batch_size=32, epochs=10, verbose=1)

# evaluate our data
score = model.evaluate(X_test, Y_test, verbose=0)
