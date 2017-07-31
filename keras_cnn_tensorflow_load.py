from __future__ import print_function
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
from keras.models import load_model
import cv2

#extra
import numpy as np
from keras.models import model_from_json
import os

batch_size = 32 #was 128
num_classes = 10 
epochs = 1 #was 12

img_rows = 28
img_cols = 28

# preparing the data...
from keras.datasets import mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# let's test individually # so it looks like it is a 7
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
#plt.imshow(x_test[0])
#plt.show()


# reshape the data
x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
input_shape = (img_rows, img_cols, 1)


x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255

y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)


# load json and create model
json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("model.h5")
print("Loaded model from disk")


loaded_model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])
score = loaded_model.evaluate(x_test, y_test, verbose=0)
print("%s: %.2f%%" % (loaded_model.metrics_names[1], score[1]*100))
print('Test loss:', score[0])
print('Test accuracy:', score[1])

for filename in os.listdir("mnist_test"):
    img = cv2.imread("mnist_test/%s" % filename)
    img = img[:,:,0]
    img = np.expand_dims(img,2)
    img.shape = (1,28,28,1)
    prediction = loaded_model.predict(img)
    print(prediction)


