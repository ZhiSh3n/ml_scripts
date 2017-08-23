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
    if filename.endswith('.png'):
        img = cv2.imread("mnist_test/%s" % filename)
        img = img[:,:,0]
        img = np.expand_dims(img,2)
        img.shape = (1,28,28,1)
        prediction = loaded_model.predict(img)
        print(prediction)
    
