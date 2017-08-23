import os,cv2
import numpy as np
import keras
from sklearn.utils import shuffle
from sklearn.cross_validation import train_test_split
from keras import backend as K
K.set_image_dim_ordering('tf')
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.optimizers import SGD,RMSprop,adam
from keras.layers import Conv2D, MaxPooling2D
from keras.datasets import mnist
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.models import load_model

# dimensions of mnist images
image_width, image_height = 28, 28

# get data path
# this will depend on your data configuration
PATH = os.getcwd()
main_directory = '../../../data/mnist_png_sorted'
directory_list = os.listdir(main_directory)

# training parameters
epochs = 5
batch_size = 128
number_classes = 10
#input_shape = (image_width, image_height, 1)


# get our data
image_data_list = [] # stores all the images
class_population = [] # used to connect each image to a class
for dataset in directory_list : # iterate through each dataset (folder) in directory
    counter = 0 # counter to count how many samples are in each class
    if os.path.isdir(main_directory + '/' + dataset) :
        image_list = os.listdir(main_directory + '/' + dataset)
        print ('Loaded the images of dataset: '+'{}\n'.format(dataset))
        for image in image_list:
            if image.endswith('.png') :
                counter = counter + 1
                input_image = cv2.imread(main_directory + '/' + dataset + '/' + image)
                # left out resizing because we are doing mnist.
                # may need to add resizing if doing other photos
                input_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2GRAY) #changing to grayscale
                input_image_resize = cv2.resize(input_image,(image_width, image_height))
                image_data_list.append(input_image_resize)
        class_population.append(counter)
number_samples = len(image_data_list)
print('Total number of images loaded:', number_samples)

# image formatting
image_data = np.array(image_data_list)
image_data = image_data.astype('float32')
image_data /= 255
print ('Shape of [image_data]:', image_data.shape)
image_data= np.expand_dims(image_data, axis=4) # adding extra dim


# split our data
print(class_population)
labels = np.ones((number_samples,),dtype='int64')
# create labels for each sample
for first_counter in range(0,10) :
    if first_counter == 0 :
        # the first class_population[0] number of samples are 0
        labels[0:class_population[first_counter]] = 0
    else :
        if first_counter == 9 :
            final_sum = sum(class_population) - class_population[9]
            labels[final_sum:] = 9
        else :
            small_sum = 0
            big_sum = 0
            for second_counter in range(0,first_counter+1) :
                big_sum = big_sum + class_population[second_counter]
            small_sum = big_sum - class_population[first_counter]
            labels[small_sum:big_sum] = first_counter
        
        
"""
# manual test data
labels[0:class_population[0]] = 0
labels[class_population[0]:class_population[1]] = 1
labels[class_population[2]:class_population[3]] = 2
labels[class_population[3]:class_population[4]] = 3
labels[class_population[4]:class_population[5]] = 4
labels[class_population[5]:class_population[6]] = 5
labels[class_population[6]:class_population[7]] = 6
labels[class_population[7]:class_population[8]] = 7
labels[class_population[8]:class_population[9]] = 8
labels[class_population[9]:] = 9
print(labels[3500])
print(labels[10500])
print(labels[17500])
print(labels[24500])
print(labels[31500])
print(labels[38500])
print(labels[45500])
print(labels[52500])
print(labels[59500])
print(labels[66500])
"""

# assign labels to each class to each sample
Y = np_utils.to_categorical(labels, number_classes)
x, y = shuffle(image_data, Y, random_state = 2)
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=2)
X_train = X_train.reshape(X_train.shape[0], image_width, image_height, 1)
X_test = X_test.reshape(X_test.shape[0], image_width, image_height, 1)

# define the model
input_shape = image_data[0].shape
print(image_data[0].shape)

"""
model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=input_shape))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(number_classes, activation='softmax'))

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])

model.fit(X_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(X_test, y_test))
score = model.evaluate(X_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
"""

model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=input_shape))
model.add(Conv2D(32, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
#model.add(Dropout(0.25))

#model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
#model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(128))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(number_classes))
model.add(Activation('softmax'))

# this is the mnist compile
model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])
model.fit(X_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(X_test, y_test))

# score
score = model.evaluate(X_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

# extra stuff -- seeing if we can save the model
# serialize model to JSON
model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("model.h5")
print("Saved model to disk")
