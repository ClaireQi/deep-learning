# -*- coding: utf-8 -*-
"""
keras minist woth CNN
Created on Sat Aug 25 11:05:05 2018

@author: claire
"""

from keras.model import Sequential
from keras.layers.core import Dense, Dropout, Flatten, Activation
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D
from keras.utlis import np_utils
from keras.datasets import mnist
from keras import backend as K

import numpy as np

import matplotlib.pyplot as plt

# matplotlib inline
def plot_sample(x):
    plt.figure()
    plt.imshow(x,cmap='gray')


# load data from mnist
(x_train, y_train),(x_test, y_test) = mnist.load_data()
# print the information of the mnist data
print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)
# plot an example of the train data
plot_sample(x_train[20])

# the data_format in tensorflow is 'channles_last' (sample_size,img_rows,img_cols,img_channels)
img_rows = 28
img_cols = 28
img_channel = 1
if K.image_data_format() == 'channels_first':
    shape_ord = (img_channel,img_rows,img_cols)
else:
    shape_ord = (img_rows, img_cols,img_channel)
    
# preprocessing
def preprocess_data(x):
    return x/255
x_train = x_train.reshape((x_train.shape[0],) + shape_ord)
x_test = x_test.reshape((x_test.shape[0],) + shape_ord)
x_train = x_train.astype('float')
x_test = x_test.astype('float')
x_train = preprocess_data(x_train)
x_test = preprocess_data(x_test)

#one-hot coding
nb_classes = 10
y_train = np_utils.to_categorical(y_train, nb_classes)
y_test = np_utils.to_categoriacal(y_test,nb_classes)

# setting parameters
kernel_size = (3,3)
pool_size = (2,2)
epochs = 3
batch_size = 128
nb_filters = 32

# setting network
def build_model():
    model = Sequential()
    
    model.add(Conv2D(nb_filters,kernel_size = kernel_size, input_shape = shape_ord))
    model.add(Activation('relu'))
    
    model.add(Conv2D(nb_filters//2, kernel_size = kernel_size))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size = pool_size))
    model.add(Dropout(0.25))
    
    model.add(Flatten())
    model.add(Dense(128))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    
    model.add(Dense(nb_classes))
    model.add(Activation('softmax'))
    
# training model
model = build_model()
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(x_train,y_train,batch_size=batch_size,epochs=epochs, verbose=1, validation_split=0.05)


    









