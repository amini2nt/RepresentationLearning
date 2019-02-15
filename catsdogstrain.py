#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 25 17:13:23 2018

@author: Zeyu Li
"""

import os
import matplotlib.pyplot as plt
import numpy as np
import timeit
from matplotlib.image import imread
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D, ZeroPadding2D
from keras import optimizers

x_data = []
y_data = []
x_test = []
y_test = []

directory = "Cat/"
filelist = os.listdir(directory)

for filename in filelist:
    im = imread(directory + filename)
    
    # CNN
    if im.shape == (64, 64, 3):
        x_data += [im]
        y_data += [[1, 0]]
    elif im.shape == (64, 64):
        x_data += [np.stack([im, im, im], -1)]
        y_data += [[0, 1]]

directory = "Dog/"
filelist = os.listdir(directory)

for filename in filelist:
    im = imread(directory + filename)
    
    # CNN
    if im.shape == (64, 64, 3):
        x_data += [im]
        y_data += [[0, 1]]
    elif im.shape == (64, 64):
        x_data += [np.stack([im, im, im], -1)]
        y_data += [[0, 1]]

x_data = np.stack(x_data, 0)
y_data = np.asarray(y_data)

x_data = (x_data - x_data.mean()) / (x_data.max() - x_data.min())

p = np.random.permutation(len(x_data))

x_data = x_data[p]
y_data = y_data[p]

#x_data = x_data.reshape(len(y_data), 3, 64, 64)

# CNN
model = Sequential()
model.add(ZeroPadding2D((2, 2), input_shape=x_data.shape[1:]))
model.add(Conv2D(32, (5, 5), activation='relu'))
model.add(ZeroPadding2D((2, 2)))
model.add(Conv2D(32, (5, 5), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(ZeroPadding2D((2, 2)))
model.add(Conv2D(64, (5, 5), activation='relu'))
model.add(ZeroPadding2D((2, 2)))
model.add(Conv2D(64, (5, 5), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(64, activation='relu'))
#model.add(Dense(64, activation='relu'))
model.add(Dense(2, activation='softmax'))

lr = 0.01
sgd = optimizers.SGD(lr=lr)

model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

start = timeit.default_timer()

epochs = 50
history = model.fit(x_data, y_data, validation_split=0.1, batch_size=16, epochs=epochs)
model.save("catsdogs4.h5")

stop = timeit.default_timer()

print("Run time: " + str(stop - start))

# summarize history for accuracy
plt.plot(1 - np.asarray(history.history['acc']))
plt.plot(1 - np.asarray(history.history['val_acc']))
plt.title('model error')
plt.ylabel('error')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()

# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()

#print(model.summary())
