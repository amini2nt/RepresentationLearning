#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 26 11:46:35 2019

@author: Zeyu Li
"""

import csv
import os
import numpy as np
import timeit
import matplotlib.pyplot as plt
from matplotlib.image import imread
from keras.models import load_model

model_catsdogs = load_model("catsdogs.h5")

imgs = []

directory = "test/"
filelist = os.listdir(directory)
idlist = [int(filename[:-4]) for filename in filelist]
idlist.sort()

for i in idlist:
    im = imread(directory + str(i) + '.jpg')
    
#    # CNN
    if im.shape == (64, 64, 3):
        imgs += [im]
    elif im.shape == (64, 64):
        imgs += [np.stack([im, im, im], -1)]
    else:
        print(i)

x_data = np.stack(imgs, 0)
x_data = (x_data - x_data.mean()) / (x_data.max() - x_data.min())

start = timeit.default_timer()

y_data = model_catsdogs.predict(x_data, batch_size=64)

stop = timeit.default_timer()
print("Run time: " + str(stop - start))

csv_catsdogs = open('catsdogs.csv', 'w')
writer_catsdogs = csv.writer(csv_catsdogs)

msgs = [[idlist[i], "Cat" if y_data[i][0] >= 0.5 else "Dog"] for i in range(len(idlist))]

writer_catsdogs.writerows([["id", "label"]])
writer_catsdogs.writerows(msgs)
csv_catsdogs.close()

n = 0

for i in range(0, len(idlist)):
    if y_data[i][0] > 0.49 and y_data[i][0] < 0.51:
        if n == 0:
            plt.figure(figsize=(18, 9))
        n += 1
        
        ax = plt.subplot(3, 6, n)
        ax.set_title(i)
        ax.axis('off')
        ax.imshow(imgs[i])
        if n == 18: break

plt.show()

cats_miss = [32, 45, 47, 191, 218, 289, 323, 388, 439, 479, 542, 602, 675, 686, 737, 810, 821, 871]

n = 0

for i in cats_miss:
    if n == 0:
        plt.figure(figsize=(18, 9))
    n += 1
    
    ax = plt.subplot(3, 6, n)
    ax.set_title(i)
    ax.axis('off')
    ax.imshow(imgs[i])
    if n == 18: break

plt.show()

dogs_miss = [59, 82, 94, 156, 280, 382, 420, 444, 446, 449, 538, 582, 698, 802, 807, 924, 925, 930]

n = 0

for i in dogs_miss:
    if n == 0:
        plt.figure(figsize=(18, 9))
    n += 1
    
    ax = plt.subplot(3, 6, n)
    ax.set_title(i)
    ax.axis('off')
    ax.imshow(imgs[i])
    if n == 18: break

plt.show()
