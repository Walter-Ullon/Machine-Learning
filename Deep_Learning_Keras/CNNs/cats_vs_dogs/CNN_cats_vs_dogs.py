#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 26 18:22:22 2018

@author: walter
"""

'''
NOTE: the dataset used in this example can be found at:
      https://www.kaggle.com/c/dogs-vs-cats/data
      (account is required - free)
'''



#=================================
#       IMPORT LIBRARIES:
#=================================
import os, shutil

#=================================
#       PRE-PROCESS FILES:
#=================================
# directory where full dataset is located:
full_dataset_dir = "/Users/walter/Desktop/Walter_stuff/Machine-Learning/Deep_Learning_Keras/CNNs/cats_vs_dogs/full_dataset/train"

# directory where smaller, working dataset will be located:
working_dataset_dir = "/Users/walter/Desktop/Walter_stuff/Machine-Learning/Deep_Learning_Keras/CNNs/cats_vs_dogs/working_dataset"
os.mkdir(working_dataset_dir)

# make working train, test, and validation subdirectories:
train_dir = os.path.join(working_dataset_dir, 'train')
os.mkdir(train_dir)

validation_dir = os.path.join(working_dataset_dir, 'validation')
os.mkdir(validation_dir)

test_dir = os.path.join(working_dataset_dir, 'test')
os.mkdir(test_dir)

# make train, test, and validation directories for cat pictures:
train_cats_dir = os.path.join(train_dir, 'cats')
validation_cats_dir = os.path.join(validation_dir, 'cats')
test_cats_dir = os.path.join(test_dir, 'cats')
os.mkdir(train_cats_dir)
os.mkdir(validation_cats_dir)
os.mkdir(test_cats_dir)

# make train, test, and validation directories for dog pictures:
train_dogs_dir = os.path.join(train_dir, 'dogs')
validation_dogs_dir = os.path.join(validation_dir, 'dogs')
test_dogs_dir = os.path.join(test_dir, 'dogs')
os.mkdir(train_dogs_dir)
os.mkdir(validation_dogs_dir)
os.mkdir(test_dogs_dir)


# split the full dataset of cat pictures into train, test, validation sets:
fnames = ['cat.{}.jpg'.format(i) for i in range(1000)]
for fname in fnames:
    src = os.path.join(full_dataset_dir, fname)
    dst = os.path.join(train_cats_dir, fname)
    shutil.copyfile(src, dst)
    
    
fnames = ['cat.{}.jpg'.format(i) for i in range(1000, 1500)]
for fname in fnames:
    src = os.path.join(full_dataset_dir, fname)
    dst = os.path.join(validation_cats_dir, fname)
    shutil.copyfile(src, dst)


fnames = ['cat.{}.jpg'.format(i) for i in range(1500, 2000)]
for fname in fnames:
    src = os.path.join(full_dataset_dir, fname)
    dst = os.path.join(test_cats_dir, fname)
    shutil.copyfile(src, dst)


# split the full dataset of dog pictures into train, test, validation sets:
fnames = ['dog.{}.jpg'.format(i) for i in range(1000)]
for fname in fnames:
    src = os.path.join(full_dataset_dir, fname)
    dst = os.path.join(train_dogs_dir, fname)
    shutil.copyfile(src, dst)
    
    
fnames = ['dog.{}.jpg'.format(i) for i in range(1000, 1500)]
for fname in fnames:
    src = os.path.join(full_dataset_dir, fname)
    dst = os.path.join(validation_dogs_dir, fname)
    shutil.copyfile(src, dst)


fnames = ['dog.{}.jpg'.format(i) for i in range(1500, 2000)]
for fname in fnames:
    src = os.path.join(full_dataset_dir, fname)
    dst = os.path.join(test_dogs_dir, fname)
    shutil.copyfile(src, dst)



#=================================
#       CONSTRUCT NETWORK:
#=================================
from keras import layers, models, optimizers

model = models.Sequential()
model.add(layers.Conv2D(32, (3,3), activation='relu',
                        input_shape=(150, 150, 3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Flatten())
model.add(layers.Dense(512, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', 
              optimizer=optimizers.RMSprop(lr=1e-4),
              metrics=['acc'])

# print model summary:
print(model.summary())














