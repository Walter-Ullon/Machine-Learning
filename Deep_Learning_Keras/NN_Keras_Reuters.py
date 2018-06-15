#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 14 20:45:58 2018

@author: walter
"""

#=================================
#       IMPORT LIBRARIES:
#=================================
from keras.datasets import reuters
import numpy as np
import matplotlib.pyplot as plt
from keras.utils.np_utils import to_categorical

#=================================
#       IMPORT DATASET:
#=================================
# create partitions of train sets and test sets.
# from the reuters dataset, bring only the 10,000 most common words:
(train_data, train_labels), (test_data, test_labels) = reuters.load_data(num_words=10000)

#=================================
#        TOKENIZE DATA:
#=================================
# function to tokenize data.
def tokenize(data, dimension=10000):
    rows = len(data)
    columns = dimension
    # set empty tensor:
    tokenized_tensor = np.zeros((rows, columns))
    # turn every word in list into 1 if present and save in index:
    for row_num, words_lst in enumerate(data):
        tokenized_tensor[row_num, words_lst] = 1
        
    return tokenized_tensor


#========================================
#   DEFINE TRAIN AND TEST SET INPUTS:
#========================================
x_train = tokenize(train_data)
x_test = tokenize(test_data)

#==============================================
#   ONE-HOT ENCODE TRAIN AND TEST SET LABELS:
#==============================================
one_hot_train = to_categorical(train_labels)
one_hot_test = to_categorical(test_labels)

#========================================
#       DEFINE VALIDATION SET:
#========================================
x_val = x_train[:1000]
partial_x_train = x_train[1000:]

y_val = one_hot_train[:1000]
partial_y_train = one_hot_train[1000:]
#========================================
#       BUILD NEURAL NETWORK:
#========================================
from keras import models, layers
# number of classification categories:
num_categories = 46

network = models.Sequential()
network.add(layers.Dense(128, activation='relu', input_shape=(10000,)))
#network.add(layers.Dense(64, activation='relu'))
#network.add(layers.Dense(64, activation='relu'))
network.add(layers.Dense(num_categories, activation='softmax'))

#========================================
#         COMPILE NETWORK:
#========================================
# set optimizer (for adjusting weights in the backward pass),
# set loss function (for calculating error),
# set metric (for evaluating performance):
network.compile(optimizer='rmsprop',
                loss='categorical_crossentropy',
                metrics=['accuracy'])

#========================================
#              TRAIN:
#========================================
history = network.fit(partial_x_train, 
                      partial_y_train,
                      epochs=8,
                      batch_size=512,
                      validation_data=(x_val, y_val))


#========================================
#         PLOT MODEL RESULTS:
#========================================
results = network.evaluate(x_test, one_hot_test)
print(results)


# Training and validation loss
history_dict = history.history
history_dict.keys()
loss_values = history_dict['loss']
val_loss_values = history_dict['val_loss']
acc = history_dict['acc']
epochs = range(1, len(acc) + 1)

plt.plot(epochs, loss_values, 'bo', label='Training loss')
plt.plot(epochs, val_loss_values, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.show()


# Training and validation accuracy
plt.clf()
acc_values = history_dict['acc']
val_acc_values = history_dict['val_acc']

plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc_values, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.show()