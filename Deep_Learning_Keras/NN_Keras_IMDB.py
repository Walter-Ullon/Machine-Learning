#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 13 18:07:12 2018

@author: walter
"""

#=================================
#       IMPORT LIBRARIES:
#=================================
from keras.datasets import imdb
import numpy as np
import matplotlib.pyplot as plt

#=================================
#       IMPORT DATASET:
#=================================
# create partitions of train sets and test sets.
# from the imdb dataset, bring only the 10,000 most common words:
(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=10000)

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

#========================================
#   DEFINE TRAIN AND TEST SET LABELS:
#========================================
y_train = np.asarray(train_labels).astype('float32')
y_test = np.asarray(test_labels).astype('float32')

#========================================
#       DEFINE VALIDATION SET:
#========================================
x_val = x_train[:10000]
partial_x_train = x_train[10000:]

y_val = y_train[:10000]
partial_y_train = y_train[10000:]
#========================================
#       BUILD NEURAL NETWORK:
#========================================
from keras import models, layers

network = models.Sequential()
network.add(layers.Dense(16, activation='relu', input_shape=(10000,)))
network.add(layers.Dense(16, activation='relu'))
network.add(layers.Dense(1, activation='sigmoid'))

#========================================
#         COMPILE NETWORK:
#========================================
# set optimizer (for adjusting weights in the backward pass),
# set loss function (for calculating error),
# set metric (for evaluating performance):
network.compile(optimizer='rmsprop',
                loss='binary_crossentropy',
                metrics=['acc'])

#========================================
#              TRAIN:
#========================================
history = network.fit(partial_x_train, 
                      partial_y_train,
                      epochs=5,
                      batch_size=512,
                      validation_data=(x_val, y_val))


#========================================
#         PLOT MODEL RESULTS:
#========================================
results = network.evaluate(x_test, y_test)
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



