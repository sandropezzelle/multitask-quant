from __future__ import print_function
import sys
import csv, numpy as np, tensorflow as tf, os, time
import os
from scipy import spatial
import keras
from keras.models import Sequential
from keras.utils.np_utils import to_categorical
from keras.layers import Dense, Activation
from keras.callbacks import ModelCheckpoint
import keras.utils
import sklearn
from sklearn.metrics import confusion_matrix
from keras import utils as np_utils


def read_data(data_path):
    """
    it reads train/validation/test files
    """
    tr = data_path + 'train_vectors.txt'
    v = data_path + 'val_vectors.txt'
    tst = data_path + 'test_vectors.txt'
    return tr, v, tst

def load_data(split):
    """
    load frozen vectors and labels
    """
    with open(split,'r') as splitfile:
        reader = [line.split() for line in splitfile]
        x_sp = np.zeros((len(reader),2048)) # size of frozen vectors
        y_sp = np.zeros((len(reader)))
        for counter, row in enumerate(reader):
           path = row[0]
           ml = path.split('/')[-1]
           ml2 = ml.split('_')
           targ = int(ml2[1])
           tot = int(ml2[2])
           nontarg = tot - targ
           if targ > nontarg:
             ans = 1
           elif nontarg > targ:
             ans = 0
           elif targ == nontarg:
             ans = 2
           y_sp[counter] = ans
           feat_vector = [float(x) for x in row[1:]]
           x_sp[counter] = feat_vector
         
    x_split = x_sp.reshape((len(reader),2048))
    y_split = y_sp.reshape((len(reader)))
    return x_split, y_split

if __name__ == '__main__':
    """
    it reads the parameters,
    initializes the hyperparameters,
    preprocesses the input,
    trains the model
    """
    data_path = sys.argv[1]
    tr, v, tst = read_data(data_path)
    x_train, y_train = load_data(tr)
    x_val, y_val = load_data(v)

    # data parameters
    dim_vectors = 2048

    # model parameters
    batch_size = 32
    half_size = dim_vectors/2
    hidden_units = 256
    opt = "sgd"

    # define model
    model = Sequential()
    model.add(Dense(units=half_size, input_dim=dim_vectors))
    model.add(Activation('relu'))
    model.add(Dense(units=hidden_units, input_dim=half_size))
    model.add(Activation('relu'))
    model.add(Dense(units=3))
    model.add(Activation('softmax'))

    model.compile(loss=keras.losses.sparse_categorical_crossentropy,
                                              optimizer=opt,
                                              metrics=['accuracy'])

    classes = model.predict(x_val, batch_size=batch_size)
    loss_and_metrics = model.evaluate(x_val,y_val, batch_size=batch_size)
    print('validation loss:',loss_and_metrics)

    # save best weigths
    filepath = "best_model/weight.best.hdf5"
    checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
    callbacks_list = [checkpoint]

    # train for 100 epochs
    print('Train the model...')
    model.fit(x_train, y_train,
              batch_size=batch_size,
              epochs=100,
              validation_data=[x_val, y_val],
              callbacks=callbacks_list)

