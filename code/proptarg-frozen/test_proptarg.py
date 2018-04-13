import sys
import csv, numpy as np, tensorflow as tf, os, time
import os
from scipy import spatial
import keras
from keras.models import Sequential
from keras.utils.np_utils import to_categorical
from keras.layers import Dense, Activation, Dropout
from keras.callbacks import ModelCheckpoint
import keras.utils
import sklearn
from sklearn.metrics import confusion_matrix
from keras import utils as np_utils

"""
classes assigned to each ratio
"""
ratio0Y=0
ratio19=1
ratio15=2
ratio14=3
ratio13=4
ratio12=5
ratio23=6
ratio34=7
ratio11=8
ratio43=9
ratio32=10
ratio21=11
ratio31=12
ratio41=13
ratio51=14
ratio91=15
ratioX0=16


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
        print("importing training feature vectors...")
        reader = [line.split() for line in splitfile]
        x_sp = np.zeros((len(reader),2048)) # size of frozen vectors
        y_sp = np.zeros((len(reader)))
        for counter, row in enumerate(reader):
           path = row[0]
           ratio = path.split('/')[-2]
           prob = eval(ratio)
           y_sp[counter] = prob
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
    x_test, y_test = load_data(tst)

    # data parameters
    dim_vectors = 2048

    # model parameters
    batch_size = 64
    half_size = dim_vectors/2
    hidden_units = 64
    opt = "sgd"
    drop = 0.5

    # define model
    model = Sequential()
    model.add(Dense(units=half_size, input_dim=dim_vectors))
    model.add(Activation('relu'))
    model.add(Dropout(drop))
    model.add(Dense(units=hidden_units, input_dim=half_size))
    model.add(Activation('relu'))
    model.add(Dense(units=17))
    model.add(Activation('softmax'))

    filepath= 'best_model/weight.best.hdf5'

    model.load_weights(filepath)
    model.compile(loss=keras.losses.sparse_categorical_crossentropy,
                                              optimizer=opt,
                                              metrics=['accuracy'])


    print("Created model and loaded weights from file")
    scores = model.evaluate(x_test, y_test, verbose=1)
    print(scores)
    print("%s: %.4f%%" % (model.metrics_names[1], scores[1]*100))

    """
    # comment out this code for the confusion matrix
    
    y_pred = model.predict_classes(x_test, verbose=1)
    y_predarr = np.asarray(y_pred)
    y_tstarr = np.array(y_test, dtype=np.float16)
    print(y_pred,y_test)
    print(confusion_matrix(y_tstarr, y_predarr))
    """

