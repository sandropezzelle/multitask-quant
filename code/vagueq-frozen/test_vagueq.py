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

"""
uncomment this line above for saving
the predictions for the vagueQ task 

"""
#output = open('vagueq_predictions.txt', 'w')


"""
probability distributions for each ratio relative to each quantifier
quantifiers: none,almost_none,the_smaller_part,few,some,many,most,almost_all,all

"""
ratio0Y=[0.9765886288,0.0050167224,0,0.0033444816,0.0100334448,0,0,0,0.0050167224]
ratio19=[0.0051020408,0.5595238095,0.1139455782,0.2823129252,0.0238095238,0,0.0017006803,0.0136054422,0]
ratio15=[0.0034542314,0.3195164076,0.2227979275,0.3730569948,0.0759930915,0,0,0.0051813472,0]
ratio14=[0.0051194539,0.2679180887,0.2457337884,0.3788395904,0.0836177474,0,0.0119453925,0.0068259386,0]
ratio13=[0.0016949153,0.1593220339,0.286440678,0.4084745763,0.1169491525,0.0084745763,0.0118644068,0.0033898305,0.0033898305]
ratio12=[0.0034364261,0.087628866,0.3453608247,0.2508591065,0.2577319588,0.0171821306,0.0257731959,0.0120274914,0]
ratio23=[0,0.0086956522,0.3426086957,0.172173913,0.3304347826,0.0660869565,0.0626086957,0.0156521739,0.0017391304]
ratio34=[0.0034722222,0.0086805556,0.2447916667,0.1371527778,0.4670138889,0.0572916667,0.0694444444,0.0104166667,0.0017361111]
ratio11=[0,0.0052631579,0.0614035088,0.0456140351,0.4842105263,0.1578947368,0.2263157895,0.0192982456,0]
ratio43=[0,0.0051369863,0.0188356164,0.0154109589,0.2859589041,0.1969178082,0.4280821918,0.0479452055,0.0017123288]
ratio32=[0,0.0052631579,0.0140350877,0.0175438596,0.1842105263,0.2035087719,0.5228070175,0.050877193,0.001754386]
ratio21=[0.0051635112,0.0034423408,0.017211704,0.0223752151,0.0671256454,0.1686746988,0.5938037866,0.1204819277,0.0017211704]
ratio31=[0.0033840948,0,0.0203045685,0.0084602369,0.0101522843,0.1539763113,0.5397631134,0.2622673435,0.0016920474]
ratio41=[0,0.006779661,0.0084745763,0.006779661,0.0118644068,0.1254237288,0.4898305085,0.3474576271,0.0033898305]
ratio51=[0,0.0118243243,0.0084459459,0,0.0050675676,0.1081081081,0.3969594595,0.4695945946,0]
ratio91=[0.0016863406,0.0151770658,0.0016863406,0.0033726813,0.0016863406,0.0573355818,0.2293423272,0.6846543002,0.0050590219]
ratioX0=[0.0016722408,0.0016722408,0,0.0016722408,0.0016722408,0.0033444816,0.0050167224,0.0016722408,0.983277592]


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
        y_sp = np.zeros((len(reader),9)) # no. of classes (quantifiers)
        for counter, row in enumerate(reader):
           path = row[0]
           ratio = path.split('/')[-2]
           prob = eval(ratio)
           y_sp[counter] = prob
           feat_vector = [float(x) for x in row[1:]]
           x_sp[counter] = feat_vector

    x_split = x_sp.reshape((len(reader),2048))
    y_split = y_sp.reshape((len(reader),9)) # prob distributions
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
    hidden_units = 128
    opt = "sgd"
    filepath = 'best_model/weight.best.hdf5'

    # define model
    model = Sequential()
    model.add(Dense(units=half_size, input_dim=dim_vectors))
    model.add(Activation('relu'))
    model.add(Dense(units=hidden_units, input_dim=half_size))
    model.add(Activation('relu'))
    model.add(Dense(units=9))
    model.add(Activation('softmax'))


    model.load_weights(filepath)
    model.compile(loss=keras.losses.categorical_crossentropy,
                                              optimizer=opt,
                                              metrics=['accuracy'])


    print("Created model and loaded weights from file")
    scores = model.evaluate(x_test, y_test, verbose=1)
    #print(scores)
    print("%s: %.4f%%" % (model.metrics_names[1], scores[1]*100))

    probabilities = model.predict(x_test, batch_size=batch_size)


    """
    uncomment this line above for saving
    the predictions for the vagueQ task 

    """
    """
    for i in range(t_size):
       for j in range(9):
          output.write(str(probabilities[i][j]) + '\t')
       output.write('\n')

       for j in range(9):
          output.write(str(y_test[i][j]) + '\t')
       output.write('\n')
    """

