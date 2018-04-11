import keras
from keras.models import Model
from keras import layers
from keras.layers import Input, Dense, Flatten, merge, Dropout
from keras.optimizers import Adam, Adadelta, Nadam
from keras.applications.inception_v3 import InceptionV3
from keras.layers.wrappers import TimeDistributed
from keras.layers.pooling import GlobalAveragePooling2D
from keras.layers import LSTM
from keras.layers import Reshape
from keras.layers import Lambda
import tensorflow as tf
from keras import backend as K
from keras.regularizers import l2

class MultitaskInc:
    def __init__(self, input_shape = (203,203, 3), act_f = "relu", batch_size = 100):
        """
        intialization of hyperparameters
        """
        self.act_f = act_f
        self.batch_size = batch_size
        self.input_shape = input_shape
        self.num_classes = 17
        self.q_classes = 9
        self.l2reg = 1e-8
        self.dropout = 0.5
        self.more_classes = 3

    def build(self):
        """
        loads the inception network,
        contains the fully connected layers,
        for each task, it predicts the class
        it returns the prediction for 3 tasks
        """
        model_inception = InceptionV3(weights = None, include_top = False)
        inp = Input(self.input_shape, name = 'more_input')

        out_inc = model_inception(inp)
        out_res = Reshape((25,2048))(out_inc)


        td_dense = TimeDistributed(Dense(1024, W_regularizer = l2(self.l2reg), activation = 'relu'))
        drop_td_dense = Dropout(self.dropout)
        l_td_dense = td_dense(out_res)
        drop_l_td_dense = drop_td_dense(l_td_dense)

        td_dense2 = TimeDistributed(Dense(512, W_regularizer = l2(self.l2reg), activation = 'relu'))
        drop_td_dense2 = Dropout(self.dropout)
        l_td_dense2 = td_dense2(drop_l_td_dense)
        drop_l_td_dense2 = drop_td_dense2(l_td_dense2)

        td_dense3 = TimeDistributed(Dense(256, W_regularizer = l2(self.l2reg), activation = 'relu'))
        drop_td_dense3 = Dropout(self.dropout)
        l_td_dense3 = td_dense3(drop_l_td_dense2)
        drop_l_td_dense3 = drop_td_dense3(l_td_dense3)

        td_dense4 = TimeDistributed(Dense(128, W_regularizer = l2(self.l2reg), activation = 'relu'))
        drop_td_dense4 = Dropout(self.dropout)
        l_td_dense4 = td_dense4(drop_l_td_dense3)
        drop_l_td_dense4 = drop_td_dense4(l_td_dense4)

        td_dense5 = TimeDistributed(Dense(64, W_regularizer = l2(self.l2reg), activation = 'relu'))
        drop_td_dense5 = Dropout(self.dropout)
        l_td_dense5 = td_dense5(drop_l_td_dense4)
        drop_l_td_dense5 = drop_td_dense5(l_td_dense5)

        td_dense6 = TimeDistributed(Dense(32, W_regularizer = l2(self.l2reg), activation = 'relu'))
        drop_td_dense6 = Dropout(self.dropout)
        l_td_dense6 = td_dense6(drop_l_td_dense5)
        drop_l_td_dense6 = drop_td_dense6(l_td_dense6)

        more_flat = Flatten()(drop_l_td_dense2)
        hidden_more = Dense(512, W_regularizer = l2(self.l2reg),  activation = 'relu', name = 'msl')(more_flat)
        drop_hidden_more = Dropout(self.dropout)(hidden_more)
        out_more = Dense(self.more_classes, activation = 'softmax', name = 'pred1')(drop_hidden_more)

        quant_flat = Flatten()(drop_l_td_dense4)
        hidden_quant = Dense(128, W_regularizer = l2(self.l2reg), activation = 'relu', name ='quant')(quant_flat)
        drop_hidden_quant = Dropout(self.dropout)(hidden_quant)
        out_quant = Dense(self.q_classes, activation = 'softmax', name = 'pred2')(drop_hidden_quant)

        card_flat = Flatten()(drop_l_td_dense6)
        hidden_card = Dense(32, W_regularizer = l2(self.l2reg), activation = 'relu', name = 'card')(card_flat)
        drop_hidden_card = Dropout(self.dropout)(hidden_card)
        out_card = Dense(self.num_classes,  activation = 'softmax', name = 'pred3')(drop_hidden_card)
        model = Model(input = inp, output = [out_more, out_quant, out_card])
        model.compile(loss = 'categorical_crossentropy', optimizer = 'sgd', metrics = ['accuracy'])
        return model
