import pickle as p
import numpy as np

import matplotlib.pyplot as plt

from keras import backend as K
from keras.regularizers import l2
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping
from keras.models import Model, Sequential
from keras.layers import Dense, Conv2D, Input, MaxPool2D, Flatten, Concatenate

from utils import INPUT_SHAPE


class SiameseNetwork():
    def __init__(self):
        self.input_shape = INPUT_SHAPE
        self.model = []
        self.triplet_path = ''
        self.convnet()

    def initialize_bias(self, shape, name=None, dtype=None):
        """
            The paper, http://www.cs.utoronto.ca/~gkoch/files/msc-thesis.pdf
            suggests to initialize CNN layer bias with mean as 0.5 and standard deviation of 0.01
        """
        return np.random.normal(loc = 0.5, scale = 1e-2, size = shape)
        
    def triplet_loss(self, y_true, y_pred, alpha = 0.4):
        """
        Implementation of the triplet loss function
        Arguments:
        y_true -- true labels, required when you define a loss in Keras, you don't need it in this function.
        y_pred -- python list containing three objects:
                anchor -- the encodings for the anchor data
                positive -- the encodings for the positive data (similar to anchor)
                negative -- the encodings for the negative data (different from anchor)
        Returns:
        loss -- real number, value of the loss
        """
        
        total_lenght = y_pred.shape.as_list()[-1]
        
        anchor = y_pred[:,0:int(total_lenght*1/3)]
        positive = y_pred[:,int(total_lenght*1/3):int(total_lenght*2/3)]
        negative = y_pred[:,int(total_lenght*2/3):int(total_lenght*3/3)]

        # distance between the anchor and the positive
        pos_dist = K.sum(K.square(anchor-positive),axis=1)

        # distance between the anchor and the negative
        neg_dist = K.sum(K.square(anchor-negative),axis=1)

        # compute loss
        basic_loss = pos_dist-neg_dist+alpha
        loss = K.maximum(basic_loss,0.0)
    
        return loss
    
    def convnet(self):
        convnet = Sequential()
        convnet.add(Conv2D(filters=64, kernel_size=(10, 10),activation='relu',
                            input_shape=self.input_shape, padding='valid',
                           bias_initializer=self.initialize_bias,
                                     kernel_regularizer='l2',name='Conv1'))
        convnet.add(MaxPool2D())

        convnet.add(Conv2D(filters=128, kernel_size=(7, 7),activation='relu',
                           padding='valid', kernel_regularizer=l2(2e-4),
                           name='Conv2', bias_initializer=self.initialize_bias))
        convnet.add(MaxPool2D())

        convnet.add(Conv2D(filters=128, kernel_size=(4, 4),activation='relu', 
                            padding='valid',kernel_regularizer=l2(2e-4),
                            name='Conv3', bias_initializer=self.initialize_bias))
        convnet.add(MaxPool2D())

        convnet.add(Conv2D(filters=256, kernel_size=(4, 4),activation='relu',
                                     kernel_regularizer=l2(2e-4),name='Conv4',
                                     bias_initializer=self.initialize_bias))
        
        convnet.add(Flatten())
        convnet.add(Dense(units=4096, activation='relu',
                          bias_initializer=self.initialize_bias,
                          kernel_regularizer=l2(2e-3),name='Dense1'))
        
        anchor_input = Input(self.input_shape, name='anchor_input')
        positive_input = Input(self.input_shape, name='positive_input')
        negative_input = Input(self.input_shape, name='negative_input')

        # Shared embedding layer for positive and negative items
        encoded_anchor = convnet(anchor_input)
        encoded_positive = convnet(positive_input)
        encoded_negative = convnet(negative_input)

        merged_vector = Concatenate(axis=-1, name='merged_layer')([encoded_anchor, encoded_positive, encoded_negative])

        self.model = Model([anchor_input,positive_input, negative_input], merged_vector)
        self.model.compile(loss=self.triplet_loss, optimizer=Adam(lr=0.0001))

    def get_data(self):
        triplet_pairs = p.load(open(self.triplet_path+'triplet_pairs.p', 'rb'))
        return triplet_pairs

    def get_callback(self):
        es = EarlyStopping(monitor='val_loss', mode='min')
        return [es]
