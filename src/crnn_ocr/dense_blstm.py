# -*- coding: utf-8 -*-

from tensorflow import keras
from tensorflow.keras.layers import Conv2D, BatchNormalization, Activation, Dropout, Dense, \
    ZeroPadding2D, AveragePooling2D, concatenate, Permute, TimeDistributed, Bidirectional, \
    Flatten, GRU
from tensorflow.keras.regularizers import l2
import logging

logging.basicConfig(level=logging.INFO)

class denseBlstm(object):
    def __init__(self):
        pass


    def conv_block(self, input, growth_rate, dropout_rate=None, weight_decay=1e-4):
        x = BatchNormalization(axis=-1, epsilon=1.1e-5)(input)
        x = Activation('relu')(x)
        x = Conv2D(growth_rate, (3, 3), kernel_initializer='he_normal', padding='same')(x)
        if (dropout_rate):
            x = Dropout(dropout_rate)(x)
        return x


    def dense_block(self, x, nb_layers, nb_filter, growth_rate, droput_rate=0.2, weight_decay=1e-4):
        for i in range(nb_layers):
            cb = self.conv_block(x, growth_rate, droput_rate, weight_decay)
            x = concatenate([x, cb], axis=-1)
            nb_filter += growth_rate
        return x, nb_filter


    def transition_block(self, input, nb_filter, dropout_rate=None, pooltype=1, weight_decay=1e-4):
        x = BatchNormalization(axis=-1, epsilon=1.1e-5)(input)
        x = Activation('relu')(x)
        x = Conv2D(nb_filter, (1, 1), kernel_initializer='he_normal', padding='same', use_bias=False,
                   kernel_regularizer=l2(weight_decay))(x)

        if (dropout_rate):
            x = Dropout(dropout_rate)(x)

        if (pooltype == 2):
            x = AveragePooling2D((2, 2), strides=(2, 2))(x)
        elif (pooltype == 1):
            x = ZeroPadding2D(padding=(0, 1))(x)
            x = AveragePooling2D((2, 2), strides=(2, 1))(x)
        elif (pooltype == 3):
            x = AveragePooling2D((2, 2), strides=(2, 1))(x)
        return x, nb_filter


    def build_dense_blstm(self, input, nclass):
        _dropout_rate = 0.2
        _weight_decay = 1e-4
        _nb_filter = 64
        x = Conv2D(_nb_filter, (5, 5), strides=(2, 2), kernel_initializer='he_normal', padding='same',
                   use_bias=False, kernel_regularizer=l2(_weight_decay))(input)
        x, _nb_filter = self.dense_block(x, 8, _nb_filter, 8, None, _weight_decay)
        x, _nb_filter = self.transition_block(x, 128, _dropout_rate, 2, _weight_decay)
        x, _nb_filter = self.dense_block(x, 8, _nb_filter, 8, None, _weight_decay)
        x, _nb_filter = self.transition_block(x, 128, _dropout_rate, 2, _weight_decay)
        x, _nb_filter = self.dense_block(x, 8, _nb_filter, 8, None, _weight_decay)
        x = BatchNormalization(axis=-1, epsilon=1.1e-5)(x)
        x = Activation('relu')(x)
        x = Permute((2, 1, 3))(x)
        x = TimeDistributed(Flatten())(x)
        # x = Bidirectional(GRU(256, return_sequences=True, kernel_initializer='he_normal', implementation=2))(x)
        # x = Bidirectional(GRU(256, return_sequences=True, kernel_initializer='he_normal', implementation=2))(x)
        output = Dense(nclass, activation='softmax', name='output')(x)
        return output


