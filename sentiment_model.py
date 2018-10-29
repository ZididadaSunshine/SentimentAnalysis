from tensorflow import keras
import tensorflow as tf
import numpy as np
from gensim.models import KeyedVectors
from keras import Input, Model
from keras.initializers import Ones
from keras.layers import Conv2D, LSTM, Dropout, Activation, Reshape, Masking
from data_handler import DataHandler


class SentimentModel:
    def __init__(self, w2v_path = 'w2v.bin', data_path = 'data.csv'):
        # Load w2v and data
        self._w2v = KeyedVectors.load_word2vec_format(w2v_path, binary=True)
        print('Word to vector model from path "' + w2v_path + '" loaded.')
        
        self._data = DataHandler(self._w2v, data_path, 200)

        # Construct the model
        input = Input(shape=(300,256,1))

        conv1 = Conv2D(filters=1, kernel_size=(300, 3), strides=(300,1), padding='same', use_bias=True, bias_initializer=Ones(), activation='relu')(input)
        dropout = Dropout(0.5)(conv1)
        reshape = Reshape((1, 256))(dropout)
        masking = Masking(mask_value=0, input_shape=(1, 256))(reshape)
        lstm = LSTM(units=(2), activation='tanh', use_bias=True)(masking)
        act = Activation('sigmoid')(lstm)

        self._model = Model(inputs=[input], outputs=[act])
        print('Keras model constructed.')
        print(self._model.summary())

    def fit(self):
        train = self._data.get_training_data()
        validation = self._data.get_validation_data()

        self._model.compile(loss = 'MSE', optimizer = 'RMSprop', metrics=['accuracy'])

        x = [d['sentence'] for d in train]
        y = [d['label'] for d in train]
        x_val = [d['sentence'] for d in validation]
        y_val = [d['label'] for d in validation]


        """ history = self._model.fit(
            x = [x], 
            y = y,
            verbose=1,
            validation_data=([x_val], y_val)) """

SentimentModel()
