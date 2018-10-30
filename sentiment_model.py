from tensorflow import keras
import tensorflow as tf
import numpy as np
import os
import _pickle as pickle
from gensim.models import KeyedVectors
from keras import Input, Model
from keras.initializers import Ones
from keras.layers import Conv2D, LSTM, Dropout, Activation, Reshape, Masking, concatenate
from data_handler import DataHandler
from keras.utils.np_utils import to_categorical


class SentimentModel:
    def __init__(self, w2v_path = 'w2v.bin', data_path = 'data.csv'):
        # Load w2v and data
        self._w2v = KeyedVectors.load_word2vec_format(w2v_path, binary=True)
        print('Word to vector model from path "' + w2v_path + '" loaded.')
        
        self._data = DataHandler(self._w2v, data_path, 100000)

        # Construct the model
        input = Input(shape=(300,256,1))

        conv1 = Conv2D(filters=1, kernel_size=(300, 3), strides=(300,1), padding='same', use_bias=True, bias_initializer=Ones(), activation='relu')(input)
        conv2 = Conv2D(filters=1, kernel_size=(300, 4), strides=(300,1), padding='same', use_bias=True, bias_initializer=Ones(), activation='relu')(input)
        conv3 = Conv2D(filters=1, kernel_size=(300, 5), strides=(300,1), padding='same', use_bias=True, bias_initializer=Ones(), activation='relu')(input)
        con = concatenate([conv1, conv2, conv3], axis=1)
        dropout = Dropout(0.5)(con)
        reshape = Reshape((3, 256))(dropout)
        masking = Masking(mask_value=0, input_shape=(1, 256))(reshape)
        lstm = LSTM(units=(2), activation='tanh', use_bias=True)(masking)
        act = Activation('sigmoid')(lstm)

        self._model = Model(inputs=[input], outputs=[act])
        print('Keras model constructed.')
        print(self._model.summary())

    def fit(self):
        train = self._data.get_training_data()
        validation = self._data.get_validation_data()

        self._model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics=['accuracy'])

        x = np.array([d['sentence'] for d in train])
        y = to_categorical(np.array([d['label'] for d in train]), num_classes=2)
        x_val = np.array([d['sentence'] for d in validation])
        y_val = to_categorical(np.array([d['label'] for d in validation]), num_classes=2)



        history = self._model.fit(
            x = x, 
            y = y,
            verbose=1,
            epochs=5,
            shuffle=True,
            validation_data=(x_val, y_val))

        score = self._model.evaluate(x_val, y_val, verbose=1)
        print('Test loss:', score[0])     
        print('Test accuracy:', score[1])

        # Create an output folder if it doesn't exist
        if not os.path.exists('output'):
            os.makedirs('output')
        
        # Create a new folder for the model
        i = 0
        while os.path.exists(f'output/{str(i)}'):
            i = i + 1

        output_dir = f'output/{str(i)}' 
        os.makedirs(output_dir)

        self._model.save(f'{output_dir}/model.h5')

        with open(f'{output_dir}/history.pkl', 'wb') as f:         
            pickle.dump(history.history, f)

SentimentModel().fit()