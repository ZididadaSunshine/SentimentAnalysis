import tensorflow as tf
import numpy as np
import pandas as pd
import os
import _pickle as pickle
from tensorflow import keras
from gensim.models import KeyedVectors
from nltk.corpus import stopwords
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from keras.utils.np_utils import to_categorical
from sklearn.model_selection import train_test_split
from keras import Model
from keras.layers import Embedding
from keras.initializers import Ones
from keras.models import Sequential
from keras.layers import Conv2D, Input, LSTM, Embedding, Dropout, Activation, Reshape, Masking, Concatenate

# https://rajmak.wordpress.com/2017/12/07/text-classification-classifying-product-titles-using-convolutional-neural-network-and-word2vec-embedding/

MAX_SEQUENCE_LENGTH = 256
EMBEDDING_DIM = 300
STOPWORDS = set(stopwords.words("english"))

class Test:
    def __init__(self, vectors_path = 'vectors.bin', data_path = 'data.csv'):
        # Load and process data
        dataset = pd.read_csv(data_path, 
                     sep=',', 
                     encoding='latin-1',
                     usecols=[0, 5],
                     names=['category', 'sentence'],
                     converters={'category': category_converter, 'sentence': sentence_converter})

        tokenizer = Tokenizer()
        tokenizer.fit_on_texts(dataset['sentence'])
        sequences = tokenizer.texts_to_sequences(dataset['sentence'])
        sequences = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)
        categories = to_categorical(dataset['category'], 2)

        negatives = dataset.index[dataset['category'] == 0].tolist()
        positives = dataset.index[dataset['category'] == 1].tolist()

        print(f"Positives: {len(negatives)}, Negatives: {len(positives)}")

        x_pos_train, x_pos_val = train_test_split(np.take(sequences, positives, axis=0), train_size=0.6)
        y_pos_train, y_pos_val = train_test_split(np.take(categories, positives, axis=0), train_size=0.6)

        x_neg_train, x_neg_val = train_test_split(np.take(sequences, negatives, axis=0), train_size=0.6)
        y_neg_train, y_neg_val = train_test_split(np.take(categories, negatives, axis=0), train_size=0.6)



        x_train = np.concatenate((x_pos_train, x_neg_train))
        y_train = np.concatenate((y_pos_train, y_neg_train))

        x_val = np.concatenate((x_pos_val, x_neg_val))
        y_val = np.concatenate((y_pos_val, y_neg_val))
            
        word2vec = KeyedVectors.load_word2vec_format(vectors_path, binary=True)

        word_index = tokenizer.word_index
        nb_words = len(word_index)+1

        
        embedding_matrix = np.zeros((nb_words, EMBEDDING_DIM))
        for word, i in word_index.items():
            if word in word2vec.vocab:
                embedding_matrix[i] = word2vec.word_vec(word)

        main_input = Input(shape=(MAX_SEQUENCE_LENGTH,))
        embedding_layer = Embedding(embedding_matrix.shape[0],
                                    embedding_matrix.shape[1],
                                    weights=[embedding_matrix],
                                    input_length=MAX_SEQUENCE_LENGTH,
                                    trainable=False)(main_input)

        reshape1 = Reshape((EMBEDDING_DIM, MAX_SEQUENCE_LENGTH, 1))(embedding_layer)
        conv1    = Conv2D(filters=1, kernel_size=(EMBEDDING_DIM, 3), strides=(EMBEDDING_DIM, 1), padding='same', use_bias=True, bias_initializer=Ones(), activation='relu')(reshape1)
        conv2    = Conv2D(filters=1, kernel_size=(EMBEDDING_DIM, 4), strides=(EMBEDDING_DIM,1), padding='same', use_bias=True, bias_initializer=Ones(), activation='relu')(reshape1)
        conv3    = Conv2D(filters=1, kernel_size=(EMBEDDING_DIM, 5), strides=(EMBEDDING_DIM,1), padding='same', use_bias=True, bias_initializer=Ones(), activation='relu')(reshape1)
        con      = Concatenate(axis=1)([conv1, conv2, conv3])
        dropout  = Dropout(0.5)(con)
        reshape2 = Reshape((3, MAX_SEQUENCE_LENGTH))(dropout)
        masking  = Masking(mask_value=0)(reshape2)
        lstm     = LSTM(units=(2), activation='tanh', use_bias=True)(masking)
        act      = Activation('sigmoid')(lstm)

        model = Model(inputs=[main_input], outputs=[act])
        model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['acc'])

        model.summary()

        history = model.fit(x_train, y_train, validation_data=(x_val, y_val), epochs=15, batch_size=128, shuffle=True)
        score = model.evaluate(x_val, y_val, verbose=0)
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

        model.save(f'{output_dir}/model.h5')

        with open(f'{output_dir}/history.pkl', 'wb') as f:         
            pickle.dump(history.history, f)


def category_converter(val):
    val = int(val)
    return 0 if val == 0 else 1

def sentence_converter(text):
    text = text.strip().lower().split()
    text = filter(lambda word: word not in STOPWORDS, text)
    return " ".join(text)

Test()