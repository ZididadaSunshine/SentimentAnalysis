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
from keras.optimizers import SGD
from keras.layers import Embedding
from keras.initializers import Ones
from keras.models import Sequential
from keras.layers import Conv2D, Conv1D, Concatenate, Input, LSTM, Embedding, Dropout, Activation, Reshape, Masking, Concatenate

# https://rajmak.wordpress.com/2017/12/07/text-classification-classifying-product-titles-using-convolutional-neural-network-and-word2vec-embedding/

MAX_SEQUENCE_LENGTH = 256
EMBEDDING_DIM = 300
STOPWORDS = set(stopwords.words("english"))
VALIDATION_SPLIT = 0.4

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

        """ indicies = list(dataset.index.values)
        np.random.shuffle(indicies)
        sequences = sequences[indicies]
        categories = categories[indicies]
        nb_validation_samples = int(VALIDATION_SPLIT * sequences.shape[0])
        x_train = sequences[:-nb_validation_samples]
        y_train = categories[:-nb_validation_samples]
        x_val = sequences[-nb_validation_samples:]
        y_val = categories[-nb_validation_samples:] """

            
        word2vec = KeyedVectors.load_word2vec_format(vectors_path, binary=True)

        word_index = tokenizer.word_index
        nb_words = len(word_index)+1

        
        embedding_matrix = np.zeros((nb_words, EMBEDDING_DIM))
        for word, i in word_index.items():
            if word in word2vec.vocab:
                embedding_matrix[i] = word2vec.word_vec(word)

        main_input          = Input(shape=(MAX_SEQUENCE_LENGTH,))
        embedding_layer     = Embedding(embedding_matrix.shape[0],
                                        embedding_matrix.shape[1],
                                        weights=[embedding_matrix],
                                        input_length=MAX_SEQUENCE_LENGTH,
                                        trainable=False)(main_input)
        concatenate_layer   = Concatenate(axis=1)([Conv1D(1, kw, padding='same', use_bias=True, bias_initializer=Ones(), activation='relu', strides=1)(embedding_layer) for kw in (3, 4, 5)])
        dropout_layer       = Dropout(0.5)(concatenate_layer)
        masking_layer       = Masking(mask_value=0)(dropout_layer)
        lstm_layer          = LSTM(units=(2), activation='tanh', use_bias=True)(masking_layer)
        activation_layer    = Activation('sigmoid')(lstm_layer)

        model = Model(inputs=[main_input], outputs=[activation_layer])
        model.compile(loss='categorical_crossentropy', optimizer=SGD(lr=0.1), metrics=['acc'])
        model.summary()

        """ model = Sequential()
        model.add(Embedding(embedding_matrix.shape[0],
                            embedding_matrix.shape[1],
                            weights=[embedding_matrix],
                            input_length=MAX_SEQUENCE_LENGTH,
                            trainable=False))
        model.add(Conv1D(1, 3, padding='same', use_bias=True, bias_initializer=Ones(), activation='relu', strides=1))
        model.add(Dropout(0.5))
        model.add(Masking(mask_value=0))
        model.add(LSTM(units=(2), activation='tanh', use_bias=True))
        model.add(Activation('sigmoid'))

        model.compile(loss='categorical_crossentropy', optimizer=SGD(lr=0.1), metrics=['acc'])

        model.summary() """

        history = model.fit(x_train, y_train, validation_data=(x_val, y_val), epochs=5, batch_size=128, shuffle=True)
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
nb_words = 1000
embedding_matrix = np.zeros((nb_words, EMBEDDING_DIM))
""" submodels = []
for kw in (3, 4, 5):
    submodel = Sequential()
    submodel.add(Embedding(embedding_matrix.shape[0],
                    embedding_matrix.shape[1],
                    weights=[embedding_matrix],
                    input_length=MAX_SEQUENCE_LENGTH,
                    trainable=False))
    submodel.add(Conv1D(EMBEDDING_DIM, kw, padding='same', activation='relu', strides=1))
    submodels.append(submodel)  """

""" model = Sequential()
model.add(Embedding(embedding_matrix.shape[0],
                    embedding_matrix.shape[1],
                    weights=[embedding_matrix],
                    input_length=MAX_SEQUENCE_LENGTH,
                    trainable=False))
model.add(Conv1D(1, (EMBEDDING_DIM, 3), padding='same', use_bias=True, bias_initializer=Ones(), activation='relu', strides=(EMBEDDING_DIM, 1)))
model.add(Dropout(0.5))
model.add(Masking(mask_value=0))
model.add(LSTM(units=(2), activation='tanh', use_bias=True))
model.add(Activation('sigmoid'))

model.compile(loss='categorical_crossentropy', optimizer=SGD(lr=0.1), metrics=['acc'])

model.summary() """

main_input          = Input(shape=(MAX_SEQUENCE_LENGTH,))
embedding_layer     = Embedding(embedding_matrix.shape[0],
                                embedding_matrix.shape[1],
                                weights=[embedding_matrix],
                                input_length=MAX_SEQUENCE_LENGTH,
                                trainable=False)(main_input)
concatenate_layer   = Concatenate(axis=1)([Conv1D(1, kw, padding='same', use_bias=True, bias_initializer=Ones(), activation='relu', strides=1)(embedding_layer) for kw in (3, 4, 5)])
dropout_layer       = Dropout(0.5)(concatenate_layer)
masking_layer       = Masking(mask_value=0)(dropout_layer)
lstm_layer          = LSTM(units=(2), activation='tanh', use_bias=True)(masking_layer)
activation_layer    = Activation('sigmoid')(lstm_layer)

model = Model(inputs=[main_input], outputs=[activation_layer])
model.compile(loss='categorical_crossentropy', optimizer=SGD(lr=0.1), metrics=['acc'])
model.summary()