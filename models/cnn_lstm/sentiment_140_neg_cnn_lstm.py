import numpy as np

from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.layers import Embedding
from keras.layers import LSTM
from keras.layers import Conv1D, MaxPooling1D
from keras_preprocessing.text import Tokenizer

from datasets import sentiment_140_neg
from utils.data_utils import export
from w2v import google_news_vectors_negative300


# Embedding
maxlen = 100
embedding_size = 128

# Convolution
kernel_size = 5
filters = 64
pool_size = 4

# LSTM
lstm_output_size = 70

# Training
batch_size = 128
epochs = 1

print('Loading data...')
(x_train, y_train), (x_val, y_val), (x_test, y_test) = sentiment_140_neg.load_data()

print('Fitting tokenizer...')
tokenizer = Tokenizer()
tokenizer.fit_on_texts(np.concatenate((x_train, x_val, x_test)))

print('Convert text to sequences')
x_train = tokenizer.texts_to_sequences(x_train)
x_val = tokenizer.texts_to_sequences(x_val)
x_test = tokenizer.texts_to_sequences(x_test)

print(len(x_train), 'train sequences')
print(len(x_val), 'validation sequences')
print(len(x_test), 'test sequences')

print('Pad sequences (samples x time)')

x_train = sequence.pad_sequences(x_train, maxlen=maxlen)
x_val = sequence.pad_sequences(x_val, maxlen=maxlen)
x_test = sequence.pad_sequences(x_test, maxlen=maxlen)

print('x_train shape:', x_train.shape)
print('x_val shape:', x_val.shape)
print('x_test shape:', x_test.shape)

print('Loading w2v...')
word2vec = google_news_vectors_negative300.load_w2v()

print('Preparing embedding matrix')
word_index = tokenizer.word_index
nb_words = len(word_index)+1

embedding_matrix = np.zeros((nb_words, 301))
for word, i in word_index.items():
    word, *neg = word.split('_')

    if word in word2vec.vocab:
        embedding_matrix[i][:300] = word2vec.word_vec(word)

    if neg:
        embedding_matrix[i][301] = 1
print('Null word embeddings: %d' % np.sum(np.sum(embedding_matrix, axis=1) == 0))

print('Build model...')
model = Sequential()
model.add(Embedding(embedding_matrix.shape[0],
                    embedding_matrix.shape[1],
                    weights=[embedding_matrix],
                    input_length=maxlen,
                    trainable=False))
model.add(Dropout(0.25))
model.add(Conv1D(filters,
                 kernel_size,
                 padding='valid',
                 activation='relu',
                 strides=1))
model.add(MaxPooling1D(pool_size=pool_size))
model.add(LSTM(lstm_output_size))
model.add(Dense(1))
model.add(Activation('sigmoid'))

model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

model.summary()

print('Train...')
history = model.fit(x_train, y_train,
                    batch_size=batch_size,
                    epochs=epochs,
                    validation_data=(x_val, y_val))
score, acc = model.evaluate(x_test, y_test, batch_size=batch_size)
print('Test score:', score)
print('Test accuracy:', acc)

export(model, history, tokenizer)
