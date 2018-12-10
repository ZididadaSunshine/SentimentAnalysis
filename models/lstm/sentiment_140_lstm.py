import numpy as np
from keras import Input, Model
from keras.preprocessing import sequence
from keras.layers import Dense, Dropout, Activation, Conv1D, GlobalMaxPooling1D
from keras.layers import Embedding
from keras.layers import LSTM
from keras_preprocessing.text import Tokenizer

from datasets import sentiment_140
from utils.data_utils import export
from w2v import google_news_vectors_negative300

# Embedding
maxlen = 100
embedding_size = 128

# Convolution
kernel_size = 5
filters = 64
pool_size = 4

#Dropout
dropout_rate = 0.25

# LSTM
lstm_output_size = 70

# Training
batch_size = 2048
epochs = 2

print('Loading data...')
(x_train, y_train), (x_val, y_val), (x_test, y_test) = sentiment_140.load_data()

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

embedding_matrix = np.zeros((nb_words, 300))
for word, i in word_index.items():
    if word in word2vec.vocab:
        embedding_matrix[i] = word2vec.word_vec(word)
print('Null word embeddings: %d' % np.sum(np.sum(embedding_matrix, axis=1) == 0))

print("Deleting w2v ")
del word2vec

print('Build model...')
emb_length = embedding_matrix.shape[0]
emb_height = embedding_matrix.shape[1]

input = Input(shape=(None,))

emb = Embedding(input_dim=emb_length,
                output_dim=emb_height,
                weights=[embedding_matrix],
                input_length=maxlen,
                trainable=False)(input)

conv1  = Conv1D(filters=32, kernel_size=5, padding='valid', activation='relu', strides=1)(emb)
conv11 = Conv1D(filters=64, kernel_size=5, padding='valid', activation='relu', strides=2)(conv1)
conv12 = Conv1D(filters=128, kernel_size=4, padding='valid', activation='relu', strides=2)(conv11)
conv13 = Conv1D(filters=128, kernel_size=3, padding='valid', activation='relu', strides=1)(conv12)
pooled = GlobalMaxPooling1D()(conv13)
drop1 = Dropout(dropout_rate)(pooled)
dense1 = Dense(1, activation='sigmoid')(drop1)
out1 = Activation('sigmoid')

lstm1 = LSTM(32)(emb)
dense2 = Dense(1, activation='sigmoid')(lstm1)
out2 = Activation('sigmoid')


model = Model(inputs=[input], outputs=[dense1, dense2])

model.compile(loss=['binary_crossentropy', 'binary_crossentropy'],
              loss_weights=[1., 0.1],
              optimizer='adam',
              metrics=['accuracy'])

print(model.summary())

print('Train...')
history = model.fit(x_train,
                    [y_train, y_train],
                    batch_size=batch_size,
                    epochs=epochs,
                    validation_data=(x_val, [y_val, y_val]),
                    shuffle=True)
score, acc = model.evaluate(x_test, y_test, batch_size=batch_size)
print('Test score:', score)
print('Test accuracy:', acc)

export(model, history, tokenizer, name="sentiment_140_lstm", score=score, acc=acc)
