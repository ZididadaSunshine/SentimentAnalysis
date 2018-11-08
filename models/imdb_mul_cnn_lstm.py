"""Train a recurrent convolutional network on the IMDB sentiment
classification task.

Gets to 0.8498 test accuracy after 2 epochs. 41s/epoch on K520 GPU.
"""

import os
import _pickle as pickle

from keras.callbacks import EarlyStopping, TensorBoard
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.layers import Embedding
from keras.layers import LSTM
from keras.layers import Conv1D
from keras.datasets import imdb

# Embedding
max_features = 20000
maxlen = 100
embedding_size = 128

# Convolution
kernel_size = 5
filters = 64
pool_size = 4

# LSTM
lstm_output_size = 128

# Training
batch_size = 30
epochs = 5

'''
Note:
batch_size is highly sensitive.
Only 2 epochs are needed as the dataset is very small.
'''

print('Loading data...')
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_features)
print(len(x_train), 'train sequences')
print(len(x_test), 'test sequences')

print('Pad sequences (samples x time)')
x_train = sequence.pad_sequences(x_train, maxlen=maxlen)
x_test = sequence.pad_sequences(x_test, maxlen=maxlen)
print('x_train shape:', x_train.shape)
print('x_test shape:', x_test.shape)

print('Build model...')

model = Sequential()
model.add(Embedding(max_features, embedding_size, input_length=maxlen))
model.add(Conv1D(16,
                 5,
                 padding='valid',
                 activation='relu',
                 strides=1))
model.add(Conv1D(32,
                 4,
                 padding='valid',
                 activation='relu',
                 strides=2))
model.add(Conv1D(64,
                 3,
                 padding='valid',
                 activation='relu',
                 strides=2))
model.add(Dropout(0.25))
model.add(LSTM(lstm_output_size))
model.add(Dense(1))
model.add(Activation('sigmoid'))

model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

callbacks = [EarlyStopping(monitor='val_loss', patience=2),
             TensorBoard(log_dir='./logs', histogram_freq=0, batch_size=32, write_graph=True, write_grads=False,
                         write_images=False, embeddings_freq=0, embeddings_layer_names=None, embeddings_metadata=None,
                         embeddings_data=None, update_freq='epoch')]

print('Train...')
history = model.fit(x_train, y_train,
                    batch_size=batch_size,
                    epochs=epochs,
                    callbacks=callbacks,
                    validation_data=(x_test, y_test))
score, acc = model.evaluate(x_test, y_test, batch_size=batch_size)
print('Test score:', score)
print('Test accuracy:', acc)

# Create an output folder if it doesn't exist
if not os.path.exists('../output'):
    os.makedirs('../output')

# Create a new folder for the model
i = 0
while os.path.exists(f'../output/{str(i)}'):
    i = i + 1

output_dir = f'../output/{str(i)}'
os.makedirs(output_dir)

model.save(f'{output_dir}/model.h5')

with open(f'{output_dir}/history.pkl', 'wb') as f:
    pickle.dump(history.history, f)
