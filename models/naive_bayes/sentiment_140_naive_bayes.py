import numpy as np
from keras.preprocessing.text import Tokenizer

from sklearn.naive_bayes import BernoulliNB
from datasets import sentiment_140
from keras.preprocessing import sequence

maxlen = 100

(x_train, y_train), (x_val, y_val), (x_test, y_test) = sentiment_140.load_data()

x_train = np.concatenate((x_train, x_val))
y_train = np.concatenate((y_train, y_val))

print('Fitting tokenizer')
tokenizer = Tokenizer()
tokenizer.fit_on_texts(np.concatenate((x_train, x_test)))

print('Convert text to sequences')
x_train = tokenizer.texts_to_sequences(x_train)
x_test = tokenizer.texts_to_sequences(x_test)

print('Pad sequences (samples x features)')
x_train = sequence.pad_sequences(x_train, maxlen=maxlen)
x_test = sequence.pad_sequences(x_test, maxlen=maxlen)

print('x_train shape:', x_train.shape)
print('x_test shape:', x_test.shape)

print('Fitting model')
bnb = BernoulliNB(binarize=0.0)
model = bnb.fit(x_train, y_train)

score = bnb.score(x_test, y_test)
print(score)
