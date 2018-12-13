import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
import pickle

from sklearn.naive_bayes import MultinomialNB
from datasets import sentiment_140
from utils.data_utils import export

maxlen = 100

# Get data
(x_train, y_train), (x_val, y_val), (x_test, y_test) = sentiment_140.load_data()

# No need for train and validation for bayes
x_train = np.concatenate((x_train, x_val))
y_train = np.concatenate((y_train, y_val))


vectorizer = CountVectorizer()

# Creates vocabulary for words, and transform sentences in train to a word frequency vector.
x_train = vectorizer.fit_transform(x_train)
x_test = vectorizer.transform(x_test)  # transform to frequency vector.

bayes = MultinomialNB()

bayes.fit(x_train, y_train)

acc = bayes.score(x_test, y_test)

export(bayes, vocabulary=vectorizer, acc=acc)