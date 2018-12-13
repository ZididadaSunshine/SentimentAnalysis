import numpy as np
from sklearn.feature_extraction.text import CountVectorizer

from sklearn.naive_bayes import MultinomialNB
from datasets import sentiment_140_neg

maxlen = 100

(x_train, y_train), (x_val, y_val), (x_test, y_test) = sentiment_140_neg.load_data()

x_train = np.concatenate((x_train, x_val))
y_train = np.concatenate((y_train, y_val))

#x_train = [sentence.split(' ') for sentence in x_train]
#x_val = [sentence.split(' ') for sentence in x_val]
#x_test = [sentence.split(' ') for sentence in x_test]

vectorizer = CountVectorizer()
x_train = vectorizer.fit_transform(x_train)
x_test = vectorizer.transform(x_test)
#print('x_train shape:', x_train.shape)
#print('x_test shape:', x_test.shape)

bayes = MultinomialNB()

bayes.fit(x_train, y_train)

print(bayes.score(x_test, y_test))