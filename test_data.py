import csv
from random import shuffle
from nltk import tokenize, word_tokenize
from nltk.tokenize.api import StringTokenizer

from sklearn.model_selection import train_test_split
from word2vec import Word2vec


class TestData:
    def __init__(self):
        self._w2v = Word2vec()


    def read_data(self):
        with open("datter.csv", encoding='latin-1') as csvfile:
            reader = csv.reader(csvfile, delimiter=',')
            return [{'sentence': r[5], 'label':0 if r[0] == '0' else 1} for r in reader]


    def get_data(self, num_elements=None):
        if num_elements:
            data = self.read_data()[:num_elements]
        else:
            data = self.read_data()

        self.transform(data)

        negatives = [p for p in data if p['label'] == 0]
        positives = [p for p in data if p['label'] == 1]

        n_train, n_rest = train_test_split(negatives, train_size=0.6)
        n_validation, n_test = train_test_split(n_rest,test_size=0.5)

        p_train, p_rest = train_test_split(positives, train_size=0.6)
        p_validation, p_test = train_test_split(p_rest, test_size=0.5)

        train = n_train + p_train
        test = n_test + p_test
        validation = n_validation + p_validation

        shuffle(train)
        shuffle(test)
        shuffle(validation)

        return train, test, validation


    def transform(self, data):
        for p in data:
            tokens = word_tokenize(p['sentence'])
            p['sentence'] = self._w2v.get_vectors(tokens)


    def analyse_labels(self):
        labels = self.read_data()

        tmp = {}
        for pair in labels:
            l = pair['label']
            if l in tmp:
                tmp[l] += 1
            else:
                tmp[l] = 1

        return tmp


    def create_padded_batches(self, data, batch_size):
        

t = TestData()
train, test, validation = t.get_data(1000)


print(f"Train: {len(train)}, Test: {len(test)}, Validation: {len(validation)}")