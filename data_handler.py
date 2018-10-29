import csv
import numpy as np
from nltk import tokenize, word_tokenize
from nltk.tokenize.api import StringTokenizer
from sklearn.model_selection import train_test_split


class DataHandler:
    def __init__(self, word2vec, data_path, num_elements = None):
        self._w2v = word2vec

        with open(data_path, encoding='latin-1') as csvfile:
            reader = csv.reader(csvfile, delimiter=',')
            data = [{'sentence': r[5], 'label':0 if r[0] == '0' else 1} for r in reader]

        if num_elements:
            data = data[:num_elements]

        for p in data:
            tokens = word_tokenize(p['sentence'])
            p['sentence'] = np.asarray([self._w2v.get_vector(t) for t in tokens if t in self._w2v.vocab])
            

        
        
        negatives = [p for p in data if p['label'] == 0]
        positives = [p for p in data if p['label'] == 1]

        print(f"Positives: {len(negatives)}, Negatives: {len(positives)}")

        n_train, n_rest = train_test_split(negatives, train_size=0.6)
        n_validation, n_test = train_test_split(n_rest, test_size=0.5)

        p_train, p_rest = train_test_split(positives, train_size=0.6)
        p_validation, p_test = train_test_split(p_rest, test_size=0.5)

        self._training_data = n_train + p_train
        self._test_data = n_test + p_test
        self._validation_data = n_validation + p_validation

        print(f"Training data: {len(self._training_data)}, Test data: {len(self._test_data)}, Validation data: {len(self._validation_data)}")


    def get_training_data(self):
        return self._training_data

    def get_test_data(self):
        return self._test_data
        
    def get_validation_data(self):
        return self._validation_data