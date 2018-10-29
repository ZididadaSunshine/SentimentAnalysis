import numpy as np
import requests
from gensim.models import KeyedVectors
from gensim.test.utils import datapath


class Word2vec:
    def __init__(self):
        # Loads w2v model
        # NOTE: Model has to be downloaded from https://drive.google.com/file/d/0B7XkCwpI5KDYNlNUTTlSS21pQmM/edit
        self.word2vec = KeyedVectors.load_word2vec_format(datapath('GoogleNews-vectors-negative300.bin'), binary=True)
        print("Loaded w2v model")

    def get_words(self, number):
        word_site = "http://svnweb.freebsd.org/csrg/share/dict/words?view=co&content-type=text/plain"

        response = requests.get(word_site)

        words = response.content.splitlines()[:number]

        print("Words downloaded")

        return  words

    def get_vectors(self, words):
        model_input = []
        length = len(words)
        processed = 0
        for word in words:
            processed += 1
            if processed % 100 == 0:
                print(f"Processed {processed / length * 100}%")

            if word in self.word2vec.vocab:
                model_input.append(self.word2vec.get_vector(word))

        return np.matrix(model_input).transpose()

'''
w2v = Word2vec()
words = w2v.get_words(512)
w2v.get_vectors(words)
'''