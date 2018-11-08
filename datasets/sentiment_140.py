import pandas as pd
from keras.utils.data_utils import get_file
import numpy as np
import zipfile
import os

from keras_preprocessing.text import Tokenizer
from sklearn.model_selection import train_test_split
from nltk.corpus import stopwords

STOPWORDS = set(stopwords.words("english"))
ORIGIN = 'http://cs.stanford.edu/people/alecmgo/trainingandtestdata.zip'


def _category(val):
    val = int(val)
    if val != 0 and val != 4:
        print(val)
    return 0 if val == 0 else 1


def _sentence(text):
    text = text.strip().lower().split()
    text = filter(lambda word: word not in STOPWORDS, text)
    return " ".join(text)


def load_data(path='trainingandtestdata.zip'):
    """Loads the stanford large movie review dataset

    # Arguments
        path: path where to cache the dataset locally
            (relative to ~/.keras/datasets).
    # Returns
        Tuple of Numpy arrays: `(x_train, y_train), (x_test, y_test)`.
    """

    path = get_file(
        path,
        origin=ORIGIN)

    cache_dir = os.path.join(os.path.expanduser('~'), '.keras/datasets/')
    data_dir = cache_dir + 'sentiment_140.npz'

    if not os.path.exists(data_dir):
        with zipfile.ZipFile(path) as archive:
            with archive.open('training.1600000.processed.noemoticon.csv') as csv:
                df = pd.read_csv(csv,
                                 sep=',',
                                 encoding='latin-1',
                                 usecols=[0, 5],
                                 names=['category', 'sentence'],
                                 converters={'category': _category, 'sentence': _sentence})

            # Fit the tokenizer
            tokenizer = Tokenizer()
            tokenizer.fit_on_texts(df['sentence'])

            # Use the tokenizer to convert text to sequences
            sequences = tokenizer.texts_to_sequences(df['sentence'])

            df = pd.DataFrame({'category': df['category'], 'sequence': sequences})

            df = df.sample(frac=1).reset_index(drop=True)

            train, val = train_test_split(df, train_size=0.6)
            val, test = train_test_split(val, train_size=0.5)

            np.savez(cache_dir + 'sentiment_140.npz', train=train, test=test, val=val)
    else:
        with np.load(cache_dir + 'sentiment_140.npz') as f:
            train, test, val = f['train'], f['test'], f['val']

    return train, test, val


def get_test_data():
    _, test, _ = load_data()
    return test


def get_train_val_data():
    train, _, val = load_data()
    return train, val
