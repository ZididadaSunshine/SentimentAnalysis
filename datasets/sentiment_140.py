import pandas as pd
from keras.utils.data_utils import get_file
import numpy as np
import zipfile
import os
from sklearn.model_selection import train_test_split
from KeywordExtraction.preprocessing.text_preprocessing import get_processed_text
import random

count = 0
max_rows = 1600000
random.seed()


def _category(val):
    val = int(val)
    if val != 0 and val != 4:
        print(val)
    return 0 if val == 0 else 1


def _sentence(text):
    global count

    try:
        if count % 10000 == 0:
            print(f'Still processing, please be patient... progress: {(count / max_rows) * 100 : 2.2f}% ', flush=True)
    except ValueError:
        pass
    finally:
        count += 1

    return " ".join(get_processed_text(text))


def load_data(path='trainingandtestdata.zip'):
    """Loads the sentiment 140 dataset

    # Arguments
        path: path where to cache the dataset locally
            (relative to ~/.keras/datasets).
    # Returns
        Tuple of Numpy arrays: `(x_train, y_train), (x_val, y_val), (x_test, y_test)`.
    """

    path = get_file(
        path,
        origin='http://cs.stanford.edu/people/alecmgo/trainingandtestdata.zip')

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

            df = df.sample(frac=1).reset_index(drop=True)

            x_train, x_test, y_train, y_test = train_test_split(df['sentence'], df['category'], train_size=0.6)
            x_test, x_val, y_test, y_val = train_test_split(x_test, y_test, train_size=0.5)

            np.savez(cache_dir + 'sentiment_140.npz',
                     x_train=x_train, y_train=y_train,
                     x_val=x_val, y_val=y_val,
                     x_test=x_test, y_test=y_test)
    else:
        with np.load(cache_dir + 'sentiment_140.npz') as f:
            x_train, y_train = f['x_train'], f['y_train']
            x_val, y_val = f['x_val'], f['y_val']
            x_test, y_test = f['x_test'], f['y_test']

    return (x_train, y_train), (x_val, y_val), (x_test, y_test)
