from keras.datasets import imdb
from sklearn.model_selection import train_test_split


def load_data(num_words=None):
    (x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=num_words)

    x_test, x_val, y_test, y_val = train_test_split(x_test, y_test, train_size=0.5)

    return (x_train, y_train), (x_val, y_val), (x_test, y_test)


def get_word_index():
    """Retrieves the dictionary mapping words to word indices.
    # Arguments
        path: where to cache the data (relative to `~/.keras/dataset`).
    # Returns
        The word index dictionary.
    """
    return imdb.get_word_index()
