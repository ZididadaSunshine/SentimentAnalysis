from keras.utils import get_file
from gensim.models import KeyedVectors


def load_w2v(path='GoogleNews-vectors-negative300.bin.gz'):
    """Loads GoogleNews-vectors-negative300

        # Arguments
            path: path where to cache the dataset locally
                (relative to ~/.keras/w2v).
        # Returns
            Gensim keyed vector
        """

    path = get_file(
        path,
        origin='https://s3.amazonaws.com/dl4j-distribution/GoogleNews-vectors-negative300.bin.gz',
        cache_subdir='w2v')

    return KeyedVectors.load_word2vec_format(path, binary=True)
