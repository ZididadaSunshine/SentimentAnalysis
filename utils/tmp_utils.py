import errno
import os

from keras.models import load_model
import _pickle as pickle
from keras.preprocessing.text import Tokenizer
from keras_preprocessing import sequence
import numpy as np
from datasets import sentiment_140_neg, sentiment_140


def evaluate(model_path=None, tokenizer_path=None, dataset=sentiment_140):
    """Converts a keras h5 model to a protobuf model

        # Arguments
            model_path: path to the directory containing the model
                (if not specified, the sub-folder last added to the output folder is selected)
            export_path: path to the desired export directory
                (if not specified, a sub-folder is created in the model_path directory)
    """


    if not model_path:
        if not os.path.exists('../output'):
            raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), 'output')

        i = 0
        while os.path.exists(f'../output/{str(i)}'):
            i = i + 1

        model_path = f'../output/{str(i-1)}/model.h5'
        model_dir = os.path.dirname(model_path)

    if tokenizer_path is None:
        tokenizer_path = f'{model_dir}/tokenizer.pkl'

    with open(tokenizer_path, 'rb') as f:
        tokenizer = pickle.load(f)  # type: Tokenizer

    y_test: np.ndarray
    _, _, (x_test, y_test) = dataset.load_data()

    x_test = tokenizer.texts_to_sequences(x_test)
    x_test = sequence.pad_sequences(x_test, maxlen=100)
    print('x_test shape:', x_test.shape)

    positive_indices = np.where(y_test == 1)[0]
    negative_indices = np.where(y_test == 0)[0]

    px_test = np.take(x_test, positive_indices, axis=0)
    py_test = np.take(y_test, positive_indices, axis=0)
    print('px_test shape:', px_test.shape)
    print('py_test shape:', py_test.shape)

    nx_test = np.take(x_test, negative_indices, axis=0)
    ny_test = np.take(y_test, negative_indices, axis=0)
    print('nx_test shape:', nx_test.shape)
    print('ny_test shape:', ny_test.shape)

    model = load_model(model_path)

    print('Evaluating model on positives...')
    positive_score, positive_acc = model.evaluate(px_test, py_test, 128)
    print('Evaluating model on negatives...')
    negative_score, negative_acc = model.evaluate(nx_test, ny_test, 128)

    print('Positive test score:', positive_score)
    print('Negative test score:', negative_score)
    print('Positive test accuracy:', positive_acc)
    print('Negative test accuracy:', negative_acc)

evaluate(dataset=sentiment_140_neg)