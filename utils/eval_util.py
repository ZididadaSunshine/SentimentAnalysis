import pickle
import json

from keras import Sequential
from keras.models import load_model
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

from datasets import sentiment_140
from keras.preprocessing import sequence
import time


def test_keras_model(model: Sequential, tokenizer, arguments):
    """Test model on sentiment140 dataset. Both accuracy and throughput."""

    if arguments["maxlen"]:
        maxlen = arguments["maxlen"]
    else:
        maxlen = 0

    # Get data, ignore train and test
    _, _, (x_test, y_test) = sentiment_140.load_data()

    starttime = time.time()
    x_test = tokenizer.texts_to_sequences(x_test)

    x_test = sequence.pad_sequences(x_test, maxlen=maxlen)

    score, acc = model.evaluate(x_test, y_test)
    endtime = time.time()

    return score, acc, endtime - starttime


def load_keras_items(path):
    model = load_model(f'{path}/model.h5')

    with open(f'{path}/tokenizer.pkl', 'rb') as f:
        tokenizer = pickle.load(f)

    with open(f'{path}/kwargs.json', 'r') as f:
        arguments = json.load(f)

    return model, tokenizer, arguments


def start_keras_test(path):
    model, tokenizer, arguments = load_keras_items(path)
    score, acc, ex_time = test_keras_model(model, tokenizer, arguments)
    dump_json(path=path, score=score, acc=acc, ex_time=ex_time)


def test_naive_model(model: MultinomialNB, vocabulary: CountVectorizer):
    # Get data, ignore train and test
    _, _, (x_test, y_test) = sentiment_140.load_data()

    starttime = time.time()
    x_test = vocabulary.transform(x_test)

    acc = model.score(x_test, y_test)
    endtime = time.time()

    return acc, endtime - starttime


def load_naive_items(path):
    with open(f'{path}/model.pkl', 'rb') as f:
        model = pickle.load(f)

    with open(f'{path}/vocab.pkl', 'rb') as f:
        vocabulary = pickle.load(f)

    return model, vocabulary


def start_naive_test(path):
    model, vocab = load_naive_items(path)
    acc, ex_time = test_naive_model(model, vocab)
    dump_json(path=path, acc=acc, ex_time=ex_time)


def dump_json(path, **kwargs):
    with open(f'{path}/eval.json', 'w') as f:
        json.dump(kwargs, f)


#start_naive_test('/home/theis/Projects/SentimentAnalysis/output/0')
start_keras_test("/home/theis/Projects/SentimentAnalysis/output/1")
