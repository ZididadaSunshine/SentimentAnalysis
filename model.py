import gensim
import tensorflow as tf
import numpy as np

sentence = ["hello", "how", "are", "you"]
'''
#Model has to be downloaded from https://drive.google.com/file/d/0B7XkCwpI5KDYNlNUTTlSS21pQmM/edit
word2vec = gensim.models.KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary=True)

print("loaded")

model_input = []
for word in sentence:
    if word in word2vec.vocab:
        model_input.append(word2vec.get_vector(word))

model_input = np.matrix(model_input).transpose()

print(model_input.shape)'''


width1 = 2
width2 = 3
width4 = 4
vector_length = 300
channels = 1


#Initialize model
X1 = tf.placeholder(tf.float32, [None, vector_length, None, channels])
b1 = tf.Variable(tf.ones([vector_length]) / 2)

Y = tf.nn.tanh(tf.nn.conv2d(X1, W1, strides=[1, 1, 1, 1], padding='VALID') + b1)

Y2 = tf.nn.rnn_cell.BasicLSTMCell()

print(Y.shape)