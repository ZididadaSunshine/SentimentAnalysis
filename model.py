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
'''


width1 = 2
width2 = 3
width3 = 4
vector_length = 300
channels = 1


# Placeholders for the input and dropout parameter.
input = tf.placeholder(tf.float32, [None, vector_length, None, channels])

dropout_p = tf.placeholder(tf.float32)


# Initialize weights and biases
b1 = tf.Variable(tf.ones([1]) / 2)
W1 = tf.Variable(tf.truncated_normal([vector_length, width1, 1, 1]))
W2 = tf.Variable(tf.truncated_normal([vector_length, width2, 1, 1]))
W3 = tf.Variable(tf.truncated_normal([vector_length, width3, 1, 1]))


# Initialize CNN models
Y1 = tf.nn.tanh(tf.nn.conv2d(input, W1, strides=[1, vector_length, 1, 1], padding='SAME') + b1)
Y2 = tf.nn.tanh(tf.nn.conv2d(input, W2, strides=[1, vector_length, 1, 1], padding='SAME') + b1)
Y3 = tf.nn.tanh(tf.nn.conv2d(input, W3, strides=[1, vector_length, 1, 1], padding='SAME') + b1)

out = tf.concat([Y1, Y2, Y3], 1)

# Initialize LSTM cell


print(out)

#Y2 = tf.nn.rnn_cell.BasicLSTMCell()


