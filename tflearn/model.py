from __future__ import division, print_function, absolute_import

import tensorflow as tf
import tflearn
from tflearn.data_utils import to_categorical, pad_sequences
from tflearn.datasets import imdb
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.embedding_ops import embedding
from tflearn.layers.recurrent import bidirectional_rnn, BasicLSTMCell
from tflearn.activations import linear, sigmoid, relu, softmax
from tflearn.layers.merge_ops import merge
from tflearn.layers.conv import conv_1d, global_max_pool
from tflearn.layers.estimator import regression


# IMDB Dataset loading
train, test, _ = imdb.load_data(
    path='imdb.pkl', 
    n_words=10000,
    valid_portion=0.1)
trainX, trainY = train
testX, testY = test


# Data preprocessing
# Sequence padding
trainX = pad_sequences(trainX, maxlen=100, value=0.)
testX = pad_sequences(testX, maxlen=100, value=0.)

# Converting labels to binary vectors
trainY = to_categorical(trainY, nb_classes=2)
testY = to_categorical(testY, nb_classes=2)


# Building convolutional network
network = input_data(shape=[None, 100], name='input')   # [sample, word]

network = embedding(
    network, 
    input_dim=10000, 
    output_dim=128, 
    trainable=False)    # [sample, word, embedding dim]

# incoming: [sample, timestep, input dim]
network = bidirectional_rnn(
    network, 
    BasicLSTMCell(128, activation=linear, inner_activation=sigmoid), 
    BasicLSTMCell(128, activation=linear, inner_activation=sigmoid), 
    return_seq=True,
    dynamic=True)
# output: [sample, timestep, output dim], output is depth concat for fw and bw
network = tf.pack(network, axis=1)

fw_outputs, bw_outputs = tf.split(split_dim=2, num_split=2, value=network) 

# network = merge([fw_outputs, bw_outputs], mode='elementwise_sum', axis=2)
network = tf.add(fw_outputs, bw_outputs)

branch1 = conv_1d(network, 128, 3, padding='valid', activation=relu, regularizer="L2")
branch2 = conv_1d(network, 128, 4, padding='valid', activation=relu, regularizer="L2")
branch3 = conv_1d(network, 128, 5, padding='valid', activation=relu, regularizer="L2")

network = merge([branch1, branch2, branch3], mode='concat', axis=1)

network = tf.expand_dims(network, 2)

network = global_max_pool(network)

network = dropout(network, 0.5)

network = fully_connected(network, 2, activation=softmax)

network = regression(
    network, 
    optimizer='adam', 
    learning_rate=0.001, 
    loss='categorical_crossentropy', 
    name='target')

# Training
model = tflearn.DNN(network, tensorboard_verbose=0, tensorboard_dir='runs')
model.fit(
    trainX, 
    trainY, 
    validation_set=(testX, testY),
    n_epoch = 5, 
    shuffle=True,  
    show_metric=True, 
    batch_size=32)
