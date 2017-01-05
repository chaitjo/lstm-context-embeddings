from __future__ import division, print_function, absolute_import

import tensorflow as tf
import tflearn
from tflearn.data_utils import to_categorical, pad_sequences
from tflearn.datasets import imdb
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.embedding_ops import embedding
from tflearn.layers.recurrent import bidirectional_rnn, BasicLSTMCell
from tflearn.layers.merge_ops import merge
from tflearn.layers.conv import conv_1d, global_max_pool
from tflearn.layers.estimator import regression


tf.flags.DEFINE_integer("maxlen", 100, "Maximum Sentence Length")
tf.flags.DEFINE_integer("vocab_size", 10000, "Size of Vocabulary")
tf.flags.DEFINE_integer("embedding_dim", 128, "Word Embedding Size")
tf.flags.DEFINE_integer("rnn_hidden_size", 128, "Size of biRNN hidden layer")
tf.flags.DEFINE_integer("num_filters", 128, "Number of CNN filters")
tf.flags.DEFINE_float("dropout_prob", 0.5, "Dropout Probability")
tf.flags.DEFINE_float("learning_rate", 0.001, "Learning Rate")
tf.flags.DEFINE_integer("batch_size", 32, "Batch Size")
tf.flags.DEFINE_integer("num_epochs", 100, "Number of Training Epochs")

FLAGS = tf.flags.FLAGS
FLAGS._parse_flags()
print("\nParameters:")
for attr, value in sorted(FLAGS.__flags.items()):
    print("{}={}".format(attr.upper(), value))
print("")

maxlen = FLAGS.maxlen
vocab_size = FLAGS.vocab_size
embedding_dim = FLAGS.embedding_dim
rnn_hidden_size = FLAGS.rnn_hidden_size
num_filters = FLAGS.num_filters
dropout_prob = FLAGS.dropout_prob
learning_rate = FLAGS.learning_rate
batch_size = FLAGS.batch_size
num_epochs = FLAGS.num_epochs


# IMDB Dataset loading
train, test, _ = imdb.load_data(path='imdb.pkl', n_words=vocab_size, valid_portion=0.1)
trainX, trainY = train
testX, testY = test

# Sequence padding
trainX = pad_sequences(trainX, maxlen=maxlen, value=0.)
testX = pad_sequences(testX, maxlen=maxlen, value=0.)

# Converting labels to binary vectors
trainY = to_categorical(trainY, nb_classes=2)
testY = to_categorical(testY, nb_classes=2)


# Building network
network = input_data(shape=[None, maxlen], name='input')   

network = embedding(
    network, 
    input_dim=vocab_size, 
    output_dim=embedding_dim, 
    trainable=True)    

network = bidirectional_rnn(
    network, 
    BasicLSTMCell(rnn_hidden_size, activation='tanh', inner_activation='sigmoid'), 
    BasicLSTMCell(rnn_hidden_size, activation='tanh', inner_activation='sigmoid'), 
    return_seq=True,
    dynamic=True)
network = tf.pack(network, axis=1)

fw_outputs, bw_outputs = tf.split(split_dim=2, num_split=2, value=network) 
network = tf.add(fw_outputs, bw_outputs)

branch1 = conv_1d(network, num_filters, 3, padding='valid', activation='relu', regularizer="L2")
branch2 = conv_1d(network, num_filters, 4, padding='valid', activation='relu', regularizer="L2")
branch3 = conv_1d(network, num_filters, 5, padding='valid', activation='relu', regularizer="L2")

network = merge([branch1, branch2, branch3], mode='concat', axis=1)

network = tf.expand_dims(network, 2)

network = global_max_pool(network)

network = dropout(network, dropout_prob)

network = fully_connected(network, 2, activation='softmax')

network = regression(
    network, 
    optimizer='adam', 
    learning_rate=learning_rate, 
    loss='categorical_crossentropy', 
    name='target')


# Training
model = tflearn.DNN(network, tensorboard_verbose=0, tensorboard_dir='runs')
model.fit(
    trainX, 
    trainY, 
    validation_set=(testX, testY),
    n_epoch = num_epochs, 
    shuffle=True,  
    show_metric=True, 
    batch_size=batch_size)
