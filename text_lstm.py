import numpy as np
import tensorflow as tf
from tensorflow.python.ops import rnn_cell, rnn, nn, array_ops


class TextLSTM(object):
    def __init__(
        self, sequence_length, num_classes, vocab_size, 
        embedding_size, hidden_size, l2_reg_lambda=0.0):

        # Placeholders for input, sequence length, output and dropout
        self.input_x = tf.placeholder(tf.int32, [None, sequence_length], name="input_x")
        self.seqlen = tf.placeholder(tf.int32, [None], name="seqlen")
        self.input_y = tf.placeholder(tf.float32, [None, num_classes], name="input_y")
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")

        # Keeping track of l2 regularization loss (optional)
        l2_loss = tf.constant(0.0)

        # Embedding layer
        with tf.device('/cpu:0'), tf.name_scope("embedding"):
            # W is our embedding matrix that we learn during training. We initialize it using a random uniform distribution
            self.W = tf.Variable(
                tf.random_uniform([vocab_size, embedding_size], -1.0, 1.0),
                trainable=False, 
                name="W")
            
            # nn.embedding_lookup creates the actual embedding operation
            # The result of the embedding operation is a 3-dimensional tensor of shape [None, sequence_length, embedding_size]
            self.embedded_chars = nn.embedding_lookup(self.W, self.input_x)            
            
            # self.inputs = tf.split(1, sequence_length, embedded_chars)
            # self.inputs = [tf.squeeze(input_, [1]) for input_ in self.inputs]

        with tf.name_scope("bidirectional-lstm"):
            # Forward direction LSTM cell
            lstm_fw_cell = rnn_cell.BasicLSTMCell(hidden_size, forget_bias=1.0)
            # Backward direction LSTM cell
            lstm_bw_cell = rnn_cell.BasicLSTMCell(hidden_size, forget_bias=1.0)

            # self.lstm_outputs, _ = rnn.bidirectional_dynamic_rnn(
            #     lstm_fw_cell, 
            #     lstm_bw_cell, 
            #     self.embedded_chars,
            #     sequence_length=self.seqlen,
            #     dtype=tf.float32)

            lstm_outputs_fw, _ = rnn.dynamic_rnn(
                lstm_fw_cell, 
                self.embedded_chars, 
                dtype=tf.float32, 
                sequence_length=self.seqlen)

            self.embedded_chars_rev = array_ops.reverse_sequence(self.embedded_chars, self.seqlen)
            tmp, _ = rnn.dynamic_rnn(
                lstm_bw_cell, 
                self.embedded_chars_rev,
                dtype=tf.float32, 
                sequence_length=self.seqlen)
            lstm_outputs_bw = array_ops.reverse_sequence(tmp, self.seqlen)

            # Concatenate outputs
            self.lstm_outputs = tf.add(outputs_fw, outputs_bw, name="lstm_outputs")
            self.lstm_outputs_mean = tf.reduce_mean(lstm_outputs, 1, name="lstm_outputs_mean")
            # TODO: Outputs after seqlen will be zeroes...
            # But must implement CNN after this point so taking mean can be ignored 

    	# Add dropout
        with tf.name_scope("dropout"):
            self.outputs_drop = nn.dropout(self.lstm_outputs_mean, self.dropout_keep_prob)

        # Final (unnormalized) scores and predictions
        with tf.name_scope("output"):
            W = tf.get_variable(
                "W", 
                shape=[hidden_size, num_classes], 
                initializer=tf.contrib.layers.xavier_initializer())
            b = tf.Variable(tf.constant(0.1, shape=[num_classes]), name="b")
            
            l2_loss += nn.l2_loss(W)
            l2_loss += nn.l2_loss(b)
            
            self.scores = nn.xw_plus_b(self.outputs_drop, W, b, name="scores")
            self.predictions = tf.argmax(self.scores, reduction_indices=1, name="predictions")

        # Calculate mean cross-entropy loss
        with tf.name_scope("loss"):
            losses = nn.softmax_cross_entropy_with_logits(self.scores, self.input_y)
            self.loss = tf.reduce_mean(losses) + l2_reg_lambda * l2_loss

        # Accuracy
        with tf.name_scope("accuracy"):
            correct_predictions = tf.equal(self.predictions, tf.argmax(self.input_y, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")
