import tensorflow as tf
from tensorflow.nn import rnn, rnn_cell

# the data
# when preprocessing, must provide all be same sentence length (padded) but seqlen list bhi provide karni padegi with each batch

# Parameters
learning_rate = 0.001
training_iters = 100000
batch_size = 128
display_step = 10

# Network Parameters
n_input = 300 # word embedding size
n_steps = 30 # timesteps, ie max sentence length
n_hidden = 150 # hidden layer num of features
n_classes = 2 # total classes

# tf Graph input
x = tf.placeholder("float", [None, n_steps, n_input])
y = tf.placeholder("float", [None, n_classes])

# Define weights
weights = {
    # Hidden layer weights => 2*n_hidden because of foward + backward cells
    'hidden': tf.Variable(tf.random_normal([n_input, 2*n_hidden])),
    'out': tf.Variable(tf.random_normal([2*n_hidden, n_classes]))
}
biases = {
    'hidden': tf.Variable(tf.random_normal([2*n_hidden])),
    'out': tf.Variable(tf.random_normal([n_classes]))
}

def BiRNN_default(x, seqlen, weights, biases):

	# data preprocessing stuff

	# Define lstm cells with tensorflow
    # Forward direction cell
    lstm_fw_cell = rnn_cell.BasicLSTMCell(n_hidden, forget_bias=1.0)
    # Backward direction cell
    lstm_bw_cell = rnn_cell.BasicLSTMCell(n_hidden, forget_bias=1.0)

    outputs, _, _ = rnn.bidirectional_rnn(lstm_fw_cell, lstm_bw_cell, x, dtype=tf.float32, sequence_length=seqlen)

    # 'outputs' is a list of output at every timestep, we pack them in a Tensor
    # and change back dimension to [batch_size, n_step, n_input]
    # Something might break here because of the bidirectional concatenation...
    # But I don't really want to concatenate them this way
    outputs = tf.pack(outputs) # Packs a list of rank-R tensors into one rank-(R+1) tensor
    outputs = tf.transpose(outputs, [1, 0, 2]) # [1, 0, 2] is a permutation of the dimensions of outputs

    # Hack to build the indexing and retrieve the right output.
    batch_size = tf.shape(outputs)[0]
    # Start indices for each sample
    index = tf.range(0, batch_size) * n_steps + (seqlen - 1)
    # tf.range creates a sequence of integers

    # Indexing
    outputs = tf.gather(tf.reshape(outputs, [-1, n_hidden]), index)
    # Gather slices from params(reshaped output) according to indices

    # mean = tf.reduce_mean(outputs, 0)
    # Linear activation, using rnn inner loop last output
    # Must change to an average of all the output vectors
    return tf.matmul(outputs[-1], weights['out']) + biases['out']

def BiRNN_custom(x, seqlen, weights, biases):

	# data preprocessing stuff

	# Define lstm cells with tensorflow
    # Forward direction cell
    lstm_fw_cell = rnn_cell.BasicLSTMCell(n_hidden, forget_bias=1.0)
    # Backward direction cell
    lstm_bw_cell = rnn_cell.BasicLSTMCell(n_hidden, forget_bias=1.0)

    outputs_fw, _ = rnn.rnn(lstm_fw_cell, x, dtype=tf.float32, sequence_length=seqlen)
    
    tmp, _ = rnn.rnn(lstm_bw_cell, _reverse_seq(x, seqlen), dtype=tf.float32, sequence_length=seqlen)
    output_bw = _reverse_seq(tmp, seqlen)

    # with the seqlen stuff dynamic calculation stuff, as soon as time step exceeds seqlen, output at that timestep is just zeroes

    # how to concatenate/join the two outputs, along which dimension
    # how to use that to make a prediction
    # maybe the reduce_mean of vectors for each sentence then linear activation

	return    

pred = BiRNN(x, seqlen, weights, biases)

# Define loss and optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(pred, y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

# Evaluate model
correct_pred = tf.equal(tf.argmax(pred,1), tf.argmax(y,1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# Initializing the variables
init = tf.initialize_all_variables()

# Launch the graph
with tf.Session() as sess:
    sess.run(init)
    step = 1
    # Keep training until reach max iterations
    while step * batch_size < training_iters:
        batch_x, batch_y = #
        # Reshape data
        batch_x = batch_x.reshape( #(batch_size, n_steps, n_input))
        # Run optimization op (backprop)
        sess.run(optimizer, feed_dict={x: batch_x, y: batch_y})
        if step % display_step == 0:
            # Calculate batch accuracy
            acc = sess.run(accuracy, feed_dict={x: batch_x, y: batch_y})
            # Calculate batch loss
            loss = sess.run(cost, feed_dict={x: batch_x, y: batch_y})
            print "Iter " + str(step*batch_size) + ", Minibatch Loss= " + \
                  "{:.6f}".format(loss) + ", Training Accuracy= " + \
                  "{:.5f}".format(acc)
        step += 1
    print "Optimization Finished!"

    # Calculate accuracy for test set
    test_len = #
    test_data = #
    test_label = #
    print "Testing Accuracy:", \
        sess.run(accuracy, feed_dict={x: test_data, y: test_label})
