# Overview
This repository contains code and results for a novel method to enrich the word embeddings of a word in a sentence with its surrounding context using a biderectional Recurrent Neural Network (RNN). 

Given the word embeddings for each word in a sentence/sequence of words, it can be represented as a 2-D tensor of shape (`seq_len`, `embedding_dim`). Then the following steps can be performed to add infomation about the surrounding words to each embedding- 

1. Pass the embedding of each word sequentially into a forward-directional RNN (fRNN). For each sequential timestep, we obtain the hidden state of the fRNN, a tensor of shape (`hidden_size`). The hidden state encodes information about the current word and all the words previously encountered in the sequence. Our final output from the fRNN is a 2-D tensor of shape (`seq_len`, `hidden_size`). 
2. Pass the embedding of each word sequentially (after reversing the sequence of words) into a backward-directional RNN (bRNN). For each sequential timestep, we again obtain the hidden state of the bRNN, a tensor of shape (`hidden_size`). The hidden state encodes information about the current word and all the words previously encountered in the sequence. Our output is a 2-D tensor of shape (`seq_len`, `hidden_size`). This output is reversed again to obtain the final output of the bRNN. 
3. Concatenate the fRNN and bRNN outputs element-wise for each of the `seq_len` timesteps in the two outputs. The final output is another 2-D tensor of shape (`seq_len`, `hidden_size`).

**The fRNN and bRNN together form a bidirectional RNN. The difference between the final outputs of fRNN and bRNN is that at each timestep they are encoding information about two different sub-sequences (which are formed by splitting the sequence at the word at that timestep).**

![Bidirectional RNN](https://raw.githubusercontent.com/chaitjo/lstm-context-embeddings/master/res/bidirectional-rnn.png)

Concatenating these outputs at each timestep results in a tensor encoding information about the word at that timestep and all the words in the sequence to its left and right. Thus, the bidirectional RNN modifies an independent word's embedding to encode information from surrounding word embeddings in a given sequence.

# Experiments
The code in this repository implements the proposed model as a pre-processing layer before feeding it into a [Convolutional Neural Network for Sentence Classification](https://arxiv.org/abs/1408.5882) (Kim, 2014). Two implementations are provided to run experiments- one with [tensorflow](https://www.tensorflow.org/) and one with [tflearn](http://tflearn.org/) (A high-level API for tensorflow).

The cells used in the RNNs are the Long Short-term Memory (LSTM) cells, which are better at capturing long-term dependencies than vanilla RNN cells. This ensures our model doesn't just consider the nearest neighbours while modifying a word's embedding. 

(The tensorflow version also allows loading pre-trained word embeddings like [word2vec](https://code.google.com/archive/p/word2vec/).)

The dataset chosen for training and testing the tensorflow code is the [Pang & Lee Movie Reviews](http://www.cs.cornell.edu/people/pabo/movie-review-data/) dataset. For the tflearn version, we experiment on the [IMDb Movie Reviews Dataset](http://www.iro.umontreal.ca/~lisa/deep/data/imdb.pkl) by UMontreal. Classification involves detecting positive/negative
reviews in both cases.

Although both versions work exactly as intended, the repository currently contains results from experiments with the tflearn version only. More results will be added soon.

# Results
The following three models were considered-

1. A [baseline CNN model](https://raw.githubusercontent.com/chaitjo/lstm-context-embeddings/master/res/cnn-128.png) without the RNN layer, `embedding_dim = 128`, `num_filters = 128`
2. The [proposed model](https://raw.githubusercontent.com/chaitjo/lstm-context-embeddings/master/res/lstm%2Bcnn-128.png), `embedding_dim = 128`, `rnn_hidden_size = 128`, `rnn_hidden_size = 128`, `num_filters = 128`
2. The [proposed model with more capacity](https://raw.githubusercontent.com/chaitjo/lstm-context-embeddings/master/res/lstm%2Bcnn-300.png), `embedding_dim = 300`, `rnn_hidden_size = 300`, `num_filters = 150`

![Training Accuracy](https://raw.githubusercontent.com/chaitjo/lstm-context-embeddings/master/res/acc.png)
![Validation Accuracy](https://raw.githubusercontent.com/chaitjo/lstm-context-embeddings/master/res/acc-val.png)
![Training Loss](https://raw.githubusercontent.com/chaitjo/lstm-context-embeddings/master/res/loss.png)
![Validation Loss](https://raw.githubusercontent.com/chaitjo/lstm-context-embeddings/master/res/loss-val.png)

# Usage
- tf (model, cnn, alt_ver)
- tflearn

# Ideas
- gru/vanilla instead of lstm
- deeper rnn
- visualizing
- cross validation
