# Overview
This repository contains code and results for a novel method to modify the word embeddings of a word in a sentence with its surrounding context using a biderectional Recurrent Neural Network (RNN). 

Given the word embeddings for each word in a sentence/sequence of words, it can be represented as a 2-D tensor of shape (`seq_len`, `embedding_dim`). Then the following steps can be performed to add infomation about the surrounding words to each embedding - 

1. Pass the embedding of each word sequentially into a forward-directional RNN (fRNN). For each sequential timestep, we obtain the hidden state of the fRNN, a tensor of shape (`hidden_size`). The hidden state encodes information about the current word and all the words previously encountered in the sequence. Our final output from the fRNN is a 2-D tensor of shape (`seq_len`, `hidden_size`). 
2. Pass the embedding of each word sequentially (after reversing the sequence of words) into a backward-directional RNN (bRNN). For each sequential timestep, we again obtain the hidden state of the bRNN, a tensor of shape (`hidden_size`). The hidden state encodes information about the current word and all the words previously encountered in the sequence. Our output is a 2-D tensor of shape (`seq_len`, `hidden_size`). This output is reversed again to obtain the final output of the bRNN. 
3. Concatenate the fRNN and bRNN outputs element-wise for each of the `seq_len` timesteps in the two outputs. The final output is another 2-D tensor of shape (`seq_len`, `hidden_size`).

**The fRNN and bRNN together form a bidirectional RNN. The difference between the final outputs of fRNN and bRNN is that at each timestep they are encoding information about two different sub-sequences (which are formed by splitting the sequence at the word at that timestep).**

![Bidirectional RNN](https://raw.githubusercontent.com/chaitjo/lstm-context-embeddings/master/res/bidirectional-rnn.png)

Concatenating these outputs at each timestep results in a tensor encoding information about the word at that timestep and all the words in the sequence to its left and right. Thus, the bidirectional RNN modifies an independent word's embedding to encode information from surrounding word embeddings in a given sequence.

# Implementation
The code in this repository implements the proposed model as a pre-processing layer before feeding it into a [Convolutional Neural Network for Sentence Classification](https://arxiv.org/abs/1408.5882) (Kim, 2014). Two implementations are provided to run experiments- one with [tensorflow](https://www.tensorflow.org/) and one with [tflearn](http://tflearn.org/) (A high-level API for tensorflow).

The cells used in the RNNs are the Long Short-term Memory (LSTM) cells, which are better at capturing long-term dependencies than vanilla RNN cells. This ensures our model doesn't just consider the nearest neighbours while modifying a word's embedding. 

The tensorflow version is built on top of [Denny Britz's implementation of Kim's CNN](https://github.com/dennybritz/cnn-text-classification-tf), and also allows loading pre-trained word embeddings like [word2vec](https://code.google.com/archive/p/word2vec/).

Training happens end-to-end in a supervised manner - the RNN layer is simply inserted as part of the existing model's architecture for text classification. 

Although both versions work exactly as intended, the repository currently contains results from experiments with the tflearn version only. More results will be added soon.

# Datasets
The dataset chosen for training and testing the tensorflow code is the [Pang & Lee Movie Reviews](http://www.cs.cornell.edu/people/pabo/movie-review-data/) dataset. For the tflearn version, we experiment on the [IMDb Movie Reviews Dataset](http://www.iro.umontreal.ca/~lisa/deep/data/imdb.pkl) by UMontreal. Classification involves detecting positive/negative
reviews in both cases.

# Experiments
The following three models were considered (Implementations can be found in `/tflearn`) -

1. A [baseline CNN model](https://raw.githubusercontent.com/chaitjo/lstm-context-embeddings/master/res/cnn-128.png) without the RNN layer, `embedding_dim = 128`, `num_filters = 128` **[ORANGE]**
2. The [proposed model](https://raw.githubusercontent.com/chaitjo/lstm-context-embeddings/master/res/lstm%2Bcnn-128.png), `embedding_dim = 128`, `rnn_hidden_size = 128`, `rnn_hidden_size = 128`, `num_filters = 128` **[PURPLE]**
2. The [proposed model with more capacity](https://raw.githubusercontent.com/chaitjo/lstm-context-embeddings/master/res/lstm%2Bcnn-300.png), `embedding_dim = 300`, `rnn_hidden_size = 300`, `num_filters = 150` **[BLUE]**

All models were trained with the following hyperparameters using the Adam optimizer - `num_epochs = 100`, `batch_size = 32`, `learning_rate = 0.001`. Ten percent of the data was held out for validation.

# Results
Training Accuracy - 
![Training Accuracy](https://raw.githubusercontent.com/chaitjo/lstm-context-embeddings/master/res/acc.png)

Training Loss -
![Training Loss](https://raw.githubusercontent.com/chaitjo/lstm-context-embeddings/master/res/loss.png)

Validation Accuracy -
![Validation Accuracy](https://raw.githubusercontent.com/chaitjo/lstm-context-embeddings/master/res/acc-val.png)

Vallidation Loss -
![Validation Loss](https://raw.githubusercontent.com/chaitjo/lstm-context-embeddings/master/res/loss-val.png)

Higher Validation Accuracy (~3%) and lower Validation Loss for the model compared to the baseline suggests that adding the bidirectional RNN pre-processing layer after the word embedding layer improve a generic text classification model's performance. However, more rigourous experimentation needs to be done to confirm this hyposthesis.

An unanswered question is whether the bump in accuracy is because the RNN layer actually adds contextual information to independent word embeddings or simply because of more processing and matrix multiplications by the network. However, adding more hidden units to the RNN layer does not lead to drastic changes in accuracy, suggesting that the former is true.

It is also extremely worrying to see the validation loss increasing instead of decreasing as training continues.

# Ideas and Next Steps
1. Visualizations of the word embeddings obtained after the RNN layer in a text sequence can be compared to their standard embeddings to confirm that their modification is due to their surrounding words and makes sense.
2. An `n` layer vanilla neural network for text classification can be compared to a model with the RNN layer followed by an `n-1` layer vanilla network. This should be a 'fairer fight' than CNN vs RNN+CNN.
3. Experiments can be carried out with static vs non-static word embeddings being passed to the RNN layer and initialization using pre-trained embeddings. 
3. Experiments can be carried out to determine the optimum depth of the RNN layer for different models. (Currently it is a single layer, but the concept can be easily extended for multilayer bidirectional RNNs.)
4. Cross validation should be performed to present results instead of randomly splitting the dataset.

# Usage
Tensorflow code is divided into `model.py`, which abstracts the model as a class and `train.py` which is used to train the model. It can be executed by running the `train.py` script (with optional hyperparameter flags) -
```
python train.py [--flag=0]
```

Tflearn code can be found in the `/tflearn` folder and can be run directly to start training (with optional hyperparameter flags) - 
```
python model.py [--flag=0]
```
