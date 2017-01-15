# Overview
This repository contains code and results for a novel method to enrich the word embeddings of a word in a sentence with its surrounding context using a biderectional Recurrent Neural Network (RNN). 

Given the word embeddings for each word in a sentence/sequence of words, we can represent it as a 2-D tensor of shape (`seq_len`, `embedding_dim`). We can then perform the following steps to add infomation about the surrounding words to each embedding- 

1. Pass the embedding of each word sequentially into a forward-directional RNN (fRNN). For each sequential timestep, we obtain the hidden state of the fRNN, a tensor of shape (`hidden_size`). The hidden state encodes information about the current word and all the words previously encountered in the sequence. Our final output from the fRNN is a 2-D tensor of shape (`seq_len`, `hidden_size`). 
2. Pass the embedding of each word sequentially (after reversing the sequence of words) into a backward-directional RNN (bRNN). For each sequential timestep, we again obtain the hidden state of the bRNN, a tensor of shape (`hidden_size`). The hidden state encodes information about the current word and all the words previously encountered in the sequence. Our output from the bRNN is a 2-D tensor of shape (`seq_len`, `hidden_size`). This output is reversed again to obtain the final output of the bRNN. 
3. Concatenate the fRNN and bRNN outputs element-wise for each of the `seq_len` timesteps in the two outputs. The output is another 2-D tensor of shape (`seq_len`, `hidden_size`).

**The difference between the final outputs of fRNN and bRNN are that at each timestep, they are encoding information about two different sub-sequences which are formed by splitting the sequence at the word at that timestep.**


# Usage
- tf (model, cnn, alt_ver)
- tflearn

# Results
- standard comparision w cnn
- different datasets
- tf and tflearn results
- make word embs static

# Ideas
- gru instead of lstm
- visualizing
- cross validation
