#!/bin/bash
python tflearn/model.py
python tflearn/cnn.py
python tflearn/model.py --embedding_dim=300 --rnn_hidden_size=150 --num_filters=100
python tflearn/cnn.py --embedding_dim=300 --num_filters=100
python train.py --word2vec=GoogleNews-vectors-negative300.bin
python cnn-text-classification/train.py --word2vec=GoogleNews-vectors-negative300.bin

