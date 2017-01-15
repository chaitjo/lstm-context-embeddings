#!/bin/bash
python train.py --embedding_dim=50 --hidden_dim=30 --num_filters=20
python cnn-model/train.py --embedding_dim=50 --num_filters=20
python train.py --num_epochs=30 --word2vec=GoogleNews-vectors-negative300.bin
python cnn-model/train.py --num_epochs=30 --word2vec=GoogleNews-vectors-negative300.bin
