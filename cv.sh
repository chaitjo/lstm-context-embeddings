#!/bin/bash
cd ~/lstm-context-embeddings
python cnn-text-classification-tf/train.py 2>&1 | tee outputs/output-cnn.txt
python train.py 2>&1 | tee outputs/output-normal.txt
for i in `seq 0 9`;
do
	python train_cv.py --cv_index=$i 2>&1 | tee outputs/output-cv-$i.txt
done

