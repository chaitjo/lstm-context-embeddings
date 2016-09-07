#!/bin/bash
python train.py 2>&1 | tee outputs/output-normal.txt
python train.py 2>&1 --dropout_keep_prob=0.7 | tee outputs/output-drp0.7.txt
python train.py 2>&1 --dropout_keep_prob=0.3 | tee outputs/output-drp0.3.txt
python train.py 2>&1 --l2_reg_lambda=0.15 | tee outputs/output-l20.15.txt
python train.py 2>&1 --l2_reg_lambda=0.5 | tee outputs/output-l20.5.txt
python train.py 2>&1 --num_filters=150 | tee outputs/output-numfil150.txt
