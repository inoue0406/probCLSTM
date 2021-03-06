#!/bin/bash

# python env
#conda activate pytorch

case="result_20190501_tst3_vae"

# running script for VAE
python ../src/main_VAE_jma.py --data_path ../data/data_h5/ --train_path ../data/train_simple_JMARadar.csv --valid_path ../data/valid_simple_JMARadar.csv --test --eval_threshold 0.5 --test_path ../data/valid_simple_JMARadar.csv --result_path $case --tdim_use 12 --learning_rate 0.001 --batch_size 20 --n_epochs 10 --n_threads 4 --checkpoint 10 --hidden_channels 8 --kernel_size 3 --optimizer adam
