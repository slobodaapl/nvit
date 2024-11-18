#!/bin/bash

# read number of GPUs from args on command line, if present, otherwise default to 8
num_gpus=$1

if [ -z "$num_gpus" ]; then
    num_gpus=1
fi

torchrun --nnodes 1 --nproc_per_node $num_gpus --rdzv_endpoint=localhost:29501 train.py
