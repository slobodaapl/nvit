#!/bin/bash

num_gpus=$1

if [ -z "$num-gpus" ]; then
    num_gpus=1
fi

torchrun --nnodes 1 --nproc_per_node $num_gpus --rdzv_endpoint=localhost:29501 nvit/train.py
