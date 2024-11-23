#!/bin/bash

# Default values
num_gpus=1
wandb_token=""

# Load environment variables from .env file if it exists
if [ -f .env ]; then
    export $(cat .env | xargs)
fi

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --num_gpus)
            num_gpus="$2"
            shift 2
            ;;
        --wandb_token)
            wandb_token="$2"
            shift 2
            ;;
        *)
            echo "Unknown argument: $1"
            exit 1
            ;;
    esac
done

# Run docker container with local directory mounted and execute training command
docker run --rm \
    --gpus all \
    -v $(pwd):/app \
    -w /app \
    -e WANDB_API_KEY=$wandb_token \
    --env-file .env \
    nvit:latest \
    torchrun --nnodes 1 --nproc_per_node $num_gpus --rdzv_endpoint=localhost:29501 nvit/train.py
