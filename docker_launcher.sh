#!/bin/bash

# Default values
num_gpus=1

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
        *)
            echo "Unknown argument: $1"
            exit 1
            ;;
    esac
done

# Get current user's UID and GID
USER_ID=$(id -u)
GROUP_ID=$(id -g)

# Run docker container with local directory mounted and execute training command
docker run --rm \
    --gpus all \
    -v $(pwd):/app \
    -w /app \
    -e HOME=/app \
    -e TORCHINDUCTOR_CACHE_DIR=/app/.cache \
    --env-file .env \
    --user ${USER_ID}:${GROUP_ID} \
    nvit:latest \
    torchrun --nnodes 1 --nproc_per_node $num_gpus --rdzv_endpoint=localhost:29501 nvit/train.py
