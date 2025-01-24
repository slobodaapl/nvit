#!/bin/bash

# Default values
num_gpus=1
visible_gpus="all"
env_file=".env"
remove_container=true  # Default to removing container after exit

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --num-gpus)
            num_gpus="$2"
            shift 2
            ;;
        --visible-gpus)
            visible_gpus="$2"
            shift 2
            ;;
        --env-file)
            env_file="$2"
            shift 2
            ;;
        -d|--detached)
            detached=true
            shift
            ;;
        --no-rm)
            remove_container=false
            shift
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

if ! [[ -f "$env_file" || -p "$env_file" ]]; then
    echo "Warning: Environment file $env_file does not exist or is not accessible"
fi

# Run docker container with local directory mounted and execute training command
docker run \
    ${remove_container:+--rm} \
    --gpus "\"device=$visible_gpus\"" \
    --shm-size=16gb \
    -v $(pwd):/app \
    -w /app \
    -e HOME=/app \
    -e TORCHINDUCTOR_CACHE_DIR=/app/.cache \
    -e NCCL_TIMEOUT=1200 \
    -e NCCL_DEBUG=INFO \
    --env-file "$env_file" \
    --user ${USER_ID}:${GROUP_ID} \
    ${detached:+-d} \
    nvit:latest \
    torchrun --nnodes 1 --nproc_per_node $num-gpus --rdzv_endpoint=localhost:29501 nvit/train.py
